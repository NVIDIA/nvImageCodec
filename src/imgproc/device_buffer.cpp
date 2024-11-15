/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "imgproc/device_buffer.h"
#include <cassert>
#include "imgproc/device_guard.h"
#include "imgproc/stream_device.h"
#include "imgproc/exception.h"

namespace nvimgcodec {

DeviceBuffer::DeviceBuffer(const nvimgcodecExecutionParams_t* exec_params)
    : allocator(exec_params ? exec_params->device_allocator : nullptr)
{
}

void DeviceBuffer::resize(size_t new_size, cudaStream_t new_stream)
{
    if (new_size <= capacity) {
        // TODO(janton): could do cudaStreamWaitEvent
        if (new_stream != stream)
            CHECK_CUDA(cudaStreamSynchronize(stream));
        size = new_size;
        return;
    }
    alloc_impl(new_size, new_stream);
}

void DeviceBuffer::alloc_impl(size_t new_size, cudaStream_t new_stream)
{
    if (allocator && allocator->device_malloc) {
        allocator->device_malloc(allocator->device_ctx, &data, new_size, new_stream);
        size = new_size;
        capacity = new_size;
        stream = new_stream;
        d_ptr.reset(data, [allocator = this->allocator, new_size, new_stream](void* ptr) {
            allocator->device_free(allocator->device_ctx, ptr, new_size, new_stream);
        });
    } else if (nvimgcodec::can_use_async_mem_ops(new_stream)) {
        CHECK_CUDA(cudaMallocAsync(&data, new_size, new_stream));
        size = new_size;
        capacity = new_size;
        stream = new_stream;
        d_ptr.reset(data, [new_stream](void* ptr) {
            CHECK_CUDA(cudaFreeAsync(ptr, new_stream));
        });
    } else {
        nvimgcodec::DeviceGuard device_guard(nvimgcodec::get_stream_device_id(new_stream));
        CHECK_CUDA(cudaMalloc(&data, new_size));
        size = new_size;
        capacity = new_size;
        stream = new_stream;
        d_ptr.reset(data, cudaFree);
    }
}


} // end nvimgcodec