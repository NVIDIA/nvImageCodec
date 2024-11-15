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

#include "imgproc/pinned_buffer.h"
#include <cassert>
#include "imgproc/device_guard.h"
#include "imgproc/stream_device.h"
#include "imgproc/exception.h"

namespace nvimgcodec {


PinnedBuffer::PinnedBuffer(const nvimgcodecExecutionParams_t* exec_params)
    : allocator(exec_params ? exec_params->pinned_allocator : nullptr)
{
}

void PinnedBuffer::resize(size_t new_size, cudaStream_t new_stream)
{
    if (new_size <= capacity) {
        // no need to sync. Pinned buffers are always synced after allocation
        stream = new_stream;
        size = new_size;
        return;
    }
    alloc_impl(new_size, new_stream);
}

void PinnedBuffer::alloc_impl(size_t new_size, cudaStream_t new_stream)
{
    if (allocator && allocator->pinned_malloc) {
        allocator->pinned_malloc(allocator->pinned_ctx, &data, new_size, new_stream);
        size = new_size;
        capacity = new_size;
        stream = new_stream;
        d_ptr.reset(data, [allocator = this->allocator, new_size, new_stream](void* ptr) {
            allocator->pinned_free(allocator->pinned_ctx, ptr, new_size, new_stream);
        });
        CHECK_CUDA(cudaStreamSynchronize(new_stream));  // want to access on host
    } else {
        nvimgcodec::DeviceGuard device_guard(nvimgcodec::get_stream_device_id(new_stream));
        CHECK_CUDA(cudaMallocHost(&data, new_size));
        size = new_size;
        capacity = new_size;
        stream = new_stream;
        d_ptr.reset(data, cudaFreeHost);
    }
}

} // end nvimgcodec