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

#pragma once

#include <nvimgcodec.h>
#include <memory>

namespace nvimgcodec {

struct PinnedBuffer
{
    explicit PinnedBuffer(const nvimgcodecExecutionParams_t* exec_params = nullptr);
    void resize(size_t new_size, cudaStream_t new_stream);
    void alloc_impl(size_t new_size, cudaStream_t new_stream);
    nvimgcodecPinnedAllocator_t* allocator;
    std::shared_ptr<void> d_ptr;
    void* data = nullptr;
    size_t capacity = 0;
    size_t size = 0;
    cudaStream_t stream = 0;
};

} // end nvimgcodec