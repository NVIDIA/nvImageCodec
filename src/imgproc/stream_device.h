/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>

namespace nvimgcodec {

CUdevice get_stream_device(cudaStream_t stream);

// Check if the device (and current driver) associated with the given stream
// can use asynchronous memory allocations and deallocations
bool can_use_async_mem_ops(cudaStream_t stream);

// Retrieve the device id associated with the cudaStream_t
int get_stream_device_id(cudaStream_t stream);

} // namespace nvimgcodec