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

#define CHECK_CUDA(call)                                                                                       \
    {                                                                                                          \
        cudaError_t _e = (call);                                                                               \
        if (_e != cudaSuccess) {                                                                               \
            std::stringstream _error;                                                                          \
            _error << "CUDA Runtime failure: '#" << std::to_string(_e) << "' at " << __FILE__ << ":" << __LINE__; \
            throw std::runtime_error(_error.str());                                                            \
        }                                                                                                      \
    }

#define CHECK_NVIMGCODEC(call)                                   \
    {                                                           \
        nvimgcodecStatus_t _e = (call);                          \
        if (_e != NVIMGCODEC_STATUS_SUCCESS) {                   \
            std::stringstream _error;                           \
            _error << "nvImageCodec failure: '#" << std::to_string(_e) << "'"; \
            throw std::runtime_error(_error.str());             \
        }                                                       \
    }
    
void check_cuda_buffer(const void* ptr);
