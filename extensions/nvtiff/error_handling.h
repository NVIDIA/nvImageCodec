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
#include "log.h"

#define XM_CHECK_NULL(ptr)                            \
    {                                                 \
        if (!ptr)                                     \
            throw std::runtime_error("null pointer"); \
    }

#define XM_CHECK_NVTIFF(call)                                             \
    {                                                                     \
        nvtiffStatus_t _e = (call);                                       \
        if (_e != NVTIFF_STATUS_SUCCESS) {                                \
            throw std::runtime_error("nvTiff call failed with code "      \
                 + std::to_string(_e) + ": " #call);                      \
        }                                                                 \
    }

#define XM_CHECK_CUDA(call)                                                    \
    {                                                                          \
        cudaError_t _e = (call);                                               \
        if (_e != cudaSuccess) {                                               \
            cudaGetLastError(); /* clean that error for any further calls */   \
            throw std::runtime_error("Cuda call failed with code "             \
                 + std::to_string(_e) + ": " #call);                           \
        }                                                                      \
    }

#define XM_NVTIFF_LOG_DESTROY(call)                                                      \
    {                                                                                    \
        nvtiffStatus_t _e = (call);                                                      \
        if (_e != NVTIFF_STATUS_SUCCESS) {                                               \
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "nvTiff call failed with code " \
                 + std::to_string(_e) + ": " #call);                                     \
        }                                                                                \
    }

#define XM_CUDA_LOG_DESTROY(call)                                                        \
    {                                                                                    \
        cudaError_t _e = (call);                                                         \
        if (_e != cudaSuccess) {                                                         \
            cudaGetLastError(); /* clean that error for any further calls */             \
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Cuda call failed with code "   \
                + std::to_string(_e) + ": " #call);                                      \
        }                                                                                \
    }
