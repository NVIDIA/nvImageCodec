/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <map>
#include "exception.h"
#include "log.h"

#define XM_CHECK_NULL(ptr)                                                 \
    {                                                                      \
        if (!ptr)                                                          \
            FatalError(NVJPEG2K_STATUS_INVALID_PARAMETER, "null pointer"); \
    }

#define XM_CHECK_CUDA(call)                                    \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << std::to_string(_e) << "'"; \
            FatalError(_e, _error.str());                      \
        }                                                      \
    }

#define XM_CHECK_NVJPEG2K(call)                                    \
    {                                                              \
        nvjpeg2kStatus_t _e = (call);                              \
        if (_e != NVJPEG2K_STATUS_SUCCESS) {                       \
            std::stringstream _error;                              \
            _error << "nvjpeg2k Runtime failure: '#" << std::to_string(_e) << "'"; \
            FatalError(_e, _error.str());                          \
        }                                                          \
    }

#define XM_NVJPEG2K_D_LOG_DESTROY(call)                            \
    {                                                              \
        nvjpeg2kStatus_t _e = (call);                              \
        if (_e != NVJPEG2K_STATUS_SUCCESS) {                       \
            std::stringstream _error;                              \
            _error << "nvjpeg2k Runtime failure: '#" << std::to_string(_e) << "'"; \
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, _error.str());    \
        }                                                          \
    }

#define XM_NVJPEG2K_E_LOG_DESTROY(call)                            \
    {                                                              \
        nvjpeg2kStatus_t _e = (call);                              \
        if (_e != NVJPEG2K_STATUS_SUCCESS) {                       \
            std::stringstream _error;                              \
            _error << "nvjpeg2k Runtime failure: '#" << std::to_string(_e) << "'"; \
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, _error.str());    \
        }                                                          \
    }

#define XM_CUDA_LOG_DESTROY(call)                               \
    {                                                           \
        cudaError_t _e = (call);                                \
        if (_e != cudaSuccess) {                                \
            std::stringstream _error;                           \
            _error << "CUDA Runtime failure: '#" << std::to_string(_e) << "'";  \
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, _error.str()); \
        }                                                       \
    }
