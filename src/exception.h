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

#include <iostream>
#include <sstream>
#include <string>
#include <cuda_runtime.h>

namespace nvimgcodec {

enum Status
{
    STATUS_OK = 0,
    NOT_VALID_FORMAT_STATUS = 1,
    UNSUPPORTED_FORMAT_STATUS = 2,
    BAD_FORMAT_STATUS = 3,
    PARSE_STATUS = 4,
    ALLOCATION_ERROR = 5,
    INTERNAL_ERROR = 6,
    INVALID_PARAMETER = 7,
    CUDA_CALL_ERROR = 8,
    BAD_STATE = 9
};

const char* getErrorString(Status);

class Exception : public std::exception
{
  public:
    explicit Exception(
        Status eStatus, const std::string& rMessage = "", const std::string& rLoc = "");

    inline virtual ~Exception() throw() { ; }

    virtual const char* what() const throw();

    Status status() const;

    const char* message() const;

    const char* where() const;

  private:
    Exception();
    Status eStatus_;
    std::string sMessage_;
    std::string sLocation_;
};

#define FatalError(status, message)                     \
    {                                                   \
        std::stringstream _where;                       \
        _where << "At " << __FILE__ << ":" << __LINE__; \
        throw Exception(status, message, _where.str()); \
    }

#define CHECK_NULL(ptr)                                    \
    {                                                      \
        if (!ptr)                                          \
            FatalError(INVALID_PARAMETER, "null pointer"); \
    }


#define CHECK_CUDA(call)                                       \
    {                                                          \
        cudaError_t _e = (call);                               \
        if (_e != cudaSuccess) {                               \
            std::stringstream _error;                          \
            _error << "CUDA Runtime failure: '#" << std::to_string(_e) << "'"; \
            FatalError(CUDA_CALL_ERROR, _error.str());         \
        }                                                      \
    }

#define LOG_CUDA_ERROR(call)                                            \
    {                                                                   \
        cudaError_t _e = (call);                                        \
        if (_e != cudaSuccess) {                                        \
            std::stringstream _error;                                   \
            std::cerr << "CUDA Runtime failure: '#" << std::to_string(_e) << std::endl; \
        }                                                               \
    }

 #define CHECK_CU(call)                                                                             \
     {                                                                                              \
         CUresult _e = (call);                                                                      \
         if (_e != CUDA_SUCCESS) {                                                                  \
             std::stringstream _error;                                                              \
             _error << "CUDA Driver API failure: '#" << std::to_string(_e) << "'";                  \
             FatalError(CUDA_CALL_ERROR, _error.str());                                             \
         }                                                                                          \
     }

} // namespace nvimgcodec
