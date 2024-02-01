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
#include <map>
#include <nvimgcodec.h>
#include <nvjpeg.h>

const char* getErrorString(nvjpegStatus_t);

class NvJpegException : public std::exception
{
  public:
    explicit NvJpegException(
        nvjpegStatus_t eStatus, const std::string& rMessage = "", const std::string& rLoc = "");
    explicit NvJpegException(
        cudaError_t eStatus, const std::string& rMessage = "", const std::string& rLoc = "");

    inline virtual ~NvJpegException() throw() { ; }

    virtual const char* what() const throw();

    nvjpegStatus_t status() const;
    
    cudaError_t cudaStatus() const;

    const char* message() const;

    const char* where() const;

    std::string info() const throw();

    nvimgcodecStatus_t nvimgcodecStatus() const;

  private:
    NvJpegException();
    nvjpegStatus_t eStatus_;
    cudaError_t eCudaStatus_;
    bool isCudaStatus_;
    std::string sMessage_;
    std::string sLocation_;
};

#define FatalError(status, message)                     \
    {                                                   \
        std::stringstream _where;                       \
        _where << "At " << __FILE__ << ":" << __LINE__; \
        throw NvJpegException(status, message, _where.str()); \
    }     
