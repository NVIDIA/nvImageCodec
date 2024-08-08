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
#include <nvjpeg2k.h>

const char* getErrorString(nvjpeg2kStatus_t);

class NvJpeg2kException : public std::exception
{
  public:
    static NvJpeg2kException FromNvJpeg2kError(nvjpeg2kStatus_t status, const std::string& where);
    static NvJpeg2kException FromCUDAError(cudaError_t status, const std::string& where);

    inline virtual ~NvJpeg2kException() throw() {}

    virtual const char* what() const throw() { return info_.c_str(); }
    const std::string& info() const { return info_; }
    nvimgcodecStatus_t nvimgcodecStatus() const { return status_; }

  private:
    NvJpeg2kException() = default;
    nvimgcodecStatus_t status_ = NVIMGCODEC_STATUS_SUCCESS;
    std::string info_;
};
