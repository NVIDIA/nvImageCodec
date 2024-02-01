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

#include "exception.h"

const std::map<nvjpeg2kStatus_t, nvimgcodecStatus_t> nvjpeg2k_to_nvimgcodec_error_map = 
    {{NVJPEG2K_STATUS_SUCCESS, NVIMGCODEC_STATUS_SUCCESS},    
    {NVJPEG2K_STATUS_NOT_INITIALIZED, NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED},
    {NVJPEG2K_STATUS_INVALID_PARAMETER, NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER},
    {NVJPEG2K_STATUS_INTERNAL_ERROR, NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR},    
    {NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED},
    {NVJPEG2K_STATUS_EXECUTION_FAILED, NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED},
    {NVJPEG2K_STATUS_BAD_JPEG, NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM},    
    {NVJPEG2K_STATUS_ALLOCATOR_FAILURE, NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE},
    {NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED}};

namespace StatusStrings {
const std::string sExtNotInit        = "nvjpeg2k extension: not initialized";
const std::string sExtInvalidParam   = "nvjpeg2k extension: Invalid parameter";
const std::string sExtBadJpeg        = "nvjpeg2k extension: Bad jpeg";
const std::string sExtJpegUnSupp     = "nvjpeg2k extension: Jpeg not supported";
const std::string sExtAllocFail      = "nvjpeg2k extension: allocator failure";
const std::string sExtIntErr         = "nvjpeg2k extension: internal error";
const std::string sExtImplNA         = "nvjpeg2k extension: implementation not supported";
const std::string sExtExeFailed      = "nvjpeg2k extension: execution failed";
const std::string sExtCudaCallError  = "nvjpeg2k extension: cuda call error";
} // namespace StatusStrings

const char* getErrorString(nvjpeg2kStatus_t eStatus_)
{
    switch (eStatus_) {        
    case NVJPEG2K_STATUS_ALLOCATOR_FAILURE:
        return StatusStrings::sExtAllocFail.c_str();
    case NVJPEG2K_STATUS_BAD_JPEG:
        return StatusStrings::sExtBadJpeg.c_str();      
    case NVJPEG2K_STATUS_JPEG_NOT_SUPPORTED:
        return StatusStrings::sExtJpegUnSupp.c_str();
    case NVJPEG2K_STATUS_EXECUTION_FAILED:
        return StatusStrings::sExtExeFailed.c_str();
    case NVJPEG2K_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return StatusStrings::sExtImplNA.c_str();
    case NVJPEG2K_STATUS_INTERNAL_ERROR:
        return StatusStrings::sExtIntErr.c_str();
    case NVJPEG2K_STATUS_INVALID_PARAMETER:
        return StatusStrings::sExtInvalidParam.c_str();
    case NVJPEG2K_STATUS_NOT_INITIALIZED:
        return StatusStrings::sExtNotInit.c_str();
    default:
        return StatusStrings::sExtIntErr.c_str();

    }
}

NvJpeg2kException::NvJpeg2kException(nvjpeg2kStatus_t eStatus, const std::string& rMessage, const std::string& rLoc)
    : eStatus_(eStatus)
    , eCudaStatus_(cudaSuccess)
    , isCudaStatus_(false)
    , sMessage_(rMessage)
    , sLocation_(rLoc)

{
}

NvJpeg2kException::NvJpeg2kException(cudaError_t eCudaStatus, const std::string& rMessage, const std::string& rLoc)
    : eStatus_(NVJPEG2K_STATUS_SUCCESS)
    , eCudaStatus_(eCudaStatus)
    , isCudaStatus_(true)
    , sMessage_(rMessage)
    , sLocation_(rLoc)

{
}

const char* NvJpeg2kException::what() const throw()
{
    if (isCudaStatus_)
        return StatusStrings::sExtCudaCallError.c_str();
    else
        return getErrorString(eStatus_);
};

nvjpeg2kStatus_t NvJpeg2kException::status() const
{    
    return eStatus_;
}

cudaError_t NvJpeg2kException::cudaStatus() const
{
    return eCudaStatus_;
}

const char* NvJpeg2kException::message() const
{
    return sMessage_.c_str();
}

const char* NvJpeg2kException::where() const
{
    return sLocation_.c_str();
}

std::string NvJpeg2kException::info() const throw()
{   
    std::string info(getErrorString(eStatus_)); 
    if (isCudaStatus_)
        info = StatusStrings::sExtCudaCallError;        
    return info + " " + sLocation_;   
}

nvimgcodecStatus_t NvJpeg2kException::nvimgcodecStatus() const
{    
    if (isCudaStatus_)
        return NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR;
    else
    {
        auto it = nvjpeg2k_to_nvimgcodec_error_map.find(eStatus_);
        if (it != nvjpeg2k_to_nvimgcodec_error_map.end())
            return it->second;
        else
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}
