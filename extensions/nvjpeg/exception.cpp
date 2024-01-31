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

const std::map<nvjpegStatus_t, nvimgcodecStatus_t> nvjpeg_status_to_nvimgcodec_error_map = 
    {{NVJPEG_STATUS_SUCCESS, NVIMGCODEC_STATUS_SUCCESS},    
    {NVJPEG_STATUS_NOT_INITIALIZED, NVIMGCODEC_STATUS_EXTENSION_NOT_INITIALIZED},
    {NVJPEG_STATUS_INVALID_PARAMETER, NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER},
    {NVJPEG_STATUS_INTERNAL_ERROR, NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR},
    {NVJPEG_STATUS_INCOMPLETE_BITSTREAM, NVIMGCODEC_STATUS_EXTENSION_INCOMPLETE_BITSTREAM},
    {NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_IMPLEMENTATION_NOT_SUPPORTED},
    {NVJPEG_STATUS_EXECUTION_FAILED, NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED},
    {NVJPEG_STATUS_BAD_JPEG, NVIMGCODEC_STATUS_EXTENSION_BAD_CODE_STREAM},
    {NVJPEG_STATUS_ARCH_MISMATCH, NVIMGCODEC_STATUS_EXTENSION_ARCH_MISMATCH},
    {NVJPEG_STATUS_ALLOCATOR_FAILURE, NVIMGCODEC_STATUS_EXTENSION_ALLOCATOR_FAILURE},
    {NVJPEG_STATUS_JPEG_NOT_SUPPORTED, NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED}};

namespace StatusStrings {
const std::string sExtNotInit        = "nvjpeg extension: not initialized";
const std::string sExtInvalidParam   = "nvjpeg extension: Invalid parameter";
const std::string sExtBadJpeg        = "nvjpeg extension: Bad jpeg";
const std::string sExtJpegUnSupp     = "nvjpeg extension: Jpeg not supported";
const std::string sExtAllocFail      = "nvjpeg extension: allocator failure";
const std::string sExtArchMis        = "nvjpeg extension: arch mismatch";
const std::string sExtIntErr         = "nvjpeg extension: internal error";
const std::string sExtImplNA         = "nvjpeg extension: implementation not supported";
const std::string sExtIncBits        = "nvjpeg extension: incomplete bitstream";
const std::string sExtExeFailed      = "nvjpeg extension: execution failed";
const std::string sExtCudaCallError  = "nvjpeg extension: cuda call error";
} // namespace StatusStrings

const char* getErrorString(nvjpegStatus_t eStatus_)
{
    switch (eStatus_) {        
    case NVJPEG_STATUS_ALLOCATOR_FAILURE:
        return StatusStrings::sExtAllocFail.c_str();
    case NVJPEG_STATUS_ARCH_MISMATCH:
        return StatusStrings::sExtArchMis.c_str();
    case NVJPEG_STATUS_BAD_JPEG:
        return StatusStrings::sExtBadJpeg.c_str();      
    case NVJPEG_STATUS_JPEG_NOT_SUPPORTED:
        return StatusStrings::sExtJpegUnSupp.c_str();
    case NVJPEG_STATUS_EXECUTION_FAILED:
        return StatusStrings::sExtExeFailed.c_str();
    case NVJPEG_STATUS_INCOMPLETE_BITSTREAM:
        return StatusStrings::sExtIncBits.c_str();
    case NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED:
        return StatusStrings::sExtImplNA.c_str();
    case NVJPEG_STATUS_INTERNAL_ERROR:
        return StatusStrings::sExtIntErr.c_str();
    case NVJPEG_STATUS_INVALID_PARAMETER:
        return StatusStrings::sExtInvalidParam.c_str();
    case NVJPEG_STATUS_NOT_INITIALIZED:
        return StatusStrings::sExtNotInit.c_str();
    default:
        return StatusStrings::sExtIntErr.c_str();

    }
}

NvJpegException::NvJpegException(nvjpegStatus_t eStatus, const std::string& rMessage, const std::string& rLoc)
    : eStatus_(eStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    ;
}

NvJpegException::NvJpegException(cudaError_t eCudaStatus, const std::string& rMessage, const std::string& rLoc)
    : eCudaStatus_(eCudaStatus)
    , sMessage_(rMessage)
    , sLocation_(rLoc)
{
    isCudaStatus_ = true;;
}

const char* NvJpegException::what() const throw()
{
    if (isCudaStatus_)
        return StatusStrings::sExtCudaCallError.c_str();
    else
        return getErrorString(eStatus_);
};

nvjpegStatus_t NvJpegException::status() const
{    
    return eStatus_;
}

cudaError_t NvJpegException::cudaStatus() const
{
    return eCudaStatus_;
}

const char* NvJpegException::message() const
{
    return sMessage_.c_str();
}

const char* NvJpegException::where() const
{
    return sLocation_.c_str();
}

std::string NvJpegException::info() const throw()
{   
    std::string info(getErrorString(eStatus_)); 
    if (isCudaStatus_)
        info = StatusStrings::sExtCudaCallError;        
    return info + " " + sLocation_;   
}

nvimgcodecStatus_t NvJpegException::nvimgcodecStatus() const
{    
    if (isCudaStatus_)
        return NVIMGCODEC_STATUS_EXTENSION_CUDA_CALL_ERROR;
    else
    {
        auto it = nvjpeg_status_to_nvimgcodec_error_map.find(eStatus_);
        if (it != nvjpeg_status_to_nvimgcodec_error_map.end())
            return it->second;
        else
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}
