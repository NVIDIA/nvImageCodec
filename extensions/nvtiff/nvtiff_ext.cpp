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

#include "nvtiff_ext.h"
#include "error_handling.h"
#include "log.h"
#include "cuda_decoder.h"
#include "cuda_encoder.h"

namespace nvtiff {

struct NvTiffImgCodecsExtension
{
  public:
    explicit NvTiffImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , nvtiff_decoder_(framework)
        , nvtiff_encoder_(framework)
    {
        framework->registerDecoder(framework->instance, nvtiff_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        framework->registerEncoder(framework->instance, nvtiff_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
    }
    ~NvTiffImgCodecsExtension() 
    { 
        framework_->unregisterDecoder(framework_->instance, nvtiff_decoder_.getDecoderDesc()); 
        framework_->unregisterEncoder(framework_->instance, nvtiff_encoder_.getEncoderDesc()); 
    }

    static nvimgcodecStatus_t nvtiff_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvtiff_ext", "nvtiff_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new NvTiffImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvtiff_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<NvTiffImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvtiff_ext", "nvtiff_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    NvTiffCudaDecoderPlugin nvtiff_decoder_;
    NvTiffCudaEncoderPlugin nvtiff_encoder_;
};

} // namespace nvtiff

  // clang-format off
nvimgcodecExtensionDesc_t nvtiff_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvtiff_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER,

    nvtiff::NvTiffImgCodecsExtension::nvtiff_extension_create,
    nvtiff::NvTiffImgCodecsExtension::nvtiff_extension_destroy
};
// clang-format on  

nvimgcodecStatus_t get_nvtiff_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvtiff_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

