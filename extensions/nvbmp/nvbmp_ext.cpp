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

#include <nvimgcodec.h>
#include "error_handling.h"
#include "log.h"
#include "decoder.h"
#include "encoder.h"

namespace nvbmp {

struct BmpImgCodecsExtension
{
  public:
    explicit BmpImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , nvbmp_encoder_(framework)
        , nvbmp_decoder_(framework)

    {
        framework->registerEncoder(framework->instance, nvbmp_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_VERY_LOW);
        framework->registerDecoder(framework->instance, nvbmp_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_VERY_LOW);
    }
    ~BmpImgCodecsExtension()
    {
        framework_->unregisterEncoder(framework_->instance, nvbmp_encoder_.getEncoderDesc());
        framework_->unregisterDecoder(framework_->instance, nvbmp_decoder_.getDecoderDesc());
    }

    static nvimgcodecStatus_t nvbmp_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvbmp_ext", "nvbmp_extension_create");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new BmpImgCodecsExtension(framework));

        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvbmp_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<BmpImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvbmp_ext", "nvbmp_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    NvBmpEncoderPlugin nvbmp_encoder_;
    NvBmpDecoderPlugin nvbmp_decoder_;
};

} // namespace nvbmp


// clang-format off
nvimgcodecExtensionDesc_t nvbmp_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvbmp_extension",  
    NVIMGCODEC_VER,       
    NVIMGCODEC_VER,

    nvbmp::BmpImgCodecsExtension::nvbmp_extension_create,
    nvbmp::BmpImgCodecsExtension::nvbmp_extension_destroy
};
// clang-format on  

nvimgcodecStatus_t get_nvbmp_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvbmp_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
