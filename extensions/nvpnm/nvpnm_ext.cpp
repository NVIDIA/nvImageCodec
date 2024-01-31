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

#include "nvpnm_ext.h"
#include "error_handling.h"
#include "log.h"
#include "encoder.h"

namespace nvpnm {

struct PnmImgCodecsExtension
{
  public:
    explicit PnmImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , nvpnm_encoder_(framework)
    {
        framework->registerEncoder(framework->instance, nvpnm_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_VERY_LOW);
    }
    ~PnmImgCodecsExtension() { framework_->unregisterEncoder(framework_->instance, nvpnm_encoder_.getEncoderDesc()); }

    static nvimgcodecStatus_t nvpnm_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvpnm_ext", "nvpnm_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new PnmImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvpnm_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<PnmImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvpnm_ext", "nvpnm_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    NvPnmEncoderPlugin nvpnm_encoder_;
};

} // namespace nvpnm

  // clang-format off
nvimgcodecExtensionDesc_t nvpnm_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvpnm_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_EXT_API_VER,

    nvpnm::PnmImgCodecsExtension::nvpnm_extension_create,
    nvpnm::PnmImgCodecsExtension::nvpnm_extension_destroy
};
// clang-format on  

nvimgcodecStatus_t get_nvpnm_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvpnm_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

