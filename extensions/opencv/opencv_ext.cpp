/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "opencv_encoder.h"
#include "opencv_decoder.h"
#include "log.h"
#include "error_handling.h"

namespace opencv {

struct OpenCVImgCodecsExtension
{
  public:
    explicit OpenCVImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , decoders_{
            OpenCVDecoderPlugin("jpeg", framework),
            OpenCVDecoderPlugin("jpeg2k", framework),
            OpenCVDecoderPlugin("png", framework),
            OpenCVDecoderPlugin("bmp", framework),
            OpenCVDecoderPlugin("pnm", framework),
            OpenCVDecoderPlugin("tiff", framework),
            OpenCVDecoderPlugin("webp", framework)
        }, encoders_{
            OpenCVEncoderPlugin("jpeg", framework),
            OpenCVEncoderPlugin("jpeg2k", framework),
            OpenCVEncoderPlugin("png", framework),
            OpenCVEncoderPlugin("bmp", framework),
            OpenCVEncoderPlugin("pnm", framework),
            OpenCVEncoderPlugin("tiff", framework),
            OpenCVEncoderPlugin("webp", framework)
        }
    {
        for (const auto& decoder : decoders_) {
            framework_->registerDecoder(framework_->instance, decoder.getDecoderDesc(), NVIMGCODEC_PRIORITY_LOW);
        }

        for (const auto& encoder : encoders_) {
            framework_->registerEncoder(framework_->instance, encoder.getEncoderDesc(), NVIMGCODEC_PRIORITY_LOW);
        }
    }

    ~OpenCVImgCodecsExtension()
    {
        for (const auto& decoder : decoders_) {
            framework_->unregisterDecoder(framework_->instance, decoder.getDecoderDesc());
        }
        for (const auto& encoder : encoders_) {
            framework_->unregisterEncoder(framework_->instance, encoder.getEncoderDesc());
        }
    }

    static nvimgcodecStatus_t opencvExtensionCreate(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "opencv_ext", "nvimgcodecExtensionCreate");

            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new opencv::OpenCVImgCodecsExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t opencvExtensionDestroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<opencv::OpenCVImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "opencv_ext", "nvimgcodecExtensionDestroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    OpenCVDecoderPlugin decoders_[7];
    OpenCVEncoderPlugin encoders_[7];
};

} // namespace opencv


// clang-format off
nvimgcodecExtensionDesc_t opencv_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "opencv_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER,

    opencv::OpenCVImgCodecsExtension::opencvExtensionCreate,
    opencv::OpenCVImgCodecsExtension::opencvExtensionDestroy
};
// clang-format on

nvimgcodecStatus_t get_opencv_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = opencv_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
