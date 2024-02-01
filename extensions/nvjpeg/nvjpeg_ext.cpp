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

#include <nvimgcodec.h>
#include "cuda_decoder.h"
#if NVJPEG_LOSSLESS_SUPPORTED
    #include "lossless_decoder.h"
#endif
#include "cuda_encoder.h"
#include "errors_handling.h"
#include "hw_decoder.h"
#include "log.h"

namespace nvjpeg {

struct NvJpegImgCodecsExtension
{
  public:
    explicit NvJpegImgCodecsExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg_hw_decoder_(framework)
        , jpeg_cuda_decoder_(framework)
        , jpeg_cuda_encoder_(framework)
#if NVJPEG_LOSSLESS_SUPPORTED
        , jpeg_lossless_decoder_(framework)
#endif
    {
        framework->registerEncoder(framework->instance, jpeg_cuda_encoder_.getEncoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
        if (jpeg_hw_decoder_.isPlatformSupported())
            framework->registerDecoder(framework->instance, jpeg_hw_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_VERY_HIGH);
        framework->registerDecoder(framework->instance, jpeg_cuda_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
#if NVJPEG_LOSSLESS_SUPPORTED
        framework->registerDecoder(framework->instance, jpeg_lossless_decoder_.getDecoderDesc(), NVIMGCODEC_PRIORITY_HIGH);
#endif
    }
    ~NvJpegImgCodecsExtension()
    {
        framework_->unregisterEncoder(framework_->instance, jpeg_cuda_encoder_.getEncoderDesc());
        if (jpeg_hw_decoder_.isPlatformSupported())
            framework_->unregisterDecoder(framework_->instance, jpeg_hw_decoder_.getDecoderDesc());
        framework_->unregisterDecoder(framework_->instance, jpeg_cuda_decoder_.getDecoderDesc());
#if NVJPEG_LOSSLESS_SUPPORTED
        framework_->unregisterDecoder(framework_->instance, jpeg_lossless_decoder_.getDecoderDesc());
#endif
    }

    static nvimgcodecStatus_t nvjpeg_extension_create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            XM_CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "nvjpeg-ext", "nvjpeg_extension_create");
            XM_CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new nvjpeg::NvJpegImgCodecsExtension(framework));
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t nvjpeg_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            XM_CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<nvjpeg::NvJpegImgCodecsExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "nvjpeg_ext", "nvjpeg_extension_destroy");
            delete ext_handle;
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    NvJpegHwDecoderPlugin jpeg_hw_decoder_;
    NvJpegCudaDecoderPlugin jpeg_cuda_decoder_;
    NvJpegCudaEncoderPlugin jpeg_cuda_encoder_;
#if NVJPEG_LOSSLESS_SUPPORTED
    NvJpegLosslessDecoderPlugin jpeg_lossless_decoder_;
#endif
};
} // namespace nvjpeg

  // clang-format off
nvimgcodecExtensionDesc_t nvjpeg_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "nvjpeg_extension",
    NVIMGCODEC_VER,    
    NVIMGCODEC_EXT_API_VER,

    nvjpeg::NvJpegImgCodecsExtension::nvjpeg_extension_create,
    nvjpeg::NvJpegImgCodecsExtension::nvjpeg_extension_destroy
};
// clang-format on  

nvimgcodecStatus_t get_nvjpeg_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = nvjpeg_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}
