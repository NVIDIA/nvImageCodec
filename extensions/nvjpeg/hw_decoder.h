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

#include <nvimgcodec.h>
#include <nvjpeg.h>
#include <future>
#include <vector>
#include <optional>
#include "../utils/stream_ctx.h"

namespace nvjpeg {

class NvJpegHwDecoderPlugin
{
  public:
    explicit NvJpegHwDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecDecoderDesc_t* getDecoderDesc();
    static bool isPlatformSupported();

  private:
    struct Decoder
    {
        Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
            const char* options = nullptr);
        ~Decoder();

        nvimgcodecStatus_t canDecodeImpl(CodeStreamCtx& ctx, nvjpegJpegStream_t& nvjpeg_stream);
        nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        nvimgcodecStatus_t decodeBatch(
            nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

        static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
        static nvimgcodecStatus_t static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
            nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

        void parseOptions(const char* options);

        const char* plugin_id_;
        nvjpegHandle_t handle_;
        nvjpegDevAllocatorV2_t device_allocator_;
        nvjpegPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcodecFrameworkDesc_t* framework_;
        
        nvjpegJpegState_t state_;
        cudaStream_t stream_;
        cudaEvent_t event_;
        CodeStreamCtxManager code_stream_mgr_;
        std::vector<nvjpegJpegStream_t> nvjpeg_streams_;

        const nvimgcodecExecutionParams_t* exec_params_;
        unsigned int num_hw_engines_;
        unsigned int num_cores_per_hw_engine_;
        float hw_load_ = 1.0f;

        int preallocate_batch_size_ = 1;
        int preallocate_width_ = 1;
        int preallocate_height_ = 1;
    };

    nvimgcodecStatus_t create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg_hw_decoder";
    nvimgcodecDecoderDesc_t decoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace nvjpeg
