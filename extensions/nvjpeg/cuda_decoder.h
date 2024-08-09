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
#include <array>
#include <map>
#include <optional>
#include "../utils/stream_ctx.h"

#define DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD 1000u * 1000u

namespace nvjpeg {

struct ParseState {
    std::optional<uint64_t> parsed_stream_id_;
    nvjpegJpegStream_t nvjpeg_stream_;
};

struct DecoderData
{
    nvjpegJpegDecoder_t decoder = nullptr;
    nvjpegJpegState_t state = nullptr;
};

// Set of resources per-thread.
// Some of them are double-buffered, so that we can simultaneously decode
// the host part of the next sample, while the GPU part of the previous
// is still consuming the data from the previous iteration pinned buffer.
struct PerThreadResources
{
    // double-buffered

    struct Page
    {
        struct DecoderData
        {
            nvjpegJpegDecoder_t decoder = nullptr;
            nvjpegJpegState_t state = nullptr;
        };
        // indexing via nvjpegBackend_t (NVJPEG_BACKEND_GPU_HYBRID and NVJPEG_BACKEND_HYBRID)
        std::array<DecoderData, 3> decoder_data;
        nvjpegBufferPinned_t pinned_buffer_;
        ParseState parse_state_;
    };
    std::array<Page, 2> pages_;
    int current_page_idx = 0;
    cudaStream_t stream_;
    cudaEvent_t event_;
    nvjpegBufferDevice_t device_buffer_;
};

struct Decoder
{
    Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        const char* options = nullptr);
    ~Decoder();

    nvimgcodecStatus_t canDecodeImpl(CodeStreamCtx& ctx);
    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    void decodeImpl(BatchItemCtx& batch_item, PerThreadResources& t);
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
    std::vector<PerThreadResources> per_thread_;

    const nvimgcodecExecutionParams_t* exec_params_;
    size_t gpu_hybrid_huffman_threshold_ = DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD;
    bool preallocate_buffers_ = true;
    std::optional<size_t> device_mem_padding_;
    std::optional<size_t> pinned_mem_padding_;
    CodeStreamCtxManager code_stream_mgr_;
};

class NvJpegCudaDecoderPlugin
{
  public:
    explicit NvJpegCudaDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecDecoderDesc_t* getDecoderDesc();

  private:
    nvimgcodecStatus_t create(nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg_cuda_decoder";
    nvimgcodecDecoderDesc_t decoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace nvjpeg
