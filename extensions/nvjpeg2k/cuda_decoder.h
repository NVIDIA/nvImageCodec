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
#include <nvjpeg2k.h>
#include <memory>
#include <vector>
#include <future>
#include <mutex>
#include <optional>
#include <queue>
#include <condition_variable>
#include "error_handling.h"
#include "../utils/stream_ctx.h"

namespace nvjpeg2k {

class NvJpeg2kDecoderPlugin
{
  public:
    explicit NvJpeg2kDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecDecoderDesc_t* getDecoderDesc();

  private:
    struct Decoder;

    struct PerThreadResources
    {
        cudaStream_t stream_;
        cudaEvent_t event_;
        nvjpeg2kDecodeState_t state_;
        nvjpeg2kStream_t nvjpeg2k_stream_;
        std::optional<uint64_t> parsed_stream_id_;

    };

    struct PerTileResources
    {
        cudaStream_t stream_;
        cudaEvent_t event_;
        nvjpeg2kDecodeState_t state_;
    };

    struct PerTileResourcesPool
    {
        std::vector<PerTileResources> res_;
        std::queue<PerTileResources*> free_;
        std::mutex mtx_;
        std::condition_variable cv_;

        size_t size() const { return res_.size(); }

        PerTileResources* Acquire()
        {
            std::unique_lock<std::mutex> lock(mtx_);
            cv_.wait(lock, [&]() { return !free_.empty(); });
            auto res_ptr = free_.front();
            free_.pop();
            return res_ptr;
        }

        void Release(PerTileResources* res_ptr)
        {
            std::lock_guard<std::mutex> lock(mtx_);
            free_.push(res_ptr);
            cv_.notify_one();
        }
    };

    struct Decoder
    {
        Decoder(
            const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
        ~Decoder();

        nvimgcodecStatus_t canDecodeImpl(CodeStreamCtx& ctx);
        nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        void decodeImpl(BatchItemCtx& ctx, int tid);
        nvimgcodecStatus_t decodeBatch(
            nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        nvjpeg2kHandle_t getNvjpeg2kHandle();

        void parseOptions(const char* options);

        static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
        static nvimgcodecStatus_t static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
            nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

        const char* plugin_id_;
        nvjpeg2kHandle_t handle_;
        nvjpeg2kDeviceAllocatorV2_t device_allocator_;
        nvjpeg2kPinnedAllocatorV2_t pinned_allocator_;
        const nvimgcodecFrameworkDesc_t* framework_;
        const nvimgcodecExecutionParams_t* exec_params_;
        int num_parallel_tiles_;
        std::optional<size_t> device_mem_padding_;
        std::optional<size_t> pinned_mem_padding_;

        std::vector<PerThreadResources> per_thread_;
        PerTileResourcesPool per_tile_res_;

        CodeStreamCtxManager code_stream_mgr_;
   };

    nvimgcodecStatus_t create(
        nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static nvimgcodecStatus_t static_create(
        void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options);

    static constexpr const char* plugin_id_ = "nvjpeg2k_decoder";
    nvimgcodecDecoderDesc_t decoder_desc_;
    const nvimgcodecFrameworkDesc_t* framework_;
};

} // namespace nvjpeg2k
