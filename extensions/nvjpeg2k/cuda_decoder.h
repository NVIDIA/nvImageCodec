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
#include <queue>
#include <condition_variable>
#include "error_handling.h"

namespace nvjpeg2k {

class NvJpeg2kDecoderPlugin
{
  public:
    explicit NvJpeg2kDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework);
    nvimgcodecDecoderDesc_t* getDecoderDesc();

  private:
    struct Decoder;

    struct ParseState
    {
        explicit ParseState(const char* id, const nvimgcodecFrameworkDesc_t* framework);
        ~ParseState();

        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
        nvjpeg2kStream_t nvjpeg2k_stream_;
        std::vector<unsigned char> buffer_;
    };

    struct DecodeState
    {
        explicit DecodeState(const char* id, const nvimgcodecFrameworkDesc_t* framework, nvjpeg2kHandle_t handle,
            nvimgcodecDeviceAllocator_t* device_allocator, nvimgcodecPinnedAllocator_t* pinned_allocator, int device_id, int num_threads,
            int num_parallel_tiles);
        ~DecodeState();

        struct PerThreadResources
        {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpeg2kDecodeState_t state_;
            std::unique_ptr<ParseState> parse_state_;
        };

        struct PerTileResources {
            cudaStream_t stream_;
            cudaEvent_t event_;
            nvjpeg2kDecodeState_t state_;
        };

        struct PerTileResourcesPool {
            const char* plugin_id_;
            const nvimgcodecFrameworkDesc_t* framework_;
            nvjpeg2kHandle_t handle_ = nullptr;

            std::vector<PerTileResources> res_;
            std::queue<PerTileResources*> free_;
            std::mutex mtx_;
            std::condition_variable cv_;

            PerTileResourcesPool(const char* id, const nvimgcodecFrameworkDesc_t* framework, nvjpeg2kHandle_t handle, int num_parallel_tiles)
                : plugin_id_(id)
                , framework_(framework)
                , handle_(handle)
                , res_(num_parallel_tiles) {
                for (auto& tile_res : res_) {
                    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&tile_res.stream_, cudaStreamNonBlocking));
                    XM_CHECK_CUDA(cudaEventCreate(&tile_res.event_));
                    XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle, &tile_res.state_));
                    free_.push(&tile_res);
                }
            }

            ~PerTileResourcesPool() {
                for (auto& tile_res : res_) {
                    if (tile_res.event_) {
                        XM_CUDA_LOG_DESTROY(cudaEventDestroy(tile_res.event_));
                    }
                    if (tile_res.stream_) {
                        XM_CUDA_LOG_DESTROY(cudaStreamDestroy(tile_res.stream_));
                    }
                    if (tile_res.state_) {
                        XM_NVJPEG2K_D_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(tile_res.state_));
                    }
                }
            }

            size_t size() const {
                return res_.size();
            }

            PerTileResources* Acquire() {
                std::unique_lock<std::mutex> lock(mtx_);
                cv_.wait(lock, [&]() { return !free_.empty(); });
                auto res_ptr = free_.front();
                free_.pop();
                return res_ptr;
            }

            void Release(PerTileResources* res_ptr) {
                std::lock_guard<std::mutex> lock(mtx_);
                free_.push(res_ptr);
                cv_.notify_one();
            }
        };

        struct Sample
        {
            nvimgcodecCodeStreamDesc_t* code_stream;
            nvimgcodecImageDesc_t* image;
            const nvimgcodecDecodeParams_t* params;
        };

        const char* plugin_id_;
        const nvimgcodecFrameworkDesc_t* framework_;
        nvjpeg2kHandle_t handle_ = nullptr;
        nvimgcodecDeviceAllocator_t* device_allocator_;
        nvimgcodecPinnedAllocator_t* pinned_allocator_;
        int device_id_;
        std::vector<PerThreadResources> per_thread_;
        std::vector<Sample> samples_;
        PerTileResourcesPool per_tile_res_;
    };

    struct Decoder
    {
        Decoder(
            const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
        ~Decoder();

        nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
            nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params);
        nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
            nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
        nvimgcodecStatus_t decode(int sample_idx, bool immediate);
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
        std::unique_ptr<DecodeState> decode_state_batch_;
        const nvimgcodecExecutionParams_t* exec_params_;
        int num_parallel_tiles_;

        struct CanDecodeCtx {
            Decoder *this_ptr;
            nvimgcodecProcessingStatus_t* status;
            nvimgcodecCodeStreamDesc_t** code_streams;
            nvimgcodecImageDesc_t** images;
            const nvimgcodecDecodeParams_t* params;
            int num_samples;
            int num_blocks;
            std::vector<std::promise<void>> promise;
        };
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
