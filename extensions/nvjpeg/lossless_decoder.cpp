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

#include "lossless_decoder.h"
#include <library_types.h>
#include <nvimgcodec.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <vector>
#include "errors_handling.h"
#include "log.h"
#include "nvjpeg_utils.h"
#include "type_convert.h"
#include "../utils/parallel_exec.h"

#if WITH_DYNAMIC_NVJPEG_ENABLED
    #include "dynlink/dynlink_nvjpeg.h"
#else
    #define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

NvJpegLosslessDecoderPlugin::NvJpegLosslessDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
          static_create, Decoder::static_destroy, Decoder::static_can_decode, Decoder::static_decode_batch}
    , framework_(framework)
{
}

bool NvJpegLosslessDecoderPlugin::isPlatformSupported()
{
    return true;
}

nvimgcodecDecoderDesc_t* NvJpegLosslessDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}


nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::Decoder::canDecodeImpl(CodeStreamCtx& ctx, nvjpegJpegStream_t& nvjpeg_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        auto* code_stream = ctx.code_stream_;
        if (!ctx.load()) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        XM_CHECK_NULL(code_stream);

        nvimgcodecJpegImageInfo_t jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t),static_cast<void*>(&jpeg_info)};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
        bool is_lossless_huffman = jpeg_info.encoding == NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN;

        for (size_t i = 0; i < ctx.size(); i++) {
            auto *status = &ctx.batch_items_[i]->processing_status;
            auto *image = ctx.batch_items_[i]->image;
            const auto *params = ctx.batch_items_[i]->params;

            XM_CHECK_NULL(status);
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(params);

            *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

            if (!is_jpeg) {
                *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                continue;
            } else if (!is_lossless_huffman) {
                *status = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                continue;
            }

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
            image->getImageInfo(image->instance, &image_info);
            if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444 && image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;

            bool is_unchanged = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED && image_info.num_planes <= 2;
            bool is_y = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y && image_info.num_planes == 1;
            if (!(is_unchanged || is_y))
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;

            if (image_info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;

            if (*status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
                continue;

            assert(ctx.encoded_stream_data_ != nullptr);
            XM_CHECK_NVJPEG(nvjpegJpegStreamParse(
                handle_, static_cast<const unsigned char*>(ctx.encoded_stream_data_), ctx.encoded_stream_data_size_, 0, 0, nvjpeg_stream));
            int isSupported = -1;
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, nvjpeg_stream, &isSupported));
            if (isSupported == 0) {
                *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "decoding this lossless jpeg image is supported");
            } else {
                *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "decoding this lossless jpeg image is NOT supported");
            }
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if lossless nvjpeg can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        nvtx3::scoped_range marker{"nvjpeg_lossless_can_decode"};
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        // Groups samples belonging to the same stream
        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);

        auto task = [](int tid, int sample_idx, void* context) {
            auto this_ptr = reinterpret_cast<Decoder*>(context);
            auto& nvjpeg_stream = tid < 0 ? this_ptr->nvjpeg_streams_.back() : this_ptr->nvjpeg_streams_[tid];
            this_ptr->canDecodeImpl(*this_ptr->code_stream_mgr_[sample_idx], nvjpeg_stream);
        };
        BlockParallelExec(this, task, code_stream_mgr_.size(), exec_params_);
        for (int i = 0; i < batch_size; i++) {
            status[i] = code_stream_mgr_.get_batch_item(i).processing_status;
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if lossless nvjpeg can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpegLosslessDecoderPlugin::ParseState::ParseState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle)
{
    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &nvjpeg_stream_));
}

NvJpegLosslessDecoderPlugin::ParseState::~ParseState()
{
    if (nvjpeg_stream_) {
        XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(nvjpeg_stream_));
    }
}

NvJpegLosslessDecoderPlugin::Decoder::Decoder(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
{
    bool use_nvjpeg_create_ex_v2 = false;
    if (nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
        if (exec_params->device_allocator && exec_params->device_allocator->device_malloc && exec_params->device_allocator->device_free) {
            device_allocator_.dev_ctx = exec_params->device_allocator->device_ctx;
            device_allocator_.dev_malloc = exec_params->device_allocator->device_malloc;
            device_allocator_.dev_free = exec_params->device_allocator->device_free;
        }

        if (exec_params->pinned_allocator && exec_params->pinned_allocator->pinned_malloc && exec_params->pinned_allocator->pinned_free) {
            pinned_allocator_.pinned_ctx = exec_params->pinned_allocator->pinned_ctx;
            pinned_allocator_.pinned_malloc = exec_params->pinned_allocator->pinned_malloc;
            pinned_allocator_.pinned_free = exec_params->pinned_allocator->pinned_free;
        }
        use_nvjpeg_create_ex_v2 =
            device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free;
    }

    unsigned int nvjpeg_flags = get_nvjpeg_flags("nvjpeg_lossless_decoder", options);
    if (use_nvjpeg_create_ex_v2) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_LOSSLESS_JPEG, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_LOSSLESS_JPEG, nullptr, nullptr, nvjpeg_flags, &handle_));
    }

    if (exec_params->device_allocator && (exec_params->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(exec_params->device_allocator->device_mem_padding, handle_));
    }
    if (exec_params->pinned_allocator && (exec_params->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(exec_params->pinned_allocator->pinned_mem_padding, handle_));
    }

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    XM_CHECK_CUDA(cudaEventCreate(&event_));

    nvjpeg_streams_.resize(num_threads + 1);
    for (auto& nvjpeg_stream : nvjpeg_streams_) {
        XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &nvjpeg_stream));
    }
}

nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        *decoder =
            reinterpret_cast<nvimgcodecDecoder_t>(new NvJpegLosslessDecoderPlugin::Decoder(plugin_id_, framework_, exec_params, options));
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpegException& e) {
        if (e.nvimgcodecStatus() == NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER) {
            // invalid parameter, probably NVJPEG_BACKEND_LOSSLESS_JPEG not available, only info message
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "Could not create nvjpeg lossless decoder: " << e.info());
        } else {
            // unexpected error
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg lossless decoder: " << e.info());
        }
        return e.nvimgcodecStatus();
    }
}

nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        NvJpegLosslessDecoderPlugin* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpegLosslessDecoderPlugin::Decoder::~Decoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_destroy");
        for (auto& nvjpeg_stream : nvjpeg_streams_)
            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(nvjpeg_stream));
        if (event_)
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
        if (stream_)
            XM_CUDA_LOG_DESTROY(cudaStreamDestroy(stream_));
        if (state_)
            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(state_));

        if (handle_)
            XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg lossless decoder - " << e.info());
    }
}

nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::Decoder::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVTX3_FUNC_RANGE();
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);
        for (size_t i = 0; i < code_stream_mgr_.size(); i++) {
            code_stream_mgr_[i]->load();
        }

        std::vector<const unsigned char*> batched_bitstreams;
        std::vector<size_t> batched_bitstreams_size;
        std::vector<nvjpegImage_t> batched_output;
        std::vector<nvimgcodecImageInfo_t> batched_image_info;

        nvjpegOutputFormat_t nvjpeg_format;

        std::vector<int> sample_idxs;
        sample_idxs.reserve(batch_size);

        std::set<cudaStream_t> sync_streams;

        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            auto &sample = code_stream_mgr_.get_batch_item(sample_idx);
            auto* ctx = sample.code_stream_ctx;
            nvimgcodecImageDesc_t* image = sample.image;

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            image->getImageInfo(image->instance, &image_info);
            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

            nvjpegImage_t nvjpeg_image;
            unsigned char* ptr = device_buffer;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                nvjpeg_image.channel[c] = ptr;
                nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
                ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
            }

            if ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED ||
                    (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y && image_info.num_planes == 1)) &&
                image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
                nvjpeg_format = NVJPEG_OUTPUT_UNCHANGEDI_U16;
                batched_bitstreams.push_back(static_cast<const unsigned char*>(ctx->encoded_stream_data_));
                batched_bitstreams_size.push_back(ctx->encoded_stream_data_size_);
                batched_output.push_back(nvjpeg_image);
                batched_image_info.push_back(image_info);
                sample_idxs.push_back(sample_idx);
                sync_streams.insert(image_info.cuda_stream);
            } else {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
        }

        if (batched_bitstreams.size() > 0) {
            // Synchronize with previous iteration
            XM_CHECK_CUDA(cudaEventSynchronize(event_));

            // Synchronize with user stream (e.g. device buffer could be allocated asynchronously on that stream)
            for (cudaStream_t stream : sync_streams) {
                XM_CHECK_CUDA(cudaEventRecord(event_, stream));
                XM_CHECK_CUDA(cudaStreamWaitEvent(stream_, event_));
            }

            XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle_, state_, batched_bitstreams.size(), 1, nvjpeg_format));

            XM_CHECK_NVJPEG(nvjpegDecodeBatched(handle_, state_, batched_bitstreams.data(), batched_bitstreams_size.data(),
                batched_output.data(), stream_));

            XM_CHECK_CUDA(cudaEventRecord(event_, stream_));
            // Synchronize with user stream
            for (cudaStream_t stream : sync_streams) {
                XM_CHECK_CUDA(cudaStreamWaitEvent(stream, event_));
            }
        }

        for (size_t i = 0; i < sample_idxs.size(); i++) {
            auto sample_idx = sample_idxs[i];
            nvimgcodecImageDesc_t* image = code_stream_mgr_.get_batch_item(sample_idx).image;
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode lossless jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            nvimgcodecImageDesc_t* image = code_stream_mgr_.get_batch_item(i).image;
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}
nvimgcodecStatus_t NvJpegLosslessDecoderPlugin::Decoder::static_decode_batch(nvimgcodecDecoder_t decoder,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegLosslessDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin::Decoder*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}
} // namespace nvjpeg
