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

#include "hw_decoder.h"
#include <nvimgcodec.h>
#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <vector>
#include <set>
#include "nvjpeg_utils.h"

#include <nvtx3/nvtx3.hpp>

#include "errors_handling.h"
#include "log.h"
#include "type_convert.h"

#if WITH_DYNAMIC_NVJPEG_ENABLED
    #include "dynlink/dynlink_nvjpeg.h"
#else
    #define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

NvJpegHwDecoderPlugin::NvJpegHwDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
          static_create, Decoder::static_destroy, Decoder::static_can_decode, Decoder::static_decode_batch}
    , framework_(framework)
{}

bool NvJpegHwDecoderPlugin::isPlatformSupported()
{
    nvjpegHandle_t handle;
    nvjpegStatus_t status = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &handle);
    if (status == NVJPEG_STATUS_SUCCESS) {
        XM_CHECK_NVJPEG(nvjpegDestroy(handle));
        return true;
    } else {
        return false;
    }
}

nvimgcodecDecoderDesc_t* NvJpegHwDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
    nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params, ParseState& parse_state)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_hw_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);

        *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        if (strcmp(cs_image_info.codec_name, "jpeg") != 0) {
            *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
            NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_GRAY};
        if (supported_css.find(cs_image_info.chroma_subsampling) == supported_css.end()) {
            *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }

        nvjpegDecodeParams_t nvjpeg_params_;
        XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &nvjpeg_params_));
        std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params(
            nvjpeg_params_, &nvjpegDecodeParamsDestroy);

        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t encoded_stream_data_size = 0;
        io_stream->size(io_stream->instance, &encoded_stream_data_size);
        void* encoded_stream_data = nullptr;
        void* mapped_encoded_stream_data = nullptr;
        io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size);

        if (!mapped_encoded_stream_data) {
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            size_t read_nbytes = 0;
            parse_state.buffer_.resize(encoded_stream_data_size);
            io_stream->read(io_stream->instance, &read_nbytes, &parse_state.buffer_[0], encoded_stream_data_size);
            if (read_nbytes != encoded_stream_data_size) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected end-of-stream");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }
            encoded_stream_data = &parse_state.buffer_[0];

        } else {
            encoded_stream_data = mapped_encoded_stream_data;
        }

        XM_CHECK_NVJPEG(nvjpegJpegStreamParseHeader(
            handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, parse_state.nvjpeg_stream_));

        if (mapped_encoded_stream_data) {
            io_stream->unmap(io_stream->instance, &mapped_encoded_stream_data, encoded_stream_data_size);
        }
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &image_info);

        bool need_params = false;
        if (params->apply_exif_orientation) {
            nvjpegExifOrientation_t orientation = nvimgcodec_to_nvjpeg_orientation(image_info.orientation);
            if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                if (!nvjpegIsSymbolAvailable("nvjpegDecodeParamsSetExifOrientation")) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeParamsSetExifOrientation not available");
                    *status = NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
                    return NVIMGCODEC_STATUS_SUCCESS;
                }
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Setting up EXIF orientation " << orientation);
                if (NVJPEG_STATUS_SUCCESS != nvjpegDecodeParamsSetExifOrientation(nvjpeg_params.get(), orientation)) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeParamsSetExifOrientation failed");
                    *status = NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
                    return NVIMGCODEC_STATUS_SUCCESS;
                }
                need_params = true;
            }
        }

        if (params->enable_roi && image_info.region.ndim > 0) {
            if (!nvjpegIsSymbolAvailable("nvjpegDecodeBatchedEx")) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "ROI HW decoding not supported in this nvjpeg version");
                *status = NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            need_params = true;
            auto region = image_info.region;
            auto roi_width = region.end[1] - region.start[1];
            auto roi_height = region.end[0] - region.start[0];
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), region.start[1], region.start[0], roi_width, roi_height));
        } else {
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), 0, 0, -1, -1));
        }

        int isSupported = -1;
        if (nvjpegIsSymbolAvailable("nvjpegDecodeBatchedSupportedEx")) {
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupportedEx(handle_, parse_state.nvjpeg_stream_, nvjpeg_params.get(), &isSupported));
        } else {
            if (need_params) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "API is not supported");
                *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, parse_state.nvjpeg_stream_, &isSupported));
        }
        if (isSupported == 0) {
            *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is supported");
        } else {
            *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is NOT supported");
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if hw nvjpeg can decode - " << e.info());
        *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        nvtx3::scoped_range marker{"nvjpeg_hw_can_decode"};
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_hw_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        int max_hw_dec_load = static_cast<int>(std::round(hw_load_ * batch_size));
        // Adjusting the load to utilize all the cores available
        size_t tail = max_hw_dec_load % num_cores_per_hw_engine_;
        if (tail > 0)
            max_hw_dec_load = max_hw_dec_load + num_cores_per_hw_engine_ - tail;
        if (max_hw_dec_load > batch_size)
            max_hw_dec_load = batch_size;
        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "max_hw_dec_load=" << max_hw_dec_load);

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        if (batch_size < (num_threads + 1)) {  // not worth parallelizing
            for (int i = 0; i < batch_size; i++)
                canDecode(&status[i], code_streams[i], images[i], params, *parse_state_[0]);
        } else {
            int num_blocks = num_threads + 1;  // the last block is processed in the current thread
            CanDecodeCtx canDecodeCtx{this, status, code_streams, images, params, max_hw_dec_load, num_blocks};
            canDecodeCtx.promise.resize(num_threads);
            std::vector<std::future<void>> fut;
            fut.reserve(num_threads);
            for (auto& pr : canDecodeCtx.promise)
                fut.push_back(pr.get_future());
            auto task = [](int tid, int block_idx, void* context) -> void {
                auto* ctx = reinterpret_cast<CanDecodeCtx*>(context);
                int64_t i_start = ctx->num_samples * block_idx / ctx->num_blocks;
                int64_t i_end = ctx->num_samples * (block_idx + 1) / ctx->num_blocks;
                for (int i = i_start; i < i_end; i++) {
                    ctx->this_ptr->canDecode(&ctx->status[i], ctx->code_streams[i], ctx->images[i], ctx->params, *ctx->this_ptr->parse_state_[tid]);
                }
                if (block_idx < static_cast<int>(ctx->promise.size()))
                    ctx->promise[block_idx].set_value();
            };
            int block_idx = 0;
            for (; block_idx < num_threads; ++block_idx) {
                executor->launch(executor->instance, exec_params_->device_id, block_idx, &canDecodeCtx, std::move(task));
            }
            task(num_threads, block_idx, &canDecodeCtx);

            // wait for it to finish
            for (auto& f : fut)
                f.wait();
        }
        for (int i = max_hw_dec_load; i < batch_size; i++) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Dropping sample " << i << " to be picked by the next decoder");
            status[i] = NVIMGCODEC_PROCESSING_STATUS_SATURATED;
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if hw nvjpeg can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpegHwDecoderPlugin::ParseState::ParseState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &nvjpeg_stream_));
}

NvJpegHwDecoderPlugin::ParseState::~ParseState()
{
    if (nvjpeg_stream_) {
        XM_NVJPEG_D_LOG_DESTROY(nvjpegJpegStreamDestroy(nvjpeg_stream_));
    }
}


void NvJpegHwDecoderPlugin::Decoder::parseOptions(const char* options)
{
    std::istringstream iss(options ? options : "");
    std::string token;
    while (std::getline(iss, token, ' ')) {
        std::string::size_type colon = token.find(':');
        std::string::size_type equal = token.find('=');
        if (colon == std::string::npos || equal == std::string::npos || colon > equal)
            continue;
        std::string module = token.substr(0, colon);
        if (module != "" && module != "nvjpeg_hw_decoder")
            continue;
        std::string option = token.substr(colon + 1, equal - colon - 1);
        std::string value_str = token.substr(equal + 1);

        std::istringstream value(value_str);
        if (option == "preallocate_width_hint") {
            value >> preallocate_width_;
        } else if (option == "preallocate_height_hint") {
            value >> preallocate_height_;
        } else if (option == "preallocate_batch_size") {
            value >> preallocate_batch_size_;
        }
    }
}

NvJpegHwDecoderPlugin::Decoder::Decoder(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
{
    parseOptions(options);
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

    unsigned int nvjpeg_flags = get_nvjpeg_flags("nvjpeg_cuda_decoder", options);
    if (use_nvjpeg_create_ex_v2) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_HARDWARE, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, nvjpeg_flags, &handle_));
    }

    if (exec_params->device_allocator && (exec_params->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(exec_params->device_allocator->device_mem_padding, handle_));
    }
    if (exec_params->pinned_allocator && (exec_params->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(exec_params->pinned_allocator->pinned_mem_padding, handle_));
    }

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    nvjpegStatus_t hw_dec_info_status = NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    hw_load_ = 1.0f;
    if (nvjpeg_at_least(11, 9, 0) && nvjpegIsSymbolAvailable("nvjpegGetHardwareDecoderInfo")) {
        hw_dec_info_status = nvjpegGetHardwareDecoderInfo(handle_, &num_hw_engines_, &num_cores_per_hw_engine_);
        if (hw_dec_info_status != NVJPEG_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegGetHardwareDecoderInfo failed with return code " << hw_dec_info_status);
            num_hw_engines_ = 0;
            num_cores_per_hw_engine_ = 0;
            hw_load_ = 0.0f;
        }
    } else {
        num_hw_engines_ = 1;
        num_cores_per_hw_engine_ = 5;
        hw_load_ = 1.0f;
        hw_dec_info_status = NVJPEG_STATUS_SUCCESS;
    }

    const nvimgcodecBackendParams_t* backend_params = nullptr;
    auto backend = exec_params_->backends;
    for (auto b = 0; b < exec_params_->num_backends; ++b) {
        if (backend->kind == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY) {
            backend_params = &backend->params;
            break;
        }
        ++backend;
    }

    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
        "HW decoder available num_hw_engines=" << num_hw_engines_ << " num_cores_per_hw_engine=" << num_cores_per_hw_engine_);
    if (backend_params != nullptr) {
        hw_load_ = backend_params->load_hint;
        if (hw_load_ < 0.0f)
            hw_load_ = 0.0f;
        else if (hw_load_ > 1.0f)
            hw_load_ = 1.0f;
    }
    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "HW decoder is enabled, hw_load=" << hw_load_);

    decode_state_batch_ = std::make_unique<NvJpegHwDecoderPlugin::DecodeState>(
        plugin_id_, framework_, handle_, &device_allocator_, &pinned_allocator_, num_threads);

    // call nvjpegDecodeBatchedPreAllocate to use memory pool for HW decoder even if hint is 0
    // due to considerable performance benefit - >20% for 8GPU training
    if (nvjpegIsSymbolAvailable("nvjpegDecodeBatchedPreAllocate")) {
        if (preallocate_batch_size_ < 1)
            preallocate_batch_size_ = 1;
        if (preallocate_width_ < 1)
            preallocate_width_ = 1;
        if (preallocate_height_ < 1)
            preallocate_height_ = 1;
        nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_444;
        nvjpegOutputFormat_t format = NVJPEG_OUTPUT_RGBI;
        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
            "nvjpegDecodeBatchedPreAllocate batch_size=" << preallocate_batch_size_ << " width=" << preallocate_width_
                                                         << " height=" << preallocate_height_);
        XM_CHECK_NVJPEG(nvjpegDecodeBatchedPreAllocate(decode_state_batch_->handle_, decode_state_batch_->state_, preallocate_batch_size_,
            preallocate_width_, preallocate_height_, subsampling, format));
    }

    parse_state_.reserve(num_threads + 1);
    for (int i = 0; i < num_threads + 1; i++)
        parse_state_.push_back(std::make_unique<NvJpegHwDecoderPlugin::ParseState>(plugin_id_, framework_, handle_));
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new NvJpegHwDecoderPlugin::Decoder(plugin_id_, framework_, exec_params, options));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        NvJpegHwDecoderPlugin* handle = reinterpret_cast<NvJpegHwDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpegHwDecoderPlugin::Decoder::~Decoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_destroy");
        parse_state_.clear();
        decode_state_batch_.reset();
        if (handle_)
            XM_NVJPEG_D_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.info());
    }
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegHwDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvJpegHwDecoderPlugin::DecodeState::DecodeState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle,
    nvjpegDevAllocatorV2_t* device_allocator, nvjpegPinnedAllocatorV2_t* pinned_allocator, int num_threads)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
{
    XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    XM_CHECK_CUDA(cudaEventCreate(&event_));
}

NvJpegHwDecoderPlugin::DecodeState::~DecodeState()
{
    if (event_)
        XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
    if (stream_)
        XM_CUDA_LOG_DESTROY(cudaStreamDestroy(stream_));
    if (state_)
        XM_NVJPEG_D_LOG_DESTROY(nvjpegJpegStateDestroy(state_));
    samples_.clear();
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVTX3_FUNC_RANGE();
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_hw_decode_batch, " << batch_size << " samples");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            decode_state_batch_->samples_.push_back(
                NvJpegHwDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }

        int nsamples = decode_state_batch_->samples_.size();
        std::vector<const unsigned char*> batched_bitstreams;
        std::vector<size_t> batched_bitstreams_size;
        std::vector<nvjpegImage_t> batched_output;
        std::vector<nvimgcodecImageInfo_t> batched_image_info;
        nvjpegOutputFormat_t nvjpeg_format = NVJPEG_OUTPUT_UNCHANGED;
        bool need_params = false;

        using nvjpeg_params_ptr = std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)>;
        std::vector<nvjpegDecodeParams_t> batched_nvjpeg_params;
        std::vector<nvjpeg_params_ptr> batched_nvjpeg_params_ptrs;
        batched_nvjpeg_params.resize(nsamples);
        batched_nvjpeg_params_ptrs.reserve(nsamples);

        auto& state = decode_state_batch_->state_;
        auto& handle = decode_state_batch_->handle_;

        std::set<cudaStream_t> sync_streams;
        code_stream_buffers_.resize(nsamples);
        for (int i = 0; i < nsamples; i++) {
            XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &batched_nvjpeg_params[i]));
            batched_nvjpeg_params_ptrs.emplace_back(batched_nvjpeg_params[i], &nvjpegDecodeParamsDestroy);
            auto& nvjpeg_params_ptr = batched_nvjpeg_params_ptrs.back();

            nvimgcodecCodeStreamDesc_t* code_stream = decode_state_batch_->samples_[i].code_stream;
            nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
            nvimgcodecImageDesc_t* image = decode_state_batch_->samples_[i].image;

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
            image->getImageInfo(image->instance, &image_info);
            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);
            const auto* params = decode_state_batch_->samples_[i].params;

            if (params->apply_exif_orientation) {
                nvjpegExifOrientation_t orientation = nvimgcodec_to_nvjpeg_orientation(image_info.orientation);

                // This is a workaround for a known bug in nvjpeg.
                if (!nvjpeg_at_least(12, 2, 0)) {
                    if (orientation == NVJPEG_ORIENTATION_ROTATE_90)
                        orientation = NVJPEG_ORIENTATION_ROTATE_270;
                    else if (orientation == NVJPEG_ORIENTATION_ROTATE_270)
                        orientation = NVJPEG_ORIENTATION_ROTATE_90;
                }

                if (orientation == NVJPEG_ORIENTATION_UNKNOWN) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                    continue;
                }

                if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                    need_params = true;
                    if (!nvjpegIsSymbolAvailable("nvjpegDecodeParamsSetExifOrientation")) {
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                        continue;
                    }
                    if (NVJPEG_STATUS_SUCCESS != nvjpegDecodeParamsSetExifOrientation(nvjpeg_params_ptr.get(), orientation)) {
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                        continue;
                    }
                }
            }

            if (params->enable_roi && image_info.region.ndim > 0) {
                need_params = true;
                auto region = image_info.region;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                    "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
                auto roi_width = region.end[1] - region.start[1];
                auto roi_height = region.end[0] - region.start[0];
                if (NVJPEG_STATUS_SUCCESS !=
                    nvjpegDecodeParamsSetROI(nvjpeg_params_ptr.get(), region.start[1], region.start[0], roi_width, roi_height)) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                    continue;
                }
            } else {
                if (NVJPEG_STATUS_SUCCESS != nvjpegDecodeParamsSetROI(nvjpeg_params_ptr.get(), 0, 0, -1, -1)) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                    continue;
                }
            }

            // get output image
            nvjpegImage_t nvjpeg_image;
            unsigned char* ptr = device_buffer;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                nvjpeg_image.channel[c] = ptr;
                nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
                ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
            }

            nvjpeg_format = nvimgcodec_to_nvjpeg_format(image_info.sample_format);

            size_t encoded_stream_data_size = 0;
            void* encoded_stream_data = nullptr;
            io_stream->size(io_stream->instance, &encoded_stream_data_size);
 
            void* mapped_encoded_stream_data = nullptr;
            io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size);

            if (!mapped_encoded_stream_data) {
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                size_t read_nbytes = 0;
                code_stream_buffers_[i].resize(encoded_stream_data_size);
                io_stream->read(io_stream->instance, &read_nbytes, &code_stream_buffers_[i][0], encoded_stream_data_size);
                if (read_nbytes != encoded_stream_data_size) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected end-of-stream");
                    return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                }
                encoded_stream_data = &code_stream_buffers_[i][0];

            } else {
                encoded_stream_data = mapped_encoded_stream_data;
            }

            batched_bitstreams.push_back(static_cast<const unsigned char*>(encoded_stream_data));
            batched_bitstreams_size.push_back(encoded_stream_data_size);
            batched_output.push_back(nvjpeg_image);
            batched_image_info.push_back(image_info);
            sync_streams.insert(image_info.cuda_stream);
        }

        try {
            if (batched_bitstreams.size() > 0) {
                XM_CHECK_CUDA(cudaEventSynchronize(decode_state_batch_->event_));

                XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle, state, batched_bitstreams.size(), 1, nvjpeg_format));

                if (nvjpegIsSymbolAvailable("nvjpegDecodeBatchedEx")) {
                    nvtx3::scoped_range marker{"nvjpegDecodeBatchedEx"};
                    XM_CHECK_NVJPEG(nvjpegDecodeBatchedEx(handle, state, batched_bitstreams.data(), batched_bitstreams_size.data(),
                        batched_output.data(), batched_nvjpeg_params.data(), decode_state_batch_->stream_));
                } else {
                    if (need_params)
                        throw std::logic_error("Unexpected error");
                    nvtx3::scoped_range marker{"nvjpegDecodeBatched"};
                    XM_CHECK_NVJPEG(nvjpegDecodeBatched(handle, state, batched_bitstreams.data(), batched_bitstreams_size.data(),
                        batched_output.data(), decode_state_batch_->stream_));
                }
                XM_CHECK_CUDA(cudaEventRecord(decode_state_batch_->event_, decode_state_batch_->stream_));
            }

            // sync with user stream
            for (cudaStream_t stream : sync_streams) {
                XM_CHECK_CUDA(cudaStreamWaitEvent(stream, decode_state_batch_->event_));
            }

            for (size_t sample_idx = 0; sample_idx < batched_bitstreams.size(); sample_idx++) {
                nvimgcodecImageDesc_t* image = decode_state_batch_->samples_[sample_idx].image;
                nvimgcodecCodeStreamDesc_t* code_stream = decode_state_batch_->samples_[sample_idx].code_stream;
                nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
                io_stream->unmap(io_stream->instance, &batched_bitstreams[sample_idx], batched_bitstreams_size[sample_idx]);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            }
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.info());
            for (size_t sample_idx = 0; sample_idx < batched_bitstreams.size(); sample_idx++) {
                nvimgcodecImageDesc_t* image = decode_state_batch_->samples_[sample_idx].image;
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
            return e.nvimgcodecStatus();
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegHwDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegHwDecoderPlugin::Decoder*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

} // namespace nvjpeg
