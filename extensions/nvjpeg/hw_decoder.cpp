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
#include "../utils/parallel_exec.h"

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

nvimgcodecStatus_t NvJpegHwDecoderPlugin::Decoder::canDecodeImpl(CodeStreamCtx& ctx, nvjpegJpegStream_t& nvjpeg_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        auto* code_stream = ctx.code_stream_;
        XM_CHECK_NULL(code_stream);

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;

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

            if (!ctx.load()) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to read from stream");
                *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                continue;
            }

            XM_CHECK_NVJPEG(nvjpegJpegStreamParseHeader(
                handle_, static_cast<const unsigned char*>(ctx.encoded_stream_data_), ctx.encoded_stream_data_size_, nvjpeg_stream));

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
                XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupportedEx(handle_, nvjpeg_stream, nvjpeg_params.get(), &isSupported));
            } else {
                if (need_params) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "API is not supported");
                    *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                    return NVIMGCODEC_STATUS_SUCCESS;
                }
                XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, nvjpeg_stream, &isSupported));
            }
            if (isSupported == 0) {
                *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is supported");
            } else {
                *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is NOT supported");
            }
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if hw nvjpeg can decode - " << e.info());
        for (size_t i = 0; i < ctx.size(); i++) {
            auto *status = &ctx.batch_items_[i]->processing_status;
            *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
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

        // Groups samples belonging to the same stream
        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);

        auto task = [](int tid, int sample_idx, void* context) {
            auto this_ptr = reinterpret_cast<Decoder*>(context);
            auto& nvjpeg_stream = tid < 0 ? this_ptr->nvjpeg_streams_.back() : this_ptr->nvjpeg_streams_[tid];
            this_ptr->canDecodeImpl(*this_ptr->code_stream_mgr_[sample_idx], nvjpeg_stream);
        };
        BlockParallelExec(this, task, code_stream_mgr_.size(), exec_params_);

        int ok_samples = 0;
        for (int i = 0; i < batch_size; i++) {
            status[i] = code_stream_mgr_.get_batch_item(i).processing_status;
            if (status[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                ok_samples++;
                if (ok_samples > max_hw_dec_load) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Dropping sample " << i << " to be picked by the next decoder");
                    status[i] = NVIMGCODEC_PROCESSING_STATUS_SATURATED;
                }
            }
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

    XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    XM_CHECK_CUDA(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    XM_CHECK_CUDA(cudaEventCreate(&event_));

    nvjpeg_streams_.resize(num_threads + 1);
    for (auto& nvjpeg_stream : nvjpeg_streams_) {
        XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &nvjpeg_stream));
    }

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
        XM_CHECK_NVJPEG(nvjpegDecodeBatchedPreAllocate(handle_, state_, preallocate_batch_size_,
            preallocate_width_, preallocate_height_, subsampling, format));
    }
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
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder:" << e.info());
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

        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);
        for (size_t i = 0; i < code_stream_mgr_.size(); i++) {
            code_stream_mgr_[i]->load();
        }

        std::vector<const unsigned char*> batched_bitstreams;
        std::vector<size_t> batched_bitstreams_size;
        std::vector<nvjpegImage_t> batched_output;
        std::vector<nvimgcodecImageInfo_t> batched_image_info;
        nvjpegOutputFormat_t nvjpeg_format = NVJPEG_OUTPUT_UNCHANGED;
        bool need_params = false;

        using nvjpeg_params_ptr = std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)>;
        std::vector<nvjpegDecodeParams_t> batched_nvjpeg_params;
        std::vector<nvjpeg_params_ptr> batched_nvjpeg_params_ptrs;
        batched_nvjpeg_params.resize(batch_size);
        batched_nvjpeg_params_ptrs.reserve(batch_size);

        std::set<cudaStream_t> sync_streams;
        for (int i = 0; i < batch_size; i++) {
            XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &batched_nvjpeg_params[i]));
            batched_nvjpeg_params_ptrs.emplace_back(batched_nvjpeg_params[i], &nvjpegDecodeParamsDestroy);
            auto& nvjpeg_params_ptr = batched_nvjpeg_params_ptrs.back();

            auto &sample = code_stream_mgr_.get_batch_item(i);
            CodeStreamCtx* ctx = sample.code_stream_ctx;
            nvimgcodecImageDesc_t* image = sample.image;
            const auto* params = sample.params;

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
            image->getImageInfo(image->instance, &image_info);
            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

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

            assert(ctx->encoded_stream_data_ != nullptr);
            assert(ctx->encoded_stream_data_size_ > 0);

            batched_bitstreams.push_back(static_cast<const unsigned char*>(ctx->encoded_stream_data_));
            batched_bitstreams_size.push_back(ctx->encoded_stream_data_size_);
            batched_output.push_back(nvjpeg_image);
            batched_image_info.push_back(image_info);
            sync_streams.insert(image_info.cuda_stream);
        }

        try {


            if (batched_bitstreams.size() > 0) {
                // Synchronize with previous iteration
                XM_CHECK_CUDA(cudaEventSynchronize(event_));

                // Synchronize with user stream (e.g. device buffer could be allocated asynchronously on that stream)
                for (cudaStream_t stream : sync_streams) {
                    XM_CHECK_CUDA(cudaEventRecord(event_, stream));
                    XM_CHECK_CUDA(cudaStreamWaitEvent(stream_, event_));
                }

                XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle_, state_, batched_bitstreams.size(), 1, nvjpeg_format));

                if (nvjpegIsSymbolAvailable("nvjpegDecodeBatchedEx")) {
                    nvtx3::scoped_range marker{"nvjpegDecodeBatchedEx"};
                    XM_CHECK_NVJPEG(nvjpegDecodeBatchedEx(handle_, state_, batched_bitstreams.data(), batched_bitstreams_size.data(),
                        batched_output.data(), batched_nvjpeg_params.data(), stream_));
                } else {
                    if (need_params)
                        throw std::logic_error("Unexpected error");
                    nvtx3::scoped_range marker{"nvjpegDecodeBatched"};
                    XM_CHECK_NVJPEG(nvjpegDecodeBatched(handle_, state_, batched_bitstreams.data(), batched_bitstreams_size.data(),
                        batched_output.data(), stream_));
                }
                XM_CHECK_CUDA(cudaEventRecord(event_, stream_));

                // sync with the user stream
                for (cudaStream_t stream : sync_streams) {
                    XM_CHECK_CUDA(cudaStreamWaitEvent(stream, event_));
                }

                for (size_t sample_idx = 0; sample_idx < batched_bitstreams.size(); sample_idx++) {
                    nvimgcodecImageDesc_t* image = code_stream_mgr_.get_batch_item(sample_idx).image;
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
                }
            }
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.info());
            for (size_t sample_idx = 0; sample_idx < batched_bitstreams.size(); sample_idx++) {
                nvimgcodecImageDesc_t* image = code_stream_mgr_.get_batch_item(sample_idx).image;
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
