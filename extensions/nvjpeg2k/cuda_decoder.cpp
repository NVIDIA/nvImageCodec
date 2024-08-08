/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#define NOMINMAX

#include <nvimgcodec.h>
#include <nvjpeg2k.h>
#include <algorithm>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <vector>

#include <imgproc/stream_device.h>
#include <imgproc/device_guard.h>

#include "log.h"
#include "cuda_decoder.h"
#include "error_handling.h"

#include "../utils/parallel_exec.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/sample_format_utils.h"

namespace nvjpeg2k {

NvJpeg2kDecoderPlugin::NvJpeg2kDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg2k",
          NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, Decoder::static_destroy, Decoder::static_can_decode,
          Decoder::static_decode_batch}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvJpeg2kDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::canDecodeImpl(CodeStreamCtx& ctx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
        XM_CHECK_NULL(ctx.code_stream_);

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ctx.code_stream_->getImageInfo(ctx.code_stream_->instance, &cs_image_info);
        bool is_jpeg2k = strcmp(cs_image_info.codec_name, "jpeg2k") == 0;

        for (size_t i = 0; i < ctx.size(); i++) {
            auto* status = &ctx.batch_items_[i]->processing_status;
            auto* image = ctx.batch_items_[i]->image;
            const auto* params = ctx.batch_items_[i]->params;

            XM_CHECK_NULL(status);
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(params);

            *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            if (!is_jpeg2k) {
                *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                return NVIMGCODEC_STATUS_SUCCESS;
            }

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            image->getImageInfo(image->instance, &image_info);
            static const std::set<nvimgcodecColorSpec_t> supported_color_space{
                NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC};
            if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
            if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV) {
                static const std::set<nvimgcodecChromaSubsampling_t> supported_css{
                    NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420};
                if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
                    *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
                }
            }

            static const std::set<nvimgcodecSampleFormat_t> supported_sample_format{
                NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
                NVIMGCODEC_SAMPLEFORMAT_P_RGB,
                NVIMGCODEC_SAMPLEFORMAT_I_RGB,
                NVIMGCODEC_SAMPLEFORMAT_P_Y,
                NVIMGCODEC_SAMPLEFORMAT_P_YUV,
            };
            if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }

            static const std::set<nvimgcodecSampleDataType_t> supported_sample_type{
                NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16};
            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                auto sample_type = image_info.plane_info[p].sample_type;
                if (supported_sample_type.find(sample_type) == supported_sample_type.end()) {
                    *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }
            }
        }
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        nvtx3::scoped_range marker{"jpeg2k_can_decode"};
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        // Groups samples belonging to the same stream
        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);

        auto task = [](int tid, int sample_idx, void* context) {
            auto this_ptr = reinterpret_cast<Decoder*>(context);
            this_ptr->canDecodeImpl(*this_ptr->code_stream_mgr_[sample_idx]);
        };
        BlockParallelExec(this, task, code_stream_mgr_.size(), exec_params_);
        for (int i = 0; i < batch_size; i++) {
            status[i] = code_stream_mgr_.get_batch_item(i).processing_status;
        }
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
}

void NvJpeg2kDecoderPlugin::Decoder::parseOptions(const char* options)
{
    num_parallel_tiles_ = 16; // default 16 tiles in parallel for all threads
    std::istringstream iss(options ? options : "");
    std::string token;
    while (std::getline(iss, token, ' ')) {
        std::string::size_type colon = token.find(':');
        std::string::size_type equal = token.find('=');
        if (colon == std::string::npos || equal == std::string::npos || colon > equal)
            continue;
        std::string module = token.substr(0, colon);
        if (module != "" && module != "nvjpeg2k_cuda_decoder")
            continue;
        std::string option = token.substr(colon + 1, equal - colon - 1);
        std::string value_str = token.substr(equal + 1);

        std::istringstream value(value_str);
        if (option == "num_parallel_tiles") {
            value >> num_parallel_tiles_;
        } else if (option == "device_memory_padding") {
            size_t padding_value = 0;
            value >> padding_value;
            device_mem_padding_ = padding_value;
        } else if (option == "host_memory_padding") {
            size_t padding_value = 0;
            value >> padding_value;
            pinned_mem_padding_ = padding_value;
        }
    }
}

NvJpeg2kDecoderPlugin::Decoder::Decoder(
    const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
{
    parseOptions(options);
    if (exec_params_->device_allocator && exec_params_->device_allocator->device_malloc && exec_params_->device_allocator->device_free) {
        device_allocator_.device_ctx = exec_params_->device_allocator->device_ctx;
        device_allocator_.device_malloc = exec_params_->device_allocator->device_malloc;
        device_allocator_.device_free = exec_params_->device_allocator->device_free;
    }

    if (exec_params_->pinned_allocator && exec_params_->pinned_allocator->pinned_malloc && exec_params_->pinned_allocator->pinned_free) {
        pinned_allocator_.pinned_ctx = exec_params_->pinned_allocator->pinned_ctx;
        pinned_allocator_.pinned_malloc = exec_params_->pinned_allocator->pinned_malloc;
        pinned_allocator_.pinned_free = exec_params_->pinned_allocator->pinned_free;
    }

    if (device_allocator_.device_malloc && device_allocator_.device_free && pinned_allocator_.pinned_malloc &&
        pinned_allocator_.pinned_free) {
        XM_CHECK_NVJPEG2K(nvjpeg2kCreateV2(NVJPEG2K_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, &handle_));
    } else {
        XM_CHECK_NVJPEG2K(nvjpeg2kCreateSimple(&handle_));
    }

    if (!device_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->device_allocator->device_mem_padding != 0)
        device_mem_padding_ = exec_params_->device_allocator->device_mem_padding;
    if (!pinned_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->pinned_allocator->pinned_mem_padding != 0)
        pinned_mem_padding_ = exec_params_->pinned_allocator->pinned_mem_padding;

    if (device_mem_padding_.has_value() && device_mem_padding_.value() > 0) {
        XM_CHECK_NVJPEG2K(nvjpeg2kSetDeviceMemoryPadding(device_mem_padding_.value(), handle_));
    }
    if (pinned_mem_padding_.has_value() && pinned_mem_padding_.value() > 0) {
        XM_CHECK_NVJPEG2K(nvjpeg2kSetPinnedMemoryPadding(pinned_mem_padding_.value(), handle_));
    }

    int device_id = exec_params_->device_id;
    // create resources per thread
    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);
    per_thread_.reserve(num_threads);

    int nCudaDevAttrComputeCapabilityMajor, nCudaDevAttrComputeCapabilityMinor;
    XM_CHECK_CUDA(cudaDeviceGetAttribute(&nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id));
    XM_CHECK_CUDA(cudaDeviceGetAttribute(&nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id));
    cudaDeviceProp device_properties{};
    XM_CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device_id));

    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &res.state_));
        XM_CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&res.nvjpeg2k_stream_));
    }

    per_tile_res_.res_.resize(num_parallel_tiles_);
    for (auto& tile_res : per_tile_res_.res_) {
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&tile_res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&tile_res.event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &tile_res.state_));
        per_tile_res_.free_.push(&tile_res);
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new NvJpeg2kDecoderPlugin::Decoder(plugin_id_, framework_, exec_params, options));
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg2k decoder - " << e.info());
        return e.nvimgcodecStatus();
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin*>(instance);
        return handle->create(decoder, exec_params, options);
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpeg2kDecoderPlugin::Decoder::~Decoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_destroy");

        for (auto& tile_res : per_tile_res_.res_) {
            if (tile_res.event_) {
                XM_CUDA_LOG_DESTROY(cudaEventDestroy(tile_res.event_));
            }
            if (tile_res.stream_) {
                XM_CUDA_LOG_DESTROY(cudaStreamDestroy(tile_res.stream_));
            }
            if (tile_res.state_) {
                XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(tile_res.state_));
            }
        }

        for (auto& res : per_thread_) {
            if (res.event_) {
                XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
            }
            if (res.stream_) {
                XM_CUDA_LOG_DESTROY(cudaStreamDestroy(res.stream_));
            }
            if (res.state_) {
                XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(res.state_));
            }
            if (res.nvjpeg2k_stream_) {
                XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kStreamDestroy(res.nvjpeg2k_stream_));
            }
        }
        if (handle_)
            XM_CHECK_NVJPEG2K(nvjpeg2kDestroy(handle_));
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg2k decoder - " << e.info());
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

void NvJpeg2kDecoderPlugin::Decoder::decodeImpl(BatchItemCtx& batch_item, int tid)
{
    int sample_idx = batch_item.index;
    nvtx3::scoped_range marker{"nvjpeg2k decode " + std::to_string(sample_idx)};
    auto& t = per_thread_[tid];
    auto& per_tile_res = per_tile_res_;
    auto jpeg2k_state = t.state_;
    nvimgcodecImageDesc_t* image = batch_item.image;
    const nvimgcodecDecodeParams_t* params = batch_item.params;
    void* decode_tmp_buffer = nullptr;
    size_t decode_tmp_buffer_sz = 0;
    try {
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &image_info);

        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected buffer kind");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }
        unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        if (batch_item.code_stream_ctx->code_stream_id_ != t.parsed_stream_id_) {
            nvtx3::scoped_range marker{"nvjpegJpegStreamParse"};
            XM_CHECK_NVJPEG2K(
                nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(batch_item.code_stream_ctx->encoded_stream_data_),
                    batch_item.code_stream_ctx->encoded_stream_data_size_, false, false, t.nvjpeg2k_stream_));
            t.parsed_stream_id_ = batch_item.code_stream_ctx->code_stream_id_;
        }

        nvjpeg2kImageInfo_t jpeg2k_info;
        XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(t.nvjpeg2k_stream_, &jpeg2k_info));

        nvjpeg2kImageComponentInfo_t comp;
        XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(t.nvjpeg2k_stream_, &comp, 0));
        auto height = comp.component_height;
        auto width = comp.component_width;
        auto bpp = comp.precision;
        auto sgn = comp.sgn;
        auto num_components = jpeg2k_info.num_components;
        if (bpp > 16) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected bitdepth");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        nvjpeg2kImage_t output_image;

        nvimgcodecSampleDataType_t out_data_type = image_info.plane_info[0].sample_type;
        size_t out_bytes_per_sample = static_cast<unsigned int>(out_data_type) >> (8 + 3);
        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != out_data_type) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }
        }

        nvimgcodecSampleDataType_t orig_data_type;
        if (sgn) {
            if ((bpp > 8) && (bpp <= 16)) {
                output_image.pixel_type = NVJPEG2K_INT16;
                orig_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
            } else {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "unsupported bit depth for a signed type. It must be 8 > bpp <= 16");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }
        } else {
            if (bpp <= 8) {
                output_image.pixel_type = NVJPEG2K_UINT8;
                orig_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            } else if (bpp <= 16) {
                output_image.pixel_type = NVJPEG2K_UINT16;
                orig_data_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
            } else {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "bit depth not supported");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }
        }
        size_t bytes_per_sample = static_cast<unsigned int>(orig_data_type) >> (8 + 3);
        size_t bits_per_sample = bytes_per_sample << 3;

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        batch_item.code_stream_ctx->code_stream_->getImageInfo(batch_item.code_stream_ctx->code_stream_->instance, &cs_image_info);
        bool convert_dtype =
            image_info.plane_info[0].sample_type != orig_data_type || (bpp != bits_per_sample && image_info.plane_info[0].precision != bpp);
        // Can decode directly to interleaved
        bool native_interleaved =
            !convert_dtype && ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB && num_components == 3) ||
                                  (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED && num_components > 1));
        // Will decode to planar then convert
        bool convert_interleaved = !native_interleaved && (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB ||
                                                              image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED);
        bool convert_gray =
            image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y && cs_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_Y;

        if (convert_dtype && out_data_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Only original dtype or conversion to uint8 is allowed");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        nvjpeg2kDecodeParams_t decode_params;
        nvjpeg2kDecodeParamsCreate(&decode_params);
        std::unique_ptr<std::remove_pointer<nvjpeg2kDecodeParams_t>::type, decltype(&nvjpeg2kDecodeParamsDestroy)> decode_params_raii(
            decode_params, &nvjpeg2kDecodeParamsDestroy);

        int rgb_output = image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB || image_info.color_spec == NVIMGCODEC_COLORSPEC_UNKNOWN ||
                         (image_info.color_spec == NVIMGCODEC_COLORSPEC_UNCHANGED && cs_image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB);
        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, rgb_output));
        XM_CHECK_NVJPEG2K(
            nvjpeg2kDecodeParamsSetOutputFormat(decode_params, native_interleaved ? NVJPEG2K_FORMAT_INTERLEAVED : NVJPEG2K_FORMAT_PLANAR));

        if (image_info.num_planes > 1 && num_components < image_info.num_planes) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of planes");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        } else if (convert_interleaved && (num_components < image_info.plane_info[0].num_channels && num_components != 1)) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        } else if (native_interleaved && image_info.plane_info[0].num_channels != num_components) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }
        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != image_info.plane_info[0].sample_type) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }
        }
        auto region = image_info.region;
        bool has_roi = params->enable_roi && image_info.region.ndim > 0;
        uint32_t roi_y_begin = has_roi ? static_cast<uint32_t>(image_info.region.start[0]) : 0;
        uint32_t roi_x_begin = has_roi ? static_cast<uint32_t>(image_info.region.start[1]) : 0;
        uint32_t roi_y_end = has_roi ? static_cast<uint32_t>(image_info.region.end[0]) : jpeg2k_info.image_height;
        uint32_t roi_x_end = has_roi ? static_cast<uint32_t>(image_info.region.end[1]) : jpeg2k_info.image_width;
        uint32_t roi_height = roi_y_end - roi_y_begin;
        uint32_t roi_width = roi_x_end - roi_x_begin;

        if (has_roi) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
            XM_CHECK_NVJPEG2K(
                nvjpeg2kDecodeParamsSetDecodeArea(decode_params, region.start[1], region.end[1], region.start[0], region.end[0]));
            for (size_t p = 0; p < image_info.num_planes; p++) {
                if (roi_height != image_info.plane_info[p].height || roi_width != image_info.plane_info[p].width) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected plane info dimensions");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return;
                }
            }
        } else {
            for (size_t p = 0; p < image_info.num_planes; p++) {
                if (height != image_info.plane_info[p].height || width != image_info.plane_info[p].width) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected plane info dimensions");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return;
                }
            }
        }

        size_t planar_row_nbytes = roi_width * bytes_per_sample;
        size_t planar_plane_nbytes = roi_width * roi_height * bytes_per_sample;

        size_t expected_buffer_size =
            roi_height * roi_width * out_bytes_per_sample * std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
        if (image_info.buffer_size < expected_buffer_size) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_,
                "The provided buffer can't hold the decoded image: " << image_info.buffer_size << " < " << expected_buffer_size);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        uint8_t* decode_buffer = nullptr;
        bool needs_convert = convert_gray || convert_interleaved || convert_dtype;
        bool planar_subset = image_info.num_planes > 1 && image_info.num_planes < num_components;
        if (needs_convert || planar_subset) {
            size_t current_offset = 0;
            if (needs_convert) {
                // If there are conversions needed, we decode to a temporary buffer first
                current_offset = planar_plane_nbytes * num_components;
            } else if (planar_subset) {
                // If there are more components than we want, we allocate temp memory for the planes we don't need
                current_offset = (num_components - image_info.num_planes) * planar_plane_nbytes;
            }

            decode_tmp_buffer_sz = current_offset; // allocate a single chunk of memory for all the temporary buffers we need
            if (device_allocator_.device_malloc) {
                device_allocator_.device_malloc(device_allocator_.device_ctx, &decode_tmp_buffer, decode_tmp_buffer_sz, t.stream_);
            } else {
                bool use_async_mem_ops = nvimgcodec::can_use_async_mem_ops(t.stream_);
                if (use_async_mem_ops) {
                    XM_CHECK_CUDA(cudaMallocAsync(&decode_tmp_buffer, decode_tmp_buffer_sz, t.stream_));
                } else {
                    nvimgcodec::DeviceGuard device_guard(nvimgcodec::get_stream_device_id(t.stream_));
                    XM_CHECK_CUDA(cudaMalloc(&decode_tmp_buffer, decode_tmp_buffer_sz));
                }
            }
        }

        std::vector<unsigned char*> decode_output(num_components);
        std::vector<size_t> pitch_in_bytes(num_components);
        if (native_interleaved) {
            decode_buffer = reinterpret_cast<uint8_t*>(device_buffer);
            decode_output[0] = decode_buffer;
            pitch_in_bytes[0] = roi_width * num_components * bytes_per_sample;
        } else if (needs_convert) {
            decode_buffer = reinterpret_cast<uint8_t*>(decode_tmp_buffer);
            for (size_t p = 0; p < num_components; p++) {
                decode_output[p] = decode_buffer + p * planar_plane_nbytes;
                pitch_in_bytes[p] = planar_row_nbytes;
            }
        } else {
            uint32_t p = 0;
            decode_buffer = device_buffer;
            for (; p < image_info.num_planes; ++p) {
                decode_output[p] = device_buffer + p * planar_plane_nbytes;
                pitch_in_bytes[p] = planar_row_nbytes;
            }
            for (; p < num_components; ++p) {
                decode_output[p] = reinterpret_cast<uint8_t*>(decode_tmp_buffer) + (p - image_info.num_planes) * planar_plane_nbytes;
                pitch_in_bytes[p] = planar_row_nbytes;
            }
        }
        output_image.num_components = num_components;
        output_image.pixel_data = (void**)&decode_output[0];
        output_image.pitch_in_bytes = &pitch_in_bytes[0];

        // Waits for GPU stage from previous iteration (on this thread)
        XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

        // Synchronize thread stream with the user stream
        XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
        XM_CHECK_CUDA(cudaStreamWaitEvent(t.stream_, t.event_));

        bool tiled = (jpeg2k_info.num_tiles_y > 1 || jpeg2k_info.num_tiles_x > 1);
        if (!tiled || per_tile_res.size() <= 1 || image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) {
            nvtx3::scoped_range marker{"nvjpeg2kDecodeImage"};
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpeg2kDecodeImage");
            XM_CHECK_NVJPEG2K(
                nvjpeg2kDecodeImage(handle_, jpeg2k_state, t.nvjpeg2k_stream_, decode_params_raii.get(), &output_image, t.stream_));
        } else {
            std::vector<uint8_t*> tile_decode_output(jpeg2k_info.num_components, nullptr);
            for (uint32_t tile_y = 0; tile_y < jpeg2k_info.num_tiles_y; tile_y++) {
                for (uint32_t tile_x = 0; tile_x < jpeg2k_info.num_tiles_x; tile_x++) {
                    uint32_t tile_y_begin = tile_y * jpeg2k_info.tile_height;
                    uint32_t tile_y_end = std::min(tile_y_begin + jpeg2k_info.tile_height, jpeg2k_info.image_height);
                    uint32_t tile_x_begin = tile_x * jpeg2k_info.tile_width;
                    uint32_t tile_x_end = std::min(tile_x_begin + jpeg2k_info.tile_width, jpeg2k_info.image_width);
                    uint32_t offset_y = tile_y_begin > roi_y_begin ? tile_y_begin - roi_y_begin : 0;
                    uint32_t offset_x = tile_x_begin > roi_x_begin ? tile_x_begin - roi_x_begin : 0;
                    if (has_roi) {
                        tile_y_begin = std::max(roi_y_begin, tile_y_begin);
                        tile_x_begin = std::max(roi_x_begin, tile_x_begin);
                        tile_y_end = std::min(roi_y_end, tile_y_end);
                        tile_x_end = std::min(roi_x_end, tile_x_end);
                    }
                    if (tile_y_begin >= tile_y_end || tile_x_begin >= tile_x_end)
                        continue;

                    XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetDecodeArea(decode_params, tile_x_begin, tile_x_end, tile_y_begin, tile_y_end));

                    nvjpeg2kImage_t output_tile;
                    output_tile.pixel_type = output_image.pixel_type;
                    output_tile.pitch_in_bytes = output_image.pitch_in_bytes;
                    output_tile.num_components = output_image.num_components;
                    output_tile.pixel_data = reinterpret_cast<void**>(&tile_decode_output[0]);
                    uint32_t c = 0;
                    if (native_interleaved) {
                        output_tile.pixel_data[0] = decode_buffer + (offset_y * roi_width + offset_x) * num_components * bytes_per_sample;
                    } else if (planar_subset) {
                        // Decode subset of planes directly to the output
                        for (; c < image_info.num_planes; c++) {
                            output_tile.pixel_data[c] =
                                decode_buffer + c * planar_plane_nbytes + offset_y * planar_row_nbytes + offset_x * bytes_per_sample;
                        }
                        // Decode remaining planes to a temp buffer
                        for (; c < output_image.num_components; c++) {
                            output_tile.pixel_data[c] = reinterpret_cast<uint8_t*>(decode_tmp_buffer) +
                                                        (c - image_info.num_planes) * planar_plane_nbytes + offset_y * planar_row_nbytes +
                                                        offset_x * bytes_per_sample;
                        }
                    } else {
                        for (uint32_t c = 0; c < output_image.num_components; c++) {
                            output_tile.pixel_data[c] =
                                decode_buffer + c * planar_plane_nbytes + offset_y * planar_row_nbytes + offset_x * bytes_per_sample;
                        }
                    }
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                        "nvjpeg2kDecodeTile: y=[" << tile_y_begin << ", " << tile_y_end << "), x=[" << tile_x_begin << ", " << tile_x_end
                                                  << ")");

                    // Acquire tile resources
                    PerTileResources* tile_res = per_tile_res.Acquire();
                    // sync with previous tile work
                    XM_CHECK_CUDA(cudaEventSynchronize(tile_res->event_));

                    // sync tile stream with thread stream
                    XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, t.stream_));
                    XM_CHECK_CUDA(cudaStreamWaitEvent(tile_res->stream_, tile_res->event_));
                    {
                        auto tile_idx = tile_y * jpeg2k_info.num_tiles_x + tile_x;
                        nvtx3::scoped_range marker{"nvjpeg2kDecodeTile #" + std::to_string(tile_idx)};
                        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeTile(handle_, tile_res->state_, t.nvjpeg2k_stream_, decode_params_raii.get(),
                            tile_idx, 0, &output_tile, tile_res->stream_));
                    }
                    // sync thread stream with tile stream
                    XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, tile_res->stream_));
                    XM_CHECK_CUDA(cudaStreamWaitEvent(t.stream_, tile_res->event_));
                    // Release tile resources
                    per_tile_res.Release(tile_res);
                }
            }
        }

        if (needs_convert) {
            nvimgcodecImageInfo_t dec_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
            strcpy(dec_image_info.codec_name, "jpeg2k");
            dec_image_info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN; // not used by the converter
            dec_image_info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
            dec_image_info.sample_format = cs_image_info.sample_format; // original (planar) sample format
            dec_image_info.orientation = image_info.orientation;
            dec_image_info.region.struct_type = NVIMGCODEC_STRUCTURE_TYPE_REGION;
            dec_image_info.region.struct_size = sizeof(nvimgcodecStructureType_t);
            dec_image_info.region.struct_next = nullptr;
            dec_image_info.region.ndim = 0;
            dec_image_info.num_planes = num_components;
            for (size_t p = 0; p < dec_image_info.num_planes; p++) {
                dec_image_info.plane_info[p].struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO;
                dec_image_info.plane_info[p].struct_size = sizeof(nvimgcodecImagePlaneInfo_t);
                dec_image_info.plane_info[p].struct_next = nullptr;
                dec_image_info.plane_info[p].width = roi_width;
                dec_image_info.plane_info[p].height = roi_height;
                dec_image_info.plane_info[p].row_stride = planar_row_nbytes;
                dec_image_info.plane_info[p].num_channels = 1;
                dec_image_info.plane_info[p].sample_type = orig_data_type;
                dec_image_info.plane_info[p].precision = bpp;
            }
            dec_image_info.buffer = decode_buffer;
            dec_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
            dec_image_info.buffer_size = planar_plane_nbytes * num_components;
            dec_image_info.cuda_stream = image_info.cuda_stream;

            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "LaunchConvertNormKernel");
            nvimgcodec::LaunchConvertNormKernel(image_info, dec_image_info, t.stream_);
        }

        // Synchronize user stream with thread stream
        XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
        XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, t.event_));

        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg2k code stream - " << e.info());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }
    try {
        if (decode_tmp_buffer) {
            if (device_allocator_.device_free) {
                device_allocator_.device_free(device_allocator_.device_ctx, decode_tmp_buffer, decode_tmp_buffer_sz, t.stream_);
            } else {
                bool use_async_mem_ops = nvimgcodec::can_use_async_mem_ops(t.stream_);
                if (use_async_mem_ops) {
                    XM_CHECK_CUDA(cudaFreeAsync(decode_tmp_buffer, t.stream_));
                } else {
                    nvimgcodec::DeviceGuard device_guard(nvimgcodec::get_stream_device_id(t.stream_));
                    XM_CHECK_CUDA(cudaFree(decode_tmp_buffer));
                }

            }
            decode_tmp_buffer = nullptr;
            decode_tmp_buffer_sz = 0;
        }
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not free buffer - " << e.info());
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_decode_batch");
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

        auto task = [](int tid, int sample_idx, void* context) -> void {
            auto* this_ptr = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(context);
            auto& batch_item = this_ptr->code_stream_mgr_.get_batch_item(sample_idx);
            this_ptr->decodeImpl(batch_item, tid);
        };

        if (batch_size == 1) {
            task(0, 0, this);
        } else {
            auto executor = exec_params_->executor;
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
                executor->launch(executor->instance, exec_params_->device_id, sample_idx, this, task);
            }
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg2k batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::static_decode_batch(nvimgcodecDecoder_t decoder,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    if (decoder) {
        auto* handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } else {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
}

} // namespace nvjpeg2k
