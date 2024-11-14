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

#include "cuda_decoder.h"
#include <imgproc/device_guard.h>
#include <imgproc/stream_device.h>
#include <nvimgcodec.h>
#include <nvjpeg2k.h>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <sstream>
#include <vector>
#include "error_handling.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/sample_format_utils.h"
#include "log.h"

using nvimgcodec::DeviceBuffer;

namespace nvjpeg2k {

struct DecoderImpl;

struct PerThreadResources
{
    const nvimgcodecFrameworkDesc_t* framework_;
    const char* plugin_id_;
    nvjpeg2kHandle_t handle_;
    nvimgcodec::DeviceBuffer device_buffer_;

    cudaEvent_t event_;
    nvjpeg2kDecodeState_t state_;
    nvjpeg2kStream_t nvjpeg2k_stream_;
    std::optional<uint64_t> parsed_stream_id_;

    PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id, const nvimgcodecExecutionParams_t* exec_params,
        nvjpeg2kHandle_t handle)
        : framework_(framework)
        , plugin_id_(plugin_id)
        , handle_(handle)
        , device_buffer_(exec_params)
    {
        XM_CHECK_CUDA(cudaEventCreate(&event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &state_));
        XM_CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&nvjpeg2k_stream_));
    }

    ~PerThreadResources() {
        if (event_) {
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
        }
        if (state_) {
            XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(state_));
        }
        if (nvjpeg2k_stream_) {
            XM_NVJPEG2K_LOG_DESTROY(nvjpeg2kStreamDestroy(nvjpeg2k_stream_));
        }
    }
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

    size_t size() const {
        return res_.size();
    }

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

struct DecoderImpl
{
    DecoderImpl(
        const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~DecoderImpl();

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);

    nvimgcodecProcessingStatus_t canDecode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->canDecode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    nvimgcodecStatus_t decode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->decode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;;
        }
    }

    void parseOptions(const char* options);

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
};

NvJpeg2kDecoderPlugin::NvJpeg2kDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg2k",
          NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_sample, nullptr, nullptr}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvJpeg2kDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecProcessingStatus_t DecoderImpl::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        XM_CHECK_NULL(code_stream);
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        bool is_jpeg2k = strcmp(cs_image_info.codec_name, "jpeg2k") == 0;
        if (!is_jpeg2k)
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        XM_CHECK_NULL(params);

        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

        static const std::set<nvimgcodecColorSpec_t> supported_color_space{
            NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV) {
            static const std::set<nvimgcodecChromaSubsampling_t> supported_css{
                NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420};
            if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
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
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        static const std::set<nvimgcodecSampleDataType_t> supported_sample_type{
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16};
        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (supported_sample_type.find(sample_type) == supported_sample_type.end()) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
        return status;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can decode - " << e.info());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

void DecoderImpl::parseOptions(const char* options)
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

DecoderImpl::DecoderImpl(
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

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back(framework_, plugin_id_, exec_params_, handle_);
    }

    if (num_parallel_tiles_ >= num_threads) {
        per_tile_res_.res_.resize(num_parallel_tiles_);
        for (auto& tile_res : per_tile_res_.res_) {
            XM_CHECK_CUDA(cudaStreamCreateWithFlags(&tile_res.stream_, cudaStreamNonBlocking));
            XM_CHECK_CUDA(cudaEventCreate(&tile_res.event_));
            XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &tile_res.state_));
            per_tile_res_.free_.push(&tile_res);
        }
    } else {
        NVIMGCODEC_LOG_INFO(framework_, plugin_id_,
            "num_parallel_tiles(" << num_parallel_tiles_ << ") < num_threads(" << num_threads
                                  << "). Per-tile parallelization will be disabled");
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
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params, options));
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg2k decoder - " << e.info());
        return e.nvimgcodecStatus();
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<NvJpeg2kDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

DecoderImpl::~DecoderImpl()
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

        per_thread_.clear();

        if (handle_)
            XM_CHECK_NVJPEG2K(nvjpeg2kDestroy(handle_));
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg2k decoder - " << e.info());
    }
}

nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::decode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    assert(code_stream->io_stream);
    void* encoded_stream_data = nullptr;
    size_t encoded_stream_data_size = 0;
    if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
        NVIMGCODEC_STATUS_SUCCESS) {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    assert(encoded_stream_data != nullptr);
    assert(encoded_stream_data_size > 0);

    auto& t = per_thread_[thread_idx];
    auto& per_tile_res = per_tile_res_;
    auto jpeg2k_state = t.state_;

    try {
        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected buffer kind");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        if (code_stream->id != t.parsed_stream_id_) {
            nvtx3::scoped_range marker{"nvjpegJpegStreamParse"};
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size,
                false, false, t.nvjpeg2k_stream_));
            t.parsed_stream_id_ = code_stream->id;
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
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        nvjpeg2kImage_t output_image;

        nvimgcodecSampleDataType_t out_data_type = image_info.plane_info[0].sample_type;
        size_t out_bytes_per_sample = static_cast<unsigned int>(out_data_type) >> (8 + 3);
        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != out_data_type) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }
        size_t bytes_per_sample = static_cast<unsigned int>(orig_data_type) >> (8 + 3);
        size_t bits_per_sample = bytes_per_sample << 3;

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

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
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        } else if (convert_interleaved && (num_components < image_info.plane_info[0].num_channels && num_components != 1)) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        } else if (native_interleaved && image_info.plane_info[0].num_channels != num_components) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of channels");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != image_info.plane_info[0].sample_type) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }
        } else {
            for (size_t p = 0; p < image_info.num_planes; p++) {
                if (height != image_info.plane_info[p].height || width != image_info.plane_info[p].width) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected plane info dimensions");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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
            t.device_buffer_.resize(current_offset, image_info.cuda_stream);
        }

        std::vector<unsigned char*> decode_output(num_components);
        std::vector<size_t> pitch_in_bytes(num_components);
        if (native_interleaved) {
            decode_buffer = reinterpret_cast<uint8_t*>(device_buffer);
            decode_output[0] = decode_buffer;
            pitch_in_bytes[0] = roi_width * num_components * bytes_per_sample;
        } else if (needs_convert) {
            decode_buffer = reinterpret_cast<uint8_t*>(t.device_buffer_.data);
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
                decode_output[p] = reinterpret_cast<uint8_t*>(t.device_buffer_.data) + (p - image_info.num_planes) * planar_plane_nbytes;
                pitch_in_bytes[p] = planar_row_nbytes;
            }
        }
        output_image.num_components = num_components;
        output_image.pixel_data = (void**)&decode_output[0];
        output_image.pitch_in_bytes = &pitch_in_bytes[0];

        // Waits for GPU stage from previous iteration
        XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

        bool tiled = (jpeg2k_info.num_tiles_y > 1 || jpeg2k_info.num_tiles_x > 1);
        // If per-tile parallelization is smaller than per-thread parallelization, it is not worth to decode tile-by-tile.
        if (!tiled || per_tile_res.size() <= 1 || image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) {
            nvtx3::scoped_range marker{"nvjpeg2kDecodeImage"};
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpeg2kDecodeImage");
            XM_CHECK_NVJPEG2K(
                nvjpeg2kDecodeImage(handle_, jpeg2k_state, t.nvjpeg2k_stream_, decode_params_raii.get(), &output_image, image_info.cuda_stream));
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
                            output_tile.pixel_data[c] = reinterpret_cast<uint8_t*>(t.device_buffer_.data) +
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

                    // sync tile stream with user stream
                    XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, image_info.cuda_stream));
                    XM_CHECK_CUDA(cudaStreamWaitEvent(tile_res->stream_, tile_res->event_));
                    {
                        auto tile_idx = tile_y * jpeg2k_info.num_tiles_x + tile_x;
                        nvtx3::scoped_range marker{"nvjpeg2kDecodeTile #" + std::to_string(tile_idx) + " @" + std::to_string((uint64_t) tile_res)};
                        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeTile(handle_, tile_res->state_, t.nvjpeg2k_stream_, decode_params_raii.get(),
                            tile_idx, 0, &output_tile, tile_res->stream_));
                    }
                    // sync thread stream with tile stream
                    XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, tile_res->stream_));
                    XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, tile_res->event_));
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
            nvimgcodec::LaunchConvertNormKernel(image_info, dec_image_info, image_info.cuda_stream);
        }

        // Record event on user stream to synchronize on next iteration
        XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg2k code stream - " << e.info());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace nvjpeg2k
