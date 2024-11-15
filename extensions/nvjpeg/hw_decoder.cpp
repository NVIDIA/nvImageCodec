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
namespace {

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        const char* options = nullptr);
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

    nvimgcodecStatus_t decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size,
        const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t** images,
        const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->decodeBatch(images, code_streams, batch_size, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
    }

    nvimgcodecStatus_t getMiniBatchSize(int* batch_size);
    static nvimgcodecStatus_t static_get_mini_batch_size(nvimgcodecDecoder_t decoder, int* batch_size)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->getMiniBatchSize(batch_size);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
    }

    void parseOptions(const char* options);

    const char* plugin_id_;
    nvjpegHandle_t handle_;
    nvjpegDevAllocatorV2_t device_allocator_;
    nvjpegPinnedAllocatorV2_t pinned_allocator_;
    const nvimgcodecFrameworkDesc_t* framework_;

    nvjpegJpegState_t state_;
    cudaEvent_t event_;
    std::vector<nvjpegJpegStream_t> nvjpeg_streams_;

    const nvimgcodecExecutionParams_t* exec_params_;
    unsigned int num_hw_engines_;
    unsigned int num_cores_per_hw_engine_;
    int preallocate_batch_size_ = 1;
    int preallocate_width_ = 1;
    int preallocate_height_ = 1;

    std::vector<const unsigned char*> batched_bitstreams_;
    std::vector<size_t> batched_bitstreams_size_;
    std::vector<nvjpegImage_t> batched_output_;
    std::vector<nvjpegDecodeParams_t> batched_nvjpeg_params_;
    std::vector<size_t> sample_idxs_;

    bool has_batched_ex_api_;
    nvjpegStatus_t hw_dec_info_status_ = NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;

    // Used in parallel CanDecode
    struct PerThreadResources {
        nvjpegJpegStream_t nvjpeg_stream_;
        nvjpegDecodeParams_t decode_params_;
        void* pinned_buffer_ = nullptr;
        size_t pinned_buffer_sz_ = 0;
        cudaStream_t pinned_buffer_stream_ = 0;

        explicit PerThreadResources(nvjpegHandle_t handle) {
            XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle, &nvjpeg_stream_));
            XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &decode_params_));
        }

        ~PerThreadResources() {
            if (pinned_buffer_sz_ > 0) {
                cudaStreamSynchronize(pinned_buffer_stream_);
                cudaFreeHost(pinned_buffer_);
            }
            nvjpegDecodeParamsDestroy(decode_params_);
            nvjpegJpegStreamDestroy(nvjpeg_stream_);
        }
    };
    std::vector<PerThreadResources> per_thread_;

    bool setDecodeParams(bool& need_params, nvimgcodecProcessingStatus_t& processing_status, nvjpegDecodeParams_t& nvjpeg_params,
        const nvimgcodecImageInfo_t& image_info, const nvimgcodecDecodeParams_t* user_params)
    {
        nvjpegDecodeParamsSetExifOrientation(nvjpeg_params, NVJPEG_ORIENTATION_NORMAL);
        if (user_params->apply_exif_orientation) {
            nvjpegExifOrientation_t orientation = nvimgcodec_to_nvjpeg_orientation(image_info.orientation);

            // This is a workaround for a known bug in nvjpeg.
            if (!nvjpeg_at_least(12, 2, 0)) {
                if (orientation == NVJPEG_ORIENTATION_ROTATE_90)
                    orientation = NVJPEG_ORIENTATION_ROTATE_270;
                else if (orientation == NVJPEG_ORIENTATION_ROTATE_270)
                    orientation = NVJPEG_ORIENTATION_ROTATE_90;
            }

            if (orientation == NVJPEG_ORIENTATION_UNKNOWN) {
                processing_status = NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
                return false;
            }

            if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                need_params = true;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Setting up EXIF orientation " << orientation);
                if (NVJPEG_STATUS_SUCCESS != nvjpegDecodeParamsSetExifOrientation(nvjpeg_params, orientation)) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeParamsSetExifOrientation failed");
                    processing_status = NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
                    return false;
                }
            }
        }

        if (user_params->enable_roi && image_info.region.ndim > 0) {
            need_params = true;
            auto region = image_info.region;
            auto roi_width = region.end[1] - region.start[1];
            auto roi_height = region.end[0] - region.start[0];
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
            if (NVJPEG_STATUS_SUCCESS !=
                nvjpegDecodeParamsSetROI(nvjpeg_params, region.start[1], region.start[0], roi_width, roi_height)) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeParamsSetROI failed");
                processing_status = NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                return false;
            }
        } else {
            nvjpegDecodeParamsSetROI(nvjpeg_params, 0, 0, -1, -1);
        }
        return true;
    }
};

}  // namespace

NvJpegHwDecoderPlugin::NvJpegHwDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg",
          NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_can_decode, nullptr,
          DecoderImpl::static_decode_batch, DecoderImpl::static_get_mini_batch_size}
    , framework_(framework)
{
}

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

nvimgcodecProcessingStatus_t DecoderImpl::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    auto& t = per_thread_[thread_idx];
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
        if (!is_jpeg) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
            NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_GRAY};
        if (supported_css.find(cs_image_info.chroma_subsampling) == supported_css.end()) {
            return NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }

        bool need_params = false;
        if (!setDecodeParams(need_params, status, t.decode_params_, image_info, params)) {
            return status;
        }

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

        XM_CHECK_NVJPEG(nvjpegJpegStreamParseHeader(
            handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, t.nvjpeg_stream_));

        int isSupported = -1;
        if (has_batched_ex_api_) {
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupportedEx(handle_, t.nvjpeg_stream_, t.decode_params_, &isSupported));
        } else {
            if (need_params) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpegDecodeBatchedSupportedEx API is not supported");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, t.nvjpeg_stream_, &isSupported));
        }
        if (isSupported == 0) {
            status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is supported");
        } else {
            status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "decoding image on HW is NOT supported");
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if hw nvjpeg can decode - " << e.info());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return status;
}

void DecoderImpl::parseOptions(const char* options)
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

DecoderImpl::DecoderImpl(
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
    int num_threads = executor->getNumThreads(executor->instance) + 1;  // +1 for the caller thread, which can also run canDecode
    per_thread_.reserve(num_threads);
    while (static_cast<int>(per_thread_.size()) < num_threads) {
        per_thread_.emplace_back(handle_);
    }

    hw_dec_info_status_ = NVJPEG_STATUS_IMPLEMENTATION_NOT_SUPPORTED;
    if (nvjpeg_at_least(11, 9, 0) && nvjpegIsSymbolAvailable("nvjpegGetHardwareDecoderInfo")) {
        hw_dec_info_status_ = nvjpegGetHardwareDecoderInfo(handle_, &num_hw_engines_, &num_cores_per_hw_engine_);
        if (hw_dec_info_status_ != NVJPEG_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "nvjpegGetHardwareDecoderInfo failed with return code " << hw_dec_info_status_);
            num_hw_engines_ = 0;
            num_cores_per_hw_engine_ = 0;
        }
    } else {
        num_hw_engines_ = 1;
        num_cores_per_hw_engine_ = 5;
        hw_dec_info_status_ = NVJPEG_STATUS_SUCCESS;
    }

    NVIMGCODEC_LOG_INFO(framework_, plugin_id_,
        "HW decoder available num_hw_engines=" << num_hw_engines_ << " num_cores_per_hw_engine=" << num_cores_per_hw_engine_);

    if (hw_dec_info_status_ == NVJPEG_STATUS_SUCCESS) {
        float hw_load = exec_params->num_backends == 0 ? 1.0f : 0.0f;
        for (int b = 0; b < exec_params->num_backends; b++) {
            auto &backend = exec_params->backends[b];
            if (backend.kind == NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY) {
                hw_load = backend.params.load_hint;
                break;
            }
        }
        int preferred_mini_batch = 0;
        getMiniBatchSize(&preferred_mini_batch);

        int full_batch_size = preallocate_batch_size_;
        preallocate_batch_size_ = static_cast<int>(std::round(hw_load * full_batch_size));
        if (preferred_mini_batch > 0) {
            int tail = preallocate_batch_size_ % preferred_mini_batch;
            if (tail > 0) {
                preallocate_batch_size_ = preallocate_batch_size_ + preferred_mini_batch - tail;
            }
        }
        NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "adjust preallocate_batch_size " << full_batch_size << " to " << preallocate_batch_size_);
    } else {
        NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "adjust preallocate_batch_size " << preallocate_batch_size_ << " to 0");
        preallocate_batch_size_ = 0;
    }

    XM_CHECK_NVJPEG(nvjpegJpegStateCreate(handle_, &state_));
    XM_CHECK_CUDA(cudaEventCreate(&event_));

    has_batched_ex_api_ = nvjpegIsSymbolAvailable("nvjpegDecodeBatchedEx")
        && nvjpegIsSymbolAvailable("nvjpegDecodeBatchedSupportedEx");

    // call nvjpegDecodeBatchedPreAllocate to use memory pool for HW decoder even if hint is 0
    // due to considerable performance benefit - >20% for 8GPU training
    if (hw_dec_info_status_ == NVJPEG_STATUS_SUCCESS && nvjpegIsSymbolAvailable("nvjpegDecodeBatchedPreAllocate")) {
        if (preallocate_width_ < 1)
            preallocate_width_ = 1;
        if (preallocate_height_ < 1)
            preallocate_height_ = 1;
        nvjpegChromaSubsampling_t subsampling = NVJPEG_CSS_444;
        nvjpegOutputFormat_t format = NVJPEG_OUTPUT_RGBI;
        std::stringstream ss;
        ss << "nvjpegDecodeBatchedPreAllocate batch_size=" << preallocate_batch_size_ << " width=" << preallocate_width_
           << " height=" << preallocate_height_;
        auto msg = ss.str();
        NVIMGCODEC_LOG_INFO(framework_, plugin_id_, msg);
        nvtx3::scoped_range marker{msg};
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

        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params, options));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder:" << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegHwDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    NvJpegHwDecoderPlugin* handle = reinterpret_cast<NvJpegHwDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

DecoderImpl::~DecoderImpl()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_destroy");
        for (auto& nvjpeg_param : batched_nvjpeg_params_)
            XM_NVJPEG_LOG_DESTROY(nvjpegDecodeParamsDestroy(nvjpeg_param));
        if (event_)
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
        if (state_)
            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(state_));

        if (handle_)
            XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.info());
    }
}

nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        DecoderImpl* handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::getMiniBatchSize(int* batch_size)
{
    *batch_size = num_hw_engines_ * num_cores_per_hw_engine_;
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
    int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    if (thread_idx != 0) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Logic error: Implementation not multithreaded");
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_hw_decode_batch, " << batch_size << " samples");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        batched_bitstreams_.clear();
        batched_bitstreams_size_.clear();
        batched_output_.clear();
        sample_idxs_.clear();
        size_t pinned_buffer_sz = 0;

        nvjpegOutputFormat_t nvjpeg_format = NVJPEG_OUTPUT_UNCHANGED;
        bool need_params = false;

        batched_nvjpeg_params_.reserve(batch_size);
        while (static_cast<int>(batched_nvjpeg_params_.size()) < batch_size) {
            batched_nvjpeg_params_.emplace_back();
            nvjpegDecodeParamsCreate(handle_, &batched_nvjpeg_params_.back());
        }

        // bool pageable = false;
        cudaStream_t stream;
        for (int sample_idx = 0, i = 0; sample_idx < batch_size; sample_idx++) {
            auto* image = images[sample_idx];
            XM_CHECK_NULL(image);

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            auto ret = image->getImageInfo(image->instance, &image_info);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                continue;
            }

            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

            if (sample_idx == 0) {
                nvjpeg_format = nvimgcodec_to_nvjpeg_format(image_info.sample_format);
                stream = image_info.cuda_stream;
            } else {
                if (stream != image_info.cuda_stream) {
                    throw std::logic_error("Expected the same CUDA stream for all the samples in the minibatch (" + std::to_string((uint64_t)stream) + "!=" + std::to_string((uint64_t)image_info.cuda_stream) + ")");
                }
                if (nvjpeg_format != nvimgcodec_to_nvjpeg_format(image_info.sample_format)) {
                    throw std::logic_error("Expected the same format for all the samples in the minibatch");
                }
            }

            XM_CHECK_NULL(code_streams[sample_idx]);
            const auto* code_stream = code_streams[sample_idx];
            assert(code_stream->io_stream);
            void* encoded_stream_data = nullptr;
            size_t encoded_stream_data_size = 0;
            if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                continue;
            }
            if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
                NVIMGCODEC_STATUS_SUCCESS) {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
                continue;
            }
            assert(encoded_stream_data != nullptr);
            assert(encoded_stream_data_size > 0);

            // cudaPointerAttributes attributes;
            // pageable |= cudaPointerGetAttributes(&attributes, encoded_stream_data) == cudaErrorInvalidValue ||
            //             attributes.type == cudaMemoryTypeUnregistered;

            nvimgcodecProcessingStatus_t tmp_status;
            bool need_params_tmp = false;
            if (!setDecodeParams(need_params_tmp, tmp_status, batched_nvjpeg_params_[i], image_info, params)) {
                image->imageReady(image->instance, tmp_status);
                continue;
            }
            need_params |= need_params_tmp;

            // get output image
            nvjpegImage_t nvjpeg_image;
            unsigned char* ptr = device_buffer;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                nvjpeg_image.channel[c] = ptr;
                nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
                ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
            }

            batched_bitstreams_.push_back(static_cast<const unsigned char*>(encoded_stream_data));
            batched_bitstreams_size_.push_back(encoded_stream_data_size);
            batched_output_.push_back(nvjpeg_image);
            sample_idxs_.push_back(sample_idx);
            pinned_buffer_sz += encoded_stream_data_size;
            i++;
        }
        if (batched_bitstreams_.size() > 0) {
            // Synchronize with previous iteration
            XM_CHECK_CUDA(cudaEventSynchronize(event_));
            XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle_, state_, batched_bitstreams_.size(), 1, nvjpeg_format));

            if (has_batched_ex_api_) {
                nvtx3::scoped_range marker{"nvjpegDecodeBatchedEx"};
                XM_CHECK_NVJPEG(nvjpegDecodeBatchedEx(handle_, state_, batched_bitstreams_.data(), batched_bitstreams_size_.data(),
                    batched_output_.data(), batched_nvjpeg_params_.data(), stream));
            } else {
                if (need_params)
                    throw std::logic_error("Unexpected error");
                nvtx3::scoped_range marker{"nvjpegDecodeBatched"};
                XM_CHECK_NVJPEG(nvjpegDecodeBatched(
                    handle_, state_, batched_bitstreams_.data(), batched_bitstreams_size_.data(), batched_output_.data(), stream));
            }
            XM_CHECK_CUDA(cudaEventRecord(event_, stream));
        }
        for (int sample_idx : sample_idxs_)
            images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.info());
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
            images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return e.nvimgcodecStatus();
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.what());
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++)
            images[sample_idx]->imageReady(images[sample_idx]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
}

} // namespace nvjpeg
