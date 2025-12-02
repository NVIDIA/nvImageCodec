/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "metadata_extractor.h"

#include <library_types.h>
#include <nvimgcodec.h>
#include <nvjpeg.h>
#include <nvtiff.h>
#include <array>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <optional>
#include <set>
#include <vector>
#include "error_handling.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/out_of_bound_roi_fill.h"
#include "imgproc/sample_format_utils.h"
#include "imgproc/stream_device.h"
#include "imgproc/type_utils.h"
#include "dynlink/dynlink_nvtiff.h"
#include "log.h"
#include <utility>

namespace nvtiff {

struct Decoder
{
    using MetadataExtractor = nvtiff::MetadataExtractor;

    Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        const char* options = nullptr);
    ~Decoder();

    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count);
    static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->getMetadata(code_stream, metadata, metadata_count);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
    }

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
    nvimgcodecProcessingStatus_t canDecode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<Decoder*>(decoder);
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
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->decode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;;
        }
    }

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;
    bool nvtiffDecodeImageExAvailable_; // available since nvtiff 0.6.0

    struct PerThreadResources
    {
        const nvimgcodecFrameworkDesc_t* framework_;
        const char* plugin_id_;

        nvtiffStream_t nvtiff_stream_ = nullptr;
        nvtiffDecodeParams_t decode_params_ = nullptr;
        nvtiffDecoder_t decoder_ = nullptr;
        nvtiffImageInfo_t iinfo_ = {};
        cudaEvent_t event_ = nullptr;
        nvimgcodec::DeviceBuffer device_buffer_;
        std::optional<cudaStream_t> cuda_stream_ = std::nullopt;
        nvtiffDeviceAllocator_t device_allocator_ = {};
        nvtiffPinnedAllocator_t pinned_allocator_ = {};
        nvtiffDeviceAllocator_t* device_allocator_ptr_ = nullptr;
        nvtiffPinnedAllocator_t* pinned_allocator_ptr_ = nullptr;
        MetadataExtractor metadata_extractor_;
        std::optional<uint64_t> parsed_stream_id_ = std::nullopt;
        std::vector<unsigned char*> channels;
        std::vector<size_t> pitches;

        PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id,
            const nvimgcodecExecutionParams_t* exec_params)
            : framework_(framework)
            , plugin_id_(plugin_id)
            , device_buffer_(exec_params ? exec_params->device_allocator : nullptr)
            , metadata_extractor_(framework_, plugin_id_, &nvtiff_stream_)
        {
            XM_CHECK_NVTIFF(nvtiffStreamCreate(&nvtiff_stream_));
            XM_CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params_));
            XM_CHECK_CUDA(cudaEventCreate(&event_));

            if (exec_params && exec_params->device_allocator) {
                device_allocator_ = {
                    exec_params->device_allocator->device_malloc,
                    exec_params->device_allocator->device_free,
                    exec_params->device_allocator->device_ctx
                };
                device_allocator_ptr_ = &device_allocator_;
            } else {
                device_allocator_ptr_ = nullptr;
            }

            if (exec_params && exec_params->pinned_allocator) {
                pinned_allocator_ = {
                    exec_params->pinned_allocator->pinned_malloc,
                    exec_params->pinned_allocator->pinned_free,
                    exec_params->pinned_allocator->pinned_ctx
                };
                pinned_allocator_ptr_ = &pinned_allocator_;
            } else {
                pinned_allocator_ptr_ = nullptr;
            }
        }

        // Helper function to move members from other to this
        void moveFrom(PerThreadResources&& other) noexcept 
        {
            framework_ = std::exchange(other.framework_, nullptr);
            plugin_id_ = std::exchange(other.plugin_id_, nullptr);
            nvtiff_stream_ = std::exchange(other.nvtiff_stream_, nullptr);
            decode_params_ = std::exchange(other.decode_params_, nullptr);
            decoder_ = std::exchange(other.decoder_, nullptr);
            iinfo_ = std::move(other.iinfo_);
            event_ = std::exchange(other.event_, nullptr);
            device_buffer_ = std::move(other.device_buffer_);
            cuda_stream_ = std::move(other.cuda_stream_);
            device_allocator_ = std::move(other.device_allocator_);
            pinned_allocator_ = std::move(other.pinned_allocator_);
            device_allocator_ptr_ = std::exchange(other.device_allocator_ptr_, nullptr);
            pinned_allocator_ptr_ = std::exchange(other.pinned_allocator_ptr_, nullptr);

            parsed_stream_id_ = std::move(other.parsed_stream_id_);
        }

        PerThreadResources(PerThreadResources&& other) noexcept 
            : metadata_extractor_(std::move(other.metadata_extractor_))
        {
            moveFrom(std::move(other));
        }

        PerThreadResources& operator=(PerThreadResources&& other) noexcept {
            if (this != &other) {
                this->~PerThreadResources();
                moveFrom(std::move(other));
                metadata_extractor_ = std::move(other.metadata_extractor_);
            }
            return *this;
        }
        
        PerThreadResources(const PerThreadResources& other) = delete;
        PerThreadResources& operator=(const PerThreadResources& other) = delete;

        ~PerThreadResources() {
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
            XM_NVTIFF_LOG_DESTROY(nvtiffDecodeParamsDestroy(decode_params_));
            if (cuda_stream_.has_value()) {
                XM_CUDA_LOG_DESTROY(cudaStreamSynchronize(cuda_stream_.value()));
                XM_NVTIFF_LOG_DESTROY(nvtiffDecoderDestroy(decoder_, cuda_stream_.value()));
            }
            XM_NVTIFF_LOG_DESTROY(nvtiffStreamDestroy(nvtiff_stream_));
        }

        nvtiffDecoder_t& decoder(cudaStream_t cuda_stream) {
            if (cuda_stream_ != cuda_stream) {
                if (cuda_stream_) {  // should not really happen but just in case
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Recreating nvtiff decoder instance");
                    XM_CHECK_CUDA(cudaStreamSynchronize(cuda_stream_.value()));
                    XM_CHECK_NVTIFF(nvtiffDecoderDestroy(decoder_, cuda_stream_.value()));
                }
                XM_CHECK_NVTIFF(nvtiffDecoderCreate(&decoder_, device_allocator_ptr_, pinned_allocator_ptr_, cuda_stream));
                cuda_stream_ = cuda_stream;
            }
            return decoder_;
        }

        nvimgcodecStatus_t ensureStreamParsed(const nvimgcodecCodeStreamDesc_t* code_stream) {
            uint64_t io_stream_id = code_stream->io_stream->id;
            
            // Check if stream has already been parsed
            if (parsed_stream_id_.has_value() && parsed_stream_id_.value() == io_stream_id) {
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            
            // Parse the stream
            try {
                void* encoded_stream_data = nullptr;
                size_t encoded_stream_data_size = 0;
                if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
                if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
                    NVIMGCODEC_STATUS_SUCCESS) {
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }
                
                XM_CHECK_NVTIFF(nvtiffStreamParse(static_cast<const uint8_t*>(encoded_stream_data), encoded_stream_data_size, nvtiff_stream_));
                parsed_stream_id_ = io_stream_id;
                
                return NVIMGCODEC_STATUS_SUCCESS;
            } catch (const std::runtime_error& e) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not parse tiff data - " << e.what());
                return NVIMGCODEC_STATUS_INTERNAL_ERROR;
            }
        }
    };

    std::vector<PerThreadResources> per_thread_;
};

Decoder::Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
    const char* options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
    , nvtiffDecodeImageExAvailable_(nvtiffIsSymbolAvailable("nvtiffDecodeImageEx"))
{
    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    per_thread_.reserve(num_threads);
    while (per_thread_.size() < static_cast<size_t>(num_threads)) {
        per_thread_.emplace_back(framework_, plugin_id_, exec_params_);
    }
}

Decoder::~Decoder()
{
}

nvimgcodecStatus_t Decoder::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        Decoder* handle = reinterpret_cast<Decoder*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t Decoder::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) 
{
    try {

        int thread_idx = 0; //TODO: Implement getting thread index from execution params
        auto& resources = per_thread_[thread_idx];

        // Ensure stream is parsed before extracting metadata
        auto parse_ret = resources.ensureStreamParsed(code_stream);
        if (parse_ret != NVIMGCODEC_STATUS_SUCCESS) {
            return parse_ret;
        }

        //get metadata count
        if (metadata == nullptr) {
            if (!metadata_count) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            *metadata_count =  resources.metadata_extractor_.getMetadataCount(code_stream); 
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        if (!metadata_count || *metadata_count <= 0) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        
        //get buffer sets
        for (int i = 0; i < *metadata_count; i++) {
            if (!metadata[i]) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }

            //Check metadata structure type and format
            if (metadata[i]->struct_type != NVIMGCODEC_STRUCTURE_TYPE_METADATA ||
                metadata[i]->struct_size != sizeof(nvimgcodecMetadata_t)) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }

            auto ret = resources.metadata_extractor_.getMetadata(code_stream, metadata[i], i);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return ret;
            }
         }

        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
}

nvimgcodecProcessingStatus_t Decoder::canDecode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    (void) thread_idx;
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        bool is_tiff = strcmp(cs_image_info.codec_name, "tiff") == 0;
        if (!is_tiff) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        switch(cs_image_info.plane_info[0].precision) {
        case 8:
        case 16:
        case 32:
            break; // supported
        default:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "nvTIFF extension can only decode 8, 16 or 32 bit input.");
            break;
        }

        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
            break; // supported

        default:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        }

        // TODO: Do we care about this for decoding? I would argue that subsampling is used during encoding.
        // nvTIFF support decoding the same type of subsampled images as nvJPEG (as nvtTIFF uses it under the hood)
        // but I think this value should be NVIMGCODEC_SAMPLING_NONE even if we are decoding subsampled image.
        if ((image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) 
            && (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }

        // This codec doesn't apply EXIF orientation
        if (params->apply_exif_orientation &&
            (image_info.orientation.flip_x || image_info.orientation.flip_y || image_info.orientation.rotated != 0)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }

        if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB &&
            image_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY &&
            image_info.color_spec != NVIMGCODEC_COLORSPEC_UNKNOWN) {
            status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        if (codestream_info.code_stream_view) {
            const auto& region = codestream_info.code_stream_view->region;
            if (region.ndim == 0) {
                // ignore
            }
            else if (region.ndim == 2) {
                if (cs_image_info.plane_info[0].height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                    cs_image_info.plane_info[0].width > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Image dimensions exceeds int32, nvtiff ROI decode is not supported.");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
                if (nvimgcodec::is_region_out_of_bounds(region, cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height)) {
                    if (auto err_message = nvimgcodec::verify_region_fill_support(region, image_info); !err_message.empty()) {
                        status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, err_message);
                    }
                    if (!nvtiffDecodeImageExAvailable_) {
                        status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "nvTiff before 0.6.0 does not support out-of-bound ROI decoding.");
                    }
                }
                uint32_t roi_height = static_cast<uint32_t>(region.end[0] - region.start[0]);
                uint32_t roi_width = static_cast<uint32_t>(region.end[1] - region.start[1]);
                if (image_info.plane_info[0].width != roi_width || image_info.plane_info[0].height != roi_height) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "ROI size must match decoded image size.");
                }
            } else {
                status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Region decoding is supported only for 2 dimensions.");
            }
        }
        return status;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvTIFF can decode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

void setOutputFormatToUINT8RGB(nvimgcodecImageInfo_t& info, nvtiffDecodeParams_t decode_params) {
    info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    info.plane_info[0].precision = 8;
    info.plane_info[0].num_channels = 3;
    info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    XM_CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_RGB_I_UINT8));
}

void setOutputFormatToUINT16RGB(nvimgcodecImageInfo_t& info, nvtiffDecodeParams_t decode_params) {
    info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    info.plane_info[0].precision = 16;
    info.plane_info[0].num_channels = 3;
    info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    XM_CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(decode_params, NVTIFF_OUTPUT_RGB_I_UINT16));
}

nvimgcodecStatus_t Decoder::decode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    try {
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
        auto& per_thread = per_thread_[thread_idx];

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected buffer kind");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        if (nvimgcodec::IsPlanar(image_info.sample_format)) {
            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                if (image_info.plane_info[p].num_channels != 1) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "In planar format, each plane should have single channel");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }
        } else {
            if (image_info.num_planes != 1) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "In non planar format, image should have a single plane");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != image_info.plane_info[0].sample_type) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            if (image_info.plane_info[p].precision != image_info.plane_info[0].precision) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same precision");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            if (image_info.plane_info[p].width != image_info.plane_info[0].width ||
                image_info.plane_info[p].height != image_info.plane_info[0].height) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same dimension");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }
        auto region = nvimgcodecRegion_t{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr};
        bool has_roi = false;
        if(codestream_info.code_stream_view) {
            region = codestream_info.code_stream_view->region;
            has_roi = region.ndim != 0;
        }

        nvimgcodecImageInfo_t dec_image_info = cs_image_info;
        // nvTIFF decoded to interleaved, but our parser assumes planar
        if (dec_image_info.num_planes == 1) {
            dec_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        } else {
            dec_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        }
        dec_image_info.plane_info[0].num_channels = dec_image_info.num_planes;
        dec_image_info.num_planes = 1;

        if (nvimgcodec::NumberOfChannels(dec_image_info) < nvimgcodec::NumberOfChannels(image_info)) {
            if (dec_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_Y && dec_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_Y) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "When increasing number of channels, can convert only from P_Y or I_Y");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            if (!nvimgcodec::IsRgb(image_info.sample_format) && !nvimgcodec::IsBgr(image_info.sample_format)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "When increasing number of channels, can convert only to RGB or BGR");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        // All plane_info should be the same, so it is enough to check with only the first one
        if (image_info.plane_info[0].precision == 0) {
            image_info.plane_info[0].precision = image_info.plane_info[0].sample_type >> 8;
        }

        // Waits for GPU stage from previous iteration (on this thread)
        XM_CHECK_CUDA(cudaEventSynchronize(per_thread.event_));

        size_t image_id = codestream_info.code_stream_view ? codestream_info.code_stream_view->image_idx : 0;

        // Ensure stream is parsed
        auto parse_ret = per_thread.ensureStreamParsed(code_stream);
        if (parse_ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not parse tiff data");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        auto& tiff_info = per_thread.iinfo_;
        try {
            XM_CHECK_NVTIFF(nvtiffStreamGetImageInfo(per_thread.nvtiff_stream_, image_id, &tiff_info));
        } catch (const std::runtime_error& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get image info - " << e.what());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_MASK) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "NVTIFF_PHOTOMETRIC_MASK is not supported");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
            setOutputFormatToUINT16RGB(dec_image_info, per_thread.decode_params_);
        } else if (per_thread.iinfo_.photometric_int == NVTIFF_PHOTOMETRIC_YCBCR) {
            if (per_thread.iinfo_.compression != NVTIFF_COMPRESSION_JPEG) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "nvTIFF support YCbCr only with JPEG compression.");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
            //nvJPEG supports only 8 bit images
            setOutputFormatToUINT8RGB(dec_image_info, per_thread.decode_params_);
        } else if ((nvimgcodec::IsRgb(image_info.sample_format) || nvimgcodec::IsBgr(image_info.sample_format)) &&
                   image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
            setOutputFormatToUINT8RGB(dec_image_info, per_thread.decode_params_);
        } else if ((nvimgcodec::IsRgb(image_info.sample_format) || nvimgcodec::IsBgr(image_info.sample_format)) &&
                   image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
            setOutputFormatToUINT16RGB(dec_image_info, per_thread.decode_params_);
        } else if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_RGB && tiff_info.compression == NVTIFF_COMPRESSION_JPEG) {
            // For now nvTIFF doesn't support unchanged interleaved in that case
            setOutputFormatToUINT8RGB(dec_image_info, per_thread.decode_params_);
        } else {
            XM_CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(per_thread.decode_params_, NVTIFF_OUTPUT_UNCHANGED_I));
        }

        // dimensions of image that will be decode by nvtiff (excluding oob roi)
        size_t decoded_image_width = image_info.plane_info[0].width;
        size_t decoded_image_height = image_info.plane_info[0].height;
        int roi_y_begin = 0;
        int roi_x_begin = 0;
        int roi_y_end = cs_image_info.plane_info[0].height;
        int roi_x_end = cs_image_info.plane_info[0].width;
        if (has_roi) {
            NVIMGCODEC_LOG_DEBUG(
                framework_,
                plugin_id_,
                "Input ROI: y=[" << region.start[0] << ", " << region.end[0] << "); x=["
                                 << region.start[1] << ", " << region.end[1] << ")"
            );
            
            roi_y_begin = std::max(0, region.start[0]);
            roi_x_begin = std::max(0, region.start[1]);
            roi_y_end = std::min((int)cs_image_info.plane_info[0].height, region.end[0]);
            roi_x_end = std::min((int)cs_image_info.plane_info[0].width, region.end[1]);

            NVIMGCODEC_LOG_DEBUG(
                framework_,
                plugin_id_,
                "ROI After clipping: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" 
                                          << roi_x_begin << ", " << roi_x_end << ")"
            );

            if (roi_x_end < roi_x_begin || roi_y_end < roi_y_begin) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "invalid ROI");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }
        decoded_image_width = roi_x_end - roi_x_begin;
        decoded_image_height = roi_y_end - roi_y_begin;
        XM_CHECK_NVTIFF(nvtiffDecodeParamsSetROI(per_thread.decode_params_, roi_x_begin, roi_y_begin,
            decoded_image_width, decoded_image_height)
        );

        auto& nvtiff_decoder = per_thread.decoder(image_info.cuda_stream);
        nvtiffStatus_t decode_check_ret = nvtiffDecodeCheckSupported(per_thread.nvtiff_stream_, nvtiff_decoder, per_thread.decode_params_, image_id);
        if (decode_check_ret != NVTIFF_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "nvtiffDecodeCheckSupported returned error code: " << decode_check_ret);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
            return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
        }

        bool need_conversion =
            (dec_image_info.sample_format != image_info.sample_format && image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) ||
            nvimgcodec::NumberOfChannels(dec_image_info) != nvimgcodec::NumberOfChannels(image_info) ||
            dec_image_info.plane_info[0].precision != image_info.plane_info[0].precision ||
            dec_image_info.plane_info[0].sample_type != image_info.plane_info[0].sample_type;

        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, (need_conversion ? "nvtiffDecode with conversion" : "Direct nvtiffDecode"));

        if (need_conversion) {
            //TODO: Make FP work with conversion
            if (dec_image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32 ||
                image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "FP32 conversion is not supported");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            // If needing conversion, we decode to a temporary dec_image_info, then convert
            dec_image_info.plane_info[0].width = image_info.plane_info[0].width;
            dec_image_info.plane_info[0].height = image_info.plane_info[0].height;

            const size_t BPP = nvimgcodec::TypeSize(dec_image_info.plane_info[0].sample_type);
            dec_image_info.plane_info[0].row_stride = BPP * dec_image_info.plane_info[0].width * dec_image_info.plane_info[0].num_channels;
            assert(nvimgcodec::GetBufferSize(dec_image_info) == nvimgcodec::GetImageSize(dec_image_info));
            per_thread.device_buffer_.resize(nvimgcodec::GetBufferSize(dec_image_info), image_info.cuda_stream);
            dec_image_info.buffer = per_thread.device_buffer_.data;
        }

        const nvimgcodecImageInfo_t& target_iinfo = need_conversion ? dec_image_info : image_info;
        per_thread.channels.resize(target_iinfo.num_planes);
        per_thread.pitches.resize(target_iinfo.num_planes);
        if (nvtiffDecodeImageExAvailable_) {
            nvtiffImage_t nvtiff_image = {
                reinterpret_cast<void**>(&per_thread.channels[0]),
                per_thread.pitches.data(),
                target_iinfo.num_planes
            };
            unsigned char* ptr = reinterpret_cast<unsigned char*>(target_iinfo.buffer);
            nvtiff_image.num_planes = target_iinfo.num_planes;
            for (uint32_t c = 0; c < target_iinfo.num_planes; ++c) {
                nvtiff_image.plane_data[c] = ptr;
                nvtiff_image.plane_pitch_bytes[c] = target_iinfo.plane_info[c].row_stride;
                ptr += nvtiff_image.plane_pitch_bytes[c] * target_iinfo.plane_info[c].height;

                // Shift the channel pointers if the ROI starts before image's top-left corner
                if (has_roi) {
                    int roi_y_begin = codestream_info.code_stream_view->region.start[0];
                    int roi_x_begin = codestream_info.code_stream_view->region.start[1];
                    unsigned char* plane_data = reinterpret_cast<unsigned char*>(nvtiff_image.plane_data[c]);
                    plane_data += (roi_y_begin < 0) * (-roi_y_begin) * nvtiff_image.plane_pitch_bytes[c];
                    int bytes_per_sample = target_iinfo.plane_info[c].sample_type >> 11;
                    plane_data += (roi_x_begin < 0) * (-roi_x_begin) * bytes_per_sample * target_iinfo.plane_info[c].num_channels;
                    nvtiff_image.plane_data[c] = plane_data;
                }
            }
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvtiffDecodeImageEx");
            XM_CHECK_NVTIFF(nvtiffDecodeImageEx(
                per_thread.nvtiff_stream_, nvtiff_decoder, per_thread.decode_params_, image_id, &nvtiff_image, image_info.cuda_stream));
        } else { // nvtiff before 0.6.0
            bool oob = nvimgcodec::is_region_out_of_bounds(region, cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height);
            assert(!oob);
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvtiffDecodeImage");
            XM_CHECK_NVTIFF(nvtiffDecodeImage(
                per_thread.nvtiff_stream_, nvtiff_decoder, per_thread.decode_params_, image_id, target_iinfo.buffer, image_info.cuda_stream));
        }

        if (need_conversion) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "LaunchConvertNormKernel");
            nvimgcodec::LaunchConvertNormKernel(image_info, dec_image_info, image_info.cuda_stream);
        }

        if (has_roi && nvtiffDecodeImageExAvailable_) {
            // Only fill out-of-bound ROI when using nvTiff >= 0.6.0
            nvtx3::scoped_range marker{ "fill_out_of_bounds_region" };
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "fill_out_of_bounds_region");
            try {
                nvimgcodec::fill_out_of_bounds_region(
                    image_info,
                    cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                    codestream_info.code_stream_view->region
                );
            }
            catch (std::runtime_error& e) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not fill out of bounds ROI - " << e.what());
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }
        }

        // Record event on user stream to synchronize on next iteration
        XM_CHECK_CUDA(cudaEventRecord(per_thread.event_, image_info.cuda_stream));

        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to decode with nvTIFF - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

NvTiffCudaDecoderPlugin::NvTiffCudaDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "tiff",
          NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, Decoder::static_destroy, Decoder::static_get_metadata, Decoder::static_can_decode,
          Decoder::static_decode_sample, nullptr}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvTiffCudaDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t NvTiffCudaDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvtiff_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new Decoder(plugin_id_, framework_, exec_params, options));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvtiff decoder:" << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvTiffCudaDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    NvTiffCudaDecoderPlugin* handle = reinterpret_cast<NvTiffCudaDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}


} // namespace nvtiff
