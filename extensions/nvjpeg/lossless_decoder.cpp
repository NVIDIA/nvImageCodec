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
#include <cassert>
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

    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const;
    static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->getMetadata(code_stream, metadata, metadata_count);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
    }

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

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);

    const char* plugin_id_;
    nvjpegHandle_t handle_;
    nvjpegDevAllocatorV2_t device_allocator_;
    nvjpegPinnedAllocatorV2_t pinned_allocator_;
    const nvimgcodecFrameworkDesc_t* framework_;

    nvjpegJpegState_t state_;
    cudaEvent_t event_;
    std::vector<nvjpegJpegStream_t> nvjpeg_streams_;
    const nvimgcodecExecutionParams_t* exec_params_;
};


} // namespace

NvJpegLosslessDecoderPlugin::NvJpegLosslessDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg",
    NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, static_create, DecoderImpl::static_destroy,  DecoderImpl::static_get_metadata, DecoderImpl::static_can_decode, nullptr,
          DecoderImpl::static_decode_batch, nullptr}
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

nvimgcodecStatus_t DecoderImpl::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const
{
    return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecProcessingStatus_t DecoderImpl::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    assert(thread_idx < static_cast<int>(nvjpeg_streams_.size()));
    auto& nvjpeg_stream = nvjpeg_streams_[thread_idx];
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        nvimgcodecJpegImageInfo_t jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&jpeg_info)};

        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
        bool is_lossless_huffman = jpeg_info.encoding == NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN;
        if (!is_jpeg) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        } else if (!is_lossless_huffman) {
            return NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
        }

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

        if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444 && image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;

        bool is_unchanged = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED && image_info.num_planes <= 2;
        bool is_y = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y && image_info.num_planes == 1;
        if (!(is_unchanged || is_y))
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;

        if (image_info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        
        XM_CHECK_NULL(code_stream); 
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        if (codestream_info.code_stream_view) {
            if (codestream_info.code_stream_view->image_idx != 0) {
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED;
            }
            auto region = codestream_info.code_stream_view->region;
            if (region.ndim > 0) {
                status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
            }
        }

        if (status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
            return status;


        auto* io_stream = code_stream->io_stream;
        XM_CHECK_NULL(io_stream);

        void* encoded_stream_data = nullptr;
        size_t encoded_stream_data_size = 0;
        if (io_stream->size(io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED;
        }
        if (io_stream->map(io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED;
        }
        assert(encoded_stream_data != nullptr);
        assert(encoded_stream_data_size > 0);

        XM_CHECK_NVJPEG(nvjpegJpegStreamParse(
            handle_, static_cast<const unsigned char*>(encoded_stream_data), encoded_stream_data_size, 0, 0, nvjpeg_stream));
        int isSupported = -1;
        XM_CHECK_NVJPEG(nvjpegDecodeBatchedSupported(handle_, nvjpeg_stream, &isSupported));
        if (isSupported == 0) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "decoding this lossless jpeg image is supported");
            return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        } else {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "decoding this lossless jpeg image is NOT supported");
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if lossless nvjpeg can decode - " << e.info());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
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
            reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params, options));
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
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    NvJpegLosslessDecoderPlugin* handle = reinterpret_cast<NvJpegLosslessDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

DecoderImpl::~DecoderImpl()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_destroy");

        for (auto& nvjpeg_stream : nvjpeg_streams_)
            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(nvjpeg_stream));

        if (event_)
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
        if (state_)
            XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(state_));

        if (handle_)
            XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg lossless decoder - " << e.info());
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

nvimgcodecStatus_t DecoderImpl::decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
    int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    if (thread_idx != 0) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Logic error: Implementation not multithreaded");
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_lossless_decode_batch, " << batch_size << " samples");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        std::vector<const unsigned char*> batched_bitstreams;
        std::vector<size_t> batched_bitstreams_size;
        std::vector<nvjpegImage_t> batched_output;
        std::vector<const nvimgcodecImageDesc_t*> batched_images;

        nvjpegOutputFormat_t nvjpeg_format{NVJPEG_OUTPUT_UNCHANGED};
        cudaStream_t stream{0};

        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
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
                    throw std::logic_error("Expected the same CUDA stream for all the samples in the minibatch");
                }
                if (nvjpeg_format != nvimgcodec_to_nvjpeg_format(image_info.sample_format)) {
                    throw std::logic_error("Expected the same format for all the samples in the minibatch");
                }
            }

            auto* code_stream = code_streams[sample_idx];
            XM_CHECK_NULL(code_stream);

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
                batched_bitstreams.push_back(static_cast<const unsigned char*>(encoded_stream_data));
                batched_bitstreams_size.push_back(encoded_stream_data_size);
                batched_output.push_back(nvjpeg_image);
                batched_images.push_back(image);
            } else {
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
        }

        if (batched_bitstreams.size() > 0) {
            // Synchronize with previous iteration
            XM_CHECK_CUDA(cudaEventSynchronize(event_));

            XM_CHECK_NVJPEG(nvjpegDecodeBatchedInitialize(handle_, state_, batched_bitstreams.size(), 1, nvjpeg_format));

            XM_CHECK_NVJPEG(nvjpegDecodeBatched(handle_, state_, batched_bitstreams.data(), batched_bitstreams_size.data(),
                batched_output.data(), stream));

            XM_CHECK_CUDA(cudaEventRecord(event_, stream));
        }

        for (auto* image : batched_images) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode lossless jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
}

} // namespace nvjpeg
