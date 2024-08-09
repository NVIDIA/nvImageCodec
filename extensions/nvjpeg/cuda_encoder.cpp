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
#define NOMINMAX
#include "cuda_encoder.h"
#include <nvimgcodec.h>
#include <cassert>
#include <cstring>
#include <future>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include <nvjpeg.h>
#include <nvtx3/nvtx3.hpp>

#include "errors_handling.h"
#include "log.h"
#include "type_convert.h"

namespace nvjpeg {

NvJpegCudaEncoderPlugin::NvJpegCudaEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_, "jpeg", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
          static_create, Encoder::static_destroy, Encoder::static_can_encode, Encoder::static_encode_batch}
    , framework_(framework)
{
}

nvimgcodecEncoderDesc_t* NvJpegCudaEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::Encoder::canEncode(nvimgcodecProcessingStatus_t* status, nvimgcodecImageDesc_t** images,
    nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_can_encode");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(params);

    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg") != 0) {
            *result = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }

        nvimgcodecJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcodecJpegImageInfo_t*>(cs_image_info.struct_next);
        while (jpeg_image_info && jpeg_image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcodecJpegImageInfo_t*>(jpeg_image_info->struct_next);
        if (jpeg_image_info) {

            static const std::set<nvimgcodecJpegEncoding_t> supported_encoding{
                NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN};
            if (supported_encoding.find(jpeg_image_info->encoding) == supported_encoding.end()) {
                *result = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                continue;
            }
        }

        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &out_image_info);

        static const std::set<nvimgcodecColorSpec_t> supported_color_space{
            NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
            NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (supported_css.find(out_image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }

        static const std::set<nvimgcodecSampleFormat_t> supported_sample_format{
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_YUV,
            NVIMGCODEC_SAMPLEFORMAT_P_Y,
        };
        if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y) {
            if ((image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY) ||
                (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if ((image_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY) && (image_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC)) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
    }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpge can encode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::Encoder::static_can_encode(nvimgcodecEncoder_t encoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpegCudaEncoderPlugin::Encoder::Encoder(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
    , options_(options)
{
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

    if (device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, 0, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, 0, &handle_));
    }

    if (exec_params->device_allocator && (exec_params->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(exec_params->device_allocator->device_mem_padding, handle_));
    }
    if (exec_params->pinned_allocator && (exec_params->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(exec_params->pinned_allocator->pinned_mem_padding, handle_));
    }

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    encode_state_batch_ = std::make_unique<NvJpegCudaEncoderPlugin::EncodeState>(plugin_id_, framework_, handle_, num_threads);
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::create(
    nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_create_encoder");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(
            new NvJpegCudaEncoderPlugin::Encoder(plugin_id_, framework_, exec_params, options));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg encoder - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::static_create(
    void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(exec_params);
        NvJpegCudaEncoderPlugin* handle = reinterpret_cast<NvJpegCudaEncoderPlugin*>(instance);
        handle->create(encoder, exec_params, options);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvJpegCudaEncoderPlugin::Encoder::~Encoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg_destroy_encoder");
    encode_state_batch_.reset();
    if (handle_)
        XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg encoder - " << e.info());
    }
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::Encoder::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        NvJpegCudaEncoderPlugin::Encoder* handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        delete handle;
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvJpegCudaEncoderPlugin::EncodeState::EncodeState(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle, int num_threads)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , handle_(handle)
{
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG(nvjpegEncoderStateCreate(handle_, &res.state_, res.stream_));
    }
}

NvJpegCudaEncoderPlugin::EncodeState::~EncodeState()
{
    for (auto& res : per_thread_) {
        if (res.event_) {
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
        }

        if (res.stream_) {
            XM_CUDA_LOG_DESTROY(cudaStreamDestroy(res.stream_));
        }

        if (res.state_) {
            XM_NVJPEG_LOG_DESTROY(nvjpegEncoderStateDestroy(res.state_));
        }
    }
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::Encoder::encode(int sample_idx)
{
    auto executor = exec_params_->executor;
    executor->launch(executor->instance, exec_params_->device_id, sample_idx, encode_state_batch_.get(),
        [](int tid, int sample_idx, void* task_context) -> void {
            nvtx3::scoped_range marker{"encode " + std::to_string(sample_idx)};
            auto encode_state = reinterpret_cast<NvJpegCudaEncoderPlugin::EncodeState*>(task_context);
            nvimgcodecCodeStreamDesc_t* code_stream = encode_state->samples_[sample_idx].code_stream_;
            nvimgcodecImageDesc_t* image = encode_state->samples_[sample_idx].image_;
            const nvimgcodecEncodeParams_t* params = encode_state->samples_[sample_idx].params;
            auto& handle_ = encode_state->handle_;
            auto& framework_ = encode_state->framework_;
            auto& plugin_id_ = encode_state->plugin_id_;
            auto& t = encode_state->per_thread_[tid];
            auto& jpeg_state_ = t.state_;
            try {
                nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                image->getImageInfo(image->instance, &image_info);

                nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                code_stream->getImageInfo(code_stream->instance, &out_image_info);

                if (image_info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED);
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample data type. Only UINT8 is supported.");
                    return;
                }

                unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

                XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
                XM_CHECK_CUDA(cudaStreamWaitEvent(t.stream_, t.event_));

                nvjpegEncoderParams_t encode_params_;
                XM_CHECK_NVJPEG(nvjpegEncoderParamsCreate(handle_, &encode_params_, t.stream_));
                std::unique_ptr<std::remove_pointer<nvjpegEncoderParams_t>::type, decltype(&nvjpegEncoderParamsDestroy)> encode_params(
                    encode_params_, &nvjpegEncoderParamsDestroy);
                int num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
                auto sample_format = image_info.sample_format;
                auto color_spec = image_info.color_spec;
                nvimgcodecChromaSubsampling_t chroma_subsampling = image_info.chroma_subsampling;
                if (num_channels == 1) {
                    sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
                    color_spec = NVIMGCODEC_COLORSPEC_GRAY;
                    chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
                }
                auto nvjpeg_format = nvimgcodec_to_nvjpeg_format(image_info.sample_format);
                if (nvjpeg_format < NVJPEG_OUTPUT_UNCHANGED || nvjpeg_format > NVJPEG_OUTPUT_FORMAT_MAX) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample format.");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
                    return;
                }

                nvjpegImage_t input_image;
                unsigned char* ptr = device_buffer;
                for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                    input_image.channel[p] = ptr;
                    input_image.pitch[p] = image_info.plane_info[p].row_stride;
                    ptr += input_image.pitch[p] * image_info.plane_info[p].height;
                }

                XM_CHECK_NVJPEG(nvjpegEncoderParamsSetQuality(encode_params.get(), static_cast<int>(params->quality), t.stream_));
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - quality: " << static_cast<int>(params->quality));

                auto jpeg_image_info = static_cast<nvimgcodecJpegImageInfo_t*>(out_image_info.struct_next);
                while (jpeg_image_info && jpeg_image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
                    jpeg_image_info = static_cast<nvimgcodecJpegImageInfo_t*>(jpeg_image_info->struct_next);
                if (jpeg_image_info) {
                    nvjpegJpegEncoding_t encoding = nvimgcodec_to_nvjpeg_encoding(jpeg_image_info->encoding);
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - encoding: " << encoding);
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetEncoding(encode_params.get(), encoding, t.stream_));
                }

                auto jpeg_encode_params = static_cast<nvimgcodecJpegEncodeParams_t*>(params->struct_next);
                while (jpeg_encode_params && jpeg_encode_params->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS)
                    jpeg_encode_params = static_cast<nvimgcodecJpegEncodeParams_t*>(jpeg_encode_params->struct_next);
                if (jpeg_encode_params) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - optimized huffman: " << jpeg_encode_params->optimized_huffman);
                    XM_CHECK_NVJPEG(
                        nvjpegEncoderParamsSetOptimizedHuffman(encode_params.get(), jpeg_encode_params->optimized_huffman, t.stream_));
                } else {
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetOptimizedHuffman(encode_params.get(), 0, t.stream_));
                }
                auto out_chroma_subsampling_nvimgcodec = num_channels == 1 ? NVIMGCODEC_SAMPLING_GRAY : out_image_info.chroma_subsampling;
                nvjpegChromaSubsampling_t out_chroma_subsampling = nvimgcodec_to_nvjpeg_css(out_chroma_subsampling_nvimgcodec);
                if (out_chroma_subsampling != NVJPEG_CSS_UNKNOWN) {
                    XM_CHECK_NVJPEG(nvjpegEncoderParamsSetSamplingFactors(encode_params.get(), out_chroma_subsampling, NULL));
                }
                if (((color_spec == NVIMGCODEC_COLORSPEC_SYCC) &&
                        ((sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV) ||
                            (sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y))) ||
                    ((color_spec == NVIMGCODEC_COLORSPEC_GRAY) && (sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y))) {
                    nvjpegChromaSubsampling_t input_chroma_subsampling = nvimgcodec_to_nvjpeg_css(chroma_subsampling);
                    XM_CHECK_NVJPEG(nvjpegEncodeYUV(handle_, jpeg_state_, encode_params.get(), &input_image, input_chroma_subsampling,
                        image_info.plane_info[0].width, image_info.plane_info[0].height, t.stream_));
                } else {
                    nvjpegInputFormat_t input_format = static_cast<nvjpegInputFormat_t>(nvjpeg_format);
                    assert(input_format >= NVJPEG_INPUT_RGB && input_format <= NVJPEG_INPUT_BGRI);
                    XM_CHECK_NVJPEG(nvjpegEncodeImage(handle_, jpeg_state_, encode_params.get(), &input_image, input_format,
                        image_info.plane_info[0].width, image_info.plane_info[0].height, t.stream_));
                }

                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

                size_t length;
                XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, jpeg_state_, NULL, &length, t.stream_));

                t.compressed_data_.resize(length);
                XM_CHECK_NVJPEG(nvjpegEncodeRetrieveBitstream(handle_, jpeg_state_, t.compressed_data_.data(), &length, t.stream_));

                nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
                size_t output_size;
                io_stream->reserve(io_stream->instance, length);
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                io_stream->write(io_stream->instance, &output_size, static_cast<void*>(&t.compressed_data_[0]), t.compressed_data_.size());
                io_stream->flush(io_stream->instance);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            } catch (const NvJpegException& e) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode jpeg code stream - " << e.info());
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
        });
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::Encoder::encodeBatch(
    nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_encode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        encode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            encode_state_batch_->samples_.push_back(
                NvJpegCudaEncoderPlugin::EncodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }

    int batch_size = encode_state_batch_->samples_.size();
    for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
        this->encode(sample_idx);
    }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaEncoderPlugin::Encoder::static_encode_batch(nvimgcodecEncoder_t encoder, nvimgcodecImageDesc_t** images,
    nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpegCudaEncoderPlugin::Encoder*>(encoder);
        return handle->encodeBatch(images, code_streams, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

} // namespace nvjpeg
