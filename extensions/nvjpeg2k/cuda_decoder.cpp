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

#include <nvimgcodec.h>
#include <nvjpeg2k.h>
#include <cstring>
#include <future>
#include <iostream>
#include <memory>
#include <set>
#include <sstream>
#include <vector>

#include <nvtx3/nvtx3.hpp>
#include "log.h"

#include "cuda_decoder.h"
#include "error_handling.h"

#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/sample_format_utils.h"

namespace nvjpeg2k {

NvJpeg2kDecoderPlugin::NvJpeg2kDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg2k", NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create,
          Decoder::static_destroy, Decoder::static_can_decode, Decoder::static_decode_batch}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvJpeg2kDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
    nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);

        *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg2k") != 0) {
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
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        nvtx3::scoped_range marker{"jpeg2k_can_decode"};
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        if (batch_size < (num_threads + 1)) {  // not worth parallelizing
            for (int i = 0; i < batch_size; i++)
                canDecode(&status[i], code_streams[i], images[i], params);
        } else {
            int num_blocks = num_threads + 1;  // the last block is processed in the current thread
            CanDecodeCtx canDecodeCtx{this, status, code_streams, images, params, batch_size, num_blocks};
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
                    ctx->this_ptr->canDecode(&ctx->status[i], ctx->code_streams[i], ctx->images[i], ctx->params);
                }
                if (block_idx < static_cast<int>(ctx->promise.size()))
                    ctx->promise[block_idx].set_value();
            };
            int block_idx = 0;
            for (; block_idx < num_threads; ++block_idx) {
                executor->launch(executor->instance, exec_params_->device_id, block_idx, &canDecodeCtx, task);
            }
            task(-1, block_idx, &canDecodeCtx);

            // wait for it to finish
            for (auto& f : fut)
                f.wait();
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
    num_parallel_tiles_ = 16;  // default 16 tiles in parallel for all threads
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

    if (exec_params_->device_allocator && (exec_params_->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG2K(nvjpeg2kSetDeviceMemoryPadding(exec_params_->device_allocator->device_mem_padding, handle_));
    }
    if (exec_params_->pinned_allocator && (exec_params_->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG2K(nvjpeg2kSetPinnedMemoryPadding(exec_params_->pinned_allocator->pinned_mem_padding, handle_));
    }

    // create resources per thread
    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    decode_state_batch_ = std::make_unique<NvJpeg2kDecoderPlugin::DecodeState>(plugin_id_, framework_, handle_,
        exec_params_->device_allocator, exec_params_->pinned_allocator, exec_params_->device_id, num_threads, num_parallel_tiles_);
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
        *decoder =
            reinterpret_cast<nvimgcodecDecoder_t>(new NvJpeg2kDecoderPlugin::Decoder(plugin_id_, framework_, exec_params, options));
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
        decode_state_batch_.reset();
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

NvJpeg2kDecoderPlugin::DecodeState::DecodeState(const char* id, const nvimgcodecFrameworkDesc_t* framework, nvjpeg2kHandle_t handle,
    nvimgcodecDeviceAllocator_t* device_allocator, nvimgcodecPinnedAllocator_t* pinned_allocator, int device_id, int num_threads,
    int num_parallel_tiles)
    : plugin_id_(id)
    , framework_(framework)
    , handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
    , device_id_(device_id)
    , per_tile_res_(id, framework, handle, num_parallel_tiles)
{
    per_thread_.reserve(num_threads);

    int nCudaDevAttrComputeCapabilityMajor, nCudaDevAttrComputeCapabilityMinor;
    XM_CHECK_CUDA(cudaDeviceGetAttribute(&nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id_));
    XM_CHECK_CUDA(cudaDeviceGetAttribute(&nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id_));
    cudaDeviceProp device_properties{};
    XM_CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device_id_));

    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kDecodeStateCreate(handle_, &res.state_));
        res.parse_state_ = std::make_unique<NvJpeg2kDecoderPlugin::ParseState>(plugin_id_, framework_);
    }
}

NvJpeg2kDecoderPlugin::DecodeState::~DecodeState()
{
    for (auto& res : per_thread_) {
        if (res.event_) {
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
        }
        if (res.stream_) {
            XM_CUDA_LOG_DESTROY(cudaStreamDestroy(res.stream_));
        }
        if (res.state_) {
            XM_NVJPEG2K_D_LOG_DESTROY(nvjpeg2kDecodeStateDestroy(res.state_));
        }
    }
}

NvJpeg2kDecoderPlugin::ParseState::ParseState(const char* id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(id)
    , framework_(framework)
{
    XM_CHECK_NVJPEG2K(nvjpeg2kStreamCreate(&nvjpeg2k_stream_));
}

NvJpeg2kDecoderPlugin::ParseState::~ParseState()
{
    if (nvjpeg2k_stream_) {
        XM_NVJPEG2K_D_LOG_DESTROY(nvjpeg2kStreamDestroy(nvjpeg2k_stream_));
    }
}

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::decode(int sample_idx, bool immediate)
{
    auto task = [](int tid, int sample_idx, void* context) -> void {
        nvtx3::scoped_range marker{"nvjpeg2k decode " + std::to_string(sample_idx)};
        auto* decode_state = reinterpret_cast<NvJpeg2kDecoderPlugin::DecodeState*>(context);
        auto& t = decode_state->per_thread_[tid];
        auto& per_tile_res = decode_state->per_tile_res_;
        auto& framework_ = decode_state->framework_;
        auto& plugin_id_ = decode_state->plugin_id_;
        auto* parse_state = t.parse_state_.get();
        auto jpeg2k_state = t.state_;
        nvimgcodecCodeStreamDesc_t* code_stream = decode_state->samples_[sample_idx].code_stream;
        nvimgcodecImageDesc_t* image = decode_state->samples_[sample_idx].image;
        const nvimgcodecDecodeParams_t* params = decode_state->samples_[sample_idx].params;
        auto handle_ = decode_state->handle_;
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

            nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
            size_t encoded_stream_data_size = 0;
            io_stream->size(io_stream->instance, &encoded_stream_data_size);
            void* encoded_stream_data = nullptr;
            void* mapped_encoded_stream_data = nullptr;
            io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size);
            if (!mapped_encoded_stream_data) {
                if (parse_state->buffer_.size() != encoded_stream_data_size) {
                    parse_state->buffer_.resize(encoded_stream_data_size);
                    io_stream->seek(io_stream->instance, 0, SEEK_SET);
                    size_t read_nbytes = 0;
                    io_stream->read(io_stream->instance, &read_nbytes, &parse_state->buffer_[0], encoded_stream_data_size);
                    if (read_nbytes != encoded_stream_data_size) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected end-of-stream");
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                        return;
                    }
                } else {
                    encoded_stream_data = mapped_encoded_stream_data;
                }
                encoded_stream_data = &parse_state->buffer_[0];
            } else {
                encoded_stream_data = mapped_encoded_stream_data;
            }

            {
                nvtx3::scoped_range marker{"nvjpeg2kStreamParse"};
                XM_CHECK_NVJPEG2K(nvjpeg2kStreamParse(handle_, static_cast<const unsigned char*>(encoded_stream_data),
                    encoded_stream_data_size, false, false, parse_state->nvjpeg2k_stream_));
            }
            if (mapped_encoded_stream_data) {
                io_stream->unmap(io_stream->instance, &mapped_encoded_stream_data, encoded_stream_data_size);
            }

            nvjpeg2kImageInfo_t jpeg2k_info;
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageInfo(parse_state->nvjpeg2k_stream_, &jpeg2k_info));

            nvjpeg2kImageComponentInfo_t comp;
            XM_CHECK_NVJPEG2K(nvjpeg2kStreamGetImageComponentInfo(parse_state->nvjpeg2k_stream_, &comp, 0));
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

            std::vector<unsigned char*> decode_output(num_components);
            std::vector<size_t> pitch_in_bytes(num_components);
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
            code_stream->getImageInfo(code_stream->instance, &cs_image_info);
            bool convert_interleaved = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB ||
                               (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED && num_components > 1);
            bool convert_gray =
                (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y) && (cs_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_Y);
            bool convert_dtype = image_info.plane_info[0].sample_type != orig_data_type || (bpp != bits_per_sample && image_info.plane_info[0].precision != bpp);
            if (convert_dtype && out_data_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Only original dtype or conversion to uint8 is allowed");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }

            nvjpeg2kDecodeParams_t decode_params;
            nvjpeg2kDecodeParamsCreate(&decode_params);
            std::unique_ptr<std::remove_pointer<nvjpeg2kDecodeParams_t>::type, decltype(&nvjpeg2kDecodeParamsDestroy)> decode_params_raii(
                decode_params, &nvjpeg2kDecodeParamsDestroy);

            int rgb_output =
                image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB ||
                (image_info.color_spec == NVIMGCODEC_COLORSPEC_UNCHANGED && cs_image_info.color_spec == NVIMGCODEC_COLORSPEC_SRGB);
            XM_CHECK_NVJPEG2K(nvjpeg2kDecodeParamsSetRGBOutput(decode_params, rgb_output));
            // original dtype nbytes
            size_t row_nbytes;
            size_t component_nbytes;
            // output dtype nbytes
            size_t out_row_nbytes;
            size_t out_component_nbytes;
            if (!convert_interleaved && num_components < image_info.num_planes) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of planes");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            } else if (convert_interleaved && (num_components < image_info.plane_info[0].num_channels && num_components != 1)) {
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
            if (params->enable_roi && image_info.region.ndim > 0) {
                auto region = image_info.region;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                    "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
                uint32_t roi_width = region.end[1] - region.start[1];
                uint32_t roi_height = region.end[0] - region.start[0];
                XM_CHECK_NVJPEG2K(
                    nvjpeg2kDecodeParamsSetDecodeArea(decode_params, region.start[1], region.end[1], region.start[0], region.end[0]));
                for (size_t p = 0; p < image_info.num_planes; p++) {
                    if (roi_height != image_info.plane_info[p].height || roi_width != image_info.plane_info[p].width) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected plane info dimensions");
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                        return;
                    }
                }
                row_nbytes = roi_width * bytes_per_sample;
                component_nbytes = roi_height * row_nbytes;
                out_row_nbytes = roi_width * out_bytes_per_sample;
                out_component_nbytes = roi_height * out_row_nbytes;
            } else {
                for (size_t p = 0; p < image_info.num_planes; p++) {
                    if (height != image_info.plane_info[p].height || width != image_info.plane_info[p].width) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected plane info dimensions");
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                        return;
                    }
                }
                row_nbytes = width * bytes_per_sample;
                component_nbytes = height * row_nbytes;
                out_row_nbytes = width * out_bytes_per_sample;
                out_component_nbytes = height * out_row_nbytes;
            }

            if (image_info.buffer_size < out_component_nbytes * image_info.num_planes) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "The provided buffer can't hold the decoded image : " << image_info.num_planes);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }

            uint8_t* decode_buffer = nullptr;
            bool needs_convert = convert_gray || convert_interleaved || convert_dtype;
            bool planar_subset = image_info.num_planes > 1 && num_components > image_info.num_planes;
            if (needs_convert || planar_subset) {
                size_t current_offset = 0;
                if (needs_convert) {
                    // If there are conversions needed, we decode to a temporary buffer first
                    current_offset = num_components * component_nbytes;
                } else if (planar_subset) {
                    // If there are more components than we want, we allocate temp memory for the planes we don't need
                    current_offset = (num_components - image_info.num_planes) * component_nbytes;
                }

                decode_tmp_buffer_sz = current_offset;  // allocate a single chunk of memory for all the temporary buffers we need
                if (decode_state->device_allocator_) {
                    decode_state->device_allocator_->device_malloc(
                        decode_state->device_allocator_->device_ctx, &decode_tmp_buffer, decode_tmp_buffer_sz, t.stream_);
                } else {
                    XM_CHECK_CUDA(cudaMallocAsync(&decode_tmp_buffer, decode_tmp_buffer_sz, t.stream_));
                }
            }

            if (needs_convert) {
                decode_buffer = reinterpret_cast<uint8_t*>(decode_tmp_buffer);
                for (uint32_t p = 0; p < num_components; ++p) {
                    decode_output[p] = decode_buffer + p * component_nbytes;
                    pitch_in_bytes[p] = row_nbytes;
                }
            } else {
                uint32_t p = 0;
                decode_buffer = device_buffer;
                for (; p < image_info.num_planes; ++p) {
                    decode_output[p] = device_buffer + p * component_nbytes;
                    pitch_in_bytes[p] = row_nbytes;
                }
                for (; p < num_components; ++p) {
                    decode_output[p] = reinterpret_cast<uint8_t*>(decode_tmp_buffer) + (p - image_info.num_planes) * component_nbytes;
                    pitch_in_bytes[p] = row_nbytes;
                }
            }
            output_image.num_components = num_components;
            output_image.pixel_data = (void**)&decode_output[0];
            output_image.pitch_in_bytes = &pitch_in_bytes[0];

            // Waits for GPU stage from previous iteration (on this thread)
            XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

            bool tiled = (jpeg2k_info.num_tiles_y > 1 || jpeg2k_info.num_tiles_x > 1);
            if (!tiled || per_tile_res.size() <= 1 || image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) {
                nvtx3::scoped_range marker{"nvjpeg2kDecodeImage"};
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvjpeg2kDecodeImage");
                XM_CHECK_NVJPEG2K(nvjpeg2kDecodeImage(
                    handle_, jpeg2k_state, parse_state->nvjpeg2k_stream_, decode_params_raii.get(), &output_image, t.stream_));
            } else {
                std::vector<uint8_t*> tile_decode_output(jpeg2k_info.num_components, nullptr);

                bool has_roi = params->enable_roi && image_info.region.ndim > 0;
                std::set<cudaEvent_t*> tile_events;
                for (uint32_t tile_y = 0; tile_y < jpeg2k_info.num_tiles_y; tile_y++) {
                    for (uint32_t tile_x = 0; tile_x < jpeg2k_info.num_tiles_x; tile_x++) {
                        uint32_t tile_y_begin = tile_y * jpeg2k_info.tile_height;
                        uint32_t tile_y_end = std::min(tile_y_begin + jpeg2k_info.tile_height, jpeg2k_info.image_height);
                        uint32_t tile_x_begin = tile_x * jpeg2k_info.tile_width;
                        uint32_t tile_x_end = std::min(tile_x_begin + jpeg2k_info.tile_width, jpeg2k_info.image_width);
                        uint32_t roi_y_begin = has_roi ? static_cast<uint32_t>(image_info.region.start[0]) : 0;
                        uint32_t roi_x_begin = has_roi ? static_cast<uint32_t>(image_info.region.start[1]) : 0;
                        uint32_t roi_y_end = has_roi ? static_cast<uint32_t>(image_info.region.end[0]) : jpeg2k_info.image_height;
                        uint32_t roi_x_end = has_roi ? static_cast<uint32_t>(image_info.region.end[1]) : jpeg2k_info.image_width;
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
                        if (planar_subset) {
                            // Decode subset of planes directly to the output
                            for (; c < image_info.num_planes; c++) {
                                output_tile.pixel_data[c] =
                                    decode_buffer + c * component_nbytes + offset_y * row_nbytes + offset_x * bytes_per_sample;
                            }
                            // Decode remaining planes to a temp buffer
                            for (; c < output_image.num_components; c++) {
                                output_tile.pixel_data[c] = reinterpret_cast<uint8_t*>(decode_tmp_buffer) +
                                                            (c - image_info.num_planes) * component_nbytes + offset_y * row_nbytes +
                                                            offset_x * bytes_per_sample;
                            }
                        } else {
                            for (uint32_t c = 0; c < output_image.num_components; c++) {
                                output_tile.pixel_data[c] =
                                    decode_buffer + c * component_nbytes + offset_y * row_nbytes + offset_x * bytes_per_sample;
                            }
                        }
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                            "nvjpeg2kDecodeTile: y=[" << tile_y_begin << ", " << tile_y_end << "), x=[" << tile_x_begin << ", "
                                                      << tile_x_end << ")");

                        DecodeState::PerTileResources* tile_res = per_tile_res.Acquire();
                        XM_CHECK_CUDA(cudaEventSynchronize(tile_res->event_));
                        {
                            auto tile_idx = tile_y * jpeg2k_info.num_tiles_x + tile_x;
                            nvtx3::scoped_range marker{"nvjpeg2kDecodeTile #" + std::to_string(tile_idx)};
                            XM_CHECK_NVJPEG2K(nvjpeg2kDecodeTile(handle_, tile_res->state_, parse_state->nvjpeg2k_stream_,
                                decode_params_raii.get(), tile_idx, 0, &output_tile, tile_res->stream_));
                        }
                        XM_CHECK_CUDA(cudaEventRecord(tile_res->event_, tile_res->stream_));
                        tile_events.insert(&tile_res->event_);

                        per_tile_res.Release(tile_res);
                    }
                }
                for (auto *event : tile_events)
                    XM_CHECK_CUDA(cudaStreamWaitEvent(t.stream_, *event));
            }

            if (needs_convert) {
                auto dec_image_info = cs_image_info;
                dec_image_info.buffer = decode_buffer;
                dec_image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                dec_image_info.buffer_size = component_nbytes * num_components;
                nvimgcodec::LaunchConvertNormKernel(image_info, dec_image_info, t.stream_);
            }

            XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
            XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, t.event_));

            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg2k code stream - " << e.info());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        try {
            if (decode_tmp_buffer) {
                if (decode_state->device_allocator_) {
                    decode_state->device_allocator_->device_free(
                        decode_state->device_allocator_->device_ctx, decode_tmp_buffer, decode_tmp_buffer_sz, t.stream_);
                } else {
                    XM_CHECK_CUDA(cudaFreeAsync(decode_tmp_buffer, t.stream_));
                }
                decode_tmp_buffer = nullptr;
                decode_tmp_buffer_sz = 0;
            }
        } catch (const NvJpeg2kException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not free buffer - " << e.info());
        }
    };

    if (immediate) {
        task(0, sample_idx, decode_state_batch_.get());
    } else {
        auto executor = exec_params_->executor;
        executor->launch(executor->instance, exec_params_->device_id, sample_idx, decode_state_batch_.get(), std::move(task));
    }
    return NVIMGCODEC_STATUS_SUCCESS;
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

        decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            decode_state_batch_->samples_.push_back(
                NvJpeg2kDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }

        int batch_size = decode_state_batch_->samples_.size();
        bool immediate = batch_size == 1; //  if single image, do not use executor
        for (int i = 0; i < batch_size; i++) {
            this->decode(i, immediate);
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

nvimgcodecStatus_t NvJpeg2kDecoderPlugin::Decoder::static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    if (decoder) {
        auto* handle = reinterpret_cast<NvJpeg2kDecoderPlugin::Decoder*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } else {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
}

} // namespace nvjpeg2k
