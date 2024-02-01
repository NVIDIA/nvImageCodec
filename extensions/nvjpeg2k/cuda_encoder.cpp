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

#include "cuda_encoder.h"
#include <nvimgcodec.h>
#include <cstring>
#include <future>
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#include "error_handling.h"
#include "log.h"
#include "nvimgcodec_type_utils.h"
#include "nvjpeg2k.h"
#include "imgproc/convert_kernel_gpu.h"

namespace nvjpeg2k {

NvJpeg2kEncoderPlugin::NvJpeg2kEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_, "jpeg2k", NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create,
          Encoder::static_destroy, Encoder::static_can_encode, Encoder::static_encode_batch}
    , framework_(framework)
{
}

nvimgcodecEncoderDesc_t* NvJpeg2kEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::Encoder::canEncode(nvimgcodecProcessingStatus_t* status, nvimgcodecImageDesc_t** images,
    nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_can_encode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

    auto result = status;
    auto code_stream = code_streams;
    auto image = images;
    for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
        *result = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg2k") != 0) {
            *result = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }

        nvimgcodecJpeg2kEncodeParams_t* j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(params->struct_next);
        while (j2k_encode_params && j2k_encode_params->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS)
            j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(j2k_encode_params->struct_next);
        if (j2k_encode_params) {
            if ((j2k_encode_params->code_block_w != 32 || j2k_encode_params->code_block_h != 32) &&
                (j2k_encode_params->code_block_w != 64 || j2k_encode_params->code_block_h != 64)) {
                *result = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                        "Unsupported block size: " << j2k_encode_params->code_block_w << "x" << j2k_encode_params->code_block_h
                                                   << "(Valid values: 32, 64)");
            }
            if (j2k_encode_params->num_resolutions > NVJPEG2K_MAXRES) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_,
                        "Unsupported number of resolutions: " << j2k_encode_params->num_resolutions << " (max = " << NVJPEG2K_MAXRES
                                                              << ") ");
                *result = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
            }
        }

        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        (*image)->getImageInfo((*image)->instance, &image_info);
        nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        (*code_stream)->getImageInfo((*code_stream)->instance, &out_image_info);

        static const std::set<nvimgcodecColorSpec_t> supported_color_space{
            NVIMGCODEC_COLORSPEC_UNKNOWN, NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        static const std::set<nvimgcodecChromaSubsampling_t> supported_css{
            NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_GRAY};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (out_image_info.chroma_subsampling != image_info.chroma_subsampling) {
            *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
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
            *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y) {
            if ((image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY) ||
                (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if (image_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                *result |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
        }

        static const std::set<nvimgcodecSampleDataType_t> supported_sample_type{
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16};
        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (supported_sample_type.find(sample_type) == supported_sample_type.end()) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
    }
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg2k can encode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::Encoder::static_can_encode(nvimgcodecEncoder_t encoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpeg2kEncoderPlugin::Encoder::Encoder(
    const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(id)
    , framework_(framework)
    , exec_params_(exec_params)
    , options_(options)
{
    XM_CHECK_NVJPEG2K(nvjpeg2kEncoderCreateSimple(&handle_));

    auto executor = exec_params->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    encode_state_batch_ = std::make_unique<NvJpeg2kEncoderPlugin::EncodeState>(plugin_id_, framework_,
        handle_, exec_params->device_allocator, exec_params->pinned_allocator, exec_params->device_id, num_threads);
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::create(
    nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_create_encoder");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new NvJpeg2kEncoderPlugin::Encoder(plugin_id_, framework_, exec_params, options));
    return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg2k encoder - " << e.info());
        return e.nvimgcodecStatus();
    }
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::static_create(
    void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(exec_params);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin*>(instance);
        return handle->create(encoder, exec_params, options);
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
}

NvJpeg2kEncoderPlugin::Encoder::~Encoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_destroy_encoder");
    encode_state_batch_.reset();
    if (handle_)
            XM_CHECK_NVJPEG2K(nvjpeg2kEncoderDestroy(handle_));
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg2k decoder - " << e.info());
    }
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::Encoder::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        delete handle;
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvJpeg2kEncoderPlugin::EncodeState::EncodeState(const char* id, const nvimgcodecFrameworkDesc_t* framework, nvjpeg2kEncoder_t handle,
    nvimgcodecDeviceAllocator_t* device_allocator, nvimgcodecPinnedAllocator_t* pinned_allocator, int device_id, int num_threads)
    : plugin_id_(id)
    , framework_(framework)
    , handle_(handle)
    , device_allocator_(device_allocator)
    , pinned_allocator_(pinned_allocator)
    , device_id_(device_id)
{
    int nCudaDevAttrComputeCapabilityMajor, nCudaDevAttrComputeCapabilityMinor;
    XM_CHECK_CUDA(cudaDeviceGetAttribute(&nCudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, device_id_));
    XM_CHECK_CUDA(cudaDeviceGetAttribute(&nCudaDevAttrComputeCapabilityMinor, cudaDevAttrComputeCapabilityMinor, device_id_));
    cudaDeviceProp device_properties{};
    XM_CHECK_CUDA(cudaGetDeviceProperties(&device_properties, device_id_));

    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));
        XM_CHECK_NVJPEG2K(nvjpeg2kEncodeStateCreate(handle_, &res.state_));
    }
}

NvJpeg2kEncoderPlugin::EncodeState::~EncodeState()
{
    for (auto& res : per_thread_) {
        if (res.state_) {
            XM_NVJPEG2K_E_LOG_DESTROY(nvjpeg2kEncodeStateDestroy(res.state_));
        }
        if (res.event_) {
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
        }
        if (res.stream_) {
            XM_CUDA_LOG_DESTROY(cudaStreamDestroy(res.stream_));
        }
    }
}

static void fill_encode_config(nvjpeg2kEncodeConfig_t* encode_config, const nvimgcodecEncodeParams_t* params)
{
    nvimgcodecJpeg2kEncodeParams_t* j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(params->struct_next);
    while (j2k_encode_params && j2k_encode_params->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS)
        j2k_encode_params = static_cast<nvimgcodecJpeg2kEncodeParams_t*>(j2k_encode_params->struct_next);
    if (j2k_encode_params) {
        encode_config->stream_type = static_cast<nvjpeg2kBitstreamType>(j2k_encode_params->stream_type);
        encode_config->code_block_w = j2k_encode_params->code_block_w;
        encode_config->code_block_h = j2k_encode_params->code_block_h;
        encode_config->irreversible = j2k_encode_params->irreversible;
        encode_config->prog_order = static_cast<nvjpeg2kProgOrder>(j2k_encode_params->prog_order);
        encode_config->num_resolutions = j2k_encode_params->num_resolutions;
    }
    //Assume Gray color space when it is unknown and there is only one component
    if (encode_config->color_space == NVJPEG2K_COLORSPACE_UNKNOWN && encode_config->num_components == 1) {
        encode_config->color_space = NVJPEG2K_COLORSPACE_GRAY;
    }
}

static nvjpeg2kColorSpace_t nvimgcodec_to_nvjpeg2k_color_spec(nvimgcodecColorSpec_t nvimgcodec_color_spec)
{
    switch (nvimgcodec_color_spec) {
    case NVIMGCODEC_COLORSPEC_UNKNOWN:
        return NVJPEG2K_COLORSPACE_UNKNOWN;
    case NVIMGCODEC_COLORSPEC_SRGB:
        return NVJPEG2K_COLORSPACE_SRGB;
    case NVIMGCODEC_COLORSPEC_GRAY:
        return NVJPEG2K_COLORSPACE_GRAY;
    case NVIMGCODEC_COLORSPEC_SYCC:
        return NVJPEG2K_COLORSPACE_SYCC;
    case NVIMGCODEC_COLORSPEC_CMYK:
        return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
    case NVIMGCODEC_COLORSPEC_YCCK:
        return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
    case NVIMGCODEC_COLORSPEC_UNSUPPORTED:
        return NVJPEG2K_COLORSPACE_NOT_SUPPORTED;
    default:
        return NVJPEG2K_COLORSPACE_UNKNOWN;
    }
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::Encoder::encode(int sample_idx)
{
    auto executor = exec_params_->executor;

    executor->launch(executor->instance, exec_params_->device_id, sample_idx, encode_state_batch_.get(),
        [](int tid, int sample_idx, void* task_context) -> void {
            nvtx3::scoped_range marker{"encode " + std::to_string(sample_idx)};
            auto encode_state = reinterpret_cast<NvJpeg2kEncoderPlugin::EncodeState*>(task_context);
            auto& t = encode_state->per_thread_[tid];
            auto& framework_ = encode_state->framework_;
            auto& plugin_id_ = encode_state->plugin_id_;
            auto state_handle = t.state_;
            auto handle = encode_state->handle_;
            nvimgcodecCodeStreamDesc_t* code_stream = encode_state->samples_[sample_idx].code_stream;
            nvimgcodecImageDesc_t* image = encode_state->samples_[sample_idx].image;
            const nvimgcodecEncodeParams_t* params = encode_state->samples_[sample_idx].params;
            size_t tmp_buffer_sz = 0;
            void* tmp_buffer = nullptr;
            try {
                nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                image->getImageInfo(image->instance, &image_info);

                unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

                XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));
                XM_CHECK_CUDA(cudaStreamWaitEvent(t.stream_, t.event_));
                
                nvjpeg2kEncodeParams_t encode_params_;
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsCreate(&encode_params_));
                std::unique_ptr<std::remove_pointer<nvjpeg2kEncodeParams_t>::type, decltype(&nvjpeg2kEncodeParamsDestroy)> encode_params(
                    encode_params_, &nvjpeg2kEncodeParamsDestroy);

                auto sample_type = image_info.plane_info[0].sample_type;
                size_t bytes_per_sample = sample_type_to_bytes_per_element(sample_type);
                nvjpeg2kImageType_t nvjpeg2k_sample_type;
                switch (sample_type) {
                case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
                    nvjpeg2k_sample_type = NVJPEG2K_UINT8;
                    break;
                case NVIMGCODEC_SAMPLE_DATA_TYPE_INT16:
                    nvjpeg2k_sample_type = NVJPEG2K_INT16;
                    break;
                case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
                    nvjpeg2k_sample_type = NVJPEG2K_UINT16;
                    break;
                default:
                    FatalError(NVJPEG2K_STATUS_INVALID_PARAMETER, "Unexpected data type");
                }

                bool interleaved = image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB;
                size_t num_components = interleaved ? image_info.plane_info[0].num_channels : image_info.num_planes;
                std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info(num_components);

                uint32_t width = image_info.plane_info[0].width;
                uint32_t height = image_info.plane_info[0].height;

                std::vector<unsigned char*> encode_input(num_components);
                std::vector<size_t> pitch_in_bytes(num_components);

                if (interleaved) {
                    size_t row_nbytes = width * bytes_per_sample;
                    size_t component_nbytes = row_nbytes * height;
                    tmp_buffer_sz = component_nbytes * num_components;
                    if (encode_state->device_allocator_) {
                        encode_state->device_allocator_->device_malloc(
                            encode_state->device_allocator_->device_ctx, &tmp_buffer, tmp_buffer_sz, t.stream_);
                    } else {
                        XM_CHECK_CUDA(cudaMallocAsync(&tmp_buffer, tmp_buffer_sz, t.stream_));
                    }
                    device_buffer = reinterpret_cast<uint8_t*>(tmp_buffer);
                    for (uint32_t c = 0; c < num_components; ++c) {
                        encode_input[c] = device_buffer + c * component_nbytes;
                        pitch_in_bytes[c] = row_nbytes;
                    }

                    auto planar_info = image_info;
                    planar_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED;
                    planar_info.buffer = tmp_buffer;
                    planar_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                    planar_info.buffer_size = component_nbytes * num_components;
                    planar_info.num_planes = image_info.plane_info[0].num_channels;
                    for (size_t p = 0; p < planar_info.num_planes; p++) {
                        planar_info.plane_info[p].num_channels = 1;
                        planar_info.plane_info[p].height = image_info.plane_info[0].height;
                        planar_info.plane_info[p].width = image_info.plane_info[0].width;
                        planar_info.plane_info[p].precision = image_info.plane_info[0].precision;
                        planar_info.plane_info[p].row_stride = bytes_per_sample * image_info.plane_info[0].width;
                        planar_info.plane_info[p].sample_type = image_info.plane_info[0].sample_type;
                        planar_info.plane_info[p].struct_type = image_info.plane_info[0].struct_type;
                        planar_info.plane_info[p].struct_size = image_info.plane_info[0].struct_size;
                        planar_info.plane_info[p].struct_next = image_info.plane_info[0].struct_next;
                    }
                    nvimgcodec::LaunchConvertNormKernel(planar_info, image_info, t.stream_);
                } else {
                    size_t plane_start = 0;
                    for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                        encode_input[c] = device_buffer + plane_start;
                        pitch_in_bytes[c] = image_info.plane_info[c].row_stride;
                        plane_start += image_info.plane_info[c].height * image_info.plane_info[c].row_stride;
                    }
                }

                for (uint32_t c = 0; c < num_components; c++) {
                    image_comp_info[c].component_width = width;
                    image_comp_info[c].component_height = height;
                    image_comp_info[c].precision = sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16;
                    image_comp_info[c].sgn =
                        (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_INT8) || (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_INT16);
                }

                nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                code_stream->getImageInfo(code_stream->instance, &out_image_info);

                nvjpeg2kEncodeConfig_t encode_config;
                memset(&encode_config, 0, sizeof(encode_config));
                encode_config.color_space = nvimgcodec_to_nvjpeg2k_color_spec(image_info.color_spec);
                encode_config.image_width = width;
                encode_config.image_height = height;
                encode_config.num_components = num_components; // planar
                encode_config.image_comp_info = image_comp_info.data();
                encode_config.mct_mode =
                    ((out_image_info.color_spec == NVIMGCODEC_COLORSPEC_SYCC) || (out_image_info.color_spec == NVIMGCODEC_COLORSPEC_GRAY)) &&
                    (image_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC) && (image_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY);

                //Defaults
                encode_config.stream_type = NVJPEG2K_STREAM_JP2; // the bitstream will be in JP2 container format
                encode_config.code_block_w = 64;
                encode_config.code_block_h = 64;
                encode_config.irreversible = 0;
                encode_config.prog_order = NVJPEG2K_LRCP;
                encode_config.num_resolutions = 6;
                encode_config.num_layers = 1;
                encode_config.enable_tiling = 0;
                encode_config.enable_SOP_marker = 0;
                encode_config.enable_EPH_marker = 0;
                encode_config.encode_modes = 0;
                encode_config.enable_custom_precincts = 0;

                fill_encode_config(&encode_config, params);

                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetEncodeConfig(encode_params.get(), &encode_config));
                if (encode_config.irreversible) {
                    XM_CHECK_NVJPEG2K(nvjpeg2kEncodeParamsSetQuality(encode_params.get(), params->target_psnr));
                }

                nvjpeg2kImage_t input_image;
                input_image.num_components = num_components;
                input_image.pixel_data = reinterpret_cast<void**>(&encode_input[0]);
                input_image.pitch_in_bytes = pitch_in_bytes.data();
                input_image.pixel_type = nvjpeg2k_sample_type;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "before encode ");
                XM_CHECK_NVJPEG2K(nvjpeg2kEncode(handle, state_handle, encode_params.get(), &input_image, t.stream_));
                XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "after encode ");

                XM_CHECK_CUDA(cudaEventSynchronize(t.event_));
                size_t length;
                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(handle, state_handle, NULL, &length, t.stream_));

                t.compressed_data_.resize(length);

                XM_CHECK_NVJPEG2K(nvjpeg2kEncodeRetrieveBitstream(handle, state_handle, t.compressed_data_.data(), &length, t.stream_));

                nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
                size_t output_size;
                io_stream->reserve(io_stream->instance, length);
                io_stream->seek(io_stream->instance, 0, SEEK_SET);
                io_stream->write(io_stream->instance, &output_size, static_cast<void*>(&t.compressed_data_[0]), t.compressed_data_.size());
                io_stream->flush(io_stream->instance);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            } catch (const NvJpeg2kException& e) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode jpeg2k code stream - " << e.info());
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
            try {
                if (tmp_buffer) {
                    if (encode_state->device_allocator_) {
                        encode_state->device_allocator_->device_free(
                            encode_state->device_allocator_->device_ctx, tmp_buffer, tmp_buffer_sz, t.stream_);
                    } else {
                        XM_CHECK_CUDA(cudaFreeAsync(tmp_buffer, t.stream_));
                    }

                    tmp_buffer = nullptr;
                    tmp_buffer_sz = 0;
                }
            } catch (const NvJpeg2kException& e) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not free buffer - " << e.info());
            }
        });

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::Encoder::encodeBatch(
    nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg2k_encode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        encode_state_batch_->samples_.clear();
        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "batch size - " << batch_size);
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            encode_state_batch_->samples_.push_back(
                NvJpeg2kEncoderPlugin::EncodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }
    int batch_size = encode_state_batch_->samples_.size();
    for (int i = 0; i < batch_size; i++) {
        this->encode(i);
    }
    return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpeg2kException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode jpeg2k batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
}

nvimgcodecStatus_t NvJpeg2kEncoderPlugin::Encoder::static_encode_batch(nvimgcodecEncoder_t encoder, nvimgcodecImageDesc_t** images,
    nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvJpeg2kEncoderPlugin::Encoder*>(encoder);
        return handle->encodeBatch(images, code_streams, batch_size, params);
    } catch (const NvJpeg2kException& e) {
        return e.nvimgcodecStatus();
    }
}

} // namespace nvjpeg2k
