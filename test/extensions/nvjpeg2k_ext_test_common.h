
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

#pragma once

#include <fstream>

#include <gtest/gtest.h>

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <nvjpeg2k.h>

#include "common.h"
#include "parsers/jpeg2k.h"

namespace nvimgcodec { namespace test {

class NvJpeg2kExtTestBase : public ExtensionTestBase
{
  public:
    virtual ~NvJpeg2kExtTestBase() = default;

    virtual void SetUp()
    {
        ExtensionTestBase::SetUp();
        nvjpeg2k_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        nvjpeg2k_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        nvjpeg2k_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvjpeg2k_extension_desc(&nvjpeg2k_extension_desc_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &nvjpeg2k_extension_, &nvjpeg2k_extension_desc_));

        nvimgcodecExtensionDesc_t jpeg2k_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc));
        nvimgcodecExtensionCreate(instance_, &jpeg2k_parser_extension_, &jpeg2k_parser_extension_desc);

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    }

    virtual void TearDown()
    {
        TearDownCodecResources();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(jpeg2k_parser_extension_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(nvjpeg2k_extension_));
        ExtensionTestBase::TearDown();
    }

    nvimgcodecExtensionDesc_t nvjpeg2k_extension_desc_{};
    nvimgcodecExtension_t nvjpeg2k_extension_;
    nvimgcodecExtension_t jpeg2k_parser_extension_;
};

class NvJpeg2kTestBase
{
  public:
    virtual ~NvJpeg2kTestBase() = default;

    virtual void SetUp()
    {
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kCreateSimple(&nvjpeg2k_handle_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeStateCreate(nvjpeg2k_handle_, &nvjpeg2k_decode_state_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamCreate(&nvjpeg2k_stream_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeParamsCreate(&nvjpeg2k_decode_params_));

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncoderCreateSimple(&nvjpeg2k_encoder_handle_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeStateCreate(nvjpeg2k_encoder_handle_, &nvjpeg2k_encode_state_));
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsCreate(&nvjpeg2k_encode_params_));
    }

    virtual void TearDown()
    {
        if (nvjpeg2k_encode_params_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsDestroy(nvjpeg2k_encode_params_));
            nvjpeg2k_encode_params_ = nullptr;
        }
        if (nvjpeg2k_encode_state_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeStateDestroy(nvjpeg2k_encode_state_));
            nvjpeg2k_encode_state_ = nullptr;
        }
        if (nvjpeg2k_encoder_handle_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncoderDestroy(nvjpeg2k_encoder_handle_));
            nvjpeg2k_encoder_handle_ = nullptr;
        }
        if (nvjpeg2k_decode_params_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeParamsDestroy(nvjpeg2k_decode_params_));
            nvjpeg2k_decode_params_ = nullptr;
        }
        if (nvjpeg2k_stream_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamDestroy(nvjpeg2k_stream_));
            nvjpeg2k_stream_ = nullptr;
        }
        if (nvjpeg2k_decode_state_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeStateDestroy(nvjpeg2k_decode_state_));
            nvjpeg2k_decode_state_ = nullptr;
        }
        if (nvjpeg2k_handle_) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDestroy(nvjpeg2k_handle_));
            nvjpeg2k_handle_ = nullptr;
        }
    };

    virtual void DecodeReference(const std::string& resources_dir, const std::string& file_name, nvimgcodecSampleFormat_t output_format,
        bool enable_color_convert, nvimgcodecImageInfo_t* cs_image_info = nullptr)
    {
        std::string file_path(resources_dir + '/' + file_name);
        std::ifstream input_stream(file_path.c_str(), std::ios::in | std::ios::binary | std::ios::ate);
        ASSERT_EQ(true, input_stream.is_open());
        std::streamsize file_size = input_stream.tellg();
        input_stream.seekg(0, std::ios::beg);
        std::vector<unsigned char> compressed_buffer(file_size);
        input_stream.read(reinterpret_cast<char*>(compressed_buffer.data()), file_size);
        ASSERT_EQ(true, input_stream.good());

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kStreamParse(nvjpeg2k_handle_, compressed_buffer.data(), static_cast<size_t>(file_size), 0, 0, nvjpeg2k_stream_));

        nvjpeg2kImageInfo_t image_info;
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamGetImageInfo(nvjpeg2k_stream_, &image_info));

        std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info(image_info.num_components);
        for (uint32_t c = 0; c < image_info.num_components; c++) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kStreamGetImageComponentInfo(nvjpeg2k_stream_, &image_comp_info[c], c));
        }
        nvjpeg2kImage_t decoded_image;
        int bytes_per_element = 1;
        if (image_comp_info[0].precision > 8 && image_comp_info[0].precision <= 16) {
            bytes_per_element = 2;
            decoded_image.pixel_type = image_comp_info[0].sgn ? NVJPEG2K_UINT16 : NVJPEG2K_INT16;
        } else if (image_comp_info[0].precision == 8) {
            bytes_per_element = 1;
            decoded_image.pixel_type = NVJPEG2K_UINT8;
        } else {
            ASSERT_EQ(false, true); //Unsupported precision
        }
        if (cs_image_info) {
            cs_image_info->plane_info[0].width = image_info.image_width;
            cs_image_info->plane_info[0].height = image_info.image_height;
            switch (decoded_image.pixel_type)
            {
            case NVJPEG2K_UINT16:
                cs_image_info->plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
                break;
            case NVJPEG2K_INT16:
                cs_image_info->plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
                break;
            case NVJPEG2K_UINT8:  // fall-through
            default:
                cs_image_info->plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
                break;
            }
            cs_image_info->plane_info[0].precision = image_comp_info[0].precision;
        }
        unsigned char* pBuffer = NULL;
        size_t buffer_size = image_info.image_width * image_info.image_height * bytes_per_element * image_info.num_components;
        ASSERT_EQ(cudaSuccess, cudaMalloc(reinterpret_cast<void**>(&pBuffer), buffer_size));
        std::unique_ptr<std::remove_pointer<void*>::type, decltype(&cudaFree)> pBuffer_raii(
                        pBuffer, &cudaFree);

        std::vector<unsigned char*> decode_output(image_info.num_components);
        std::vector<size_t> decode_output_pitch(image_info.num_components);
        for (uint32_t c = 0; c < image_info.num_components; c++) {
            decode_output[c] = pBuffer + c * image_info.image_width * image_info.image_height * bytes_per_element;
            decode_output_pitch[c] = image_info.image_width * bytes_per_element;
        }

        decoded_image.pixel_data = (void**)decode_output.data();
        decoded_image.pitch_in_bytes = decode_output_pitch.data();
        decoded_image.num_components = image_info.num_components;

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kDecodeParamsSetRGBOutput(nvjpeg2k_decode_params_, enable_color_convert));

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kDecodeImage(nvjpeg2k_handle_, nvjpeg2k_decode_state_, nvjpeg2k_stream_, nvjpeg2k_decode_params_, &decoded_image, 0));
        ASSERT_EQ(cudaSuccess, cudaDeviceSynchronize());

        ref_buffer_.resize(buffer_size);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(pBuffer), buffer_size,
                                   ::cudaMemcpyDeviceToHost));
    }

    virtual void EncodeReference(const nvimgcodecImageInfo_t& input_image_info, const nvimgcodecEncodeParams_t& params,
        const nvimgcodecJpeg2kEncodeParams_t& jpeg2k_enc_params, const nvimgcodecImageInfo_t& output_image_info,
        std::vector<unsigned char>* out_buffer)
    {
        auto assert_cudaFree = [](void *devPtr) {
            ASSERT_EQ(cudaSuccess, cudaFree(devPtr));
        };

        constexpr auto nvimgcodec2nvjpeg2k_prog_order = [](nvimgcodecJpeg2kProgOrder_t nvimgcodec_prog_order) -> nvjpeg2kProgOrder {
            switch (nvimgcodec_prog_order) {
            case NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP:
                return NVJPEG2K_LRCP;
            case NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP:
                return NVJPEG2K_RLCP;
            case NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL:
                return NVJPEG2K_RPCL;
            case NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL:
                return NVJPEG2K_PCRL;
            case NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL:
                return NVJPEG2K_CPRL;
            default:
                return NVJPEG2K_LRCP;
            }
        };

        constexpr auto nvimgcodec2nvjpeg2k_color_spec = [](nvimgcodecColorSpec_t color_spec) -> nvjpeg2kColorSpace_t {
            switch (color_spec) {
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
            default:
                return NVJPEG2K_COLORSPACE_UNKNOWN;
            };
        };
        std::vector<size_t> strides(input_image_info.num_planes);
        std::vector<nvjpeg2kImageComponentInfo_t> image_comp_info(input_image_info.num_planes);
        for (uint32_t i = 0; i < input_image_info.num_planes; ++i) {
            image_comp_info[i].component_width = input_image_info.plane_info[i].width;
            image_comp_info[i].component_height = input_image_info.plane_info[i].height;
            image_comp_info[i].precision = input_image_info.plane_info[i].precision
                                               ? input_image_info.plane_info[i].precision
                                               : (static_cast<unsigned int>(input_image_info.plane_info[i].sample_type) >> 8)&0xFF;
            image_comp_info[i].sgn = input_image_info.plane_info[i].sample_type & 0b1;
            strides[i] = input_image_info.plane_info[0].row_stride;
        }

        nvjpeg2kEncodeConfig_t enc_config{};
        enc_config.image_width = input_image_info.plane_info[0].width;
        enc_config.image_height = input_image_info.plane_info[0].height;
        enc_config.num_components = input_image_info.num_planes;
        enc_config.stream_type = jpeg2k_enc_params.stream_type == NVIMGCODEC_JPEG2K_STREAM_JP2 ? NVJPEG2K_STREAM_JP2 : NVJPEG2K_STREAM_J2K;
        enc_config.color_space = nvimgcodec2nvjpeg2k_color_spec(input_image_info.color_spec);
        enc_config.code_block_w = jpeg2k_enc_params.code_block_w;
        enc_config.code_block_h = jpeg2k_enc_params.code_block_h;
        enc_config.mct_mode = jpeg2k_enc_params.mct_mode;

        enc_config.prog_order = nvimgcodec2nvjpeg2k_prog_order(jpeg2k_enc_params.prog_order);
        enc_config.num_resolutions = jpeg2k_enc_params.num_resolutions;
        enc_config.image_comp_info = image_comp_info.data();

        enc_config.rsiz = jpeg2k_enc_params.ht ? 0x4000 : 0;
        enc_config.encode_modes = jpeg2k_enc_params.ht ? 64 : 0;

        if (params.quality_type != NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
            enc_config.irreversible = 1;
        }

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsSetEncodeConfig(nvjpeg2k_encode_params_, &enc_config));

        if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_QUALITY) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsSpecifyQuality(nvjpeg2k_encode_params_, NVJPEG2K_QUALITY_TYPE_Q_FACTOR, params.quality_value));
        } else if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_DEFAULT) {
            if (enc_config.color_space == NVJPEG2K_COLORSPACE_SRGB && enc_config.mct_mode == 0) {
                if (!jpeg2k_enc_params.ht) { // for HT use default quality
                    ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsSpecifyQuality(nvjpeg2k_encode_params_, NVJPEG2K_QUALITY_TYPE_TARGET_PSNR, 50));
                }
            } else {
                ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsSpecifyQuality(nvjpeg2k_encode_params_, NVJPEG2K_QUALITY_TYPE_Q_FACTOR, 75));
            }
        } else if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsSpecifyQuality(nvjpeg2k_encode_params_, NVJPEG2K_QUALITY_TYPE_QUANTIZATION_STEP, params.quality_value));
        } else if (params.quality_type == NVIMGCODEC_QUALITY_TYPE_PSNR) {
            ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS, nvjpeg2kEncodeParamsSpecifyQuality(nvjpeg2k_encode_params_, NVJPEG2K_QUALITY_TYPE_TARGET_PSNR, params.quality_value));
        } else {
            // quality types other than lossless are not supported
            ASSERT_EQ(params.quality_type, NVIMGCODEC_QUALITY_TYPE_LOSSLESS);
        }

        unsigned char* dev_buffer = nullptr;
        ASSERT_EQ(cudaSuccess, cudaMalloc((void**)&dev_buffer, GetBufferSize(input_image_info)));
        std::unique_ptr<std::remove_pointer<void*>::type, decltype(assert_cudaFree)> dev_buffer_raii(
                        dev_buffer, assert_cudaFree);
        ASSERT_EQ(cudaSuccess, cudaMemcpy(dev_buffer, input_image_info.buffer, GetBufferSize(input_image_info), cudaMemcpyHostToDevice));

        std::vector<unsigned short*> input_buffers_u16;
        std::vector<unsigned char*> input_buffers_u8;

        nvjpeg2kImage_t img_desc;
        memset(&img_desc, 0, sizeof(nvjpeg2kImage_t));
        img_desc.num_components = input_image_info.num_planes;
        img_desc.pitch_in_bytes = strides.data();

        if (image_comp_info[0].precision > 8 && image_comp_info[0].precision <= 16) {
            input_buffers_u16.resize(input_image_info.num_planes);
            img_desc.pixel_data = (void**)input_buffers_u16.data();
            img_desc.pixel_type = NVJPEG2K_UINT16;
        } else if (image_comp_info[0].precision == 8) {
            input_buffers_u8.resize(input_image_info.num_planes);
            img_desc.pixel_data = (void**)input_buffers_u8.data();
            img_desc.pixel_type = NVJPEG2K_UINT8;
        }
        unsigned char* comp_dev_buffer = dev_buffer;
        for (uint32_t i = 0; i < input_image_info.num_planes; ++i) {
            img_desc.pixel_data[i] = (void*)comp_dev_buffer;
            comp_dev_buffer += input_image_info.plane_info[i].height * input_image_info.plane_info[i].row_stride;
        }

        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kEncode(nvjpeg2k_encoder_handle_, nvjpeg2k_encode_state_, nvjpeg2k_encode_params_, &img_desc, NULL));

        size_t length;
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kEncodeRetrieveBitstream(nvjpeg2k_encoder_handle_, nvjpeg2k_encode_state_, NULL, &length, NULL));
        out_buffer->resize(length);
        ASSERT_EQ(NVJPEG2K_STATUS_SUCCESS,
            nvjpeg2kEncodeRetrieveBitstream(nvjpeg2k_encoder_handle_, nvjpeg2k_encode_state_, out_buffer->data(), &length, NULL));
    }

    nvjpeg2kBackend_t backend_ = NVJPEG2K_BACKEND_DEFAULT;
    nvjpeg2kHandle_t nvjpeg2k_handle_ = nullptr;
    nvjpeg2kDecodeState_t nvjpeg2k_decode_state_ = nullptr;
    nvjpeg2kStream_t nvjpeg2k_stream_ = nullptr;
    nvjpeg2kDecodeParams_t nvjpeg2k_decode_params_ = nullptr;
    std::vector<unsigned char> ref_buffer_;

    nvjpeg2kEncoder_t nvjpeg2k_encoder_handle_ = nullptr;
    nvjpeg2kEncodeState_t nvjpeg2k_encode_state_ = nullptr;
    nvjpeg2kEncodeParams_t nvjpeg2k_encode_params_ = nullptr;
};
}} // namespace nvimgcodec::test
