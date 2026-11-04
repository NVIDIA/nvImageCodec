/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>
#include "nvimgcodec_tests.h"
#include "common.h"
#include "common_ext_decoder_test.h"

#define NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM 0
#define DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcodec { namespace test {

class CommonExtEncoderTest : public ExtensionTestBase
{
  public:
    CommonExtEncoderTest() {}

    // Reuse the decoder test's stride-mode vocabulary so encoder and decoder
    // stride tests stay consistent.
    using StrideMode = CommonExtDecoderTest::StrideMode;

    // Re-pad the encoder's INPUT buffer per the stride mode. image_buffer_ and
    // planar_out_buffer_ (the reference the decode is compared against) stay
    // packed; only the buffer the encoder reads from gets per-plane row padding,
    // so this exercises each encoder's strided-input handling without disturbing
    // the reference. Must be called after the packed image_buffer_ is filled and
    // the reference is built, but before the input Image is created.
    void applyInputStridePadding(StrideMode mode)
    {
        if (mode == StrideMode::Contiguous)
            return;
        const uint32_t num_planes = image_info_.num_planes;
        size_t packed_offsets[NVIMGCODEC_MAX_NUM_PLANES];
        size_t padded_offsets[NVIMGCODEC_MAX_NUM_PLANES];
        size_t packed_strides[NVIMGCODEC_MAX_NUM_PLANES];
        size_t padded_strides[NVIMGCODEC_MAX_NUM_PLANES];
        size_t packed_running = 0, padded_running = 0;
        for (uint32_t p = 0; p < num_planes; ++p) {
            packed_strides[p] = image_info_.plane_info[p].row_stride;
            padded_strides[p] = packed_strides[p] + CommonExtDecoderTest::ExtraRowPadding(mode, p);
            packed_offsets[p] = packed_running;
            padded_offsets[p] = padded_running;
            packed_running += packed_strides[p] * image_info_.plane_info[p].height;
            padded_running += padded_strides[p] * image_info_.plane_info[p].height;
        }
        strided_input_buffer_.assign(padded_running, 0);
        for (uint32_t p = 0; p < num_planes; ++p) {
            for (uint32_t y = 0; y < image_info_.plane_info[p].height; ++y) {
                memcpy(strided_input_buffer_.data() + padded_offsets[p] + y * padded_strides[p],
                       image_buffer_.data() + packed_offsets[p] + y * packed_strides[p], packed_strides[p]);
            }
            image_info_.plane_info[p].row_stride = padded_strides[p];
        }
        image_info_.buffer = strided_input_buffer_.data();
    }


    void SetUp()
    {
        ExtensionTestBase::SetUp();

        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        images_.clear();
        streams_.clear();

        image_width_ = 600;
        image_height_ = 400;
    }

    void CreateDecoderAndEncoder()
    {
        nvimgcodecExecutionParams_t exec_params = {NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = 1;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, nullptr));

        encode_params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr};
        decode_params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
    }

    void TearDown()
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        if (decoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        }
        TearDownCodecResources();
        for (auto& ext : extensions_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(ext));

        ExtensionTestBase::TearDown();
    }


    // In the future we might want to put this function in core nvimagecodec functions
    static bool isLossyEncoding(std::string codec_name, nvimgcodecImageInfo_t *image_info, nvimgcodecEncodeParams_t *encode_params) {
        if (encode_params->quality_type == NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
            return false;
        } else if (encode_params->quality_type == NVIMGCODEC_QUALITY_TYPE_DEFAULT) {
            return codec_name == "jpeg" || codec_name == "jpeg2k" || codec_name == "webp";
        } else {
            return true;
        }
    }

    void dump_file_if_debug(std::string codec_name) {
        if constexpr (NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM) {
            std::string extension = std::string(".") + codec_name;
            if (extension == ".jpeg") {
                extension = ".jpg";
            }
            if (extension == ".jpeg2k") {
                extension = ".jp2";
            }
            const ::testing::TestInfo* test_info = ::testing::UnitTest::GetInstance()->current_test_info();
            std::string test_name = test_info->name();
            std::replace(test_name.begin(), test_name.end(), '/', '_');

            std::string filename = std::string("encoded_out_") + test_name + extension;
            std::ofstream b_stream(filename.c_str(), std::fstream::out | std::fstream::binary);
            b_stream.write(reinterpret_cast<char*>(code_stream_buffer_.data()), code_stream_buffer_.size());
            b_stream.close();
            ASSERT_TRUE(b_stream);

            auto ref ="./ref_" + test_name +  ".bmp";
            auto plane_size = image_info_.plane_info[0].height * image_info_.plane_info[0].width;
            write_bmp(
                ref.c_str(),
                planar_out_buffer_.data(), image_info_.plane_info[0].width,
                planar_out_buffer_.data() + plane_size, image_info_.plane_info[0].width,
                planar_out_buffer_.data() + 2 * plane_size, image_info_.plane_info[0].width,
                image_info_.plane_info[0].width, image_info_.plane_info[0].height
            );
        }
    }

    void genRandomImage()
    {
        srand(4771);
        for(unsigned int i = 0; i < image_buffer_.size(); ++i) {
            // create some pattern but also have random data inside
            image_buffer_[i] = i < image_buffer_.size() * 2 / 5 ? 150 : 0;
            image_buffer_[i] +=  rand() % 100;
        }
    }

    void TestEncodeDecodeSingleImage(const std::string& codec_name, nvimgcodecProcessingStatus_t expected_encode_status,
        bool add_alpha = false, StrideMode stride_mode = StrideMode::Contiguous)
    {
        image_info_.plane_info[0].width = image_width_;
        image_info_.plane_info[0].height = image_height_;
        image_info_.plane_info[0].precision = 8;
        image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        PrepareImageForFormat(add_alpha);
        genRandomImage();

        // convert input image to Planar RGB for debug and comparison
        ConvertToPlanar();

        image_info_.buffer = image_buffer_.data();
        // Optionally re-pad the encoder input (image_buffer_ / planar_out_buffer_
        // stay packed as the reference); exercises strided-input handling.
        applyInputStridePadding(stride_mode);
        strcpy(image_info_.codec_name, codec_name.c_str());
        nvimgcodecJpegImageInfo_t jpeg_optional_image_info = nvimgcodecJpegImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
        if(codec_name == "jpeg") {
            jpeg_optional_image_info.encoding = jpeg_encoding_;
            image_info_.struct_next = &jpeg_optional_image_info;
        }

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));

        nvimgcodecImageInfo_t encoded_image_info(image_info_);
        // The encoded stream describes the logical (packed) image; any per-plane
        // padding applied to the input buffer must not leak into the output
        // descriptor (otherwise the decoded buffer would be sized with padding).
        for (uint32_t p = 0; p < encoded_image_info.num_planes; ++p) {
            auto& pi = encoded_image_info.plane_info[p];
            pi.row_stride = static_cast<size_t>(pi.width) * pi.num_channels * TypeSize(pi.sample_type);
        }
        strcpy(encoded_image_info.codec_name, codec_name.c_str());
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this, &ResizeBufferStatic<CommonExtEncoderTest>, &encoded_image_info));

        nvimgcodecJpeg2kEncodeParams_t jpeg2k_optional_encode_params = {NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), nullptr,
            NVIMGCODEC_JPEG2K_STREAM_JP2, NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL, 6, 64, 64, 0, 0
        };
        if(codec_name == "jpeg2k") {
            jpeg2k_optional_encode_params.stream_type = jpeg2k_stream_type_;
            jpeg2k_optional_encode_params.ht = jpeg2k_ht_;
            encode_params_.struct_next = &jpeg2k_optional_encode_params;
        }

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, &in_image_, &out_code_stream_, 1, &encode_params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        size_t status_size;
        nvimgcodecProcessingStatus_t encode_status;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));

        ASSERT_EQ(expected_encode_status, encode_status);

        if (expected_encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            return;
        }

        dump_file_if_debug(codec_name);

        // read the compressed image info
        nvimgcodecImageInfo_t decoded_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(out_code_stream_, &decoded_info));

        // compare output with original image info
        ASSERT_EQ(decoded_info.color_spec, encoded_image_info.color_spec);
        ASSERT_EQ(decoded_info.sample_format, encoded_image_info.sample_format);
        ASSERT_EQ(decoded_info.num_planes, encoded_image_info.num_planes);
        ASSERT_EQ(decoded_info.chroma_subsampling, encoded_image_info.chroma_subsampling);
        for (uint32_t p = 0; p < decoded_info.num_planes; p++) {
            ASSERT_EQ(decoded_info.plane_info[p].width, encoded_image_info.plane_info[p].width);
            ASSERT_EQ(decoded_info.plane_info[p].height, encoded_image_info.plane_info[p].height);
            ASSERT_EQ(decoded_info.plane_info[p].num_channels, encoded_image_info.plane_info[p].num_channels);
            ASSERT_EQ(decoded_info.plane_info[p].sample_type, encoded_image_info.plane_info[p].sample_type);
            ASSERT_EQ(decoded_info.plane_info[p].precision, encoded_image_info.plane_info[p].precision);
        }

        std::vector<uint8_t> decode_buffer;
        decode_buffer.resize(GetBufferSize(decoded_info));
        decoded_info.buffer = decode_buffer.data();
        decoded_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        decoded_info.sample_format = add_alpha ? NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED : NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        decoded_info.num_planes = 3 + static_cast<int>(add_alpha);
        for (uint32_t p = 0; p < decoded_info.num_planes; p++) {
            decoded_info.plane_info[p].height = decoded_info.plane_info[0].height;
            decoded_info.plane_info[p].width = decoded_info.plane_info[0].width;
            decoded_info.plane_info[p].row_stride = decoded_info.plane_info[0].width;
            decoded_info.plane_info[p].num_channels = 1;
            decoded_info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            decoded_info.plane_info[p].precision = 0;
        }

        if (image_info_.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_Y) {
            decoded_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
            decoded_info.num_planes = 1;
        }

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &out_image_, &decoded_info));

        LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());

        nvimgcodecFuture_t decoder_future = nullptr;
        nvimgcodecStatus_t decoder_decode_status = nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &out_image_, 1, &decode_params_, &decoder_future);
        std::unique_ptr<std::remove_pointer<nvimgcodecFuture_t>::type, decltype(&nvimgcodecFutureDestroy)> decoder_future_raii(
                decoder_future, &nvimgcodecFutureDestroy);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, decoder_decode_status);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(decoder_future));
        nvimgcodecProcessingStatus_t decode_status;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(decoder_future, &decode_status, &status_size));
        ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, decode_status);

        ASSERT_EQ(image_buffer_.size(), decode_buffer.size());

#if (CV_VERSION_MAJOR > 4) || (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 14)
        constexpr bool webp_lossless_alpha_is_exact = true;
#else
        // OpenCV < 4.14 has no IMWRITE_WEBP_LOSSLESS_MODE, so the encoder can't request
        // exact preservation of RGB under alpha=0 pixels (see opencv_encoder.cpp).
        constexpr bool webp_lossless_alpha_is_exact = false;
#endif
        if (isLossyEncoding(codec_name, &image_info_, &encode_params_) ||
            (codec_name == "webp" && add_alpha && !webp_lossless_alpha_is_exact)) {
            float diff_sum = 0;
            for(size_t i=0; i < planar_out_buffer_.size(); ++i) {
                if (decode_buffer[i] > planar_out_buffer_[i]) {
                    diff_sum += decode_buffer[i] - planar_out_buffer_[i];
                } else {
                    diff_sum += planar_out_buffer_[i] - decode_buffer[i];
                }
            }
            auto diff_mean = diff_sum / planar_out_buffer_.size();

            if (chroma_subsampling_ == NVIMGCODEC_SAMPLING_444 && codec_name != "webp") { // webp is using subsampling
                ASSERT_LT(diff_mean, 15.f);
            } else {
                ASSERT_LT(diff_mean, 21.1f);
            }
        } else {
            for(size_t i=0; i < planar_out_buffer_.size(); ++i) {
                ASSERT_EQ(decode_buffer[i], planar_out_buffer_[i]);
            }
        }
    }

    std::vector<nvimgcodecExtension_t> extensions_;
    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecDecoder_t decoder_;
    nvimgcodecEncodeParams_t encode_params_;
    nvimgcodecDecodeParams_t decode_params_;
    int image_width_;
    int image_height_;
    
    nvimgcodecJpegEncoding_t jpeg_encoding_;
    int jpeg_optimized_huffman_;
    nvimgcodecJpeg2kBitstreamType_t jpeg2k_stream_type_;
    int jpeg2k_ht_;
    std::vector<unsigned char> strided_input_buffer_;
};

// Emit three encode->decode round-trip test cases for a CommonExtEncoderTest-
// derived fixture (one per input plane-stride layout), so each layout is a
// distinct gtest case - the encoder analog of the decoder
// DEFINE_SINGLE_IMAGE_STRIDE_TESTS macro. The fixture's SetUp must set
// sample_format_ / color_spec_ / chroma_subsampling_ and register the codec's
// encoder, decoder and parser extensions. CODEC is the codec name string; the
// round-trip is expected to succeed.
#define DEFINE_ENCODE_DECODE_STRIDE_TESTS_FOR_CODEC(FIXTURE, CODEC, ADD_ALPHA)                                       \
    TEST_P(FIXTURE, EncodeDecodeStrideContiguous)                                                                    \
    {                                                                                                                \
        TestEncodeDecodeSingleImage(CODEC, NVIMGCODEC_PROCESSING_STATUS_SUCCESS, ADD_ALPHA, StrideMode::Contiguous); \
    }                                                                                                                \
    TEST_P(FIXTURE, EncodeDecodeStrideSamePadAllPlanes)                                                              \
    {                                                                                                                \
        TestEncodeDecodeSingleImage(CODEC, NVIMGCODEC_PROCESSING_STATUS_SUCCESS, ADD_ALPHA, StrideMode::SamePadAllPlanes); \
    }                                                                                                                \
    TEST_P(FIXTURE, EncodeDecodeStrideDiffPadPerPlane)                                                               \
    {                                                                                                                \
        TestEncodeDecodeSingleImage(CODEC, NVIMGCODEC_PROCESSING_STATUS_SUCCESS, ADD_ALPHA, StrideMode::DiffPadPerPlane); \
    }

}} // namespace nvimgcodec::test
