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

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "nvimgcodec_tests.h"
#include "nvjpeg_ext_test_common.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

#define NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM 0
#define NV_DEVELOPER_DEBUG_DUMP_DECODE_OUTPUT 0

namespace nvimgcodec { namespace test {

class NvJpegExtEncoderTestBase : public NvJpegExtTestBase
{
  public:
    NvJpegExtEncoderTestBase() {}

    virtual void SetUp()
    {
        NvJpegExtTestBase::SetUp();
        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));

        jpeg_enc_params_ = {NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), 0};
        jpeg_enc_params_.optimized_huffman = 0;
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.struct_next = &jpeg_enc_params_;
        params_.quality = 95;
        params_.target_psnr = 0;
        out_jpeg_image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    }

    virtual void TearDown()
    {
        if (encoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        NvJpegExtTestBase::TearDown();
    }

    nvimgcodecEncoder_t encoder_;
    nvimgcodecJpegEncodeParams_t jpeg_enc_params_;
    nvimgcodecEncodeParams_t params_;
    nvimgcodecJpegImageInfo_t out_jpeg_image_info_;
};

class NvJpegExtEncoderTestSingleImage : public NvJpegExtEncoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                            nvimgcodecChromaSubsampling_t, nvimgcodecChromaSubsampling_t, nvimgcodecJpegEncoding_t>>
{
  public:
    virtual ~NvJpegExtEncoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        NvJpegExtEncoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        encoded_chroma_subsampling_ = std::get<4>(GetParam());
        out_jpeg_image_info_.encoding = std::get<5>(GetParam());
        image_info_.struct_next = &out_jpeg_image_info_;
    }

    void TearDown() override
    {
        NvJpegTestBase::TearDown();
        NvJpegExtEncoderTestBase::TearDown();
    }

    nvimgcodecChromaSubsampling_t encoded_chroma_subsampling_;
};

TEST_P(NvJpegExtEncoderTestSingleImage, ValidFormatAndParameters)
{
    nvimgcodecImageInfo_t ref_cs_image_info;
    DecodeReference(resources_dir, image_file_, sample_format_, true, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    PrepareImageForFormat();
    memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name,"jpeg");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpegExtEncoderTestSingleImage::ResizeBufferStatic<NvJpegExtEncoderTestSingleImage>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, 1);
    if (NV_DEVELOPER_DUMP_OUTPUT_CODE_STREAM) {
        std::ofstream b_stream("./encoded_out.jpg", std::fstream::out | std::fstream::binary);
        b_stream.write(reinterpret_cast<char*>(code_stream_buffer_.data()), code_stream_buffer_.size());
    }

    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcodecImageInfo_t load_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecJpegImageInfo_t load_jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    load_info.struct_next = &load_jpeg_info;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &load_info));
    EXPECT_EQ(out_jpeg_image_info_.encoding, load_jpeg_info.encoding);
    EXPECT_EQ(cs_image_info.chroma_subsampling, load_info.chroma_subsampling);

    std::vector<unsigned char> ref_out_buffer;
    EncodeReference(image_info_, params_, jpeg_enc_params_, cs_image_info, out_jpeg_image_info_, &ref_out_buffer);
    ASSERT_EQ(0,
        memcmp(reinterpret_cast<void*>(ref_out_buffer.data()), reinterpret_cast<void*>(code_stream_buffer_.data()), ref_out_buffer.size()));
}

// clang-format off

static const char* css_filenames[] = {"/jpeg/padlock-406986_640_410.jpg", "/jpeg/padlock-406986_640_411.jpg",
    "/jpeg/padlock-406986_640_420.jpg", "/jpeg/padlock-406986_640_422.jpg", "/jpeg/padlock-406986_640_440.jpg",
    "/jpeg/padlock-406986_640_444.jpg", "/jpeg/padlock-406986_640_gray.jpg"};

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SRGB_INPUT_FORMATS_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        ValuesIn(css_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS444_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410,  NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS410_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_410.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_410),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS411_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_411.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_411),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS420_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_420.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS422_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_422.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_SYCC_CSS440_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        Values("/jpeg/padlock-406986_640_440.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_440),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, 
            NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));


INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_VALID_GRAY_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImage,
    Combine(
        ValuesIn(css_filenames),
        Values(NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN)));


class NvJpegExtEncoderTestSingleImageWithStatus : public NvJpegExtEncoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                            nvimgcodecChromaSubsampling_t, nvimgcodecChromaSubsampling_t, nvimgcodecJpegEncoding_t, nvimgcodecProcessingStatus_t>>
{
  public:
    virtual ~NvJpegExtEncoderTestSingleImageWithStatus() = default;

  protected:
    void SetUp() override
    {
        NvJpegExtEncoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        encoded_chroma_subsampling_ = std::get<4>(GetParam());
        out_jpeg_image_info_.encoding = std::get<5>(GetParam());
        expected_status_ =  std::get<6>(GetParam());
        image_info_.struct_next = &out_jpeg_image_info_;

    }

    virtual void TearDown()
    {
        NvJpegTestBase::TearDown();
        NvJpegExtEncoderTestBase::TearDown();
    }

    nvimgcodecChromaSubsampling_t encoded_chroma_subsampling_;
    nvimgcodecProcessingStatus_t expected_status_;
};


TEST_P(NvJpegExtEncoderTestSingleImageWithStatus, InvalidFormatsOrParameters)
{
    nvimgcodecImageInfo_t ref_cs_image_info;
    DecodeReference(resources_dir, image_file_, sample_format_, true, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    PrepareImageForFormat();
    memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name, "jpeg");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
     &NvJpegExtEncoderTestSingleImageWithStatus::ResizeBufferStatic<NvJpegExtEncoderTestSingleImageWithStatus>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(expected_status_, encode_status);
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, 1);
}

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_OUTPUT_CHROMA_FOR_P_Y_FORMAT_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_410V, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_INPUT_CHROMA_FOR_P_Y_FORMAT_WITH_VARIOUS_ENCODING, NvJpegExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410,  NVIMGCODEC_SAMPLING_410V,  NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG_ENCODE_INVALID_COLOR_SPEC_FOR_P_Y_FORMAT, NvJpegExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("/jpeg/padlock-406986_640_444.jpg"),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_YCCK, NVIMGCODEC_COLORSPEC_CMYK),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT),
        Values(NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED| NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

// clang-format on

}} // namespace nvimgcodec::test