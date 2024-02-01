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

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "nvimgcodec_tests.h"
#include "nvjpeg2k_ext_test_common.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

namespace nvimgcodec { namespace test {

class NvJpeg2kExtEncoderTestBase : public NvJpeg2kExtTestBase
{
  public:
    NvJpeg2kExtEncoderTestBase() {}

    void SetUp() override
    {
        NvJpeg2kExtTestBase::SetUp();
        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));

        jpeg2k_enc_params_ = {NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), 0};
        jpeg2k_enc_params_.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
        jpeg2k_enc_params_.prog_order = NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP;
        jpeg2k_enc_params_.num_resolutions = 2;
        jpeg2k_enc_params_.code_block_w = 32;
        jpeg2k_enc_params_.code_block_h = 32;
        bool irreversible = false;
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t),&jpeg2k_enc_params_, 0};
        params_.quality = 0;
        params_.target_psnr = 30;
    }

    void TearDown() override
    {
        if (encoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        NvJpeg2kExtTestBase::TearDown();
    }

    nvimgcodecEncoder_t encoder_;
    nvimgcodecJpeg2kEncodeParams_t jpeg2k_enc_params_;
    nvimgcodecEncodeParams_t params_;
};

class NvJpeg2kExtEncoderTestSingleImage : public NvJpeg2kExtEncoderTestBase,
                                          public NvJpeg2kTestBase,
                                          public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                              nvimgcodecChromaSubsampling_t, nvimgcodecChromaSubsampling_t, nvimgcodecJpeg2kProgOrder_t>>
{
  public:
    virtual ~NvJpeg2kExtEncoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        NvJpeg2kExtEncoderTestBase::SetUp();
        NvJpeg2kTestBase::SetUp();
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        encoded_chroma_subsampling_ = std::get<4>(GetParam());
        jpeg2k_enc_params_.prog_order = std::get<5>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpeg2kTestBase::TearDown();
        NvJpeg2kExtEncoderTestBase::TearDown();
    }

    nvimgcodecChromaSubsampling_t encoded_chroma_subsampling_;
};

TEST_P(NvJpeg2kExtEncoderTestSingleImage, ValidFormatAndParameters)
{
    nvimgcodecImageInfo_t ref_cs_image_info;
    bool enable_color_conversion = color_spec_ == NVIMGCODEC_COLORSPEC_SRGB;
    DecodeReference(resources_dir, image_file_, sample_format_, enable_color_conversion, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    PrepareImageForFormat();
    auto image_info_ref = image_info_;
    if (sample_format_ == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
        Convert_P_RGB_to_I_RGB(image_buffer_, ref_buffer_, image_info_);
        image_info_ref.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info_ref.num_planes = image_info_.plane_info[0].num_channels;
        for (int p = 0; p < image_info_ref.num_planes; p++) {
            image_info_ref.plane_info[p].height = image_info_.plane_info[0].height;
            image_info_ref.plane_info[p].width = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].row_stride = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].num_channels = 1;
            image_info_ref.plane_info[p].sample_type = image_info_.plane_info[0].sample_type;
            image_info_ref.plane_info[p].precision = 0;
        }
        image_info_ref.buffer_size = ref_buffer_.size();
        image_info_ref.buffer = ref_buffer_.data();
        image_info_ref.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else {
        memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());
    }

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;

    nvimgcodecImageInfo_t cs_image_info_ref(image_info_ref);
    cs_image_info_ref.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name, "jpeg2k");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpeg2kExtEncoderTestSingleImage::ResizeBufferStatic<NvJpeg2kExtEncoderTestSingleImage>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t encode_status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, encode_status);
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, 1);

    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcodecImageInfo_t load_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &load_info));
    //TODO uncomment when generic jpeg2k parser is in place EXPECT_EQ(cs_image_info.chroma_subsampling, load_info.chroma_subsampling);

    std::vector<unsigned char> ref_out_buffer;
    EncodeReference(image_info_ref, params_, jpeg2k_enc_params_, cs_image_info_ref, &ref_out_buffer);
    ASSERT_EQ(0,
        memcmp(reinterpret_cast<void*>(ref_out_buffer.data()), reinterpret_cast<void*>(code_stream_buffer_.data()), ref_out_buffer.size()));
}
// clang-format off

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SRGB_INPUT_FORMATS_WITH_VARIOUS_PROG_ORDERS, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP, NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP, NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL, NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL,
    NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SYCC_INPUT_FORMATS_WITH_CSS444, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SYCC_INPUT_FORMATS_WITH_CSS422, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SYCC_INPUT_FORMATS_WITH_CSS420, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_GRAY_AND_SYCC_WITH_P_Y, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640_gray.jp2"),
        Values(NVIMGCODEC_COLORSPEC_GRAY),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP)));
// clang-format on

class NvJpeg2kExtEncoderTestSingleImageWithStatus
    : public NvJpeg2kExtEncoderTestBase,
      public NvJpeg2kTestBase,
      public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t, nvimgcodecChromaSubsampling_t,
          nvimgcodecChromaSubsampling_t, nvimgcodecJpeg2kProgOrder_t, nvimgcodecProcessingStatus_t>>
{
  public:
    virtual ~NvJpeg2kExtEncoderTestSingleImageWithStatus() = default;

  protected:
    void SetUp() override
    {
        NvJpeg2kExtEncoderTestBase::SetUp();
        NvJpeg2kTestBase::SetUp();
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        encoded_chroma_subsampling_ = std::get<4>(GetParam());
        jpeg2k_enc_params_.prog_order = std::get<5>(GetParam());
        expected_status_ = std::get<6>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpeg2kTestBase::TearDown();
        NvJpeg2kExtEncoderTestBase::TearDown();
    }

    nvimgcodecChromaSubsampling_t encoded_chroma_subsampling_;
    nvimgcodecProcessingStatus_t expected_status_;
};

TEST_P(NvJpeg2kExtEncoderTestSingleImageWithStatus, InvalidFormatsOrParameters)
{
    nvimgcodecImageInfo_t ref_cs_image_info;
    DecodeReference(resources_dir, image_file_, sample_format_, true, &ref_cs_image_info);
    image_info_.plane_info[0] = ref_cs_image_info.plane_info[0];
    PrepareImageForFormat();
    memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());

    code_stream_buffer_.resize(image_info_.buffer_size);

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    strcpy(cs_image_info.codec_name, "jpeg2k");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpeg2kExtEncoderTestSingleImageWithStatus::ResizeBufferStatic<NvJpeg2kExtEncoderTestSingleImageWithStatus>, &cs_image_info));
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

// clang-format off
INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_SAMPLE_FORMATS, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCODEC_SAMPLING_NONE),
        Values(NVIMGCODEC_SAMPLING_NONE),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_INPUT_CHROMA, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_410V),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_OUTPUT_CHROMA, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_410V),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_FROM_CSS444, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_FROM_CSS422, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_FROM_CSS420, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_TO_CSS420, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422,  NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_TO_CSS422, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420,  NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_TO_CSS444, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420,  NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));     

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_COLOR_SPEC_FOR_P_Y, NvJpeg2kExtEncoderTestSingleImageWithStatus,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640_gray.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED)));

// clang-format on

}} // namespace nvimgcodec::test