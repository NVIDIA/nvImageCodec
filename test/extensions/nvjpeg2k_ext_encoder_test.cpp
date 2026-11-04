/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "common_ext_encoder_test.h"
#include "parsers/jpeg2k.h"

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
        jpeg2k_enc_params_.mct_mode = 0;
        jpeg2k_enc_params_.ht = 0;

        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), &jpeg2k_enc_params_};
    }

    void TearDown() override
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        NvJpeg2kExtTestBase::TearDown();
    }

    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecJpeg2kEncodeParams_t jpeg2k_enc_params_;
    nvimgcodecEncodeParams_t params_;
};

class NvJpeg2kExtEncoderTestSingleImage : public NvJpeg2kExtEncoderTestBase,
                                          public NvJpeg2kTestBase,
                                          public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                              nvimgcodecChromaSubsampling_t, nvimgcodecChromaSubsampling_t, nvimgcodecColorSpec_t, nvimgcodecJpeg2kProgOrder_t,
                                              int /*mct_mode*/, int /*ht*/, nvimgcodecQualityType_t, float /*quality value*/, nvimgcodecProcessingStatus>>
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
        encoded_color_spec_ = std::get<5>(GetParam());
        jpeg2k_enc_params_.prog_order = std::get<6>(GetParam());
        jpeg2k_enc_params_.mct_mode = std::get<7>(GetParam());
        jpeg2k_enc_params_.ht = std::get<8>(GetParam());
        params_.quality_type = std::get<9>(GetParam());
        params_.quality_value = std::get<10>(GetParam());
        expected_encode_status = std::get<11>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpeg2kTestBase::TearDown();
        NvJpeg2kExtEncoderTestBase::TearDown();
    }

    nvimgcodecChromaSubsampling_t encoded_chroma_subsampling_;
    nvimgcodecColorSpec_t encoded_color_spec_;
    nvimgcodecProcessingStatus expected_encode_status;
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
        Convert_P_RGB_to_I_RGB(image_buffer_, ref_buffer_);
        image_info_ref.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info_ref.num_planes = image_info_.plane_info[0].num_channels;
        for (uint32_t p = 0; p < image_info_ref.num_planes; p++) {
            image_info_ref.plane_info[p].height = image_info_.plane_info[0].height;
            image_info_ref.plane_info[p].width = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].row_stride = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].num_channels = 1;
            image_info_ref.plane_info[p].sample_type = image_info_.plane_info[0].sample_type;
            image_info_ref.plane_info[p].precision = 8;
        }
        assert(GetBufferSize(image_info_ref) == ref_buffer_.size());
        image_info_ref.buffer = ref_buffer_.data();
        image_info_ref.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else {
        memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());
    }

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    cs_image_info.chroma_subsampling = encoded_chroma_subsampling_;
    cs_image_info.color_spec = encoded_color_spec_;

    nvimgcodecImageInfo_t cs_image_info_ref(image_info_ref);
    cs_image_info_ref.chroma_subsampling = encoded_chroma_subsampling_;
    cs_image_info_ref.color_spec = encoded_color_spec_;
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
    ASSERT_EQ(expected_encode_status, encode_status);
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, 1);

    if (expected_encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        return;
    }

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
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP, NVIMGCODEC_JPEG2K_PROG_ORDER_RLCP, NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL, NVIMGCODEC_JPEG2K_PROG_ORDER_PCRL,
            NVIMGCODEC_JPEG2K_PROG_ORDER_CPRL),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SYCC_INPUT_FORMATS_WITH_CSS444, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SYCC_INPUT_FORMATS_WITH_CSS422, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_SYCC_INPUT_FORMATS_WITH_CSS420, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_VALID_GRAY_AND_SYCC_WITH_P_Y, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640_gray.jp2"),
        Values(NVIMGCODEC_COLORSPEC_GRAY),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_COLORSPEC_GRAY),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_QUALITY_LOSSLESS, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_LOSSLESS),
        Values(0), // quality value (ignored for lossless)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_QUALITY_QUALITY, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(1), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(1, 10, 30, 50, 75, 95, 100, 85.1f, 85.6f, 85.5f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_QUALITY_QUALITY_SYCC, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(1, 10, 30, 50, 75, 95, 100, 85.1f, 85.6f, 85.5f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_QUALITY_QUANTIZATION_STEP, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP),
        Values(1.f, 2.5f, 10.f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_QUALITY_PSNR, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_PSNR),
        Values(30.f, 40.f, 50.f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

/*-------------negative tests below-------------*/
INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_SAMPLE_FORMATS, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCODEC_SAMPLING_NONE),
        Values(NVIMGCODEC_SAMPLING_NONE),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_SYCC_TO_RGB, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(75), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_INPUT_CHROMA, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_410V),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_OUTPUT_CHROMA, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_410V),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_FROM_CSS444, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_FROM_CSS422, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_FROM_CSS420, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_TO_CSS420, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_422,  NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_420),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_TO_CSS422, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420,  NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_CHROMA_NO_RESAMPLING_TO_CSS444, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
        Values(NVIMGCODEC_SAMPLING_420,  NVIMGCODEC_SAMPLING_422),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_COLOR_SPEC_FOR_P_Y, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640_gray.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_SAMPLING_GRAY),
        Values(NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_QUALITY_TYPE, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO),
        Values(0), // quality value 
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_QUALITY_TYPE_WITH_HT, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(1), // ht
        Values(NVIMGCODEC_QUALITY_TYPE_PSNR),
        Values(30), // quality value 
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_QUALITY_TYPE_WITHOUT_MCT, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(0), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(75), // quality value 
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_QUALITY_VALUE, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(1), // mct_mode
        Values(0, 1), // non-ht & ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(0, 101), // quality value 
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_INVALID_QUALITY_NEGATIVE_VALUE, NvJpeg2kExtEncoderTestSingleImage,
    Combine(
        Values("jpeg2k/tiled-cat-1046544_640.jp2"),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP),
        Values(1), // mct_mode
        Values(0), // non-ht
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY, NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP, NVIMGCODEC_QUALITY_TYPE_PSNR),
        Values(-1), // quality value 
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED)
    )
);

// clang-format on

// Tests that the nvjpeg2k encoder honors a custom plane_info.precision and that the
// resulting bitstream round-trips that precision through the parser/decoder. Covers
// both sub-8 precision (e.g. 4-bit data in uint8) and supra-8 precision (e.g. 12-bit
// data in uint16).
struct CustomPrecisionCase
{
    nvimgcodecSampleDataType_t sample_type;
    uint8_t precision;
};

class NvJpeg2kExtEncoderTestCustomPrecision : public NvJpeg2kExtEncoderTestBase,
                                              public ::testing::TestWithParam<CustomPrecisionCase>
{
  public:
    void SetUp() override
    {
        NvJpeg2kExtEncoderTestBase::SetUp();
        params_.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;
        params_.quality_value = 0;
        jpeg2k_enc_params_.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
    }

    void TearDown() override { NvJpeg2kExtEncoderTestBase::TearDown(); }

    template <typename T>
    void FillPattern(T* buf, uint32_t width, uint32_t height, uint32_t channels, uint8_t precision)
    {
        const uint32_t max_value = (precision >= 32) ? 0xFFFFFFFFu : ((1u << precision) - 1u);
        for (uint32_t y = 0; y < height; ++y) {
            for (uint32_t x = 0; x < width; ++x) {
                for (uint32_t c = 0; c < channels; ++c) {
                    uint32_t v = (x * 17u + y * 31u + c * 113u) & max_value;
                    buf[(y * width + x) * channels + c] = static_cast<T>(v);
                }
            }
        }
    }
};

TEST_P(NvJpeg2kExtEncoderTestCustomPrecision, EncodesAndPreservesPrecisionInBitstream)
{
    constexpr uint32_t width = 64;
    constexpr uint32_t height = 48;
    const auto& tc = GetParam();
    const uint8_t precision = tc.precision;
    const nvimgcodecSampleDataType_t sample_type = tc.sample_type;
    const size_t bytes_per_sample = (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) ? 1 : 2;

    image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    image_info_.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
    image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_Y;
    image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
    image_info_.num_planes = 1;
    image_info_.plane_info[0].width = width;
    image_info_.plane_info[0].height = height;
    image_info_.plane_info[0].num_channels = 1;
    image_info_.plane_info[0].sample_type = sample_type;
    image_info_.plane_info[0].precision = precision;
    image_info_.plane_info[0].row_stride = width * bytes_per_sample;
    image_buffer_.resize(GetBufferSize(image_info_));
    image_info_.buffer = image_buffer_.data();
    image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    if (bytes_per_sample == 1) {
        FillPattern<uint8_t>(reinterpret_cast<uint8_t*>(image_buffer_.data()), width, height, 1, precision);
    } else {
        FillPattern<uint16_t>(reinterpret_cast<uint16_t*>(image_buffer_.data()), width, height, 1, precision);
    }

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    std::strcpy(cs_image_info.codec_name, "jpeg2k");

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpeg2kExtEncoderTestCustomPrecision::ResizeBufferStatic<NvJpeg2kExtEncoderTestCustomPrecision>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t encode_status;
    size_t status_size = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, encode_status);

    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcodecImageInfo_t parsed_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &parsed_info));

    // The jpeg2k parser reads precision from the Ssiz field of the SIZ marker. If the
    // encoder wrote `precision`, the parser will report `precision` here.
    EXPECT_EQ(precision, parsed_info.plane_info[0].precision);
    EXPECT_EQ(sample_type, parsed_info.plane_info[0].sample_type);
    EXPECT_EQ(width, parsed_info.plane_info[0].width);
    EXPECT_EQ(height, parsed_info.plane_info[0].height);
}

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_CUSTOM_PRECISION, NvJpeg2kExtEncoderTestCustomPrecision,
    ::testing::Values(
        // Sub-8 precision carried in a uint8 buffer
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, 4},
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, 5},
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, 7},
        // Boundary: precision == bitdepth
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, 8},
        // Supra-8 precision carried in a uint16 buffer
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 10},
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 12},
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 14},
        CustomPrecisionCase{NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 16}));

class NvJpeg2kExtEncoderTestCustomPrecisionNegative : public NvJpeg2kExtEncoderTestBase, public ::testing::Test
{
  public:
    void SetUp() override
    {
        NvJpeg2kExtEncoderTestBase::SetUp();
        params_.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;
        params_.quality_value = 0;
        jpeg2k_enc_params_.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
    }
    void TearDown() override { NvJpeg2kExtEncoderTestBase::TearDown(); }
};

TEST_F(NvJpeg2kExtEncoderTestCustomPrecisionNegative, CanEncodeRejectsPrecisionGreaterThanBitdepth)
{
    image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    image_info_.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
    image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_Y;
    image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
    image_info_.num_planes = 1;
    image_info_.plane_info[0].width = 32;
    image_info_.plane_info[0].height = 32;
    image_info_.plane_info[0].num_channels = 1;
    image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    image_info_.plane_info[0].precision = 12;  // invalid: 8-bit dtype cannot carry 12-bit data
    image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width;
    image_buffer_.resize(GetBufferSize(image_info_));
    image_info_.buffer = image_buffer_.data();
    image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    std::strcpy(cs_image_info.codec_name, "jpeg2k");

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpeg2kExtEncoderTestCustomPrecisionNegative::ResizeBufferStatic<NvJpeg2kExtEncoderTestCustomPrecisionNegative>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t encode_status;
    size_t status_size = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED, encode_status);
}

class NvJpeg2kExtEncoderTestOrientationNegative : public NvJpeg2kExtEncoderTestBase, public ::testing::Test
{
  public:
    void SetUp() override
    {
        NvJpeg2kExtEncoderTestBase::SetUp();
        params_.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;
        params_.quality_value = 0;
        jpeg2k_enc_params_.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
    }
    void TearDown() override { NvJpeg2kExtEncoderTestBase::TearDown(); }
};

TEST_F(NvJpeg2kExtEncoderTestOrientationNegative, CanEncodeRejectsNonIdentityOrientation)
{
    image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    image_info_.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
    image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_Y;
    image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
    image_info_.num_planes = 1;
    image_info_.plane_info[0].width = 32;
    image_info_.plane_info[0].height = 32;
    image_info_.plane_info[0].num_channels = 1;
    image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width;
    image_info_.orientation.rotated = 90;  // non-identity: no encoder applies orientation
    image_buffer_.resize(GetBufferSize(image_info_));
    image_info_.buffer = image_buffer_.data();
    image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    std::strcpy(cs_image_info.codec_name, "jpeg2k");

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvJpeg2kExtEncoderTestOrientationNegative::ResizeBufferStatic<NvJpeg2kExtEncoderTestOrientationNegative>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    nvimgcodecProcessingStatus_t encode_status;
    size_t status_size = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED, encode_status);
}

// ---------------------------------------------------------------------------
// Strided encode->decode round-trip via the shared CommonExtEncoderTest harness.
// One test case per input plane-stride layout, exercising the strided host->
// device input copy feeding the nvJPEG2000 encoder. Uses lossless encoding so
// the round-trip is exact.
// ---------------------------------------------------------------------------
class NvJpeg2kExtEncodeDecodeStrideTest :
    public CommonExtEncoderTest,
    public TestWithParam<nvimgcodecSampleFormat_t>
{
  public:
    void SetUp() override
    {
        CommonExtEncoderTest::SetUp();

        nvimgcodecExtensionDesc_t jpeg2k_parser_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg2k_parser_desc));

        nvimgcodecExtensionDesc_t nvjpeg2k_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvjpeg2k_extension_desc(&nvjpeg2k_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvjpeg2k_desc));

        color_spec_ = NVIMGCODEC_COLORSPEC_SRGB;
        sample_format_ = GetParam();
        chroma_subsampling_ = NVIMGCODEC_SAMPLING_444;
        jpeg2k_stream_type_ = NVIMGCODEC_JPEG2K_STREAM_JP2;
        jpeg2k_ht_ = 0;

        CommonExtEncoderTest::CreateDecoderAndEncoder();
        encode_params_.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;
    }

    void TearDown() override { CommonExtEncoderTest::TearDown(); }
};

DEFINE_ENCODE_DECODE_STRIDE_TESTS_FOR_CODEC(NvJpeg2kExtEncodeDecodeStrideTest, "jpeg2k", false)

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_ENCODE_DECODE_STRIDED, NvJpeg2kExtEncodeDecodeStrideTest,
    Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_RGB));

}} // namespace nvimgcodec::test
