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

#include <extensions/libjpeg_turbo/libjpeg_turbo_ext.h>
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/jpeg.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodec_tests.h"

using ::testing::WithParamInterface;
using ::testing::Combine;
using ::testing::Values;

namespace nvimgcodec { namespace test {

class LibjpegTurboExtDecoderTest : public CommonExtDecoderTest
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcodecExtensionDesc_t jpeg_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg_parser_extension_desc));

        nvimgcodecExtensionDesc_t libjpeg_turbo_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_libjpeg_turbo_extension_desc(&libjpeg_turbo_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &libjpeg_turbo_extension_desc));
    
        CommonExtDecoderTest::CreateDecoder();
    }
};

TEST_F(LibjpegTurboExtDecoderTest, ROIDecodingWholeImage)
{
    // Whole image
    nvimgcodecRegion_t region1{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region1.start[0] = 0;
    region1.start[1] = 0;
    region1.end[0] = 426;
    region1.end[1] = 640;
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCODEC_SAMPLEFORMAT_I_RGB, region1);
}

TEST_F(LibjpegTurboExtDecoderTest, ROIDecodingPortion)
{
    // Actual ROI
    nvimgcodecRegion_t region2{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region2.start[0] = 10;
    region2.start[1] = 20;
    region2.end[0] = 10 + 100;
    region2.end[1] = 20 + 100;
    TestSingleImage("jpeg/padlock-406986_640_422.jpg", NVIMGCODEC_SAMPLEFORMAT_I_RGB, region2);
}

class LibjpegTurboExtDecoderTestWithPathAndFormat :
    public LibjpegTurboExtDecoderTest,
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleFormat_t>>
{
public:
    void SetUp() override
    {
        image_path = std::get<0>(GetParam());
        sample_format = std::get<1>(GetParam());
        LibjpegTurboExtDecoderTest::SetUp();
    }

    std::string image_path;
    nvimgcodecSampleFormat_t sample_format;
};

TEST_P(LibjpegTurboExtDecoderTestWithPathAndFormat, SingleImage)
{
    TestSingleImage(image_path, sample_format);
}

INSTANTIATE_TEST_SUITE_P(LIBJPEG_TURBO_DECODE_420,
    LibjpegTurboExtDecoderTestWithPathAndFormat,
    Combine(
        Values(
            "jpeg/padlock-406986_640_420.jpg"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR
        )
    )
);

INSTANTIATE_TEST_SUITE_P(LIBJPEG_TURBO_DECODE_I_RGB,
    LibjpegTurboExtDecoderTestWithPathAndFormat,
    Combine(
        Values(
            "jpeg/padlock-406986_640_410.jpg",
            "jpeg/padlock-406986_640_411.jpg",
            "jpeg/padlock-406986_640_422.jpg",
            "jpeg/padlock-406986_640_440.jpg",
            "jpeg/padlock-406986_640_444.jpg",
            "jpeg/padlock-406986_640_gray.jpg",
            "jpeg/cmyk.jpg",
            "jpeg/ycck_colorspace.jpg",
            "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB
        )
    )
);

INSTANTIATE_TEST_SUITE_P(LIBJPEG_TURBO_DECODE_GRAY,
    LibjpegTurboExtDecoderTestWithPathAndFormat,
    Values(std::tuple{"jpeg/padlock-406986_640_gray.jpg", NVIMGCODEC_SAMPLEFORMAT_P_Y})
);

class LibjpegTurboExtDecoderTestUnsupported :
    public LibjpegTurboExtDecoderTest,
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleDataType_t, nvimgcodecProcessingStatus>>
{
public:
    void SetUp() override
    {
        image_path = std::get<0>(GetParam());
        sample_data_type = std::get<1>(GetParam());
        expected_decode_status = std::get<2>(GetParam());
        LibjpegTurboExtDecoderTest::SetUp();
    }

    std::string image_path;
    nvimgcodecSampleDataType_t sample_data_type;
    nvimgcodecProcessingStatus expected_decode_status;
};


TEST_P(LibjpegTurboExtDecoderTestUnsupported, TestNotSupported)
{
    TestNotSupported(image_path, NVIMGCODEC_SAMPLEFORMAT_I_RGB, sample_data_type, expected_decode_status);
}

INSTANTIATE_TEST_SUITE_P(LIBJPEG_TURBO_DECODE_EXIF_ORIENTATION,
    LibjpegTurboExtDecoderTestUnsupported,
    Combine(
        Values(
            "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
            "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
            "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
            "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
            "jpeg/exif/padlock-406986_640_rotate_90.jpg",
            "jpeg/exif/padlock-406986_640_rotate_180.jpg",
            "jpeg/exif/padlock-406986_640_rotate_270.jpg"
        ), Values(
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8
        ), Values (
            NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED
        )
    )
);

INSTANTIATE_TEST_SUITE_P(LIBJPEG_TURBO_DECODE_SAMPLE_TYPE_UNSUPPORTED,
    LibjpegTurboExtDecoderTestUnsupported,
    Combine(
        Values(
            "jpeg/padlock-406986_640_444.jpg"
        ), Values(
            NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32,
            NVIMGCODEC_SAMPLE_DATA_TYPE_INT16,
            NVIMGCODEC_SAMPLE_DATA_TYPE_INT8,
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16
        ), Values(
            NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED
        )
    )
);

}} // namespace nvimgcodec::test
