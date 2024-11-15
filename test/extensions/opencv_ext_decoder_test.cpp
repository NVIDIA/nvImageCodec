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

#include <extensions/opencv/opencv_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/bmp.h>
#include <parsers/jpeg.h>
#include <parsers/jpeg2k.h>
#include <parsers/parser_test_utils.h>
#include <parsers/png.h>
#include <parsers/pnm.h>
#include <parsers/tiff.h>
#include <parsers/webp.h>
#include <test_utils.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "common_ext_decoder_test.h"
#include "nvimgcodec_tests.h"

using testing::Values;
using testing::Combine;

namespace nvimgcodec { namespace test {

class OpenCVExtDecoderTest : public CommonExtDecoderTestWithPathAndFormat
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTestWithPathAndFormat::SetUp();

        nvimgcodecExtensionDesc_t jpeg_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg_parser_extension_desc);

        nvimgcodecExtensionDesc_t jpeg2k_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg2k_parser_extension_desc);

        nvimgcodecExtensionDesc_t png_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_png_parser_extension_desc(&png_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &png_parser_extension_desc);

        nvimgcodecExtensionDesc_t bmp_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_bmp_parser_extension_desc(&bmp_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &bmp_parser_extension_desc);

        nvimgcodecExtensionDesc_t pnm_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_pnm_parser_extension_desc(&pnm_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &pnm_parser_extension_desc);

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc);

        nvimgcodecExtensionDesc_t webp_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_webp_parser_extension_desc(&webp_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &webp_parser_extension_desc);

        nvimgcodecExtensionDesc_t opencv_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_opencv_extension_desc(&opencv_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &opencv_extension_desc));

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        images_.clear();
        streams_.clear();

        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params_.apply_exif_orientation= 1;
    }
};

TEST_P(OpenCVExtDecoderTest, SingleImage)
{
    TestSingleImage(image_path, sample_format);
}

INSTANTIATE_TEST_SUITE_P(OPENCV_DECODE_JPEG_I_RGB,
    OpenCVExtDecoderTest,
    Combine(
        Values(
            "jpeg/padlock-406986_640_420.jpg",
            "jpeg/padlock-406986_640_gray.jpg",
            "jpeg/cmyk.jpg",
            "jpeg/ycck_colorspace.jpg",
            "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",
            "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
            "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
            "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
            "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
            "jpeg/exif/padlock-406986_640_rotate_90.jpg",
            "jpeg/exif/padlock-406986_640_rotate_180.jpg",
            "jpeg/exif/padlock-406986_640_rotate_270.jpg"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB
        )
    )
);

INSTANTIATE_TEST_SUITE_P(OPENCV_DECODE_ALL_CODECS,
    OpenCVExtDecoderTest,
    Combine(
        Values(
            "jpeg2k/cat-1046544_640.jp2",
            "png/cat-1245673_640.png",
            "bmp/cat-111793_640.bmp",
            "webp/lossy/cat-3113513_640.webp",
            "pnm/cat-1245673_640.pgm",
            "tiff/cat-1245673_640.tiff",
            "tiff/cat-1245673_300572.tiff",
            "tiff/cat-300572_640_palette.tiff",
            "tiff/cat-300572_640_grayscale.tiff",
            "tiff/cat-300572_640_no_compression.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

INSTANTIATE_TEST_SUITE_P(OPENCV_DECODE_JPEG_ALL_FORMATS,
    OpenCVExtDecoderTest,
    Values(
        std::tuple{"jpeg/padlock-406986_640_420.jpg", NVIMGCODEC_SAMPLEFORMAT_I_RGB},
        std::tuple{"jpeg/padlock-406986_640_420.jpg", NVIMGCODEC_SAMPLEFORMAT_I_RGB},
        std::tuple{"jpeg/padlock-406986_640_gray.jpg", NVIMGCODEC_SAMPLEFORMAT_I_RGB}
    )
);

class OpenCVExtDecoderTestRegion : public OpenCVExtDecoderTest
{

};

TEST_P(OpenCVExtDecoderTestRegion, ROIDecodingWholeImage)
{
    // Whole image
    nvimgcodecRegion_t region1{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region1.start[0] = 0;
    region1.start[1] = 0;
    region1.end[0] = 426;
    region1.end[1] = 640;
    TestSingleImage(image_path, sample_format, region1);
}

TEST_P(OpenCVExtDecoderTestRegion, ROIDecodingPortion)
{
    // Actual ROI
    nvimgcodecRegion_t region2{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region2.start[0] = 10;
    region2.start[1] = 20;
    region2.end[0] = 10 + 100;
    region2.end[1] = 20 + 100;
    TestSingleImage(image_path, sample_format, region2);
}

INSTANTIATE_TEST_SUITE_P(OPENCV_DECODE_JPEG_ROI,
    OpenCVExtDecoderTestRegion,
    Values(
        std::tuple{"jpeg/padlock-406986_640_422.jpg", NVIMGCODEC_SAMPLEFORMAT_I_RGB}
    )
);

}} // namespace nvimgcodec::test

