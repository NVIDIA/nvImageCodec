/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <extensions/nvtiff/nvtiff_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/tiff.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "nvimgcodec_tests.h"
#include "common_ext_decoder_test.h"

#include <nvjpeg.h>

using ::testing::Combine;
using ::testing::Values;

namespace nvimgcodec { namespace test {

class NvTiffExtDecoderTestBase : public CommonExtDecoderTest
{
public:
    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc);

        nvimgcodecExtensionDesc_t nvtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvtiff_extension_desc(&nvtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvtiff_extension_desc));

        CommonExtDecoderTest::CreateDecoder();
    }

};
class NvTiffExtDecoderTest :
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleFormat_t>>,
    public NvTiffExtDecoderTestBase
{
public:
    void SetUp() override
    {
        image_path = std::get<0>(GetParam());
        sample_format = std::get<1>(GetParam());
        NvTiffExtDecoderTestBase::SetUp();
    }
public:
    std::string image_path;
    nvimgcodecSampleFormat_t sample_format;
};

TEST_P(NvTiffExtDecoderTest, SingleImage)
{
    if (image_path.find("jpeg") != std::string::npos) {
        int nvjpeg_major = -1, nvjpeg_minor = -1;
        if (NVJPEG_STATUS_SUCCESS != nvjpegGetProperty(MAJOR_VERSION, &nvjpeg_major) ||
            NVJPEG_STATUS_SUCCESS != nvjpegGetProperty(MINOR_VERSION, &nvjpeg_minor) ||
            nvjpeg_major * 1000 + nvjpeg_minor * 10 < 11060
        ) {
            GTEST_SKIP() << "This test requires nvJPEG from at least 11.6 CUDA.";
        }
    }

#if defined(_WIN32) || defined(_WIN64)
    if (CC_major < 7) {
        GTEST_SKIP() << "On Windows, nvCOMP deflate requires sm_70 or higher to work.";
    }
#endif
    TestSingleImage(image_path, sample_format);
}

#if SKIP_NVTIFF_WITH_NVCOMP_TESTS_ENABLED
    #pragma message("Skipping nvTIFF tests that require nvCOMP")
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(NvTiffExtDecoderTest);
#else
INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE,
    NvTiffExtDecoderTest,
    Combine(
        Values(
            "tiff/cat-1245673_640.tiff",
            "tiff/cat-1245673_300572.tiff",
            "tiff/cat-300572_640_grayscale.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_FP32_TO_UINT8,
    NvTiffExtDecoderTest,
    Combine(
        Values(
            "tiff/cat-300572_640_fp32.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR
        )
    )
);
#endif

class NvTiffExtDecoderTestWithoutNvCOMP : public NvTiffExtDecoderTest {};

TEST_P(NvTiffExtDecoderTestWithoutNvCOMP, SingleImage)
{
    if (image_path.find("jpeg") != std::string::npos) {
        int nvjpeg_major = -1, nvjpeg_minor = -1;
        if (NVJPEG_STATUS_SUCCESS != nvjpegGetProperty(MAJOR_VERSION, &nvjpeg_major) ||
            NVJPEG_STATUS_SUCCESS != nvjpegGetProperty(MINOR_VERSION, &nvjpeg_minor) ||
            nvjpeg_major * 1000 + nvjpeg_minor * 10 < 11060
        ) {
            GTEST_SKIP() << "This test requires nvJPEG from at least 11.6 CUDA.";
        }
    }
    TestSingleImage(image_path, sample_format);
}

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_WITHOUT_COMPRESSION,
    NvTiffExtDecoderTestWithoutNvCOMP,
    Combine(
        Values(
            "tiff/cat-300572_640_no_compression.tiff",
            // "tiff/cat-300572_640_jpeg_compression.tiff", // This test fails in gitlab with cuda 11.3, but should be skipped. (which it does with manual testing). It also fails for some reason with cuda 11.8 on A100 (other gpus work fine)
            // "tiff/cat-300572_640_ycbcr.tiff", // This test fails in gitlab with cuda 11.3, but should be skipped. (which it does with manual testing). It also fails for some reason with cuda 11.8 on A100 (other gpus work fine)
            "tiff/cat-300572_640_palette.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

#if SKIP_NVTIFF_WITH_NVCOMP_TESTS_ENABLED
    #pragma message("Skipping nvTIFF tests that require nvCOMP")
#else

class NvTiffExtDecoderTestROI :
    public ::testing::WithParamInterface<std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t>>,
    public NvTiffExtDecoderTestBase
{};

TEST_P(NvTiffExtDecoderTestROI, SingleImage)
{
#if defined(_WIN32) || defined(_WIN64)
    if (CC_major < 7) {
        GTEST_SKIP() << "On Windows, nvCOMP deflate requires sm_70 or higher to work.";
    }
#endif

    nvimgcodecRegion_t region1{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region1.start[0] = std::get<1>(GetParam());
    region1.start[1] = std::get<2>(GetParam());
    region1.end[0] = std::get<3>(GetParam());
    region1.end[1] = std::get<4>(GetParam());

    TestSingleImage(std::get<0>(GetParam()), NVIMGCODEC_SAMPLEFORMAT_I_RGB, region1);
    TestSingleImage(std::get<0>(GetParam()), NVIMGCODEC_SAMPLEFORMAT_P_BGR, region1);
}

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_ROI,
    NvTiffExtDecoderTestROI,
    Values(
        std::tuple{"tiff/cat-300572_640.tiff", 0, 0, 536, 640},
        std::tuple{"tiff/cat-300572_640.tiff", 100, 150, 275, 598},
        std::tuple{"tiff/cat-300572_640.tiff", 123, 456, 124, 457}
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_ROI_TILED,
    NvTiffExtDecoderTestROI,
    Values(
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 0, 0, 536, 640},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 123, 456, 124, 457},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 32, 48, 64, 96},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 30, 40, 68, 100},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 34, 60, 60, 90},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 128, 240, 320, 624},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 120, 230, 300, 620}
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_ROI_STRIPED,
    NvTiffExtDecoderTestROI,
    Values(
        std::tuple{"tiff/cat-300572_640_striped.tiff", 0, 0, 536, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 123, 456, 124, 457},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 5, 0, 10, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 3, 0, 11, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 6, 10, 9, 600},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 15, 0, 100, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 13, 42, 99, 620}
    )
);
#endif

}} // namespace nvimgcodec::test
