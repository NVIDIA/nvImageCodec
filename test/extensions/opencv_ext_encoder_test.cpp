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
#include "common_ext_encoder_test.h"
#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

using testing::Combine;
using testing::Values;
using ::testing::TestWithParam;

class OpenCVExtTestBase : public CommonExtEncoderTest
{
  public:

    void SetUp() override
    {
        CommonExtEncoderTest::SetUp();

        nvimgcodecExtensionDesc_t jpeg_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg_parser_extension_desc));

        nvimgcodecExtensionDesc_t jpeg2k_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &jpeg2k_parser_extension_desc));

        nvimgcodecExtensionDesc_t png_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_png_parser_extension_desc(&png_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &png_parser_extension_desc));

        nvimgcodecExtensionDesc_t bmp_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_bmp_parser_extension_desc(&bmp_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &bmp_parser_extension_desc));

        nvimgcodecExtensionDesc_t pnm_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_pnm_parser_extension_desc(&pnm_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &pnm_parser_extension_desc));

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc));

        nvimgcodecExtensionDesc_t webp_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_webp_parser_extension_desc(&webp_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &webp_parser_extension_desc));

        nvimgcodecExtensionDesc_t opencv_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_opencv_extension_desc(&opencv_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &opencv_extension_desc));

        color_spec_ = NVIMGCODEC_COLORSPEC_SRGB;
        sample_format_ = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        jpeg_encoding_ = NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT;
        jpeg_optimized_huffman_ = 0;
        jpeg2k_stream_type_ = NVIMGCODEC_JPEG2K_STREAM_JP2;
        jpeg2k_ht_ = 0;

        CommonExtEncoderTest::CreateDecoderAndEncoder();
    }

    void TearDown() override
    {
         CommonExtEncoderTest::TearDown();
    }
};

class OpenCVExtTest : 
    public OpenCVExtTestBase,
    public TestWithParam<std::tuple<
        nvimgcodecSampleFormat_t, nvimgcodecChromaSubsampling_t, std::string, int /*jpeg optimized huffman*/, 
        nvimgcodecQualityType_t, float /*quality value*/, nvimgcodecProcessingStatus_t
    >>
{
public:
    void SetUp() override
    {
        OpenCVExtTestBase::SetUp();
        sample_format_ = std::get<0>(GetParam());
        chroma_subsampling_ = std::get<1>(GetParam());
        codec_name = std::get<2>(GetParam());
        jpeg_optimized_huffman_ = std::get<3>(GetParam());
        encode_params_.quality_type = std::get<4>(GetParam());
        encode_params_.quality_value = std::get<5>(GetParam());
    }
    void TearDown() override
    {
        OpenCVExtTestBase::TearDown();
    }

    std::string codec_name;
};


TEST_P(OpenCVExtTest, ValidFormatAndParameters)
{
    TestEncodeDecodeSingleImage(codec_name, std::get<6>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_SIMILAR_WITH_VARIOUS_FORMATS, OpenCVExtTest,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED
        ),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("png", "bmp", "jpeg", "jpeg2k", "pnm", "tiff", "webp"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_JPEG_DIFFERENT_SAMPLING, OpenCVExtTest,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED
        ),
        Values(NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_411),
        Values("jpeg"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_JPEG_OPTIMIZED_HUFFMAN, OpenCVExtTest,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED
        ),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("jpeg"),
        Values(1), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_REVERSIBLE, OpenCVExtTest,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED
        ),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("jpeg2k", "png", "bmp", "tiff", "webp"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_LOSSLESS),
        Values(0), // quality value (ignored for lossless)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_QUALITY, OpenCVExtTest,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED
        ),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("webp", "jpeg"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(95), // quality value, high value to get similar image
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_SIZE_RATIO, OpenCVExtTest,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED
        ),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("jpeg2k"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO),
        Values(0.5f), // size ratio
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

class OpenCVExtTestWithAlpha: public OpenCVExtTest
{};

TEST_P(OpenCVExtTestWithAlpha, ValidFormatAndParameters)
{
    TestEncodeDecodeSingleImage(codec_name, std::get<6>(GetParam()), true);
}
// bmp does not support 4 channel decoding, so encoding will be skipped

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_SIMILAR_WITH_VARIOUS_FORMATS, OpenCVExtTestWithAlpha,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("png", "jpeg2k", "tiff", "webp"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT, NVIMGCODEC_QUALITY_TYPE_LOSSLESS),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(ENCODE_DECODE_WEBP_LOSSY, OpenCVExtTestWithAlpha,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("webp"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(90), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);


INSTANTIATE_TEST_SUITE_P(TEST_NEGATIVE_PLANAR_UNCHANGED, OpenCVExtTestWithAlpha,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("jpeg", "pnm"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(TEST_NEGATIVE_INTERLEAVED_UNCHANGED, OpenCVExtTestWithAlpha,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("jpeg", "pnm"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(TEST_NEGATIVE_WITHOUT_UNCHANGED, OpenCVExtTestWithAlpha,
    Combine(
        Values(
            NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR
        ),
        Values(NVIMGCODEC_SAMPLING_444),
        Values("png", "bmp", "jpeg2k", "tiff", "webp"),
        Values(0), // optimized huffman
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)
    )
);

class OpenCVExtNegativeTest :
    public OpenCVExtTestBase,
    public TestWithParam<std::tuple<
        nvimgcodecChromaSubsampling_t, nvimgcodecJpegEncoding_t, std::string, int /*HT jpeg 2000*/, 
        nvimgcodecQualityType_t, float /*quality value*/, nvimgcodecProcessingStatus_t
    >>
{
public:
    void SetUp() override
    {
        OpenCVExtTestBase::SetUp();
    }
    void TearDown() override
    {
        OpenCVExtTestBase::TearDown();
    }
};

TEST_P(OpenCVExtNegativeTest, InvalidEncodeSetting)
{
    chroma_subsampling_ = std::get<0>(GetParam());
    jpeg_encoding_ = std::get<1>(GetParam());
    jpeg2k_ht_ = std::get<3>(GetParam());
    encode_params_.quality_type = std::get<4>(GetParam());
    encode_params_.quality_value = std::get<5>(GetParam());

    TestEncodeDecodeSingleImage(std::get<2>(GetParam()), std::get<6>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(INVALID_JPEG_ENCODING, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(
            NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN,
            NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN,
            NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_HUFFMAN,
            NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_HUFFMAN,
            NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_HUFFMAN,
            NVIMGCODEC_JPEG_ENCODING_RESERVED_FOR_JPEG_EXTENSIONS,
            NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_ARITHMETIC,
            NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_ARITHMETIC,
            NVIMGCODEC_JPEG_ENCODING_LOSSLESS_ARITHMETIC,
            NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_SEQUENTIAL_DCT_ARITHMETIC,
            NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_PROGRESSIVE_DCT_ARITHMETIC,
            NVIMGCODEC_JPEG_ENCODING_DIFFERENTIAL_LOSSLESS_ARITHMETIC
        ),
        Values("jpeg"),
        Values(0), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_FORMATS_SUBSAMPLING, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_422 , NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_411),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("png", "bmp", "jpeg2k", "pnm", "tiff", "webp"),
        Values(0), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_JPEG2000_HT, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("jpeg2k"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT),
        Values(0), // quality value (ignored for default)
        Values(NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_QUALITY_TYPE_ALL, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("png", "bmp", "jpeg", "jpeg2k", "pnm", "tiff", "webp"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP, NVIMGCODEC_QUALITY_TYPE_PSNR),
        Values(0), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_QUALITY_TYPE_SIZE_RATIO, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("png", "bmp", "jpeg", "pnm", "tiff", "webp"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO),
        Values(0.5f), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_QUALITY_TYPE_QUALITY, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("png", "bmp", "jpeg2k", "pnm", "tiff"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(50), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_QUALITY_TYPE_LOSSLESS, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("jpeg"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_LOSSLESS),
        Values(0), // quality value (ignored for lossless)
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

INSTANTIATE_TEST_SUITE_P(INVALID_QUALITY_VALUE_QUALITY, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("jpeg", "webp"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_QUALITY),
        Values(-1, 0, 101), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED)
    )
);
INSTANTIATE_TEST_SUITE_P(INVALID_QUALITY_SIZE_RATIO, OpenCVExtNegativeTest,
    Combine(
        Values(NVIMGCODEC_SAMPLING_444),
        Values(NVIMGCODEC_JPEG_ENCODING_UNKNOWN),
        Values("jpeg2k"),
        Values(1), // HT jpeg2k
        Values(NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO),
        Values(-1, 2), // quality value
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_VALUE_UNSUPPORTED)
    )
);


}} // namespace nvimgcodec::test
