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

#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <tuple>

#include <gtest/gtest.h>

#include <nvimgcodec.h>

#include <extensions/nvjpeg2k/nvjpeg2k_ext.h>
#include <parsers/parser_test_utils.h>

#include "nvjpeg2k_ext_test_common.h"

#include <test_utils.h>
#include "nvimgcodec_tests.h"
#include "common_ext_decoder_test.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

#define NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP 0

namespace nvimgcodec { namespace test {

class NvJpeg2kExtDecoderTestBase : public NvJpeg2kExtTestBase
{
  public:
    virtual ~NvJpeg2kExtDecoderTestBase() = default;

    void SetUp()
    {
        NvJpeg2kExtTestBase::SetUp();
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        NvJpeg2kExtTestBase::TearDown();
    }

    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecodeParams_t params_;
    std::string image_file_;
};

class NvJpeg2kExtDecoderTestSingleImage : public NvJpeg2kExtDecoderTestBase,
                                          public NvJpeg2kTestBase,
                                          public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                              nvimgcodecChromaSubsampling_t, nvimgcodecSampleFormat_t>>
{
  public:
    virtual ~NvJpeg2kExtDecoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        reference_output_format_ = std::get<4>(GetParam());
        NvJpeg2kExtDecoderTestBase::SetUp();
        NvJpeg2kTestBase::SetUp();
    }

    void TearDown() override
    {
        NvJpeg2kTestBase::TearDown();
        NvJpeg2kExtDecoderTestBase::TearDown();
    }

};

TEST_P(NvJpeg2kExtDecoderTestSingleImage, ValidFormatAndParameters)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + image_file_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    PrepareImageForFormat();
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &out_image_, &image_info_));
    streams_.push_back(in_code_stream_);
    images_.push_back(out_image_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, streams_.data(), images_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    cudaDeviceSynchronize();
    nvimgcodecProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, 1);
    DecodeReference(resources_dir, image_file_, reference_output_format_, image_info_.color_spec == NVIMGCODEC_COLORSPEC_SRGB);
    ConvertToPlanar();
    if (NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP) {
        write_bmp("./out.bmp", image_buffer_.data(), image_info_.plane_info[0].width,
            image_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width), image_info_.plane_info[0].width,
            image_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width) * 2, image_info_.plane_info[0].width,
            image_info_.plane_info[0].width, image_info_.plane_info[0].height);

        write_bmp("./ref.bmp", ref_buffer_.data(), image_info_.plane_info[0].width,
            ref_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width), image_info_.plane_info[0].width,
            ref_buffer_.data() + (image_info_.plane_info[0].height * image_info_.plane_info[0].width) * 2, image_info_.plane_info[0].width,
            image_info_.plane_info[0].width, image_info_.plane_info[0].height);
    }
    ASSERT_EQ(image_buffer_.size(), ref_buffer_.size());
    ASSERT_EQ(0, memcmp(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(image_buffer_.data()), image_buffer_.size()));
}

static const char* css_filenames[] = {"/jpeg2k/chroma_420/artificial_420_8b3c_dwt97CPRL.jp2",
    "/jpeg2k/chroma_420/cathedral_420_8b3c_dwt53RLCP.jp2", "/jpeg2k/chroma_420/deer_420_8b3c_dwt97RPCL.jp2",
    "/jpeg2k/chroma_420/leavesISO200_420_8b3c_dwt53PCRL.jp2", "/jpeg2k/chroma_420/leavesISO200_420_8b3c_dwt97CPRL.j2k",

    "/jpeg2k/chroma_422/artificial_422_8b3c_dwt53PCRL.jp2", "/jpeg2k/chroma_422/cathedral_422_8b3c_dwt97CPRL.jp2",
    "/jpeg2k/chroma_422/deer_422_8b3c_dwt53RPCL.j2k", "/jpeg2k/chroma_422/deer_422_8b3c_dwt53RPCL.jp2",
    "/jpeg2k/chroma_422/leavesISO200_422_8b3c_dwt97PCRL.jp2"};

// clang-format off
INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_VARIOUS_CHROMA_WITH_VALID_SRGB_OUTPUT_FORMATS, NvJpeg2kExtDecoderTestSingleImage,
    Combine(::testing::ValuesIn(css_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB), //NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        //Various output chroma subsampling are ignored for SRGB 
         Values(NVIMGCODEC_SAMPLING_NONE, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY, NVIMGCODEC_SAMPLING_410V), 
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_VARIOUS_CHROMA_WITH_VALID_SYCC_OUTPUT_FORMATS, NvJpeg2kExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_SYCC),
         Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
         //Chroma subsampling should be the same as file chroma (there is not chroma convert) but nvjpeg2k accepts only 444, 422, 420 
         Values(NVIMGCODEC_SAMPLING_NONE, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420), 
         Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV)));

class NvJpeg2kExtDecoderTestSingleImageWithStatus
    : public NvJpeg2kExtDecoderTestBase,
      public TestWithParam<
          std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t, nvimgcodecChromaSubsampling_t, nvimgcodecProcessingStatus_t >>
{
  public:
    virtual ~NvJpeg2kExtDecoderTestSingleImageWithStatus() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        NvJpeg2kExtDecoderTestBase::SetUp();
        expected_status_ =  std::get<4>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpeg2kExtDecoderTestBase::TearDown();
    }
    nvimgcodecProcessingStatus_t expected_status_;
};

TEST_P(NvJpeg2kExtDecoderTestSingleImageWithStatus, InvalidFormatsOrParameters)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + image_file_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    PrepareImageForFormat();

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &out_image_, &image_info_));
    streams_.push_back(in_code_stream_);
    images_.push_back(out_image_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder_, streams_.data(), images_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    cudaDeviceSynchronize();
    nvimgcodecProcessingStatus_t status;
    size_t status_size;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
    ASSERT_EQ(expected_status_, status);
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, 1);
}

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_INVALID_OUTPUT_CHROMA, NvJpeg2kExtDecoderTestSingleImageWithStatus,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_SYCC),
         Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
         Values(NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY, NVIMGCODEC_SAMPLING_410V), 
         Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_INVALID_OUTPUT_FORMAT, NvJpeg2kExtDecoderTestSingleImageWithStatus,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_SRGB, NVIMGCODEC_COLORSPEC_SYCC),
         Values(NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
         Values(NVIMGCODEC_SAMPLING_444), 
         Values(NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED)));

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE_INVALID_COLORSPEC, NvJpeg2kExtDecoderTestSingleImageWithStatus,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_CMYK, NVIMGCODEC_COLORSPEC_YCCK),
         Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB),
         Values(NVIMGCODEC_SAMPLING_444), 
         Values(NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED)));

// clang-format on
class NvJpeg2kExtDecoderTestRef : public CommonExtDecoderTestWithPathAndFormat
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTestWithPathAndFormat::SetUp();

        nvimgcodecExtensionDesc_t nvjpeg2k_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&nvjpeg2k_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvjpeg2k_parser_extension_desc));

        nvimgcodecExtensionDesc_t nvjpeg2k_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvjpeg2k_extension_desc(&nvjpeg2k_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvjpeg2k_extension_desc));

        CommonExtDecoderTestWithPathAndFormat::CreateDecoder();
    }
};

TEST_P(NvJpeg2kExtDecoderTestRef, SingleImage)
{
    TestSingleImage(image_path, sample_format);
}

INSTANTIATE_TEST_SUITE_P(NVJPEG2K_DECODE,
    NvJpeg2kExtDecoderTestRef,
    Combine(
        Values(
            "jpeg2k/cat-1046544_640.jp2"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

}} // namespace nvimgcodec::test
