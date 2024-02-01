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

#include <gtest/gtest.h>

#include <nvimgcodec.h>

#include <extensions/nvjpeg/nvjpeg_ext.h>
#include <parsers/parser_test_utils.h>

#include "nvjpeg_ext_test_common.h"

#include <test_utils.h>
#include "nvimgcodec_tests.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;

#define NV_DEVELOPER_DUMP_OUTPUT_IMAGE_TO_BMP 0

namespace nvimgcodec { namespace test {

class NvJpegExtDecoderTestBase : public NvJpegExtTestBase
{
  public:
    virtual ~NvJpegExtDecoderTestBase() = default;

    void SetUp()
    {
        NvJpegExtTestBase::SetUp();
        // reference is decoded with GPU hybrid backend
        std::string dec_options{":fancy_upsampling=0 nvjpeg_cuda_decoder:hybrid_huffman_threshold=0"};
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.num_backends = 1;
        exec_params.backends = backends_.data();

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, dec_options.c_str()));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        params_.apply_exif_orientation= 1;
    }

    void TearDown()
    {
        if (decoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        NvJpegExtTestBase::TearDown();
    }


    nvimgcodecDecoder_t decoder_;
    nvimgcodecDecodeParams_t params_;
    std::vector<nvimgcodecBackend_t> backends_{{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), 0,
        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 1}}};
};

class NvJpegExtDecoderTestSingleImage : public NvJpegExtDecoderTestBase,
                                        public NvJpegTestBase,
                                        public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t,
                                            nvimgcodecChromaSubsampling_t, nvimgcodecSampleFormat_t, nvimgcodecColorSpec_t>>
{
  public:
    virtual ~NvJpegExtDecoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        reference_output_format_ = std::get<4>(GetParam());
        NvJpegExtDecoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
        output_color_spec_ = std::get<5>(GetParam());
    }

    virtual void TearDown()
    {
        NvJpegTestBase::TearDown();
        NvJpegExtDecoderTestBase::TearDown();
    }

    nvimgcodecColorSpec_t output_color_spec_ = NVIMGCODEC_COLORSPEC_UNCHANGED;
};

TEST_P(NvJpegExtDecoderTestSingleImage, ValidFormatAndParameters)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + image_file_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    PrepareImageForFormat();

    nvimgcodecImageInfo_t out_image_info(image_info_);
    out_image_info.color_spec = output_color_spec_;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &out_image_, &out_image_info));
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
    bool allow_cmyk = (output_color_spec_ != NVIMGCODEC_COLORSPEC_UNCHANGED) && (output_color_spec_ != NVIMGCODEC_COLORSPEC_CMYK) &&
                      ((output_color_spec_ != NVIMGCODEC_COLORSPEC_YCCK));
    DecodeReference(resources_dir, image_file_, reference_output_format_, allow_cmyk);
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

    ASSERT_EQ(
        0, memcmp(reinterpret_cast<void*>(ref_buffer_.data()), reinterpret_cast<void*>(planar_out_buffer_.data()), ref_buffer_.size()));
}

static const char* css_filenames[] = {"/jpeg/padlock-406986_640_410.jpg", "/jpeg/padlock-406986_640_411.jpg",
    "/jpeg/padlock-406986_640_420.jpg", "/jpeg/padlock-406986_640_422.jpg", "/jpeg/padlock-406986_640_440.jpg",
    "/jpeg/padlock-406986_640_444.jpg", "/jpeg/padlock-406986_640_gray.jpg"};

// clang-format off
INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_VARIOUS_CHROMA_WITH_VALID_SRGB_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
    Combine(::testing::ValuesIn(css_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        //Various output chroma subsampling are ignored for SRGB 
         Values(NVIMGCODEC_SAMPLING_NONE, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY, NVIMGCODEC_SAMPLING_410V), 
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_COLORSPEC_SRGB)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_VARIOUS_CHROMA_WITH_VALID_SYCC_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_SYCC),
         Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
         //Various output chroma subsampling should be ignored - there is not resampling in nvjpeg 
         Values(NVIMGCODEC_SAMPLING_NONE, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY, NVIMGCODEC_SAMPLING_410V), 
         Values(NVIMGCODEC_SAMPLEFORMAT_P_YUV),
         Values(NVIMGCODEC_COLORSPEC_SYCC)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_VARIOUS_CHROMA_WITH_VALID_GRAY_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_GRAY),
         Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
         //Various output chroma subsampling should be ignored - there is only luma
         Values(NVIMGCODEC_SAMPLING_NONE, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY, NVIMGCODEC_SAMPLING_410V), 
         Values(NVIMGCODEC_SAMPLEFORMAT_P_Y),
         Values(NVIMGCODEC_COLORSPEC_GRAY)));

static const char* cmyk_and_ycck_filenames[] = {"/jpeg/cmyk.jpg", "/jpeg/cmyk-dali.jpg",
    "/jpeg/ycck_colorspace.jpg"};

INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_CYMK_AND_YCCK_WITH_VALID_SRGB_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
    Combine(::testing::ValuesIn(cmyk_and_ycck_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR, NVIMGCODEC_SAMPLEFORMAT_I_BGR),
        Values(NVIMGCODEC_SAMPLING_NONE), 
        Values(NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_COLORSPEC_SRGB)));

 INSTANTIATE_TEST_SUITE_P(NVJPEG_DECODE_CYMK_AND_YCCK_WITH_VALID_CMYK_OUTPUT_FORMATS, NvJpegExtDecoderTestSingleImage,
     Combine(::testing::ValuesIn(css_filenames),
         Values(NVIMGCODEC_COLORSPEC_CMYK, NVIMGCODEC_COLORSPEC_YCCK), //for unchanged format it should be ignored
         Values(NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED),
         //Various output chroma subsampling should be ignored - there is not resampling in nvjpeg 
         Values(NVIMGCODEC_SAMPLING_NONE, NVIMGCODEC_SAMPLING_422, NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, 
                NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY, NVIMGCODEC_SAMPLING_410V), 
         Values(NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED),
         Values(NVIMGCODEC_COLORSPEC_UNCHANGED)));

// clang-format on
}} // namespace nvimgcodec::test
