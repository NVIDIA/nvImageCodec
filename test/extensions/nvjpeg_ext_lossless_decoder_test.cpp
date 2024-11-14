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
        NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), 0, 1, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}}};
};

class NvJpegExtLosslessDecoderTestSingleImage :
    public NvJpegExtDecoderTestBase,
    public NvJpegTestBase,
    public TestWithParam<std::tuple<const char*, nvimgcodecColorSpec_t, nvimgcodecSampleFormat_t, nvimgcodecChromaSubsampling_t>>
{
  public:
    virtual ~NvJpegExtLosslessDecoderTestSingleImage() = default;

  protected:
    void SetUp() override
    {
        image_file_ = std::get<0>(GetParam());
        color_spec_ = std::get<1>(GetParam());
        sample_format_ = std::get<2>(GetParam());
        chroma_subsampling_ = std::get<3>(GetParam());
        NvJpegExtDecoderTestBase::SetUp();
        NvJpegTestBase::SetUp();
    }

    virtual void TearDown()
    {
        NvJpegTestBase::TearDown();
        NvJpegExtDecoderTestBase::TearDown();
    }

    nvimgcodecColorSpec_t output_color_spec_ = NVIMGCODEC_COLORSPEC_UNCHANGED;
};

TEST_P(NvJpegExtLosslessDecoderTestSingleImage, LosslessJpegValidFormatAndParameters)
{
#if defined(_WIN32) || defined(_WIN64)
    if (CC_major < 7) {
        GTEST_SKIP() << "On Windows, nvJPEG lossless requires sm_70 or higher to work.";
    }
#endif

    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + image_file_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
    PrepareImageForFormat();

    nvimgcodecImageInfo_t out_image_info(image_info_);
    out_image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
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
}

static const char* css_lossless_filenames[] = {"/jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg",
    "/jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg",
    "/jpeg/lossless/cat-3449999_640_grayscale_16bit.jpg",
    "/jpeg/lossless/cat-3449999_640_grayscale_8bit.jpg"};

// clang-format off
INSTANTIATE_TEST_SUITE_P(NVJPEG_LOSSLESS_DECODE_VARIOUS_CHROMA_WITH_VALID_SRGB_OUTPUT_FORMATS,
    NvJpegExtLosslessDecoderTestSingleImage,
    Combine(::testing::ValuesIn(css_lossless_filenames),
        Values(NVIMGCODEC_COLORSPEC_SRGB),
        Values(NVIMGCODEC_SAMPLEFORMAT_P_Y, NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED),
        Values(NVIMGCODEC_SAMPLING_NONE)));

// clang-format on
}} // namespace nvimgcodec::test
