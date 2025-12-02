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

#include <extensions/nvpnm/nvpnm_ext.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/parser_test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <tuple>
#include <vector>

#include "nvimgcodec_tests.h"
#include "common.h"
#include "parsers/pnm.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

namespace nvimgcodec { namespace test {

class NvpnmExtTestBase : public ExtensionTestBase
{
  public:
    virtual ~NvpnmExtTestBase() = default;

    virtual void SetUp()
    {
        ExtensionTestBase::SetUp();

        nvpnm_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        nvpnm_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        nvpnm_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvpnm_extension_desc(&nvpnm_extension_desc_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &nvpnm_extension_, &nvpnm_extension_desc_));

        nvpnm_parser_extension_desc.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        nvpnm_parser_extension_desc.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        nvpnm_parser_extension_desc.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_pnm_parser_extension_desc(&nvpnm_parser_extension_desc));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &nvpnm_parser_extension_, &nvpnm_parser_extension_desc));

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    }

    virtual void TearDown()
    {
        ExtensionTestBase::TearDownCodecResources();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(nvpnm_parser_extension_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(nvpnm_extension_));
        ExtensionTestBase::TearDown();
    }

    nvimgcodecExtensionDesc_t nvpnm_extension_desc_{};
    nvimgcodecExtension_t nvpnm_extension_;
    nvimgcodecExtensionDesc_t nvpnm_parser_extension_desc{};
    nvimgcodecExtension_t nvpnm_parser_extension_;
};

class NvpnmExtEncoderTest :
    public NvpnmExtTestBase,
    public TestWithParam<std::tuple<nvimgcodecSampleFormat_t, nvimgcodecQualityType_t, nvimgcodecProcessingStatus>>
{
  public:
    NvpnmExtEncoderTest() {}

    void SetUp() override
    {
        NvpnmExtTestBase::SetUp();

        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));

        sample_format_ = std::get<0>(GetParam());
        color_spec_ = NVIMGCODEC_COLORSPEC_SRGB;
        chroma_subsampling_ = NVIMGCODEC_SAMPLING_NONE;

        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.quality_type = std::get<1>(GetParam());

        image_width_ = 256;
        image_height_ = 256;
        num_components_ = 3; 
        image_size_ = image_width_ * image_height_ * num_components_;

        expected_encode_status = std::get<2>(GetParam());
    }

    void TearDown() override
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }

        NvpnmExtTestBase::TearDown();
    }

    void genRandomImage()
    {
        ref_buffer_.resize(image_size_);

        srand(4771);
        for(size_t i = 0; i < static_cast<size_t>(image_size_); ++i) {
            ref_buffer_[i] = rand()%255;
        } 
    }

    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecEncodeParams_t params_;

    int image_width_;
    int image_height_;
    int num_components_; 
    int image_size_;
    std::vector<unsigned char> ref_buffer_;

    nvimgcodecProcessingStatus expected_encode_status;
};

TEST_P(NvpnmExtEncoderTest, ValidFormatAndParameters)
{
    // generate random image
    genRandomImage();

    image_info_.plane_info[0].width = image_width_;
    image_info_.plane_info[0].height = image_height_;
    image_info_.plane_info[0].precision = 8;
    image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    PrepareImageForFormat();

    auto image_info_ref = image_info_;
    if (sample_format_ == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
        Convert_P_RGB_to_I_RGB(image_buffer_, ref_buffer_, image_info_);
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

    // encode the image
    nvimgcodecImageInfo_t cs_image_info(image_info_);
    strcpy(cs_image_info.codec_name, "pnm");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this, &NvpnmExtEncoderTest::ResizeBufferStatic<NvpnmExtEncoderTest>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

    size_t status_size;
    nvimgcodecProcessingStatus_t encode_status;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(expected_encode_status, encode_status);
    ASSERT_EQ(status_size, 1);
    if (expected_encode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        return;
    }

    // read the compressed image info
    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcodecImageInfo_t load_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &load_info));

    // compare the image info with the original image info
    ASSERT_EQ(load_info.color_spec, image_info_ref.color_spec);
    ASSERT_EQ(load_info.sample_format, image_info_ref.sample_format);
    ASSERT_EQ(load_info.num_planes, image_info_ref.num_planes);
    for (uint32_t p = 0; p < load_info.num_planes; p++) {
        ASSERT_EQ(load_info.plane_info[p].width, image_info_ref.plane_info[p].width);
        ASSERT_EQ(load_info.plane_info[p].height, image_info_ref.plane_info[p].height);
        ASSERT_EQ(load_info.plane_info[p].num_channels, image_info_ref.plane_info[p].num_channels);
        ASSERT_EQ(load_info.plane_info[p].sample_type, image_info_ref.plane_info[p].sample_type);
        ASSERT_EQ(load_info.plane_info[p].precision, image_info_ref.plane_info[p].precision);
    }

    // parse the encoded stream and compare the encoded image content with the original
    size_t total_size = code_stream_buffer_.size();
    size_t image_size = GetBufferSize(image_info_ref);
    size_t header_size = total_size - image_size;

    std::string s_header((char*)code_stream_buffer_.data(), 0, header_size);
    std::stringstream ss_header(s_header);
    std::string magic, app;
    int w, h, p;

    ss_header >> magic >> app >> w >> h >> p;

    ASSERT_EQ(magic, "P6");
    ASSERT_EQ(app, "#nvImageCodec");
    ASSERT_EQ(w, image_width_);
    ASSERT_EQ(h, image_height_);
    ASSERT_EQ(p, 255);

    // compare the image content
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < num_components_; ++c) {
                ASSERT_EQ(code_stream_buffer_[header_size + y*w*num_components_ + x*num_components_ + c], ref_buffer_[c*h*w + y*w + x]);
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(NVPNM_ENCODE_VALID_SRGB_INPUT_FORMATS,
    NvpnmExtEncoderTest,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT, NVIMGCODEC_QUALITY_TYPE_LOSSLESS),
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVPNM_ENCODE_INVALID_QUALITY,
    NvpnmExtEncoderTest,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(
            NVIMGCODEC_QUALITY_TYPE_QUALITY,
            NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
            NVIMGCODEC_QUALITY_TYPE_PSNR,
            NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO
        ),
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

}} // namespace nvimgcodec::test
