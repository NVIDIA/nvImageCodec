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

#include <extensions/nvbmp/nvbmp_ext.h>
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
#include "parsers/bmp.h"

using ::testing::Bool;
using ::testing::Combine;
using ::testing::TestWithParam;
using ::testing::Values;
using ::testing::ValuesIn;

namespace nvimgcodec { namespace test {

class NvbmpExtTestBase : public ExtensionTestBase
{
  public:
    virtual ~NvbmpExtTestBase() = default;

    virtual void SetUp()
    {
        ExtensionTestBase::SetUp();

        nvbmp_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        nvbmp_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        nvbmp_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvbmp_extension_desc(&nvbmp_extension_desc_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &nvbmp_extension_, &nvbmp_extension_desc_));

        nvbmp_parser_extension_desc.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        nvbmp_parser_extension_desc.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        nvbmp_parser_extension_desc.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_bmp_parser_extension_desc(&nvbmp_parser_extension_desc));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &nvbmp_parser_extension_, &nvbmp_parser_extension_desc));

        image_info_ = {NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    }

    virtual void TearDown()
    {
        ExtensionTestBase::TearDownCodecResources();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(nvbmp_parser_extension_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(nvbmp_extension_));
        ExtensionTestBase::TearDown();
    }

    nvimgcodecExtensionDesc_t nvbmp_extension_desc_{};
    nvimgcodecExtension_t nvbmp_extension_;
    nvimgcodecExtensionDesc_t nvbmp_parser_extension_desc{};
    nvimgcodecExtension_t nvbmp_parser_extension_;
};

class NvbmpExtEncoderTest : public NvbmpExtTestBase, public TestWithParam<nvimgcodecSampleFormat_t>
{
  public:
    NvbmpExtEncoderTest() {}

    void SetUp() override
    {
        NvbmpExtTestBase::SetUp();

        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));

        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.quality = 0;
        params_.target_psnr = 0;

        sample_format_ = GetParam();
        color_spec_ = NVIMGCODEC_COLORSPEC_SRGB;
        chroma_subsampling_ = NVIMGCODEC_SAMPLING_NONE;

        image_width_ = 256;
        image_height_ = 256;
        num_components_ = 3; 
        image_size_ = image_width_ * image_height_ * num_components_;
    }

    void TearDown() override
    {
        if (encoder_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));

        NvbmpExtTestBase::TearDown();
    }

    void genRandomImage()
    {
        ref_buffer_.resize(image_size_);

        srand(4771);
        for(unsigned int i = 0; i < image_size_; ++i) {
            ref_buffer_[i] = rand()%255;
        } 
    }

    nvimgcodecEncoder_t encoder_;
    nvimgcodecEncodeParams_t params_;

    int image_width_;
    int image_height_;
    int num_components_; 
    int image_size_;
    std::vector<unsigned char> ref_buffer_;
};

TEST_P(NvbmpExtEncoderTest, ValidFormatAndParameters)
{
    // generate random image
    genRandomImage();

    image_info_.plane_info[0].width = image_width_;
    image_info_.plane_info[0].height = image_height_;
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
            image_info_ref.plane_info[p].precision = 8;
        }
        image_info_ref.buffer_size = ref_buffer_.size();
        image_info_ref.buffer = ref_buffer_.data();
        image_info_ref.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else {
        memcpy(image_buffer_.data(), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size());
    }

    // encode the image
    nvimgcodecImageInfo_t cs_image_info(image_info_);
    strcpy(cs_image_info.codec_name, "bmp");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this, &NvbmpExtEncoderTest::ResizeBufferStatic<NvbmpExtEncoderTest>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

    size_t status_size;
    nvimgcodecProcessingStatus_t encode_status;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, encode_status);

    // read the compressed image info
    LoadImageFromHostMemory(instance_, in_code_stream_, code_stream_buffer_.data(), code_stream_buffer_.size());
    nvimgcodecImageInfo_t load_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &load_info));

    // compare the image info with the original image info
    ASSERT_EQ(load_info.color_spec, image_info_ref.color_spec);
    ASSERT_EQ(load_info.sample_format, image_info_ref.sample_format);
    ASSERT_EQ(load_info.num_planes, image_info_ref.num_planes);
    for (int p = 0; p < load_info.num_planes; p++) {
        ASSERT_EQ(load_info.plane_info[p].width, image_info_ref.plane_info[p].width);
        ASSERT_EQ(load_info.plane_info[p].height, image_info_ref.plane_info[p].height);
        ASSERT_EQ(load_info.plane_info[p].num_channels, image_info_ref.plane_info[p].num_channels);
        ASSERT_EQ(load_info.plane_info[p].sample_type, image_info_ref.plane_info[p].sample_type);
        ASSERT_EQ(load_info.plane_info[p].precision, image_info_ref.plane_info[p].precision);
    }

    std::vector<uint8_t> decode_buffer;
    load_info.buffer_size = image_info_ref.buffer_size;
    decode_buffer.resize(load_info.buffer_size);
    load_info.buffer = decode_buffer.data();
    load_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    // decode the compressed image
    nvimgcodecDecoder_t decoder;
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
    exec_params.max_num_cpu_threads = 1;
    nvimgcodecStatus_t decoder_create_status = nvimgcodecDecoderCreate(instance_, &decoder, &exec_params, nullptr);
    std::unique_ptr<std::remove_pointer<nvimgcodecDecoder_t>::type, decltype(&nvimgcodecDecoderDestroy)> decoder_raii(
            decoder, &nvimgcodecDecoderDestroy);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, decoder_create_status);

    nvimgcodecDecodeParams_t decode_params;
    decode_params = {NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &out_image_, &load_info));

    nvimgcodecFuture_t decoder_future = nullptr;
    nvimgcodecStatus_t decoder_decode_status = nvimgcodecDecoderDecode(decoder, &in_code_stream_, &out_image_, 1, &decode_params, &decoder_future);
    std::unique_ptr<std::remove_pointer<nvimgcodecFuture_t>::type, decltype(&nvimgcodecFutureDestroy)> decoder_future_raii(
            decoder_future, &nvimgcodecFutureDestroy);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, decoder_decode_status);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(decoder_future));
    nvimgcodecProcessingStatus_t decode_status;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(decoder_future, &decode_status, &status_size));
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, decode_status);

    // compare the decoded image with the original random image
    ASSERT_EQ(0,
        memcmp(reinterpret_cast<void*>(decode_buffer.data()), reinterpret_cast<void*>(ref_buffer_.data()), ref_buffer_.size()));
}

INSTANTIATE_TEST_SUITE_P(NVBMP_ENCODE_VALID_SRGB_INPUT_FORMATS,
    NvbmpExtEncoderTest,
    Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_RGB)
);

}} // namespace nvimgcodec::test
