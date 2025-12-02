/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <memory>

namespace nvimgcodec { namespace test {

using ::testing::ByMove;
using ::testing::Eq;
using ::testing::Return;
using ::testing::ReturnPointee;
using ::testing::ReturnRef;

class APINegativeTest : public ::testing::Test
{
  public:
    APINegativeTest() {}

    void SetUp() override { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info_)); }

    void TearDown() override { ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_)); }

    nvimgcodecInstanceCreateInfo_t create_info_{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    nvimgcodecExtensionDesc_t extension_desc_{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
    nvimgcodecDebugMessengerDesc_t dbg_messanger_desc_{
        NVIMGCODEC_STRUCTURE_TYPE_DEBUG_MESSENGER_DESC, sizeof(nvimgcodecDebugMessengerDesc_t), 0};
    nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    nvimgcodecEncodeParams_t enc_params_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
    nvimgcodecDecodeParams_t dec_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};

    nvimgcodecInstance_t instance_;
    nvimgcodecDecoder_t decoder_ = nullptr;
    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecExtension_t extension_;
    nvimgcodecDebugMessenger_t dbg_messenger_;
    nvimgcodecFuture_t future_;
    nvimgcodecProcessingStatus_t processing_status_;
    nvimgcodecImageInfo_t image_info_;
    nvimgcodecImage_t image_ = nullptr;
    nvimgcodecImage_t images_ = nullptr;
    nvimgcodecCodeStream_t code_stream_{nullptr};
    nvimgcodecCodeStream_t streams_{nullptr};
    std::vector<unsigned char> data_{1, 10};
    std::string filename_;
    size_t status_size_;
    nvimgcodecResizeBufferFunc_t resize_buffer_func_;
};

TEST_F(APINegativeTest, get_properties_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecGetProperties(nullptr));
}

TEST_F(APINegativeTest, instance_create_with_null)
{

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecInstanceCreate(nullptr, &create_info_));
    nvimgcodecInstance_t instance;
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecInstanceCreate(&instance, nullptr));
}

TEST_F(APINegativeTest, instance_destroy_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecInstanceDestroy(nullptr));
}

TEST_F(APINegativeTest, extension_create_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecExtensionCreate(nullptr, &extension_, &extension_desc_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecExtensionCreate(instance_, nullptr, &extension_desc_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecExtensionCreate(instance_, &extension_, nullptr));
}

TEST_F(APINegativeTest, _extension_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecExtensionDestroy(nullptr));
}

TEST_F(APINegativeTest, debug_messegner_create_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDebugMessengerCreate(nullptr, &dbg_messenger_, &dbg_messanger_desc_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDebugMessengerCreate(instance_, nullptr, &dbg_messanger_desc_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDebugMessengerCreate(instance_, &dbg_messenger_, nullptr));
}

TEST_F(APINegativeTest, dbg_messenger_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDebugMessengerDestroy(nullptr));
}

TEST_F(APINegativeTest, future_wait_for_all_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecFutureWaitForAll(nullptr));
}

TEST_F(APINegativeTest, future_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecFutureDestroy(nullptr));
}

TEST_F(APINegativeTest, future_get_processing_status_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecFutureGetProcessingStatus(nullptr, &processing_status_, &status_size_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecFutureGetProcessingStatus(future_, nullptr, &status_size_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecFutureGetProcessingStatus(future_, &processing_status_, nullptr));
}

TEST_F(APINegativeTest, image_create_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageCreate(nullptr, &image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageCreate(instance_, nullptr, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageCreate(instance_, &image_, nullptr));
}

TEST_F(APINegativeTest, image_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageDestroy(nullptr));
}

TEST_F(APINegativeTest, get_image_info_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageGetImageInfo(nullptr, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageGetImageInfo(image_, nullptr));
}

TEST_F(APINegativeTest, code_stream_create_from_file_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateFromFile(nullptr, &code_stream_, filename_.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateFromFile(instance_, nullptr, filename_.c_str()));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateFromFile(instance_, &code_stream_, nullptr));
}

TEST_F(APINegativeTest, code_stream_create_from_host_mem_with_null_test)
{
    ASSERT_EQ(
        NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateFromHostMem(nullptr, &code_stream_, data_.data(), data_.size()));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateFromHostMem(instance_, nullptr, data_.data(), data_.size()));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateFromHostMem(instance_, &code_stream_, nullptr, data_.size()));
}

TEST_F(APINegativeTest, code_stream_create_to_file_with_null_test)
{
    ASSERT_EQ(
        NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateToFile(nullptr, &code_stream_, filename_.c_str(), &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateToFile(instance_, nullptr, filename_.c_str(), &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateToFile(instance_, &code_stream_, nullptr, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateToFile(instance_, &code_stream_, filename_.c_str(), nullptr));
}

TEST_F(APINegativeTest, code_stream_create_to_host_mem_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateToHostMem(nullptr, &code_stream_, nullptr, resize_buffer_func_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateToHostMem(instance_, nullptr, nullptr, resize_buffer_func_, &image_info_));
    ASSERT_EQ(
        NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamCreateToHostMem(instance_, &code_stream_, nullptr, nullptr, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &code_stream_, nullptr, resize_buffer_func_, nullptr));
}

TEST_F(APINegativeTest, code_stream_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamDestroy(nullptr));
}

TEST_F(APINegativeTest, code_stream_get_image_info_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamGetImageInfo(nullptr, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamGetImageInfo(code_stream_, nullptr));
}

TEST_F(APINegativeTest, decoder_create_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(nullptr, &decoder_, &exec_params_, nullptr));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance_, nullptr, &exec_params_, nullptr));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance_, &decoder_, nullptr, nullptr));
}

TEST_F(APINegativeTest, decoder_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDestroy(nullptr));
}

TEST_F(APINegativeTest, decoder_can_decode_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params_, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(nullptr, &streams_, &images_, 1, &dec_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, nullptr, &images_, 1, &dec_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &streams_, nullptr, 1, &dec_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &streams_, &images_, 1, nullptr, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &streams_, &images_, 1, &dec_params_, nullptr, 0));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
}

TEST_F(APINegativeTest, decoder_decode_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params_, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(nullptr, &streams_, &images_, 1, &dec_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(decoder_, nullptr, &images_, 1, &dec_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(decoder_, &streams_, nullptr, 1, &dec_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(decoder_, &streams_, &images_, 1, nullptr, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(decoder_, &streams_, &images_, 1, &dec_params_, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
}

TEST_F(APINegativeTest, decoder_get_metadata_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params_, nullptr));

    nvimgcodecMetadata_t* metadata{nullptr};
    int metadata_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderGetMetadata(nullptr, code_stream_, &metadata, &metadata_count));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderGetMetadata(decoder_, nullptr, &metadata, &metadata_count));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderGetMetadata(decoder_, code_stream_, &metadata, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
}

TEST_F(APINegativeTest, decoder_get_metadata_with_negative_or_zero_metadata_count_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params_, nullptr));

    nvimgcodecMetadata_t* metadata{nullptr};
    int metadata_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderGetMetadata(decoder_, code_stream_, &metadata, &metadata_count));
    metadata_count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderGetMetadata(decoder_, code_stream_, &metadata, &metadata_count));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
}

TEST_F(APINegativeTest, encoder_create_with_null)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderCreate(nullptr, &encoder_, &exec_params_, nullptr));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderCreate(instance_, nullptr, &exec_params_, nullptr));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderCreate(instance_, &encoder_, nullptr, nullptr));
}

TEST_F(APINegativeTest, encoder_destroy_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderDestroy(nullptr));
}

TEST_F(APINegativeTest, encoder_can_encode_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params_, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(nullptr, &images_, &streams_, 1, &enc_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, nullptr, &streams_, 1, &enc_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &images_, nullptr, 1, &enc_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &images_, &streams_, 1, nullptr, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &images_, &streams_, 1, &enc_params_, nullptr, 0));

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &images_, &streams_, 0, &enc_params_, &processing_status_, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &images_, &streams_, -5, &enc_params_, &processing_status_, 0));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
}

TEST_F(APINegativeTest, encoder_encode_with_null_test)
{
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params_, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(nullptr, &images_, &streams_, 1, &enc_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, nullptr, &streams_, 1, &enc_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &images_, nullptr, 1, &enc_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &images_, &streams_, 1, nullptr, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &images_, &streams_, 1, &enc_params_, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &images_, &streams_, 0, &enc_params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &images_, &streams_, -5, &enc_params_, &future_));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
}

}} // namespace nvimgcodec::test
