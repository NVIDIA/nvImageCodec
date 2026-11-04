/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <cstring>
#include <vector>

namespace nvimgcodec { namespace test {

namespace {

// Minimal valid BMP header, enough for the builtin parser to identify the codec.
unsigned char small_bmp[] = {0x42, 0x4D, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00};

unsigned char encode_output_buffer[256];

unsigned char* ResizeEncodeOutputBuffer(void*, size_t)
{
    return encode_output_buffer;
}

struct ParserCleanupState
{
    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    nvimgcodecExtension_t extension = nullptr;
    nvimgcodecStatus_t destroy_attempt_status = NVIMGCODEC_STATUS_NOT_INITIALIZED;
    nvimgcodecParserDesc_t parser_desc{};
};

nvimgcodecStatus_t cleanupParserCanParse(void*, int* result, nvimgcodecCodeStreamDesc_t*)
{
    *result = 1;
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t cleanupParserCreate(void* instance, nvimgcodecParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcodecParser_t>(instance);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t cleanupParserDestroy(nvimgcodecParser_t parser)
{
    auto* state = reinterpret_cast<ParserCleanupState*>(parser);
    state->destroy_attempt_status = nvimgcodecExtensionDestroy(state->extension);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t cleanupParserGetCodeStreamInfo(
    nvimgcodecParser_t, nvimgcodecCodeStreamInfo_t* code_stream_info, nvimgcodecCodeStreamDesc_t*)
{
    std::strcpy(code_stream_info->codec_name, "parser_cleanup_codec");
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t cleanupParserGetImageInfo(
    nvimgcodecParser_t, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t*)
{
    std::strcpy(image_info->codec_name, "parser_cleanup_codec");
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t cleanupExtensionCreate(
    void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    auto* state = static_cast<ParserCleanupState*>(instance);
    state->framework = framework;
    nvimgcodecStatus_t status =
        framework->registerParser(framework->instance, &state->parser_desc, NVIMGCODEC_PRIORITY_NORMAL);
    if (status == NVIMGCODEC_STATUS_SUCCESS) {
        *extension = reinterpret_cast<nvimgcodecExtension_t>(state);
    }
    return status;
}

nvimgcodecStatus_t cleanupExtensionDestroy(nvimgcodecExtension_t extension)
{
    auto* state = reinterpret_cast<ParserCleanupState*>(extension);
    return state->framework->unregisterParser(state->framework->instance, &state->parser_desc);
}

} // namespace

// Mixing CodeStream / Image handles from a different nvimgcodecInstance than the Decoder /
// Encoder must be rejected with NVIMGCODEC_STATUS_INVALID_PARAMETER rather than crashing.
// Decoder and Encoder under test are created on matching_instance_; foreign_* handles come from
// foreign_instance_. All objects are CPU-only.
class CrossInstanceApiTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        create_info_.load_builtin_modules = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&foreign_instance_, &create_info_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&matching_instance_, &create_info_));

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateFromHostMem(
                foreign_instance_, &foreign_dec_stream_, small_bmp, sizeof(small_bmp), nullptr));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateFromHostMem(
                matching_instance_, &matching_dec_stream_, small_bmp, sizeof(small_bmp), nullptr));

        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        image_info_.buffer = image_buffer_.data();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(foreign_instance_, &foreign_image_, &image_info_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(matching_instance_, &matching_image_, &image_info_));

        nvimgcodecImageInfo_t stream_image_info = image_info_;
        std::strcpy(stream_image_info.codec_name, "bmp");
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateToHostMem(
                foreign_instance_, &foreign_enc_stream_, nullptr, &ResizeEncodeOutputBuffer, &stream_image_info));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateToHostMem(
                matching_instance_, &matching_enc_stream_, nullptr, &ResizeEncodeOutputBuffer, &stream_image_info));

        exec_params_.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
        exec_params_.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(matching_instance_, &decoder_, &exec_params_, nullptr));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(matching_instance_, &encoder_, &exec_params_, nullptr));
    }

    void TearDown() override
    {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(matching_enc_stream_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(foreign_enc_stream_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(matching_image_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(foreign_image_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(matching_dec_stream_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(foreign_dec_stream_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(matching_instance_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(foreign_instance_));
    }

    nvimgcodecInstanceCreateInfo_t create_info_{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    nvimgcodecImageInfo_t image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecDecodeParams_t dec_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    nvimgcodecEncodeParams_t enc_params_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};

    std::vector<unsigned char> image_buffer_{std::vector<unsigned char>(1)};

    nvimgcodecInstance_t foreign_instance_ = nullptr;
    nvimgcodecInstance_t matching_instance_ = nullptr;
    nvimgcodecCodeStream_t foreign_dec_stream_ = nullptr;
    nvimgcodecCodeStream_t matching_dec_stream_ = nullptr;
    nvimgcodecCodeStream_t foreign_enc_stream_ = nullptr;
    nvimgcodecCodeStream_t matching_enc_stream_ = nullptr;
    nvimgcodecImage_t foreign_image_ = nullptr;
    nvimgcodecImage_t matching_image_ = nullptr;
    nvimgcodecDecoder_t decoder_ = nullptr;
    nvimgcodecEncoder_t encoder_ = nullptr;
};

TEST_F(CrossInstanceApiTest, decoder_can_decode_rejects_foreign_handles)
{
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &foreign_dec_stream_, &matching_image_, 1, &dec_params_, &processing_status, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &matching_dec_stream_, &foreign_image_, 1, &dec_params_, &processing_status, 0));
}

TEST_F(CrossInstanceApiTest, decoder_decode_rejects_foreign_handles)
{
    nvimgcodecFuture_t future = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderDecode(decoder_, &foreign_dec_stream_, &matching_image_, 1, &dec_params_, &future));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderDecode(decoder_, &matching_dec_stream_, &foreign_image_, 1, &dec_params_, &future));
}

TEST_F(CrossInstanceApiTest, decoder_get_metadata_rejects_foreign_stream)
{
    int metadata_count = 1;
    nvimgcodecMetadata_t* metadata = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderGetMetadata(decoder_, foreign_dec_stream_, &metadata, &metadata_count));
}

TEST_F(CrossInstanceApiTest, encoder_can_encode_rejects_foreign_handles)
{
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &foreign_image_, &matching_enc_stream_, 1, &enc_params_, &processing_status, 0));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &matching_image_, &foreign_enc_stream_, 1, &enc_params_, &processing_status, 0));
}

TEST_F(CrossInstanceApiTest, encoder_encode_rejects_foreign_handles)
{
    nvimgcodecFuture_t future = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderEncode(encoder_, &foreign_image_, &matching_enc_stream_, 1, &enc_params_, &future));
    ASSERT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderEncode(encoder_, &matching_image_, &foreign_enc_stream_, 1, &enc_params_, &future));
}

TEST(CrossInstanceCodeStreamReuseTest, OldLeaseProtectsParserDuringReplacement)
{
    nvimgcodecInstanceCreateInfo_t create_info{
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
    nvimgcodecInstance_t first_instance = nullptr;
    nvimgcodecInstance_t second_instance = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&first_instance, &create_info));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&second_instance, &create_info));

    ParserCleanupState state;
    state.parser_desc.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC;
    state.parser_desc.struct_size = sizeof(state.parser_desc);
    state.parser_desc.instance = &state;
    state.parser_desc.id = "parser_cleanup_test";
    state.parser_desc.codec = "parser_cleanup_codec";
    state.parser_desc.canParse = &cleanupParserCanParse;
    state.parser_desc.create = &cleanupParserCreate;
    state.parser_desc.destroy = &cleanupParserDestroy;
    state.parser_desc.getCodeStreamInfo = &cleanupParserGetCodeStreamInfo;
    state.parser_desc.getImageInfo = &cleanupParserGetImageInfo;

    nvimgcodecExtensionDesc_t extension_desc{
        NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr};
    extension_desc.instance = &state;
    extension_desc.id = "parser_cleanup_extension";
    extension_desc.version = 1;
    extension_desc.ext_api_version = NVIMGCODEC_VER;
    extension_desc.create = &cleanupExtensionCreate;
    extension_desc.destroy = &cleanupExtensionDestroy;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(first_instance, &state.extension, &extension_desc));

    unsigned char encoded_data[] = {0};
    nvimgcodecCodeStream_t stream = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromHostMem(
            first_instance, &stream, encoded_data, sizeof(encoded_data), nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromHostMem(
            second_instance, &stream, encoded_data, sizeof(encoded_data), nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, state.destroy_attempt_status);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(state.extension));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(second_instance));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(first_instance));
}

}} // namespace nvimgcodec::test
