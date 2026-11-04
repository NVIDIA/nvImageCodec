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

// Checks for C API inputs that a caller can get wrong (or craft) and that used to be dereferenced,
// copied or indexed without validation.

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <cstring>
#include <vector>

namespace nvimgcodec { namespace test {

namespace {

// Minimal 1x1 24-bit BMP (BITMAPCOREHEADER), enough for the builtin parser to produce image info.
const unsigned char small_bmp[] = {0x42, 0x4D, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00};

// Sink for CodeStreamCreateToHostMem. The encode-side tests below all fail validation before any
// bytes are produced, so this is never actually written through.
unsigned char encode_output_buffer[64];

unsigned char* ResizeEncodeOutputBuffer(void*, size_t)
{
    return encode_output_buffer;
}

} // namespace

class APIHardeningTest : public ::testing::Test
{
  public:
    void SetUp() override
    {
        create_info_.load_builtin_modules = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info_));
        exec_params_.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
        exec_params_.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params_, nullptr));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params_, nullptr));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateFromHostMem(instance_, &code_stream_, small_bmp, sizeof(small_bmp), nullptr));

        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        image_info_.buffer = buffer_.data();
    }

    void TearDown() override
    {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    nvimgcodecInstanceCreateInfo_t create_info_{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
    nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
    nvimgcodecDecodeParams_t dec_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
    nvimgcodecEncodeParams_t enc_params_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
    nvimgcodecImageInfo_t image_info_{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};

    std::vector<unsigned char> buffer_{std::vector<unsigned char>(64)};
    nvimgcodecInstance_t instance_ = nullptr;
    nvimgcodecDecoder_t decoder_ = nullptr;
    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecCodeStream_t code_stream_ = nullptr;
};

// The batch arrays were only checked for a null array, not for null elements: a null entry was
// dereferenced per-element.
TEST_F(APIHardeningTest, null_elements_in_decode_batch_are_rejected)
{
    nvimgcodecImage_t image = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &image_info_));

    nvimgcodecCodeStream_t null_stream = nullptr;
    nvimgcodecImage_t null_image = nullptr;
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    nvimgcodecFuture_t future = nullptr;

    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &null_stream, &image, 1, &dec_params_, &status, 0));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecDecoderCanDecode(decoder_, &code_stream_, &null_image, 1, &dec_params_, &status, 0));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(decoder_, &null_stream, &image, 1, &dec_params_, &future));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderDecode(decoder_, &code_stream_, &null_image, 1, &dec_params_, &future));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
}

TEST_F(APIHardeningTest, null_elements_in_encode_batch_are_rejected)
{
    nvimgcodecImage_t image = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &image_info_));

    nvimgcodecCodeStream_t null_stream = nullptr;
    nvimgcodecImage_t null_image = nullptr;
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    nvimgcodecFuture_t future = nullptr;

    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &null_image, &code_stream_, 1, &enc_params_, &status, 0));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecEncoderCanEncode(encoder_, &image, &null_stream, 1, &enc_params_, &status, 0));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &null_image, &code_stream_, 1, &enc_params_, &future));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderEncode(encoder_, &image, &null_stream, 1, &enc_params_, &future));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
}

// plane_info is a fixed NVIMGCODEC_MAX_NUM_PLANES-element array; num_planes past that used to be
// carried all the way into the temp-buffer allocation loops.
TEST_F(APIHardeningTest, out_of_range_num_planes_is_rejected)
{
    nvimgcodecImage_t image = nullptr;
    auto info = image_info_;
    info.num_planes = NVIMGCODEC_MAX_NUM_PLANES + 1;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageCreate(instance_, &image, &info));

    info.num_planes = 1000;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecImageCreate(instance_, &image, &info));
}

// plane_info[].num_channels indexes nothing and must NOT be bounded by NVIMGCODEC_MAX_NUM_PLANES:
// an interleaved image can legitimately carry many channels, e.g. a DLPack tensor of shape HxWx65.
TEST_F(APIHardeningTest, many_channels_in_a_single_plane_is_accepted)
{
    nvimgcodecImage_t image = nullptr;
    auto info = image_info_;
    info.num_planes = 1;
    info.plane_info[0].num_channels = 65;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &info));
    if (image) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    }
}

TEST_F(APIHardeningTest, out_of_range_num_planes_is_rejected_for_encode_streams)
{
    nvimgcodecCodeStream_t stream = nullptr;
    auto info = image_info_;
    std::strcpy(info.codec_name, "bmp");
    info.num_planes = NVIMGCODEC_MAX_NUM_PLANES + 1;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &stream, nullptr, &ResizeEncodeOutputBuffer, &info));
    if (stream) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
    }
}

// A chained struct is copied at full sizeof; a caller-declared struct_size smaller than that means
// the copy would run past the caller's buffer.
TEST_F(APIHardeningTest, chained_struct_smaller_than_sizeof_is_rejected)
{
    nvimgcodecJpegImageInfo_t jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t) - 8, nullptr};
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &jpeg_info};
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamGetImageInfo(code_stream_, &info));
}

// A caller built against a newer header declares a larger struct_size. Writing the prefix we know
// about is in bounds, so it must be accepted - and the caller's own struct_size must survive, along
// with any trailing bytes it declared.
TEST_F(APIHardeningTest, chained_struct_larger_than_sizeof_is_accepted)
{
    struct
    {
        nvimgcodecJpegImageInfo_t known;
        uint64_t future_field;
    } extended{{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t) + sizeof(uint64_t), nullptr}, 0xABCDEF};

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &extended.known};
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(code_stream_, &info));
    EXPECT_EQ(sizeof(nvimgcodecJpegImageInfo_t) + sizeof(uint64_t), extended.known.struct_size);
    EXPECT_EQ(0xABCDEFu, extended.future_field);
}

// An unknown chained struct used to hit assert(false) on the encode path.
TEST_F(APIHardeningTest, unknown_chained_struct_on_encode_stream_is_rejected)
{
    nvimgcodecImageInfo_t bogus{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecImageInfo_t), nullptr};
    auto info = image_info_;
    std::strcpy(info.codec_name, "bmp");
    info.struct_next = &bogus;
    info.num_planes = 1;

    nvimgcodecCodeStream_t stream = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &stream, nullptr, &ResizeEncodeOutputBuffer, &info));
    if (stream) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
    }
}

// Reject stale layouts before reading any members beyond the fixed structure header.
TEST_F(APIHardeningTest, execution_params_with_wrong_struct_size_is_rejected)
{
    nvimgcodecExecutionParams_t bad{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t) - 8, 0};
    bad.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    nvimgcodecDecoder_t decoder = nullptr;
    nvimgcodecEncoder_t encoder = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, nvimgcodecDecoderCreate(instance_, &decoder, &bad, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR, nvimgcodecEncoderCreate(instance_, &encoder, &bad, nullptr));
}

}} // namespace nvimgcodec::test
