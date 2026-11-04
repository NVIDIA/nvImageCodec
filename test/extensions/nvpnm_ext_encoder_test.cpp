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
#include <algorithm>
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
    public TestWithParam<
        std::tuple<nvimgcodecSampleFormat_t, nvimgcodecSampleDataType_t, nvimgcodecQualityType_t, nvimgcodecProcessingStatus>>
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
        sample_type_ = std::get<1>(GetParam());
        color_spec_ = NVIMGCODEC_COLORSPEC_SRGB;
        chroma_subsampling_ = NVIMGCODEC_SAMPLING_NONE;

        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.quality_type = std::get<2>(GetParam());

        image_width_ = 256;
        image_height_ = 256;
        num_components_ = 3;
        bytes_per_sample_ = sample_type_ == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 ? 1 : 2;
        precision_ = bytes_per_sample_ * 8;
        num_samples_ = image_width_ * image_height_ * num_components_;

        expected_encode_status = std::get<3>(GetParam());
    }

    void TearDown() override
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }

        NvpnmExtTestBase::TearDown();
    }

    // Planar reference raster, held as raw bytes in native order so that both the 8-bit and the
    // 16-bit case can share the same plumbing.
    void genRandomImage()
    {
        ref_buffer_.resize(static_cast<size_t>(num_samples_) * bytes_per_sample_);
        ref_samples_.resize(num_samples_);

        const uint32_t max_value = bytes_per_sample_ == 1 ? 255 : 65535;

        srand(4771);
        for (size_t i = 0; i < static_cast<size_t>(num_samples_); ++i) {
            // Values above 255 only exist in the 16-bit case; they are what regressed when the
            // encoder always wrote through an unsigned char pointer.
            uint32_t v = static_cast<uint32_t>(rand()) % (max_value + 1);
            // Pin the extremes into the first two samples so they are always exercised: the high
            // byte of the maximum 16-bit value is exactly what the previous encoder dropped.
            if (i == 0) {
                v = max_value;
            } else if (i == 1) {
                v = 0;
            }
            ref_samples_[i] = v;
            if (bytes_per_sample_ == 1) {
                ref_buffer_[i] = static_cast<uint8_t>(v);
            } else {
                uint16_t v16 = static_cast<uint16_t>(v);
                memcpy(ref_buffer_.data() + i * 2, &v16, 2);
            }
        }
    }

    // Reads the y-th row, x-th pixel, c-th channel out of the encoded raster, honoring PNM's
    // big-endian ordering for 16-bit samples.
    uint32_t encodedSample(size_t header_size, int y, int x, int c) const
    {
        size_t offset = header_size +
                        (static_cast<size_t>(y) * image_width_ * num_components_ + static_cast<size_t>(x) * num_components_ + c) *
                            bytes_per_sample_;
        if (bytes_per_sample_ == 1) {
            return code_stream_buffer_[offset];
        }
        return (static_cast<uint32_t>(static_cast<uint8_t>(code_stream_buffer_[offset])) << 8) |
               static_cast<uint8_t>(code_stream_buffer_[offset + 1]);
    }

    uint32_t refSample(int y, int x, int c) const
    {
        return ref_samples_[static_cast<size_t>(c) * image_height_ * image_width_ + static_cast<size_t>(y) * image_width_ + x];
    }

    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecEncodeParams_t params_;

    int image_width_;
    int image_height_;
    int num_components_;
    int bytes_per_sample_;
    int precision_;
    int num_samples_;
    nvimgcodecSampleDataType_t sample_type_;
    std::vector<unsigned char> ref_buffer_;
    std::vector<uint32_t> ref_samples_;

    nvimgcodecProcessingStatus expected_encode_status;
};

TEST_P(NvpnmExtEncoderTest, ValidFormatAndParameters)
{
    // generate random image
    genRandomImage();

    image_info_.plane_info[0].width = image_width_;
    image_info_.plane_info[0].height = image_height_;
    image_info_.plane_info[0].precision = precision_;
    image_info_.plane_info[0].sample_type = sample_type_;
    PrepareImageForFormat();

    auto image_info_ref = image_info_;
    if (sample_format_ == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
        Convert_P_RGB_to_I_RGB(image_buffer_, ref_buffer_);
        image_info_ref.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info_ref.num_planes = image_info_.plane_info[0].num_channels;
        for (uint32_t p = 0; p < image_info_ref.num_planes; p++) {
            image_info_ref.plane_info[p].height = image_info_.plane_info[0].height;
            image_info_ref.plane_info[p].width = image_info_.plane_info[0].width;
            image_info_ref.plane_info[p].row_stride = image_info_.plane_info[0].width * bytes_per_sample_;
            image_info_ref.plane_info[p].num_channels = 1;
            image_info_ref.plane_info[p].sample_type = image_info_.plane_info[0].sample_type;
            image_info_ref.plane_info[p].precision = precision_;
        }
        ASSERT_EQ(GetBufferSize(image_info_ref), ref_buffer_.size());
        image_info_ref.buffer = ref_buffer_.data();
        image_info_ref.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    } else {
        ASSERT_EQ(image_buffer_.size(), ref_buffer_.size());
        memcpy(image_buffer_.data(), ref_buffer_.data(), ref_buffer_.size());
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
    ASSERT_EQ(p, (1 << precision_) - 1);

    // compare the image content
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            for (int c = 0; c < num_components_; ++c) {
                ASSERT_EQ(encodedSample(header_size, y, x, c), refSample(y, x, c))
                    << "at y=" << y << " x=" << x << " c=" << c;
            }
        }
    }
}

INSTANTIATE_TEST_SUITE_P(NVPNM_ENCODE_VALID_SRGB_INPUT_FORMATS,
    NvpnmExtEncoderTest,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_P_RGB),
        Values(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16),
        Values(NVIMGCODEC_QUALITY_TYPE_DEFAULT, NVIMGCODEC_QUALITY_TYPE_LOSSLESS),
        Values(NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    )
);

INSTANTIATE_TEST_SUITE_P(NVPNM_ENCODE_INVALID_QUALITY,
    NvpnmExtEncoderTest,
    Combine(
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB),
        Values(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8),
        Values(
            NVIMGCODEC_QUALITY_TYPE_QUALITY,
            NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
            NVIMGCODEC_QUALITY_TYPE_PSNR,
            NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO
        ),
        Values(NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED)
    )
);

// canEncode() and encode() both enforce the plane layout, and neither can rely on the other having
// run. These cases pin that rejection at the public API, where a regression in either would
// otherwise only show up as an out-of-bounds read.
class NvpnmExtEncoderShapeTest : public NvpnmExtTestBase, public ::testing::Test
{
  public:
    void SetUp() override
    {
        NvpnmExtTestBase::SetUp();
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, nullptr));

        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.quality_type = NVIMGCODEC_QUALITY_TYPE_DEFAULT;

        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    }

    void TearDown() override
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        NvpnmExtTestBase::TearDown();
    }

    // Sets up `num_planes` uint8 planes with the given per-plane heights and row stride, and returns
    // the status the encoder reports. The buffer is sized to exactly what the declared geometry
    // implies (the running sum of row_stride * height), so an accepted-but-out-of-range read would
    // genuinely run off the end rather than landing in slack.
    nvimgcodecProcessingStatus_t EncodeStatusFor(uint32_t num_planes, const std::vector<uint32_t>& heights, size_t row_stride = kWidth,
        uint32_t num_channels = 1, const std::vector<uint32_t>& widths = {})
    {
        image_info_.num_planes = num_planes;
        size_t buffer_size = 0;
        for (uint32_t p = 0; p < num_planes; p++) {
            image_info_.plane_info[p].width = widths.empty() ? kWidth : widths[p];
            image_info_.plane_info[p].height = heights[p];
            image_info_.plane_info[p].num_channels = num_channels;
            image_info_.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            image_info_.plane_info[p].precision = 8;
            image_info_.plane_info[p].row_stride = row_stride;
            buffer_size += row_stride * heights[p];
        }
        // Keep the allocation non-empty so that image creation succeeds and the encoder is the thing
        // doing the rejecting, even when the declared geometry implies no bytes at all.
        image_buffer_.resize(std::max<size_t>(buffer_size, 1));
        image_info_.buffer = image_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

        nvimgcodecImageInfo_t cs_image_info(image_info_);
        strcpy(cs_image_info.codec_name, "pnm");
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
                &NvpnmExtEncoderShapeTest::ResizeBufferStatic<NvpnmExtEncoderShapeTest>, &cs_image_info));
        images_.push_back(in_image_);
        streams_.push_back(out_code_stream_);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));

        size_t status_size;
        nvimgcodecProcessingStatus_t encode_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
        EXPECT_EQ(status_size, 1);
        return encode_status;
    }

    static constexpr uint32_t kWidth = 256;

    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecEncodeParams_t params_;
};

// Planes that are shorter than plane 0 would be indexed over plane 0's rectangle while their byte
// offsets come from their own row_stride * height, reading roughly 64 KB past the buffer.
TEST_F(NvpnmExtEncoderShapeTest, PlanesShorterThanFirstPlaneAreRejected)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED, EncodeStatusFor(3, {256, 1, 1}));
}

// Fewer planes than components would read plane_offsets[] entries SafePlaneByteOffsets() never
// wrote, i.e. an uninitialized stack value used as a buffer offset.
TEST_F(NvpnmExtEncoderShapeTest, TooFewPlanesAreRejected)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED, EncodeStatusFor(2, {256, 256}));
}

// A row_stride that does not span one row packs the planes tighter than they are read, so the last
// plane's final rows run off the end of the buffer even though every plane has the same extent.
TEST_F(NvpnmExtEncoderShapeTest, RowStrideShorterThanARowIsRejected)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED, EncodeStatusFor(3, {256, 256, 256}, kWidth / 2));
}

// The core passes a caller-supplied row_stride through to the plugin verbatim, zero included.
TEST_F(NvpnmExtEncoderShapeTest, ZeroRowStrideIsRejected)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED, EncodeStatusFor(3, {256, 256, 256}, 0));
}

// A plane narrower than plane 0 is indexed over plane 0's width, so its rows are read past their end.
TEST_F(NvpnmExtEncoderShapeTest, PlanesNarrowerThanFirstPlaneAreRejected)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED,
        EncodeStatusFor(3, {256, 256, 256}, kWidth, 1, {256, 128, 128}));
}

// Planar input carries one channel per plane; anything else does not describe an RGB raster.
TEST_F(NvpnmExtEncoderShapeTest, WrongChannelCountIsRejected)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED, EncodeStatusFor(3, {256, 256, 256}, kWidth, 2));
}

TEST_F(NvpnmExtEncoderShapeTest, MatchingPlanesAreAccepted)
{
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, EncodeStatusFor(3, {256, 256, 256}));
}

class NvpnmExtEncoderOrientationTest : public NvpnmExtTestBase, public ::testing::Test
{
  public:
    void SetUp() override
    {
        NvpnmExtTestBase::SetUp();
        const char* options = nullptr;
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, options));
        params_ = {NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        params_.quality_type = NVIMGCODEC_QUALITY_TYPE_DEFAULT;
    }
    void TearDown() override
    {
        if (encoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        NvpnmExtTestBase::TearDown();
    }
    nvimgcodecEncoder_t encoder_ = nullptr;
    nvimgcodecEncodeParams_t params_;
};

TEST_F(NvpnmExtEncoderOrientationTest, CanEncodeRejectsNonIdentityOrientation)
{
    sample_format_ = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    color_spec_ = NVIMGCODEC_COLORSPEC_SRGB;
    chroma_subsampling_ = NVIMGCODEC_SAMPLING_NONE;

    image_info_.plane_info[0].width = 32;
    image_info_.plane_info[0].height = 32;
    image_info_.plane_info[0].precision = 8;
    image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    PrepareImageForFormat();
    image_info_.orientation.rotated = 90;  // non-identity: no encoder applies orientation

    nvimgcodecImageInfo_t cs_image_info(image_info_);
    strcpy(cs_image_info.codec_name, "pnm");
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, (void*)this,
            &NvpnmExtEncoderOrientationTest::ResizeBufferStatic<NvpnmExtEncoderOrientationTest>, &cs_image_info));
    images_.push_back(in_image_);
    streams_.push_back(out_code_stream_);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), 1, &params_, &future_));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
    size_t status_size = 0;
    nvimgcodecProcessingStatus_t encode_status;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &encode_status, &status_size));
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED, encode_status);
}

}} // namespace nvimgcodec::test
