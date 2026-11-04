/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <extensions/nvtiff/nvtiff_ext.h>
#include <extensions/nvtiff/batching_heuristics.h>
#include <extensions/nvtiff/nvtiff_utils.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/tiff.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include <cuda_runtime_api.h>
#include <cuda.h>

#include "nvimgcodec_tests.h"
#include "common_ext_decoder_test.h"

#include <nvjpeg.h>

using ::testing::Combine;
using ::testing::Values;

namespace nvimgcodec { namespace test {

class NvTiffExtDecoderTestBase : public CommonExtDecoderTest
{
public:
    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc));

        nvimgcodecExtensionDesc_t nvtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvtiff_extension_desc(&nvtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvtiff_extension_desc));

        CommonExtDecoderTest::CreateDecoder();
    }

};
class NvTiffExtDecoderTest :
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleFormat_t>>,
    public NvTiffExtDecoderTestBase
{
public:
    void SetUp() override
    {
        image_path = std::get<0>(GetParam());
        sample_format = std::get<1>(GetParam());
        NvTiffExtDecoderTestBase::SetUp();
    }
public:
    std::string image_path;
    nvimgcodecSampleFormat_t sample_format;
};

#if defined(_WIN32) || defined(_WIN64)
#define NVTIFF_SKIP_IF_NEEDED()                                                                       \
    do {                                                                                              \
        if (CC_major < 7)                                                                             \
            GTEST_SKIP() << "On Windows, nvCOMP deflate requires sm_70 or higher to work.";           \
    } while (0)
#else
#define NVTIFF_SKIP_IF_NEEDED() do {} while (0)
#endif

TEST_P(NvTiffExtDecoderTest, SingleImageStrideContiguous)
{
    NVTIFF_SKIP_IF_NEEDED();
    TestSingleImage(image_path, sample_format, StrideMode::Contiguous);
}
TEST_P(NvTiffExtDecoderTest, SingleImageStrideSamePadAllPlanes)
{
    NVTIFF_SKIP_IF_NEEDED();
    TestSingleImage(image_path, sample_format, StrideMode::SamePadAllPlanes);
}
TEST_P(NvTiffExtDecoderTest, SingleImageStrideDiffPadPerPlane)
{
    NVTIFF_SKIP_IF_NEEDED();
    TestSingleImage(image_path, sample_format, StrideMode::DiffPadPerPlane);
}

// Higher-precision (uint16) reference decode tests; same extension setup as
// NvTiffExtDecoderTestBase, driven through the path+format+type fixture. The
// uint16 TIFF is deflate-compressed, so this lives under the nvCOMP guard.
class NvTiffExtDecoderTestHighPrec : public CommonExtDecoderTestWithPathFormatAndType
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTestWithPathFormatAndType::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc));

        nvimgcodecExtensionDesc_t nvtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvtiff_extension_desc(&nvtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvtiff_extension_desc));

        CommonExtDecoderTestWithPathFormatAndType::CreateDecoder();
    }
};

DEFINE_SINGLE_IMAGE_HIGHPREC_STRIDE_TESTS(NvTiffExtDecoderTestHighPrec)

#if SKIP_NVTIFF_WITH_NVCOMP_TESTS_ENABLED
    #pragma message("Skipping nvTIFF tests that require nvCOMP")
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(NvTiffExtDecoderTest);
    GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(NvTiffExtDecoderTestHighPrec);
#else
INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_HIGHPREC_UINT16,
    NvTiffExtDecoderTestHighPrec,
    Combine(
        Values("tiff/cat-300572_640_uint16.tiff"),
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB, NVIMGCODEC_SAMPLEFORMAT_I_BGR,
               NVIMGCODEC_SAMPLEFORMAT_P_RGB, NVIMGCODEC_SAMPLEFORMAT_P_BGR),
        Values(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_HIGHPREC_FP32,
    NvTiffExtDecoderTestHighPrec,
    Combine(
        Values("tiff/cat-300572_640_fp32.tiff"),
        Values(NVIMGCODEC_SAMPLEFORMAT_I_RGB), // nvtiff can decode only to interleaved, and we don't allow conversions with FP32 right now
        Values(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32)
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE,
    NvTiffExtDecoderTest,
    Combine(
        Values(
            "tiff/cat-1245673_640.tiff",
            "tiff/cat-1245673_300572.tiff",
            "tiff/cat-300572_640_grayscale.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_FP32_TO_UINT8,
    NvTiffExtDecoderTest,
    Combine(
        Values(
            "tiff/cat-300572_640_fp32.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR
        )
    )
);
#endif

class NvTiffExtDecoderTestWithoutNvCOMP : public NvTiffExtDecoderTest {};

DEFINE_SINGLE_IMAGE_STRIDE_TESTS(NvTiffExtDecoderTestWithoutNvCOMP)

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_WITHOUT_COMPRESSION,
    NvTiffExtDecoderTestWithoutNvCOMP,
    Combine(
        Values(
            "tiff/cat-300572_640_no_compression.tiff",
            "tiff/cat-300572_640_jpeg_compression.tiff",
            "tiff/cat-300572_640_ycbcr.tiff",
            "tiff/cat-300572_640_palette.tiff"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_Y
        )
    )
);

#if SKIP_NVTIFF_WITH_NVCOMP_TESTS_ENABLED
    #pragma message("Skipping nvTIFF tests that require nvCOMP")
#else

class NvTiffExtDecoderTestROI :
    public ::testing::WithParamInterface<std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t>>,
    public NvTiffExtDecoderTestBase
{};

TEST_P(NvTiffExtDecoderTestROI, SingleImage)
{
#if defined(_WIN32) || defined(_WIN64)
    if (CC_major < 7) {
        GTEST_SKIP() << "On Windows, nvCOMP deflate requires sm_70 or higher to work.";
    }
#endif

    nvimgcodecRegion_t region1{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    region1.start[0] = std::get<1>(GetParam());
    region1.start[1] = std::get<2>(GetParam());
    region1.end[0] = std::get<3>(GetParam());
    region1.end[1] = std::get<4>(GetParam());

    TestSingleImage(std::get<0>(GetParam()), NVIMGCODEC_SAMPLEFORMAT_I_RGB, region1);
    TestSingleImage(std::get<0>(GetParam()), NVIMGCODEC_SAMPLEFORMAT_P_BGR, region1);
}

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_ROI,
    NvTiffExtDecoderTestROI,
    Values(
        std::tuple{"tiff/cat-300572_640.tiff", 0, 0, 536, 640},
        std::tuple{"tiff/cat-300572_640.tiff", 100, 150, 275, 598},
        std::tuple{"tiff/cat-300572_640.tiff", 123, 456, 124, 457}
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_ROI_TILED,
    NvTiffExtDecoderTestROI,
    Values(
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 0, 0, 536, 640},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 123, 456, 124, 457},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 32, 48, 64, 96},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 30, 40, 68, 100},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 34, 60, 60, 90},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 128, 240, 320, 624},
        std::tuple{"tiff/cat-300572_640_tiled.tiff", 120, 230, 300, 620}
    )
);

INSTANTIATE_TEST_SUITE_P(NVTIFF_DECODE_ROI_STRIPED,
    NvTiffExtDecoderTestROI,
    Values(
        std::tuple{"tiff/cat-300572_640_striped.tiff", 0, 0, 536, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 123, 456, 124, 457},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 5, 0, 10, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 3, 0, 11, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 6, 10, 9, 600},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 15, 0, 100, 640},
        std::tuple{"tiff/cat-300572_640_striped.tiff", 13, 42, 99, 620}
    )
);
#endif

// A caller may ask for several TIFF tags in a single nvimgcodecDecoderGetMetadata call.
// Each entry names its own tag through `id`, so the batch must return exactly what the
// same tags return when fetched one at a time.
class NvTiffExtDecoderMetadataTest : public NvTiffExtDecoderTestBase
{
  public:
    static constexpr uint16_t kImageWidthTag = 256;
    static constexpr uint16_t kImageLengthTag = 257;

    static nvimgcodecMetadata_t MakeTagRequest(uint16_t tag_id)
    {
        nvimgcodecMetadata_t md{NVIMGCODEC_STRUCTURE_TYPE_METADATA, sizeof(nvimgcodecMetadata_t), nullptr};
        md.kind = NVIMGCODEC_METADATA_KIND_TIFF_TAG;
        md.format = NVIMGCODEC_METADATA_FORMAT_RAW;
        md.id = tag_id;
        md.buffer = nullptr;
        md.buffer_size = 0;
        return md;
    }

    // Two-pass fetch: sizes first, then the bytes. Returns the buffers in request order.
    std::vector<std::vector<uint8_t>> FetchTags(const std::vector<uint16_t>& tag_ids)
    {
        std::vector<nvimgcodecMetadata_t> md;
        for (uint16_t id : tag_ids)
            md.push_back(MakeTagRequest(id));
        std::vector<nvimgcodecMetadata_t*> ptrs;
        for (auto& m : md)
            ptrs.push_back(&m);

        int count = static_cast<int>(ptrs.size());
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderGetMetadata(decoder_, in_code_stream_, ptrs.data(), &count));
        EXPECT_EQ(static_cast<int>(ptrs.size()), count);

        std::vector<std::vector<uint8_t>> buffers(md.size());
        for (size_t i = 0; i < md.size(); ++i) {
            buffers[i].resize(md[i].buffer_size);
            md[i].buffer = buffers[i].data();
        }
        count = static_cast<int>(ptrs.size());
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderGetMetadata(decoder_, in_code_stream_, ptrs.data(), &count));
        return buffers;
    }
};

TEST_F(NvTiffExtDecoderMetadataTest, MultipleTiffTagsInOneCallMatchIndividualFetches)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + "/tiff/JP2K-33003-1.svs");

    auto width_alone = FetchTags({kImageWidthTag});
    auto length_alone = FetchTags({kImageLengthTag});
    auto both = FetchTags({kImageWidthTag, kImageLengthTag});

    ASSERT_EQ(2u, both.size());
    EXPECT_EQ(width_alone[0], both[0]);
    EXPECT_EQ(length_alone[0], both[1]);
    EXPECT_NE(both[0], both[1]) << "width and length must not be the same bytes, or the test proves nothing";
}

// A caller that already knows which kind it wants may populate an entry and call straight
// into nvimgcodecDecoderGetMetadata. Nothing in the public API says a count query has to
// come first, so a kind the stream carries must be returned either way.
TEST_F(NvTiffExtDecoderMetadataTest, ExplicitKindDoesNotRequireACountQueryFirst)
{
    LoadImageFromFilename(instance_, in_code_stream_, resources_dir + "/tiff/JP2K-33003-1.svs");

    nvimgcodecMetadata_t md{NVIMGCODEC_STRUCTURE_TYPE_METADATA, sizeof(nvimgcodecMetadata_t), nullptr};
    md.kind = NVIMGCODEC_METADATA_KIND_ICC_PROFILE;
    md.format = NVIMGCODEC_METADATA_FORMAT_RAW;
    md.buffer = nullptr;
    md.buffer_size = 0;
    nvimgcodecMetadata_t* ptr = &md;

    int count = 1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderGetMetadata(decoder_, in_code_stream_, &ptr, &count));
    ASSERT_EQ(1, count) << "the stream carries an ICC profile but it was reported as absent";
    EXPECT_EQ(NVIMGCODEC_METADATA_KIND_ICC_PROFILE, md.kind);
    EXPECT_GT(md.buffer_size, 0u);
}

// nvTIFF cannot emit unsigned RGB from a signed source: nvtiffDecodeCheckSupported() refuses the
// combination, which surfaced as a silently rejected sample. Such images must instead be decoded
// with NVTIFF_OUTPUT_UNCHANGED_I and converted on the GPU. ConvertNorm maps the signed range onto
// the unsigned one with a midpoint shift, so the extremes land on 0 and 255 and zero lands
// mid-scale. Both fixtures are uncompressed, so they need no nvCOMP guard.
class NvTiffExtDecoderSignedTest : public NvTiffExtDecoderTestBase
{
  public:
    // Decodes rel_path into the given format and output sample type, returning the per-sample
    // processing status. UINT8 and UINT16 outputs take different branches in prepareDecodeSample(),
    // so both need to be reachable from here.
    nvimgcodecProcessingStatus_t DecodeTo(const std::string& rel_path, nvimgcodecSampleFormat_t sample_format,
        int num_channels, nvimgcodecSampleDataType_t out_sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8)
    {
        const int out_bytes = (out_sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16 ||
                                   out_sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_INT16)
                                   ? 2
                                   : 1;
        LoadImageFromFilename(instance_, in_code_stream_, resources_dir + "/" + rel_path);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(in_code_stream_, &image_info_));
        image_info_.sample_format = sample_format;
        image_info_.color_spec = num_channels == 1 ? NVIMGCODEC_COLORSPEC_GRAY : NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.chroma_subsampling = num_channels == 1 ? NVIMGCODEC_SAMPLING_GRAY : NVIMGCODEC_SAMPLING_444;
        image_info_.num_planes = 1;
        image_info_.plane_info[0].num_channels = num_channels;
        image_info_.plane_info[0].sample_type = out_sample_type;
        image_info_.plane_info[0].precision = 8 * out_bytes;
        image_info_.plane_info[0].row_stride = image_info_.plane_info[0].width * num_channels * out_bytes;
        out_buffer_.resize(GetBufferSize(image_info_));
        image_info_.buffer = out_buffer_.data();
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image_, &image_info_));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderDecode(decoder_, &in_code_stream_, &image_, 1, &params_, &future_));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        size_t status_size = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &status_size));
        return status;
    }
};

// The fixtures carry the column ramp {-32767, -16384, 0, 6554, 32767} repeated over 4 rows.
// Under the midpoint shift these become {0, 64, 128, 153, 255}. Tolerance of 1 absorbs the
// rounding mode rather than pinning it.
TEST_F(NvTiffExtDecoderSignedTest, SignedRgbDecodesToUint8)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, DecodeTo("tiff/synth-4x5_int16_rgb.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB, 3));
    const int expected[5] = {0, 64, 128, 153, 255};
    for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
        for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
            for (int c = 0; c < 3; c++) {
                const int got = out_buffer_[y * image_info_.plane_info[0].row_stride + x * 3 + c];
                EXPECT_NEAR(expected[x], got, 1) << " at y=" << y << " x=" << x << " c=" << c;
            }
        }
    }
}

TEST_F(NvTiffExtDecoderSignedTest, SignedGrayscaleDecodesToUint8)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, DecodeTo("tiff/synth-4x5_int16_gray.tiff", NVIMGCODEC_SAMPLEFORMAT_P_Y, 1));
    const int expected[5] = {0, 64, 128, 153, 255};
    for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
        for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
            const int got = out_buffer_[y * image_info_.plane_info[0].row_stride + x];
            EXPECT_NEAR(expected[x], got, 1) << " at y=" << y << " x=" << x;
        }
    }
}

// synth-4x5_int16_rgb.tiff carries the same value in all three channels, so it cannot tell a correct
// BGR decode from one that ignores the channel order. synth-4x5_int16_rgb_channels.tiff rotates the
// ramp by one column per channel instead, so R, G and B differ at every pixel:
//     R = ramp[x], G = ramp[x + 1], B = ramp[x + 2]   (indices mod 5)
// Under the midpoint shift ramp {-32767, -16384, 0, 6554, 32767} becomes {0, 64, 128, 153, 255} for
// a UINT8 output and {0, 16384, 32768, 39323, 65535} for a UINT16 one, so the expectation for
// channel c at column x is simply the ramp rotated by c.
constexpr int kSignedRampUint8[5] = {0, 64, 128, 153, 255};
constexpr int kSignedRampUint16[5] = {0, 16384, 32768, 39323, 65535};

// Checks an interleaved 3-channel plane against the rotated ramp. channel_order maps the position
// of a channel in the output buffer to its index in the rotation, which is what distinguishes RGB
// (0, 1, 2) from BGR (2, 1, 0).
template <typename T>
void ExpectRotatedRamp(const nvimgcodecImageInfo_t& info, const unsigned char* buffer, const int (&ramp)[5],
    const int (&channel_order)[3], int tolerance)
{
    for (uint32_t y = 0; y < info.plane_info[0].height; y++) {
        for (uint32_t x = 0; x < info.plane_info[0].width; x++) {
            for (int c = 0; c < 3; c++) {
                const T* row = reinterpret_cast<const T*>(buffer + y * info.plane_info[0].row_stride);
                const int got = row[x * 3 + c];
                const int expected = ramp[(x + channel_order[c]) % 5];
                EXPECT_NEAR(expected, got, tolerance) << " at y=" << y << " x=" << x << " c=" << c;
            }
        }
    }
}

constexpr int kRgbOrder[3] = {0, 1, 2};
constexpr int kBgrOrder[3] = {2, 1, 0};

// The guard added to prepareDecodeSample() covers BGR through wants_rgb_out, but with a fixture
// whose channels all carry the same value a broken BGR conversion would still pass.
TEST_F(NvTiffExtDecoderSignedTest, SignedBgrDecodesToUint8)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_rgb_channels.tiff", NVIMGCODEC_SAMPLEFORMAT_I_BGR, 3));
    ExpectRotatedRamp<uint8_t>(image_info_, out_buffer_.data(), kSignedRampUint8, kBgrOrder, 1);
}

// The RGB counterpart on the same fixture, so the two together pin the channel order in both
// directions rather than just asserting that some decode happened.
TEST_F(NvTiffExtDecoderSignedTest, SignedRgbChannelOrderIsPreserved)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_rgb_channels.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB, 3));
    ExpectRotatedRamp<uint8_t>(image_info_, out_buffer_.data(), kSignedRampUint8, kRgbOrder, 1);
}

// The UINT16 branch of prepareDecodeSample() gained the same guard as the UINT8 one and routes to
// NVTIFF_OUTPUT_UNCHANGED_I, so LaunchConvertNormKernel runs INT16 -> UINT16 here. Tolerance of 2
// absorbs the rounding of the float intermediate at this magnitude.
TEST_F(NvTiffExtDecoderSignedTest, SignedRgbDecodesToUint16)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_rgb_channels.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB, 3,
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16));
    ExpectRotatedRamp<uint16_t>(image_info_, out_buffer_.data(), kSignedRampUint16, kRgbOrder, 2);
}

TEST_F(NvTiffExtDecoderSignedTest, SignedBgrDecodesToUint16)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_rgb_channels.tiff", NVIMGCODEC_SAMPLEFORMAT_I_BGR, 3,
            NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16));
    ExpectRotatedRamp<uint16_t>(image_info_, out_buffer_.data(), kSignedRampUint16, kBgrOrder, 2);
}

// The libTIFF (CPU) decoder decodes signed samples natively without any midpoint conversion.
// These cover the GPU decoder's own native-signed passthrough: requesting a signed INT16
// output (rather than converting to UINT8/UINT16) makes sample_format/precision/sample_type
// match the natively-decoded image, so need_conversion is false and NVTIFF_OUTPUT_UNCHANGED_I
// should hand back the raw signed samples unmodified - no midpoint shift onto the unsigned range.
constexpr int kSignedRampInt16[5] = {-32767, -16384, 0, 6554, 32767};

TEST_F(NvTiffExtDecoderSignedTest, SignedGrayscaleDecodesToInt16Natively)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_gray.tiff", NVIMGCODEC_SAMPLEFORMAT_P_Y, 1, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16));
    const int16_t* pixels = reinterpret_cast<const int16_t*>(out_buffer_.data());
    for (uint32_t y = 0; y < image_info_.plane_info[0].height; y++) {
        const int16_t* row = pixels + y * (image_info_.plane_info[0].row_stride / sizeof(int16_t));
        for (uint32_t x = 0; x < image_info_.plane_info[0].width; x++) {
            EXPECT_EQ(kSignedRampInt16[x % 5], row[x]) << " at y=" << y << " x=" << x;
        }
    }
}

TEST_F(NvTiffExtDecoderSignedTest, SignedRgbChannelOrderPreservedAsInt16Natively)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_rgb_channels.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB, 3, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16));
    ExpectRotatedRamp<int16_t>(image_info_, out_buffer_.data(), kSignedRampInt16, kRgbOrder, 0);
}

TEST_F(NvTiffExtDecoderSignedTest, SignedBgrChannelOrderPreservedAsInt16Natively)
{
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS,
        DecodeTo("tiff/synth-4x5_int16_rgb_channels.tiff", NVIMGCODEC_SAMPLEFORMAT_I_BGR, 3, NVIMGCODEC_SAMPLE_DATA_TYPE_INT16));
    ExpectRotatedRamp<int16_t>(image_info_, out_buffer_.data(), kSignedRampInt16, kBgrOrder, 0);
}

}} // namespace nvimgcodec::test

TEST(NvTiffUtils, IdentifiesHamamatsuByMcuStartsTag)
{
    auto classify = [](nvtiffCompression_t compression, std::vector<uint16_t> tags) {
        return nvtiff::isHamamatsu(compression, [&](uint16_t tag) {
            return std::find(tags.begin(), tags.end(), tag) != tags.end();
        });
    };
    EXPECT_TRUE(classify(NVTIFF_COMPRESSION_JPEG, {65426}));
    EXPECT_TRUE(classify(NVTIFF_COMPRESSION_JPEG, {273, 65426}));
    EXPECT_FALSE(classify(NVTIFF_COMPRESSION_JPEG, {273}));
    EXPECT_TRUE(classify(NVTIFF_COMPRESSION_JPEG, {324, 65426}));
    EXPECT_TRUE(classify(NVTIFF_COMPRESSION_JPEG, {273, 324, 65426}));
    EXPECT_FALSE(classify(NVTIFF_COMPRESSION_LZW, {273, 65426}));
}

TEST(NvTiffBatchingHeuristics, AutomaticChoicesRespectGroupSizeAndWorkers)
{
    EXPECT_EQ(1u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, false, 1, 8));
    EXPECT_EQ(1u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, false, 31, 8));
    EXPECT_EQ(2u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, false, 32, 8));
    EXPECT_EQ(1u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, false, 4096, 1));
    EXPECT_EQ(2u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, false, 64, 8));
    EXPECT_EQ(3u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, true, 48, 8));
    EXPECT_EQ(4u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_JPEG, true, 64, 8));
    EXPECT_EQ(1u, nvtiff::preferredSubbatchCount(NVTIFF_COMPRESSION_LZW, false, 4096, 8));
}

TEST(NvTiffBatchingHeuristics, ImageLimitOnlyReducesAutomaticSize)
{
    for (auto compression : {NVTIFF_COMPRESSION_JPEG, NVTIFF_COMPRESSION_LZW}) {
        const size_t automatic = nvtiff::subbatchSize(compression, false, 64, 8, 0);
        EXPECT_EQ(1u, nvtiff::subbatchSize(compression, false, 64, 8, 1));
        EXPECT_EQ(3u, nvtiff::subbatchSize(compression, false, 64, 8, 3));
        EXPECT_EQ(automatic, nvtiff::subbatchSize(compression, false, 64, 8, 128));
        EXPECT_EQ(3u, nvtiff::subbatchSize(compression, false, 3, 8, 8));
        EXPECT_EQ(1u, nvtiff::subbatchSize(compression, false, 1, 1, 8));
    }
}

TEST(NvTiffBatchingHeuristics, ParsesOnlyScopedOptions)
{
    EXPECT_EQ(0u, nvtiff::parseUIntOption(nullptr, "max_subbatch_images"));
    EXPECT_EQ(0u, nvtiff::parseUIntOption(":max_subbatch_images=4", "max_subbatch_images"));
    EXPECT_EQ(4u, nvtiff::parseUIntOption(":fancy_upsampling=0 nvtiff_cuda_decoder:max_subbatch_images=4", "max_subbatch_images"));
    EXPECT_EQ(0u, nvtiff::parseUIntOption("nvtiff_cuda_decoder:max_subbatch_images=0", "max_subbatch_images"));
    for (const char* value : {"", "-1", "two", "2x", "184467440737095516160"}) {
        auto option = std::string("nvtiff_cuda_decoder:max_subbatch_images=") + value;
        std::string invalid;
        EXPECT_EQ(0u, nvtiff::parseUIntOption(option.c_str(), "max_subbatch_images", &invalid));
        EXPECT_EQ(option, invalid);
    }
}

TEST(NvTiffBatchingHeuristics, DuplicateOptionsKeepLastOccurrence)
{
    for (const char* name : {"max_subbatch_images", "trace_subbatches"}) {
        const auto prefix = std::string("nvtiff_cuda_decoder:") + name + "=";
        const auto malformed = prefix + "bogus";
        const auto valid = prefix + "8";
        std::string invalid;
        EXPECT_EQ(8u, nvtiff::parseUIntOption((malformed + " " + valid).c_str(), name, &invalid));
        EXPECT_EQ(malformed, invalid);
        invalid.clear();
        EXPECT_EQ(0u, nvtiff::parseUIntOption((valid + " " + malformed).c_str(), name, &invalid));
        EXPECT_EQ(malformed, invalid);
    }
}
