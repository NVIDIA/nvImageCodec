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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodec_tests.h"
#include "parsers/parser_test_utils.h"
#include "parsers/webp.h"

namespace nvimgcodec { namespace test {

class WebpParserPluginTest : public ::testing::Test
{
  public:
    WebpParserPluginTest() {}

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        webp_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        webp_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        webp_parser_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_webp_parser_extension_desc(&webp_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &webp_parser_extension_, &webp_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        }
        nvimgcodecExtensionDestroy(webp_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }


    nvimgcodecImageInfo_t expected_cat_2184682_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.num_planes = 1;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

        info.plane_info[0].height = 398;
        info.plane_info[0].width = 640;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;

        return info;
    }

    nvimgcodecImageInfo_t expected_cat_3113513_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.num_planes = 1;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

        info.plane_info[0].height = 299;
        info.plane_info[0].width = 640;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;

        return info;
    }

    nvimgcodecImageInfo_t expected_camel_1987672_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGBA;
        info.num_planes = 1;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

        info.plane_info[0].height = 426;
        info.plane_info[0].width = 640;
        info.plane_info[0].num_channels = 4;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;

        return info;
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t webp_parser_extension_desc_{};
    nvimgcodecExtension_t webp_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};

TEST_F(WebpParserPluginTest, OversizedExifChunkSizeReturnsBadCodestream)
{
    // Minimal extended WebP (VP8X + EXIF layout bit) with an EXIF chunk that
    // claims size 0xFFFFFFFF. The parser must reject this before allocating.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        30, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x08,       // layout_mask: EXIF metadata present
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
        'E', 'X', 'I', 'F',
        0xff, 0xff, 0xff, 0xff, // claimed chunk size (~4 GiB)
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, ExifChunkSizeEqualsRemainingAccepted)
{
    // EXIF payload size exactly equals the remaining stream bytes (even size, no pad).
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        32, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x08,       // layout_mask: EXIF metadata present
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
        'E', 'X', 'I', 'F',
        2, 0, 0, 0, // EXIF payload size (== remaining)
        0x00, 0x00, // EXIF payload
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(1u, info.plane_info[0].width);
    EXPECT_EQ(1u, info.plane_info[0].height);
}

TEST_F(WebpParserPluginTest, ExifChunkSizeExceedsRemainingByOneReturnsBadCodestream)
{
    // EXIF claims one more byte than remains after the chunk header.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        32, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x08,       // layout_mask: EXIF metadata present
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
        'E', 'X', 'I', 'F',
        3, 0, 0, 0, // EXIF payload size (remaining + 1)
        0x00, 0x00, // EXIF payload (only 2 bytes present)
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, OversizedVp8xChunkSizeReturnsBadCodestream)
{
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        12, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        0xff, 0xff, 0xff, 0xff, // claimed VP8X chunk size (~4 GiB)
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, TruncatedExifChunkHeaderReturnsBadCodestream)
{
    // EXIF flag set, but fewer than 8 bytes remain for the next chunk header.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        25, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x08,       // layout_mask: EXIF metadata present
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
        'E', 'X', 'I', // truncated EXIF chunk header (4+4 bytes expected)
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, UnexpectedEndOfLossyChunkReturnsBadCodestream)
{
    // VP8 frame tag is present, but the following three-byte sync code is absent.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        15, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', ' ',
        3, 0, 0, 0, // VP8 payload size
        0, 0, 0,    // frame_tag only; sync code missing
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, OddSizedChunkBeforeExifUsesRiffPadding)
{
    // Odd-sized unknown chunk before EXIF requires a pad byte; seek must skip it.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        42, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x08,       // layout_mask: EXIF metadata present
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
        'U', 'N', 'K', 'N',
        1, 0, 0, 0, // odd payload size
        0xAB,       // payload
        0x00,       // RIFF pad
        'E', 'X', 'I', 'F',
        2, 0, 0, 0, // EXIF payload size
        0x00, 0x00, // EXIF payload
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(1u, info.plane_info[0].width);
    EXPECT_EQ(1u, info.plane_info[0].height);
}

TEST_F(WebpParserPluginTest, OddSizedChunkWithoutRiffPaddingReturnsBadCodestream)
{
    // The one-byte payload consumes the stream; the required RIFF pad is absent.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        31, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x08,       // layout_mask: EXIF metadata present
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
        'U', 'N', 'K', 'N',
        1, 0, 0, 0, // odd payload size
        0xAB,       // payload; required RIFF pad missing
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, Vp8ChunkSizeBelowMinReturnsBadCodestream)
{
    // Declared VP8 size 9 with 10 payload bytes present; must not read across the boundary.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        22, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', ' ',
        9, 0, 0, 0, // VP8 payload size (min is 10)
        0, 0, 0,    // frame_tag
        0x9D, 0x01, 0x2A, // sync
        1, 0,       // width
        1, 0,       // height
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, Vp8ChunkSizeEqualsMinAccepted)
{
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        22, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', ' ',
        10, 0, 0, 0, // VP8 payload size (== min)
        0, 0, 0,     // frame_tag
        0x9D, 0x01, 0x2A, // sync
        1, 0,        // width
        1, 0,        // height
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(1u, info.plane_info[0].width);
    EXPECT_EQ(1u, info.plane_info[0].height);
}

TEST_F(WebpParserPluginTest, Vp8lChunkSizeBelowMinReturnsBadCodestream)
{
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        17, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'L',
        4, 0, 0, 0, // VP8L payload size (min is 5)
        0x2F,       // signature
        0, 0, 0, 0, // features
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, Vp8lChunkSizeEqualsMinAccepted)
{
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        17, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'L',
        5, 0, 0, 0, // VP8L payload size (== min)
        0x2F,       // signature
        0, 0, 0, 0, // features -> 1x1
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(1u, info.plane_info[0].width);
    EXPECT_EQ(1u, info.plane_info[0].height);
}

TEST_F(WebpParserPluginTest, Vp8xChunkSizeBelowMinReturnsBadCodestream)
{
    // Size 0 with a full 10-byte VP8X body present must not be accepted as 1x1.
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        22, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        0, 0, 0, 0, // VP8X payload size (min is 10)
        0x00,       // layout_mask
        0, 0, 0,    // reserved
        0, 0, 0,    // canvas width - 1
        0, 0, 0,    // canvas height - 1
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, Vp8xChunkSizeEqualsMinAccepted)
{
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        22, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size (== min)
        0x00,        // layout_mask
        0, 0, 0,     // reserved
        0, 0, 0,     // canvas width - 1 -> 1
        0, 0, 0,     // canvas height - 1 -> 1
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(1u, info.plane_info[0].width);
    EXPECT_EQ(1u, info.plane_info[0].height);
}

TEST_F(WebpParserPluginTest, Vp8xCanvasAreaAtLimitAccepted)
{
    // 65535 * 65537 == 2^32 - 1
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        22, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x00,        // layout_mask
        0, 0, 0,     // reserved
        0xFE, 0xFF, 0x00, // width - 1 = 65534 -> width 65535
        0x00, 0x00, 0x01, // height - 1 = 65536 -> height 65537
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(65535u, info.plane_info[0].width);
    EXPECT_EQ(65537u, info.plane_info[0].height);
}

TEST_F(WebpParserPluginTest, Vp8xCanvasAreaOverLimitReturnsBadCodestream)
{
    // 65536 * 65536 == 2^32
    // clang-format off
    const std::vector<uint8_t> webp = {
        'R', 'I', 'F', 'F',
        22, 0, 0, 0, // file size (ignored by parser)
        'W', 'E', 'B', 'P',
        'V', 'P', '8', 'X',
        10, 0, 0, 0, // VP8X payload size
        0x00,        // layout_mask
        0, 0, 0,     // reserved
        0xFF, 0xFF, 0x00, // width - 1 = 65535 -> width 65536
        0xFF, 0xFF, 0x00, // height - 1 = 65535 -> height 65536
    };
    // clang-format on
    LoadImageFromHostMemory(instance_, stream_handle_, webp.data(), webp.size());

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(WebpParserPluginTest, Lossy)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/lossy/cat-3113513_640.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_3113513_640(), info);
}

TEST_F(WebpParserPluginTest, Lossy_FromHostMem)
{
    auto buffer = read_file(resources_dir + "/webp/lossy/cat-3113513_640.webp");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_3113513_640(), info);
}

TEST_F(WebpParserPluginTest, Lossless)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/lossless/cat-3113513_640.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_3113513_640(), info);
}

TEST_F(WebpParserPluginTest, LosslessAlpha)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/lossless_alpha/camel-1987672_640.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_camel_1987672_640(), info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_Horizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_horizontal.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_MirrorHorizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_mirror_horizontal.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x= 1;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_Rotate180)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_rotate_180.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 180;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_MirrorVertical)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_mirror_vertical.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate270)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_mirror_horizontal_rotate_270.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_Rotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_rotate_90.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate90)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_mirror_horizontal_rotate_90.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_Rotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_rotate_270.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(WebpParserPluginTest, EXIF_Orientation_NoOrientation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/webp/exif_orientation/cat-lossy-2184682_640_no_orientation.webp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_2184682_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

}} // namespace nvimgcodec::test
