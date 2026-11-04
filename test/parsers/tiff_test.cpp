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
#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "nvimgcodec_tests.h"
#include "parsers/parser_test_utils.h"
#include "parsers/tiff.h"

using ::testing::Values;

namespace nvimgcodec { namespace test {

namespace {

nvimgcodecStatus_t IgnoreLog(
    void*, const nvimgcodecDebugMessageSeverity_t, const nvimgcodecDebugMessageCategory_t, const nvimgcodecDebugMessageData_t*)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

struct FailingSeekMemoryStream
{
    std::vector<uint8_t> data;
    ptrdiff_t pos = 0;
    bool fail_next_seek = true;

    static nvimgcodecStatus_t read(void* instance, size_t* output_size, void* buf, size_t bytes)
    {
        auto* stream = reinterpret_cast<FailingSeekMemoryStream*>(instance);
        const auto remaining = stream->data.size() - static_cast<size_t>(stream->pos);
        const auto to_read = std::min(bytes, remaining);
        std::memcpy(buf, stream->data.data() + stream->pos, to_read);
        stream->pos += static_cast<ptrdiff_t>(to_read);
        *output_size = to_read;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t seek(void* instance, ptrdiff_t offset, int whence)
    {
        auto* stream = reinterpret_cast<FailingSeekMemoryStream*>(instance);
        ptrdiff_t new_pos = offset;
        if (whence == SEEK_CUR) {
            new_pos += stream->pos;
        } else if (whence == SEEK_END) {
            new_pos += static_cast<ptrdiff_t>(stream->data.size());
        }
        if (new_pos < 0 || new_pos > static_cast<ptrdiff_t>(stream->data.size())) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        if (stream->fail_next_seek) {
            stream->fail_next_seek = false;
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        stream->pos = new_pos;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t tell(void* instance, ptrdiff_t* offset)
    {
        auto* stream = reinterpret_cast<FailingSeekMemoryStream*>(instance);
        *offset = stream->pos;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t size(void* instance, size_t* size)
    {
        auto* stream = reinterpret_cast<FailingSeekMemoryStream*>(instance);
        *size = stream->data.size();
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t write(void*, size_t*, void*, size_t)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecStatus_t putc(void*, size_t*, unsigned char)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecStatus_t skip(void*, size_t)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecStatus_t reserve(void*, size_t)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecStatus_t flush(void*)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecStatus_t map(void*, void**, size_t, size_t)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecStatus_t unmap(void*, void*, size_t)
    {
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
};

nvimgcodecIoStreamDesc_t makeFailingSeekIoStream(FailingSeekMemoryStream& stream)
{
    return {NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC,
        sizeof(nvimgcodecIoStreamDesc_t),
        nullptr,
        &stream,
        0,
        FailingSeekMemoryStream::read,
        FailingSeekMemoryStream::write,
        FailingSeekMemoryStream::putc,
        FailingSeekMemoryStream::skip,
        FailingSeekMemoryStream::seek,
        FailingSeekMemoryStream::tell,
        FailingSeekMemoryStream::size,
        FailingSeekMemoryStream::reserve,
        FailingSeekMemoryStream::flush,
        FailingSeekMemoryStream::map,
        FailingSeekMemoryStream::unmap};
}

nvimgcodecFrameworkDesc_t makeTestFramework()
{
    return {NVIMGCODEC_STRUCTURE_TYPE_FRAMEWORK_DESC, sizeof(nvimgcodecFrameworkDesc_t), nullptr, nullptr, "test", NVIMGCODEC_VER, 0,
        IgnoreLog, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
}

} // namespace

class TIFFParserPluginTest : public ::testing::Test
{
  public:

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        tiff_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        tiff_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        tiff_parser_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &tiff_parser_extension_, &tiff_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        }
        nvimgcodecExtensionDestroy(tiff_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecImageInfo_t expected_cat_1245673_640()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_1046544_640()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 475;
            info.plane_info[p].width = 640;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_300572_640()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 536;
            info.plane_info[p].width = 640;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_300572_640_grayscale()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_GRAY;
        info.num_planes = 1;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 536;
            info.plane_info[p].width = 640;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_8x8_uninitialized()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 8;
            info.plane_info[p].width = 8;
        }
        return info;
    }

    // SampleFormat present but with fewer values than SamplesPerPixel: the first value applies
    // to every channel, so all three planes are INT16.
    nvimgcodecImageInfo_t expected_cat_8x8_partial_sample_format()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 8;
            info.plane_info[p].width = 8;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
            info.plane_info[p].precision = 16;
        }
        return info;
    }

    // BitsPerSample present but with fewer values than SamplesPerPixel: the first value applies
    // to every channel, so all three planes are UINT16 at precision 16.
    nvimgcodecImageInfo_t expected_cat_8x8_partial_bits_per_sample()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 8;
            info.plane_info[p].width = 8;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
            info.plane_info[p].precision = 16;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_16x16_long8()
    {
        auto info = default_expected_image_info();
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 16;
            info.plane_info[p].width = 16;
        }
        return info;
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t tiff_parser_extension_desc_{};
    nvimgcodecExtension_t tiff_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;

private:
    // width and height are left unspecified
    nvimgcodecImageInfo_t default_expected_image_info()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x = 0;
        info.orientation.flip_y = 0;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
        }
        return info;
    }
};

TEST_F(TIFFParserPluginTest, RGB)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_640.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

TEST_F(TIFFParserPluginTest, Grayscale)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-300572_640_grayscale.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_300572_640_grayscale(), info);
}

TEST_F(TIFFParserPluginTest, Grayscale_MissingSPP)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-300572_640_grayscale_no_spp.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_300572_640_grayscale(), info);
}

TEST_F(TIFFParserPluginTest, BigTiff)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_640_bigtiff.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

TEST_F(TIFFParserPluginTest, RGB_FromHostMem)
{
    auto buffer = read_file(resources_dir + "/tiff/cat-1245673_640.tiff");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

// If there are multiple images, parser should return imageInfo of the first image
TEST_F(TIFFParserPluginTest, MULIT_IMAGE)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_300572.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

// Test TIFF with UNINITIALIZED (0) sample format, parser should treat it as UINT
TEST_F(TIFFParserPluginTest, UninitializedSampleFormat)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-8x8_uninitialized_sample_format.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_8x8_uninitialized(), info);
}

// A TIFF whose SampleFormat tag carries fewer values than SamplesPerPixel (one value for a
// 3-channel image) must apply that value to every channel, not leave the remaining channels
// at the UNINITIALIZED default. Real-world GeoTIFFs do this; libtiff/tifffile read the file
// below as int16 throughout.
TEST_F(TIFFParserPluginTest, PartialSampleFormatAppliesToAllChannels)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-8x8_partial_sample_format.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_8x8_partial_sample_format(), info);
}

// Same rule for BitsPerSample: a single value covers every channel. Left unhandled the missing
// entries stay 0, which convert_to_sample_type() maps to UINT8 at precision 0 rather than failing.
TEST_F(TIFFParserPluginTest, PartialBitsPerSampleAppliesToAllChannels)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-8x8_partial_bits_per_sample.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_8x8_partial_bits_per_sample(), info);
}

// Test BigTIFF with LONG8 dimension tags
TEST_F(TIFFParserPluginTest, Long8ValidDimensions)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-16x16_long8_valid.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_16x16_long8(), info);
}

// Test malformed TIFF with dimension exceeding 32-bit limit
TEST_F(TIFFParserPluginTest, DimensionOverflow)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/error/dimension_overflow.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    // Should fail with CODESTREAM_UNSUPPORTED status due to dimension > 2^32
    ASSERT_EQ(NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(TIFFParserPluginTest, WidthOrHeight0)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/error/width_or_height_is_0.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// Explicitly covers the ImageWidth=0 (ImageLength!=0) variant. The on-disk fixture above does
// not specify which dimension is zero, so this in-memory 62-byte minimal classic TIFF nails
// down the ImageWidth=0 path (would previously SIGFPE in div_up(width, ...) during GetInfoImpl).
TEST_F(TIFFParserPluginTest, ImageWidth0_FromHostMem)
{
    // Minimal little-endian classic TIFF: header (8) + 4-entry IFD (2 + 4*12) + next-IFD (4).
    const std::vector<uint8_t> tiff = {
        // Header: 'I','I', version=42, IFD offset = 8
        'I', 'I', 42, 0,
        0x08, 0x00, 0x00, 0x00,
        // IFD: 4 entries
        0x04, 0x00,
        // ImageWidth (256), LONG (4), count=1, value=0  <-- malformed
        0x00, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
        // ImageLength (257), LONG (4), count=1, value=16
        0x01, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // BitsPerSample (258), SHORT (3), count=1, value=8
        0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        // SamplesPerPixel (277), SHORT (3), count=1, value=3
        0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
        // next IFD = 0 (no more)
        0x00, 0x00, 0x00, 0x00,
    };
    LoadImageFromHostMemory(instance_, stream_handle_, tiff.data(), tiff.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(TIFFParserPluginTest, CreationViewAcceptsOnlyBitstreamOffset)
{
    const std::vector<uint8_t> tiff = {
        // Header: 'I','I', version=42, IFD offset = 8
        'I', 'I', 42, 0,
        0x08, 0x00, 0x00, 0x00,
        // IFD: 5 entries
        0x05, 0x00,
        // ImageWidth (256), LONG (4), count=1, value=16
        0x00, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // ImageLength (257), LONG (4), count=1, value=16
        0x01, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // BitsPerSample (258), SHORT (3), count=1, value=8
        0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        // SamplesPerPixel (277), SHORT (3), count=1, value=1
        0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        // RowsPerStrip (278), LONG (4), count=1, value=16
        0x16, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // next IFD = 0 (no more)
        0x00, 0x00, 0x00, 0x00,
    };

    nvimgcodecCodeStreamView_t creation_view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr};
    creation_view.image_idx = 0;
    creation_view.bitstream_offset = 8;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromHostMem(instance_, &stream_handle_, tiff.data(), tiff.size(), &creation_view));

    auto expect_full_image_info = [&]() {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
        EXPECT_EQ(16u, info.plane_info[0].width);
        EXPECT_EQ(16u, info.plane_info[0].height);
    };

    expect_full_image_info();
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
    stream_handle_ = nullptr;

    nvimgcodecCodeStream_t invalid_stream = nullptr;
    creation_view.image_idx = 7;
    creation_view.bitstream_offset = 0;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateFromHostMem(instance_, &invalid_stream, tiff.data(), tiff.size(), &creation_view));
    EXPECT_EQ(nullptr, invalid_stream);

    creation_view.image_idx = 0;
    creation_view.bitstream_offset = 8;
    creation_view.region = {NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr, 2};
    creation_view.region.start[0] = 4;
    creation_view.region.start[1] = 4;
    creation_view.region.end[0] = 8;
    creation_view.region.end[1] = 8;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateFromHostMem(instance_, &invalid_stream, tiff.data(), tiff.size(), &creation_view));
    EXPECT_EQ(nullptr, invalid_stream);

    creation_view.region.ndim = 0;
    creation_view.image_idx = 7;
    creation_view.bitstream_offset = 8;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        nvimgcodecCodeStreamCreateFromHostMem(instance_, &invalid_stream, tiff.data(), tiff.size(), &creation_view));
    EXPECT_EQ(nullptr, invalid_stream);
}

TEST_F(TIFFParserPluginTest, FailingSeekStatusPropagates)
{
    const std::vector<uint8_t> tiff = {
        // Header: 'I','I', version=42, IFD offset = 8
        'I', 'I', 42, 0,
        0x08, 0x00, 0x00, 0x00,
        // IFD: 5 entries
        0x05, 0x00,
        // ImageWidth (256), LONG (4), count=1, value=16
        0x00, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // ImageLength (257), LONG (4), count=1, value=16
        0x01, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // BitsPerSample (258), SHORT (3), count=1, value=8
        0x02, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x08, 0x00, 0x00, 0x00,
        // SamplesPerPixel (277), SHORT (3), count=1, value=1
        0x15, 0x01, 0x03, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
        // RowsPerStrip (278), LONG (4), count=1, value=16
        0x16, 0x01, 0x04, 0x00, 0x01, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
        // next IFD = 0 (no more)
        0x00, 0x00, 0x00, 0x00,
    };

    FailingSeekMemoryStream stream{tiff};
    auto io_stream = makeFailingSeekIoStream(stream);
    nvimgcodecCodeStreamDesc_t code_stream{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, nullptr, &io_stream, nullptr, nullptr};
    auto framework = makeTestFramework();
    TIFFParserPlugin parser_plugin(&framework);
    auto* parser_desc = parser_plugin.getParserDesc();

    nvimgcodecParser_t parser = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, parser_desc->create(parser_desc->instance, &parser));

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    EXPECT_EQ(NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR, parser_desc->getImageInfo(parser, &info, &code_stream));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, parser_desc->destroy(parser));
}

TEST_F(TIFFParserPluginTest, CodeStreamInfoRequiresImageTags)
{
    const std::vector<uint8_t> tiff = {
        // Header: 'I','I', version=42, IFD offset = 8
        'I', 'I', 42, 0,
        0x08, 0x00, 0x00, 0x00,
        // IFD: 0 entries, next IFD = 0. This is not enough for image info.
        0x00, 0x00,
        0x00, 0x00, 0x00, 0x00,
    };

    FailingSeekMemoryStream stream{tiff};
    stream.fail_next_seek = false;
    auto io_stream = makeFailingSeekIoStream(stream);
    nvimgcodecCodeStreamDesc_t code_stream{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, nullptr, &io_stream, nullptr, nullptr};
    auto framework = makeTestFramework();
    TIFFParserPlugin parser_plugin(&framework);
    auto* parser_desc = parser_plugin.getParserDesc();

    nvimgcodecParser_t parser = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, parser_desc->create(parser_desc->instance, &parser));

    nvimgcodecCodeStreamInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
    EXPECT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, parser_desc->getCodeStreamInfo(parser, &info, &code_stream));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, parser_desc->destroy(parser));
}

TEST_F(TIFFParserPluginTest, NestedImageIdxSubstreamRejected)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_300572.tiff");

    nvimgcodecCodeStreamView_t view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr, 1};
    nvimgcodecCodeStream_t sub_stream = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetSubCodeStream(stream_handle_, &sub_stream, &view));

    nvimgcodecCodeStream_t nested_stream = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamGetSubCodeStream(sub_stream, &nested_stream, &view));
    EXPECT_EQ(nullptr, nested_stream);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(sub_stream));
}

TEST_F(TIFFParserPluginTest, InheritedBitstreamOffsetRejectsNonzeroImageIdx)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_300572.tiff");

    // Use the file's real first-IFD offset: getSubCodeStream now eagerly validates a
    // bitstream_offset view at creation, so an arbitrary offset would be rejected.
    nvimgcodecCodeStreamInfoTiffExt_t tiff_ext{
        NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr, 0};
    nvimgcodecCodeStreamInfo_t info{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfo_t), &tiff_ext};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(stream_handle_, &info));
    ASSERT_GT(tiff_ext.ifd_offset, 0u);

    nvimgcodecCodeStreamView_t offset_view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr};
    offset_view.bitstream_offset = tiff_ext.ifd_offset;
    nvimgcodecCodeStream_t sub_stream = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetSubCodeStream(stream_handle_, &sub_stream, &offset_view));

    nvimgcodecCodeStreamView_t image_view{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW, sizeof(nvimgcodecCodeStreamView_t), nullptr, 1};
    nvimgcodecCodeStream_t nested_stream = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamGetSubCodeStream(sub_stream, &nested_stream, &image_view));
    EXPECT_EQ(nullptr, nested_stream);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(sub_stream));
}

TEST_F(TIFFParserPluginTest, MultiPageRootReportsCurrentAndNextIfdOffsets)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/multi_page.tif");

    nvimgcodecCodeStreamInfoTiffExt_t tiff_ext{
        NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr, 0};
    nvimgcodecCodeStreamInfo_t info{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfo_t), &tiff_ext};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(stream_handle_, &info));

    EXPECT_GT(info.num_images, 1u);
    EXPECT_GT(tiff_ext.ifd_offset, 0u);
    EXPECT_GT(tiff_ext.next_ifd_offset, 0u);
}

TEST_F(TIFFParserPluginTest, MultiPageImageIdxSubstreamReportsOneImage)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/multi_page.tif");

    nvimgcodecCodeStreamView_t view{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW,
        sizeof(nvimgcodecCodeStreamView_t), nullptr, 1};
    nvimgcodecCodeStream_t sub_stream = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetSubCodeStream(stream_handle_, &sub_stream, &view));

    nvimgcodecCodeStreamInfoTiffExt_t tiff_ext{
        NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr, 0};
    nvimgcodecCodeStreamInfo_t info{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfo_t), &tiff_ext};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(sub_stream, &info));

    EXPECT_EQ(info.num_images, 1u);
    EXPECT_GT(tiff_ext.ifd_offset, 0u);
    EXPECT_GT(tiff_ext.next_ifd_offset, 0u);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(sub_stream));
}

TEST_F(TIFFParserPluginTest, OutOfRangeImageIdxSubstreamRejectedAtCreation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/multi_page.tif");

    nvimgcodecCodeStreamView_t view{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW,
        sizeof(nvimgcodecCodeStreamView_t), nullptr, 100};
    nvimgcodecCodeStream_t sub_stream = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecCodeStreamGetSubCodeStream(stream_handle_, &sub_stream, &view));
    EXPECT_EQ(nullptr, sub_stream);
}

TEST_F(TIFFParserPluginTest, MultiPageTiffViewsReportBackingStreamSize)
{
    const std::string path = resources_dir + "/tiff/multi_page.tif";
    LoadImageFromFilename(instance_, stream_handle_, path);

    const auto expected_size = static_cast<size_t>(std::filesystem::file_size(path));

    nvimgcodecCodeStreamInfo_t root_info{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(stream_handle_, &root_info));
    EXPECT_EQ(expected_size, root_info.size);

    nvimgcodecCodeStreamView_t image_view{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_VIEW,
        sizeof(nvimgcodecCodeStreamView_t), nullptr, 1};
    nvimgcodecCodeStream_t image_stream = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetSubCodeStream(stream_handle_, &image_stream, &image_view));

    nvimgcodecCodeStreamInfo_t image_info{
        NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO,
        sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetCodeStreamInfo(image_stream, &image_info));
    EXPECT_EQ(expected_size, image_info.size);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(image_stream));
}

class TIFFParserPluginTestEXIF :
    public TIFFParserPluginTest,
    public ::testing::WithParamInterface<std::tuple<std::string, int, int, int>>
{};

TEST_P(TIFFParserPluginTestEXIF, info_check)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_" + std::get<0>(GetParam()) + ".tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = std::get<1>(GetParam());
    expected_info.orientation.flip_x = std::get<2>(GetParam());
    expected_info.orientation.flip_y = std::get<3>(GetParam());
    expect_eq(expected_info, info);
}

INSTANTIATE_TEST_SUITE_P(TIFF_PARSER,
    TIFFParserPluginTestEXIF,
    Values(
        std::tuple{"horizontal", 0, 0, 0},
        std::tuple{"mirror_horizontal", 0, 1, 0},
        std::tuple{"mirror_vertical", 0, 0, 1},
        std::tuple{"rotate_90", 360 - 90, 0, 0},
        std::tuple{"rotate_180", 360 - 180, 0, 0},
        std::tuple{"rotate_270", 360 - 270, 0, 0},
        std::tuple{"mirror_horizontal_rotate_90", 360 - 90, 0, 1},
        std::tuple{"mirror_horizontal_rotate_270", 360 - 270, 0, 1},
        std::tuple{"no_orientation", 0, 0, 0}
    )
);

class TIFFParserPluginTestDtype :
    public TIFFParserPluginTest,
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleDataType_t, uint8_t, nvimgcodecColorSpec_t>>
{};

TEST_P(TIFFParserPluginTestDtype, info_check)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/" + std::get<0>(GetParam()) + ".tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_300572_640();
    for (uint32_t p = 0; p < expected_info.num_planes; p++) {
        expected_info.plane_info[p].sample_type = std::get<1>(GetParam());
        expected_info.plane_info[p].precision = std::get<2>(GetParam());
    }
    expected_info.color_spec = std::get<3>(GetParam());

    expect_eq(expected_info, info);
}

INSTANTIATE_TEST_SUITE_P(TIFF_PARSER,
    TIFFParserPluginTestDtype,
    Values(
        std::tuple{"cat-300572_640", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, 8, NVIMGCODEC_COLORSPEC_SRGB},
        std::tuple{"cat-300572_640_uint16", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 16,   NVIMGCODEC_COLORSPEC_SRGB},
        std::tuple{"cat-300572_640_uint32", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32, 32,  NVIMGCODEC_COLORSPEC_SRGB},
        std::tuple{"cat-300572_640_palette", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 16, NVIMGCODEC_COLORSPEC_PALETTE},
        std::tuple{"cat-300572_640_fp32", NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, 32, NVIMGCODEC_COLORSPEC_SRGB}
    )
);

class TIFFParserPluginTestDimension :
    public TIFFParserPluginTest,
    public ::testing::WithParamInterface<std::tuple<std::string, uint32_t, uint32_t, uint32_t, uint32_t>>
{};

TEST_P(TIFFParserPluginTestDimension, info_check)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/" + std::get<0>(GetParam()) + ".tiff");
    nvimgcodecTileGeometryInfo_t tile_info{NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), 0};
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_info};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

    auto expected_info = expected_cat_300572_640();
    expected_info.struct_next = &tile_info;
    expect_eq(expected_info, info);
    EXPECT_EQ(tile_info.num_tiles_y, std::get<1>(GetParam()));
    EXPECT_EQ(tile_info.num_tiles_x, std::get<2>(GetParam()));
    EXPECT_EQ(tile_info.tile_height, std::get<3>(GetParam()));
    EXPECT_EQ(tile_info.tile_width, std::get<4>(GetParam()));
}

INSTANTIATE_TEST_SUITE_P(TIFF_PARSER,
    TIFFParserPluginTestDimension,
    Values(
        std::tuple{"cat-300572_640", 1, 1, 536, 640},
        std::tuple{"cat-300572_640_tiled", 17, 14, 32, 48},
        std::tuple{"cat-300572_640_striped", 108, 1, 5, 640}
    )
);

}} // namespace nvimgcodec::test
