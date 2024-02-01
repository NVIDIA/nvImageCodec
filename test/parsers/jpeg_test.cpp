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
#include "parsers/jpeg.h"
#include "parsers/parser_test_utils.h"
#include "nvimgcodec_tests.h"
#include <nvimgcodec.h>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>

namespace nvimgcodec {
namespace test {

static int NormalizeAngle(int degrees)
{
    return (degrees % 360 + 360) % 360;
}

class JPEGParserPluginTest : public ::testing::Test
{
  public:
    JPEGParserPluginTest()
    {
    }

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecInstanceCreate(&instance_, &create_info));

        jpeg_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        jpeg_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        jpeg_parser_extension_desc_.struct_next = nullptr;
         ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            get_jpeg_parser_extension_desc(&jpeg_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &jpeg_parser_extension_, &jpeg_parser_extension_desc_);
    }

    void TearDown() override {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
                nvimgcodecCodeStreamDestroy(stream_handle_));
        nvimgcodecExtensionDestroy(jpeg_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t jpeg_parser_extension_desc_{};
    nvimgcodecExtension_t jpeg_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};

TEST_F(JPEGParserPluginTest, YCC_410) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_410.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_410, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, YCC_410_Extended_JPEG_info) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_410.jpg");
    nvimgcodecJpegImageInfo_t jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &jpeg_info};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_410, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }

    EXPECT_EQ(NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT, jpeg_info.encoding);
}


TEST_F(JPEGParserPluginTest, YCC_411) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_411.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_411, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_420) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_420, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_422) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_422.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_422, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_440) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_440.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_440, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, YCC_444) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_444.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_444, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, Gray) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_gray.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_Y, info.sample_format);
    EXPECT_EQ(1, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_GRAY, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_GRAY, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}


TEST_F(JPEGParserPluginTest, CMYK) {  // TODO(janton) : get a permissive license free image
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/cmyk-dali.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(4, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_CMYK, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_UNSUPPORTED, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(616, info.plane_info[p].height);
        EXPECT_EQ(792, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, YCCK) {  // TODO(janton) : get a permissive license free image
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/ycck_colorspace.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(4, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_YCCK, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_UNSUPPORTED, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(512, info.plane_info[p].height);
        EXPECT_EQ(512, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, File_vs_MemoryStream)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

    auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info2{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info2));

    EXPECT_EQ(info.struct_type, info2.struct_type);
    EXPECT_EQ(info.struct_size, info2.struct_size);
    EXPECT_EQ(info.struct_next, info2.struct_next);
    EXPECT_STREQ(info.codec_name, info2.codec_name);
    EXPECT_EQ(info.color_spec, info2.color_spec);
    EXPECT_EQ(info.chroma_subsampling, info2.chroma_subsampling);
    EXPECT_EQ(info.sample_format, info2.sample_format);
    EXPECT_EQ(info.orientation.struct_type, info2.orientation.struct_type);
    EXPECT_EQ(info.orientation.struct_size, info2.orientation.struct_size);
    EXPECT_EQ(info.orientation.struct_next, info2.orientation.struct_next);
    EXPECT_EQ(info.orientation.rotated, info2.orientation.rotated);
    EXPECT_EQ(info.orientation.flip_x, info2.orientation.flip_x);
    EXPECT_EQ(info.orientation.flip_y, info2.orientation.flip_y);
    EXPECT_EQ(info.region.struct_type, info.region.struct_type);
    EXPECT_EQ(info.region.struct_size, info.region.struct_size);
    EXPECT_EQ(info.region.struct_next, info.region.struct_next);
    EXPECT_EQ(info.region.ndim, info.region.ndim);
    for (int d = 0; d < info.region.ndim; d++) {
        EXPECT_EQ(info.region.start[d], info2.region.start[d]);
        EXPECT_EQ(info.region.end[d], info2.region.end[d]);
    }
    EXPECT_EQ(info.num_planes, info2.num_planes);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(info.plane_info[p].struct_type, info2.plane_info[p].struct_type);
        EXPECT_EQ(info.plane_info[p].struct_size, info2.plane_info[p].struct_size);
        EXPECT_EQ(info.plane_info[p].struct_next, info2.plane_info[p].struct_next);
        EXPECT_EQ(info.plane_info[p].width, info2.plane_info[p].width);
        EXPECT_EQ(info.plane_info[p].height, info2.plane_info[p].height);
        EXPECT_EQ(info.plane_info[p].row_stride, info2.plane_info[p].row_stride);
        EXPECT_EQ(info.plane_info[p].num_channels, info2.plane_info[p].num_channels);
        EXPECT_EQ(info.plane_info[p].sample_type, info2.plane_info[p].sample_type);
        EXPECT_EQ(info.plane_info[p].precision, info2.plane_info[p].precision);
    }
    EXPECT_EQ(info.buffer, info2.buffer);
    EXPECT_EQ(info.buffer_size, info2.buffer_size);
    EXPECT_EQ(info.buffer_kind, info2.buffer_kind);
    EXPECT_EQ(info.cuda_stream, info2.cuda_stream);
}

TEST_F(JPEGParserPluginTest, Error_CreateStream_Empty)
{
    std::vector<uint8_t> empty;
    ASSERT_NE(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromHostMem(instance_, &stream_handle_, empty.data(), empty.size()));
}

TEST_F(JPEGParserPluginTest, Error_CreateStream_BadSOI) {
  auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
  EXPECT_EQ(0xd8, buffer[1]);  // A valid JPEG starts with ff d8 (Start Of Image marker)...
  buffer[1] = 0xc0;            // ...but we make it ff c0, which is Start Of Frame
  EXPECT_NE(NVIMGCODEC_STATUS_SUCCESS,
    nvimgcodecCodeStreamCreateFromHostMem(instance_, &stream_handle_, buffer.data(), buffer.size()));
}

TEST_F(JPEGParserPluginTest, Error_GetInfo_NoSOF) {
    auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    // We change Start Of Frame marker into a Comment marker
    auto bad = replace(buffer, {0xff, 0xc0}, {0xff, 0xfe});
    // It can match the JPEG parser
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateFromHostMem(instance_, &stream_handle_, bad.data(), bad.size()));
    // Fails to GetInfo (actual parsing) because there's no valid SOF marker
    nvimgcodecImageInfo_t info;
    ASSERT_NE(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(JPEGParserPluginTest, Padding)
{
    /* https://www.w3.org/Graphics/JPEG/itu-t81.pdf section B.1.1.2 Markers
   * Any marker may optionally be preceded by any number of fill bytes,
   * which are bytes assigned code X’FF’ */
    auto buffer = read_file(resources_dir + "/jpeg/padlock-406986_640_420.jpg");
    auto padded = replace(buffer, {0xff, 0xe0}, {0xff, 0xff, 0xff, 0xff, 0xe0});
    padded = replace(padded, {0xff, 0xe1}, {0xff, 0xff, 0xe1});
    padded = replace(padded, {0xff, 0xdb}, {0xff, 0xff, 0xff, 0xdb});
    padded = replace(padded, {0xff, 0xc0}, {0xff, 0xff, 0xff, 0xff, 0xff, 0xc0});
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromHostMem(instance_, &stream_handle_, padded.data(), padded.size()));
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SYCC, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_420, info.chroma_subsampling);
    EXPECT_EQ(0, info.orientation.rotated);
    EXPECT_FALSE(info.orientation.flip_x);
    EXPECT_FALSE(info.orientation.flip_y);
    for (int p = 0; p < info.num_planes; p++) {
        EXPECT_EQ(426, info.plane_info[p].height);
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
    }
}

TEST_F(JPEGParserPluginTest, EXIF_NoOrientation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_no_orientation.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Horizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_horizontal.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorHorizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_horizontal.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(true, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Rotate180)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_rotate_180.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(180, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorVertical)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_vertical.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(0, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(true, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorHorizontalRotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 270, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(true, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Rotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_rotate_90.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 90, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_MirrorHorizontalRotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 90, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(true, info.orientation.flip_y);
}

TEST_F(JPEGParserPluginTest, EXIF_Rotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg/exif/padlock-406986_640_rotate_270.jpg");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    EXPECT_EQ(360 - 270, NormalizeAngle(info.orientation.rotated));
    EXPECT_EQ(false, info.orientation.flip_x);
    EXPECT_EQ(false, info.orientation.flip_y);
}

}  // namespace test
}  // namespace nvimgcodec