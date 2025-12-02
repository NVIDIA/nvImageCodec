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
#include "parsers/png.h"

namespace nvimgcodec { namespace test {

class PNGParserPluginTest : public ::testing::Test
{
  public:
    PNGParserPluginTest() {}

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        png_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        png_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        png_parser_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_png_parser_extension_desc(&png_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &png_parser_extension_, &png_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        }
        nvimgcodecExtensionDestroy(png_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecImageInfo_t expected_cat_1245673_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.num_planes = 1;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

        info.plane_info[0].height = 423;
        info.plane_info[0].width = 640;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;

        return info;
    }

    nvimgcodecImageInfo_t expected_bicycle_161524_1280()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGBA;
        info.num_planes = 1;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

        info.plane_info[0].height = 901;
        info.plane_info[0].width = 1280;
        info.plane_info[0].num_channels = 4;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;
        
        return info;
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t png_parser_extension_desc_{};
    nvimgcodecExtension_t png_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};

TEST_F(PNGParserPluginTest, RGB)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/cat-1245673_640.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

TEST_F(PNGParserPluginTest, RGB_FromHostMem)
{
    auto buffer = read_file(resources_dir + "/png/cat-1245673_640.png");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640(), info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_Horizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_horizontal.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x = 0;
    expected_info.orientation.flip_y = 0;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_MirrorHorizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_mirror_horizontal.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x= 1;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_Rotate180)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_rotate_180.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 180;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_MirrorVertical)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_mirror_vertical.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x = 0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate270)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_mirror_horizontal_rotate_270.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x = 0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_Rotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_rotate_90.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate90)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_mirror_horizontal_rotate_90.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_Rotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_rotate_270.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(PNGParserPluginTest, EXIF_Orientation_NoOrientation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/png/exif_orientation/bicycle-161524_1280_no_orientation.png");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_bicycle_161524_1280();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

}} // namespace nvimgcodec::test
