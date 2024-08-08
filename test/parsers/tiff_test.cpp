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
#include "parsers/tiff.h"

namespace nvimgcodec { namespace test {

class TIFFParserPluginTest : public ::testing::Test
{
  public:
    TIFFParserPluginTest() {}

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
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
        if (stream_handle_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        nvimgcodecExtensionDestroy(tiff_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecImageInfo_t expected_cat_1245673_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_1046544_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x =0;
        info.orientation.flip_y =0;
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 475;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
        }
        return info;
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t tiff_parser_extension_desc_{};
    nvimgcodecExtension_t tiff_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};

TEST_F(TIFFParserPluginTest, RGB)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/cat-1245673_640.tiff");
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

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Horizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_horizontal.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorHorizontal)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_horizontal.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x= 1;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Rotate180)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_rotate_180.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 180;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorVertical)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_vertical.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate270)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_horizontal_rotate_270.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Rotate90)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_rotate_90.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_MirrorHorizontalRotate90)
{
    LoadImageFromFilename(
        instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_mirror_horizontal_rotate_90.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 90;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y= 1;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_Rotate270)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_rotate_270.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 360 - 270;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

TEST_F(TIFFParserPluginTest, EXIF_Orientation_NoOrientation)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/exif_orientation/cat-1046544_640_no_orientation.tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    expected_info.orientation.rotated = 0;
    expected_info.orientation.flip_x =0;
    expected_info.orientation.flip_y =0;
    expect_eq(expected_info, info);
}

}} // namespace nvimgcodec::test
