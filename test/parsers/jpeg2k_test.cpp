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
#include "parsers/jpeg2k.h"
#include "parsers/parser_test_utils.h"
#include "nvimgcodec_tests.h"
#include <nvimgcodec.h>
#include <string>
#include <fstream>
#include <vector>

#include <cstring>

namespace nvimgcodec {
namespace test {

class JPEG2KParserPluginTest : public ::testing::Test
{
  public:
    JPEG2KParserPluginTest()
    {
    }

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        jpeg2k_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        jpeg2k_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        jpeg2k_parser_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_jpeg2k_parser_extension_desc(&jpeg2k_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &jpeg2k_parser_extension_, &jpeg2k_parser_extension_desc_);
    }

    void TearDown() override {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
                nvimgcodecCodeStreamDestroy(stream_handle_));
        nvimgcodecExtensionDestroy(jpeg2k_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecImageInfo_t expected_cat_1046544_640() {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x =0;
        info.orientation.flip_y =0;
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 475;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 0;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_1245673_640_12bit()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x =0;
        info.orientation.flip_y =0;
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
            info.plane_info[p].precision = 12;
        }
        return info;
    }
    nvimgcodecImageInfo_t expected_cat_1245673_640_5bit()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x =0;
        info.orientation.flip_y =0;
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 5;
        }
        return info;
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t jpeg2k_parser_extension_desc_{};
    nvimgcodecExtension_t jpeg2k_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};


TEST_F(JPEG2KParserPluginTest, Uint8) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1046544_640(), info);
}

TEST_F(JPEG2KParserPluginTest, Uint8_FromHostMem) {
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1046544_640(), info);
}

TEST_F(JPEG2KParserPluginTest, Uint8_Precision_5_FromHostMem)
{
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1245673_640-5bit.jp2");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640_5bit(), info);
}

TEST_F(JPEG2KParserPluginTest, Uint16_Precision_12_FromHostMem)
{
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1245673_640-12bit.jp2");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1245673_640_12bit(), info);
}

TEST_F(JPEG2KParserPluginTest, Uint8_CodeStreamOnly) {
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    std::vector<uint8_t> JP2_header_until_SOC = {
        0x0, 0x0, 0x0, 0xc, 0x6a, 0x50, 0x20, 0x20, 0xd, 0xa, 0x87, 0xa,
        0x0, 0x0, 0x0, 0x14, 0x66, 0x74, 0x79, 0x70, 0x6a, 0x70, 0x32,
        0x20, 0x0, 0x0, 0x0, 0x0, 0x6a, 0x70, 0x32, 0x20, 0x0, 0x0, 0x0,
        0x2d, 0x6a, 0x70, 0x32, 0x68, 0x0, 0x0, 0x0, 0x16, 0x69, 0x68,
        0x64, 0x72, 0x0, 0x0, 0x1, 0xdb, 0x0, 0x0, 0x2, 0x80, 0x0, 0x3,
        0x7, 0x7, 0x0, 0x0, 0x0, 0x0, 0x0, 0xf, 0x63, 0x6f, 0x6c, 0x72,
        0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x10, 0x0, 0x8, 0x83, 0xe8, 0x6a,
        0x70, 0x32, 0x63, 0xff, 0x4f};
    std::vector<uint8_t> just_SOC = {0xff, 0x4f};
    buffer = replace(buffer, JP2_header_until_SOC, just_SOC);
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

    auto expected_info = expected_cat_1046544_640();
    expected_info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;  // don't have such info in codestream

    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_info, info);
}

TEST_F(JPEG2KParserPluginTest, TiledUint8) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg2k/tiled-cat-1046544_640.jp2");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_1046544_640(), info);
}

TEST_F(JPEG2KParserPluginTest, TiledUint16) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/jpeg2k/cat-1046544_640-16bit.jp2");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_1046544_640();
    for (int p = 0; p < expected_info.num_planes; p++)
        expected_info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    expect_eq(expected_info, info);
}

TEST_F(JPEG2KParserPluginTest, ErrorUnexpectedEnd) {
    const std::array<uint8_t, 12> just_the_signatures = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
    LoadImageFromHostMemory(instance_, stream_handle_, just_the_signatures.data(), just_the_signatures.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_NE(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}


}  // namespace test
}  // namespace nvimgcodec
