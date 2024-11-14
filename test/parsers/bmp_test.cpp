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
#include "parsers/bmp.h"
#include "parsers/parser_test_utils.h"
#include "nvimgcodec_tests.h"
#include <nvimgcodec.h>
#include <string>
#include <fstream>
#include <vector>
#include <cstring>

namespace nvimgcodec {
namespace test {

class BMPParserPluginTest : public ::testing::Test
{
  public:
    BMPParserPluginTest()
    {
    }

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecInstanceCreate(&instance_, &create_info));

        bmp_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        bmp_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        bmp_parser_extension_desc_.struct_next = nullptr;
         ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            get_bmp_parser_extension_desc(&bmp_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &bmp_parser_extension_, &bmp_parser_extension_desc_);
    }

    void TearDown() override {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
                nvimgcodecCodeStreamDestroy(stream_handle_));
        nvimgcodecExtensionDestroy(bmp_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecImageInfo_t expected_cat_111793_640() {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 426;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
        }
        return info;
    }

    nvimgcodecInstance_t instance_;
    nvimgcodecExtensionDesc_t bmp_parser_extension_desc_{};
    nvimgcodecExtension_t bmp_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};

TEST_F(BMPParserPluginTest, RGB) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640.bmp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_111793_640(), info);
}

TEST_F(BMPParserPluginTest, RGB_FromHostMem) {
    auto buffer = read_file(resources_dir + "/bmp/cat-111793_640.bmp");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_cat_111793_640(), info);
}

TEST_F(BMPParserPluginTest, Grayscale) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640_grayscale.bmp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_111793_640();
    expected_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
    expected_info.num_planes = 1;
    expected_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    expect_eq(expected_info, info);
}


TEST_F(BMPParserPluginTest, Palette_1Bit) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640_palette_1bit.bmp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_111793_640();
    expected_info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
    expected_info.num_planes = 1;
    expected_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    expect_eq(expected_info, info);
}


TEST_F(BMPParserPluginTest, Palette_8Bit) {
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/bmp/cat-111793_640_palette_8bit.bmp");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_111793_640();
    expected_info.num_planes = 3;
    expected_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    expect_eq(expected_info, info);
}

}  // namespace test
}  // namespace nvimgcodec
