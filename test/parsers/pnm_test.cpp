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
#include "parsers/pnm.h"

namespace nvimgcodec { namespace test {

class PNMParserPluginTest : public ::testing::Test
{
  public:
    PNMParserPluginTest() {}

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.message_severity = NVIMGCODEC_DEBUG_MESSAGE_SEVERITY_DEFAULT;
        create_info.message_category = NVIMGCODEC_DEBUG_MESSAGE_CATEGORY_ALL;


        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));

        pnm_parser_extension_desc_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        pnm_parser_extension_desc_.struct_size = sizeof(nvimgcodecExtensionDesc_t);
        pnm_parser_extension_desc_.struct_next = nullptr;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_pnm_parser_extension_desc(&pnm_parser_extension_desc_));
        nvimgcodecExtensionCreate(instance_, &pnm_parser_extension_, &pnm_parser_extension_desc_);
    }

    void TearDown() override
    {
        if (stream_handle_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        nvimgcodecExtensionDestroy(pnm_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    void TestComments(const char* data, size_t data_size)
    {
        LoadImageFromHostMemory(instance_, stream_handle_, reinterpret_cast<const uint8_t*>(data), data_size);
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
        EXPECT_EQ(NVIMGCODEC_SAMPLING_NONE, info.chroma_subsampling);
        EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
        EXPECT_EQ(0, info.orientation.rotated);
        EXPECT_EQ(false, info.orientation.flip_x);
        EXPECT_EQ(false, info.orientation.flip_y);
        EXPECT_EQ(1, info.num_planes);
        EXPECT_EQ(1, info.plane_info[0].num_channels);
        EXPECT_EQ(6, info.plane_info[0].width);
        EXPECT_EQ(10, info.plane_info[0].height);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    }

    nvimgcodecImageInfo_t expected_cat_2184682_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 398;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_1245673_640()
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
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
    nvimgcodecExtensionDesc_t pnm_parser_extension_desc_{};
    nvimgcodecExtension_t pnm_parser_extension_;
    nvimgcodecCodeStream_t stream_handle_ = nullptr;
};


TEST_F(PNMParserPluginTest, ValidPbm)
{
    auto buffer = read_file(resources_dir + "/pnm/cat-2184682_640.pbm");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected = expected_cat_2184682_640();
    expected.num_planes = 1;
    expected.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    expect_eq(expected, info);
}

TEST_F(PNMParserPluginTest, ValidPgm)
{
    auto buffer = read_file(resources_dir + "/pnm/cat-1245673_640.pgm");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected = expected_cat_1245673_640();
    expected.num_planes = 1;
    expected.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    expect_eq(expected, info);
}


TEST_F(PNMParserPluginTest, ValidPpm)
{
    auto buffer = read_file(resources_dir + "/pnm/cat-111793_640.ppm");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected = expected_cat_111793_640();
    expected.num_planes = 3;
    expected.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    expect_eq(expected, info);
}

TEST_F(PNMParserPluginTest, ValidPbmComment)
{
    const char data[] =
        "P1\n"
        "#This is an example bitmap of the letter \"J\"\n"
        "6 10\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "0 0 0 0 1 0\n"
        "1 0 0 0 1 0\n"
        "0 1 1 1 0 0\n"
        "0 0 0 0 0 0\n"
        "0 0 0 0 0 0\n";
    TestComments(data, sizeof(data));
}

TEST_F(PNMParserPluginTest, ValidPbmCommentInsideToken)
{
  const char data[] =
      "P1\n"
      "6 1#Comment can be inside of a token\n"
      "0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
    TestComments(data, sizeof(data));
}

TEST_F(PNMParserPluginTest, ValidPbmCommentInsideWhitespaces)
{
  const char data[] =
      "P1 \n"
      "#Comment can be inside of whitespaces\n"
      " 6 10\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "0 0 0 0 1 0\n"
      "1 0 0 0 1 0\n"
      "0 1 1 1 0 0\n"
      "0 0 0 0 0 0\n"
      "0 0 0 0 0 0\n";
    TestComments(data, sizeof(data));
}

TEST_F(PNMParserPluginTest, CannotParsePamFormat)
{
    const char data[] = "P7 \n";
    ASSERT_EQ(NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED,
        nvimgcodecCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                             reinterpret_cast<const uint8_t*>(data), sizeof(data)));
}

TEST_F(PNMParserPluginTest, CanParseAllKindsOfWhitespace)
{
    for (uint8_t whitespace : {' ', '\n', '\f', '\r', '\t', '\v'}) {
        const uint8_t data[] = {'P', '6', whitespace};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                                reinterpret_cast<const uint8_t*>(data), sizeof(data)));
    }
}

TEST_F(PNMParserPluginTest, MissingWhitespace)
{
    const char data[] = "P61\n";
    ASSERT_EQ(NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED,
        nvimgcodecCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                             reinterpret_cast<const uint8_t*>(data), sizeof(data)));
}

TEST_F(PNMParserPluginTest, LowercaseP)
{
    const char data[] = "p6 \n";
    ASSERT_EQ(NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED,
        nvimgcodecCodeStreamCreateFromHostMem(this->instance_, &this->stream_handle_,
                                             reinterpret_cast<const uint8_t*>(data), sizeof(data)));
}


}} // namespace nvimgcodec::test
