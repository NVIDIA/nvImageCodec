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

using ::testing::Values;

namespace nvimgcodec { namespace test {

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
        if (stream_handle_)
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        nvimgcodecExtensionDestroy(tiff_parser_extension_);
        nvimgcodecInstanceDestroy(instance_);
    }

    nvimgcodecImageInfo_t expected_cat_1245673_640()
    {
        auto info = default_expected_image_info();
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_1046544_640()
    {
        auto info = default_expected_image_info();
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 475;
            info.plane_info[p].width = 640;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_cat_300572_640()
    {
        auto info = default_expected_image_info();
        for (int p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 536;
            info.plane_info[p].width = 640;
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
        for (int p = 0; p < info.num_planes; p++) {
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
    public ::testing::WithParamInterface<std::tuple<std::string, nvimgcodecSampleDataType_t, uint8_t>>
{};

TEST_P(TIFFParserPluginTestDtype, info_check)
{
    LoadImageFromFilename(instance_, stream_handle_, resources_dir + "/tiff/" + std::get<0>(GetParam()) + ".tiff");
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    auto expected_info = expected_cat_300572_640();
    for (int p = 0; p < expected_info.num_planes; p++) {
        expected_info.plane_info[p].sample_type = std::get<1>(GetParam());
        expected_info.plane_info[p].precision = std::get<2>(GetParam());
    }
    expect_eq(expected_info, info);
}

INSTANTIATE_TEST_SUITE_P(TIFF_PARSER,
    TIFFParserPluginTestDtype,
    Values(
        std::tuple{"cat-300572_640", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, 8},
        std::tuple{"cat-300572_640_uint16", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 16},
        std::tuple{"cat-300572_640_uint32", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32, 32},
        std::tuple{"cat-300572_640_palette", NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, 16},
        std::tuple{"cat-300572_640_fp32", NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, 32}
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
