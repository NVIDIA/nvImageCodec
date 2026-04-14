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
#include <string_view>
#include <fstream>
#include <vector>

#include <cassert>
#include <cstring>

namespace nvimgcodec {
namespace test {

// Helper functions to create synthetic JPEG2000 streams for testing
namespace synthetic_jp2 {

// Write a 32-bit big-endian integer
inline void write_be32(std::vector<uint8_t>& data, uint32_t value) {
    data.push_back((value >> 24) & 0xFF);
    data.push_back((value >> 16) & 0xFF);
    data.push_back((value >> 8) & 0xFF);
    data.push_back(value & 0xFF);
}

// Write a 16-bit big-endian integer
inline void write_be16(std::vector<uint8_t>& data, uint16_t value) {
    data.push_back((value >> 8) & 0xFF);
    data.push_back(value & 0xFF);
}

// Write a box header (size + type)
inline void write_box_header(std::vector<uint8_t>& data, uint32_t size, std::string_view type) {
    assert(type.size() == 4 && "Box type must be exactly 4 bytes");
    write_be32(data, size);
    data.insert(data.end(), type.begin(), type.end());
}

// Create JP2 signature box
inline void create_jp2_signature(std::vector<uint8_t>& data) {
    write_box_header(data, 12, "jP  ");
    write_be32(data, 0x0D0A870A); // JP2 magic bytes
}

// Create file type box
inline void create_ftyp_box(std::vector<uint8_t>& data) {
    write_box_header(data, 20, "ftyp");
    data.insert(data.end(), {'j', 'p', '2', ' '}); // Brand
    write_be32(data, 0); // MinV
    data.insert(data.end(), {'j', 'p', '2', ' '}); // CL
}

// Create image header box
inline void create_ihdr_box(std::vector<uint8_t>& data, uint32_t height, uint32_t width, 
                            uint16_t num_components, uint8_t bpc) {
    write_box_header(data, 22, "ihdr");
    write_be32(data, height);
    write_be32(data, width);
    write_be16(data, num_components);
    data.push_back(bpc); // bits per component (actual_bits - 1)
    data.push_back(7);   // compression type
    data.push_back(0);   // colorspace unknown
    data.push_back(0);   // IPR
}

// Create color specification box
inline void create_colr_box(std::vector<uint8_t>& data, uint8_t method, uint32_t enumCS) {
    if (method == 1) {
        write_box_header(data, 15, "colr");
        data.push_back(method);  // METH
        data.push_back(0);       // PREC
        data.push_back(0);       // APPROX
        write_be32(data, enumCS); // EnumCS
    } else if (method == 2) {
        // ICC profile method - include minimal dummy profile
        write_box_header(data, 15, "colr");
        data.push_back(method);  // METH
        data.push_back(0);       // PREC
        data.push_back(0);       // APPROX
        // Add 4 bytes of dummy ICC profile data
        data.insert(data.end(), {0x00, 0x00, 0x00, 0x00});
    }
}

// Create JP2 header superbox
inline void create_jp2h_box(std::vector<uint8_t>& data, uint32_t height, uint32_t width,
                            uint16_t num_components, uint8_t bpc,
                            uint8_t colr_method, uint32_t enumCS) {
    // Build sub-boxes in a temporary vector to calculate size
    std::vector<uint8_t> sub_boxes;
    create_ihdr_box(sub_boxes, height, width, num_components, bpc);
    create_colr_box(sub_boxes, colr_method, enumCS);
    
    uint32_t total_size = 8 + sub_boxes.size();
    write_box_header(data, total_size, "jp2h");
    data.insert(data.end(), sub_boxes.begin(), sub_boxes.end());
}

// Create raw J2K codestream with SIZ marker (without JP2 wrapper)
inline void create_raw_j2k_codestream(std::vector<uint8_t>& data, uint32_t width, uint32_t height,
                                      uint16_t num_components,
                                      const std::vector<uint8_t>& XRSiz,
                                      const std::vector<uint8_t>& YRSiz,
                                      const std::vector<uint8_t>& Ssiz,
                                      uint16_t RSiz = 0) {
    // SOC marker (0xFF4F)
    write_be16(data, 0xFF4F);
    
    // SIZ marker (0xFF51)
    write_be16(data, 0xFF51);
    
    // Calculate Lsiz (marker segment length)
    uint16_t Lsiz = 38 + 3 * num_components;
    write_be16(data, Lsiz);
    
    // RSiz (capabilities)
    write_be16(data, RSiz); // Defaults to 0 (No profile)
    
    // XSiz, YSiz (reference grid size)
    write_be32(data, width);
    write_be32(data, height);
    
    // XOSiz, YOSiz (image offset)
    write_be32(data, 0);
    write_be32(data, 0);
    
    // XTSiz, YTSiz (tile size)
    write_be32(data, width);
    write_be32(data, height);
    
    // XTOSiz, YTOSiz (tile offset)
    write_be32(data, 0);
    write_be32(data, 0);
    
    // CSiz (number of components)
    write_be16(data, num_components);
    
    // Component parameters
    for (uint16_t i = 0; i < num_components; i++) {
        data.push_back(Ssiz[i]);   // Precision and sign
        data.push_back(XRSiz[i]);  // Horizontal separation
        data.push_back(YRSiz[i]);  // Vertical separation
    }
}

// Create a complete minimal JP2 file
inline std::vector<uint8_t> create_complete_jp2(uint32_t width, uint32_t height,
                                               uint16_t num_components, uint8_t bpc,
                                               uint8_t colr_method, uint32_t enumCS,
                                               const std::vector<uint8_t>& XRSiz,
                                               const std::vector<uint8_t>& YRSiz,
                                               const std::vector<uint8_t>& Ssiz) {
    std::vector<uint8_t> data;
    
    // JP2 signature
    create_jp2_signature(data);
    
    // File type box
    create_ftyp_box(data);
    
    // JP2 header box
    create_jp2h_box(data, height, width, num_components, bpc, colr_method, enumCS);
    
    // Codestream box (need to build it first to know its size)
    std::vector<uint8_t> codestream;
    create_raw_j2k_codestream(codestream, width, height, num_components, XRSiz, YRSiz, Ssiz);
    write_box_header(data, 8 + codestream.size(), "jp2c");
    data.insert(data.end(), codestream.begin(), codestream.end());
    
    return data;
}

// Create JP2 file with codestream box that has LBox=0 (extends to EOF)
inline std::vector<uint8_t> create_jp2_with_box_size_zero(uint32_t width, uint32_t height,
                                                          uint16_t num_components, uint8_t bpc,
                                                          uint8_t colr_method, uint32_t enumCS,
                                                          const std::vector<uint8_t>& XRSiz,
                                                          const std::vector<uint8_t>& YRSiz,
                                                          const std::vector<uint8_t>& Ssiz) {
    std::vector<uint8_t> data;
    
    // JP2 signature
    create_jp2_signature(data);
    
    // File type box
    create_ftyp_box(data);
    
    // JP2 header box
    create_jp2h_box(data, height, width, num_components, bpc, colr_method, enumCS);
    
    // Codestream box with LBox=0 (extends to end of file)
    write_box_header(data, 0, "jp2c");  // LBox = 0 means "extends to EOF"
    create_raw_j2k_codestream(data, width, height, num_components, XRSiz, YRSiz, Ssiz);
    
    return data;
}

// Create JP2 file with multiple colr boxes
inline std::vector<uint8_t> create_jp2_with_multiple_colr(uint32_t width, uint32_t height,
                                                          uint16_t num_components, uint8_t bpc,
                                                          uint32_t first_enumCS, uint32_t second_enumCS,
                                                          const std::vector<uint8_t>& XRSiz,
                                                          const std::vector<uint8_t>& YRSiz,
                                                          const std::vector<uint8_t>& Ssiz) {
    std::vector<uint8_t> data;
    
    // JP2 signature
    create_jp2_signature(data);
    
    // File type box
    create_ftyp_box(data);
    
    // JP2 header box with multiple colr boxes (need to build it to calculate size)
    std::vector<uint8_t> jp2h_content;
    create_ihdr_box(jp2h_content, height, width, num_components, bpc);
    create_colr_box(jp2h_content, 1, first_enumCS);
    create_colr_box(jp2h_content, 1, second_enumCS);
    
    uint32_t total_size = 8 + jp2h_content.size();
    write_box_header(data, total_size, "jp2h");
    data.insert(data.end(), jp2h_content.begin(), jp2h_content.end());
    
    // Codestream box (need to build it first to know its size)
    std::vector<uint8_t> codestream;
    create_raw_j2k_codestream(codestream, width, height, num_components, XRSiz, YRSiz, Ssiz);
    write_box_header(data, 8 + codestream.size(), "jp2c");
    data.insert(data.end(), codestream.begin(), codestream.end());
    
    return data;
}

// Create JP2 file with Part-2 extensions (high RSiz value)
inline std::vector<uint8_t> create_jp2_with_part2_rsiz(uint32_t width, uint32_t height,
                                                       uint16_t num_components, uint8_t bpc,
                                                       uint8_t colr_method, uint32_t enumCS,
                                                       const std::vector<uint8_t>& XRSiz,
                                                       const std::vector<uint8_t>& YRSiz,
                                                       const std::vector<uint8_t>& Ssiz,
                                                       uint16_t RSiz) {
    std::vector<uint8_t> data;
    
    // JP2 signature
    create_jp2_signature(data);
    
    // File type box
    create_ftyp_box(data);
    
    // JP2 header box
    create_jp2h_box(data, height, width, num_components, bpc, colr_method, enumCS);
    
    // Codestream box with custom RSiz (need to build it first to know its size)
    std::vector<uint8_t> codestream;
    create_raw_j2k_codestream(codestream, width, height, num_components, XRSiz, YRSiz, Ssiz, RSiz);
    write_box_header(data, 8 + codestream.size(), "jp2c");
    data.insert(data.end(), codestream.begin(), codestream.end());
    
    return data;
}

} // namespace synthetic_jp2

// These tests verify edge cases and features identified in the parser evaluation.

class JPEG2KParserPluginTest : public ::testing::Test
{
  public:
    JPEG2KParserPluginTest()
    {
    }

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.create_debug_messenger = 1;
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
        if (stream_handle_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
                nvimgcodecCodeStreamDestroy(stream_handle_));
        }
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
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 475;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
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
        for (uint32_t p = 0; p < info.num_planes; p++) {
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
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = 423;
            info.plane_info[p].width = 640;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 5;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_rgba_image(uint32_t width, uint32_t height, 
                                              nvimgcodecChromaSubsampling_t subsampling)
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGBA;
        info.num_planes = 4;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = subsampling;
        info.orientation.rotated = 0;
        info.orientation.flip_x = 0;
        info.orientation.flip_y = 0;
        for (uint32_t p = 0; p < info.num_planes; p++) {
            info.plane_info[p].height = height;
            info.plane_info[p].width = width;
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
        }
        return info;
    }

    nvimgcodecImageInfo_t expected_grayscale_image(uint32_t width, uint32_t height)
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
        info.num_planes = 1;
        info.color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
        info.orientation.rotated = 0;
        info.orientation.flip_x = 0;
        info.orientation.flip_y = 0;
        info.plane_info[0].height = height;
        info.plane_info[0].width = width;
        info.plane_info[0].num_channels = 1;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;
        return info;
    }

    nvimgcodecImageInfo_t expected_yuv_image(uint32_t width, uint32_t height,
                                             nvimgcodecChromaSubsampling_t subsampling)
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_YUV;
        info.num_planes = 3;
        info.color_spec = NVIMGCODEC_COLORSPEC_SYCC;
        info.chroma_subsampling = subsampling;
        info.orientation.rotated = 0;
        info.orientation.flip_x = 0;
        info.orientation.flip_y = 0;
        
        // Y plane (full resolution)
        info.plane_info[0].height = height;
        info.plane_info[0].width = width;
        info.plane_info[0].num_channels = 1;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;
        
        // UV planes depend on subsampling
        for (uint32_t p = 1; p < 3; p++) {
            info.plane_info[p].num_channels = 1;
            info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
            info.plane_info[p].precision = 8;
            
            if (subsampling == NVIMGCODEC_SAMPLING_420) {
                info.plane_info[p].width = width / 2;
                info.plane_info[p].height = height / 2;
            } else if (subsampling == NVIMGCODEC_SAMPLING_422) {
                info.plane_info[p].width = width / 2;
                info.plane_info[p].height = height;
            } else { // 444 or NONE
                info.plane_info[p].width = width;
                info.plane_info[p].height = height;
            }
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
    expected_info.color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN; // color spec information are in jp2 header

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
    for (uint32_t p = 0; p < expected_info.num_planes; p++) {
        expected_info.plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        expected_info.plane_info[p].precision = 16;
    }
    expect_eq(expected_info, info);
}

TEST_F(JPEG2KParserPluginTest, ErrorUnexpectedEnd) {
    const std::array<uint8_t, 12> just_the_signatures = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
    LoadImageFromHostMemory(instance_, stream_handle_, just_the_signatures.data(), just_the_signatures.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_NE(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// Test for 4-component RGBA image with 4:4:4 (no subsampling)
TEST_F(JPEG2KParserPluginTest, RGBA_NoSubsampling) {
    // Create synthetic RGBA image with 4:4:4 (no subsampling)
    // Note: Subsampling detection based on first 3 components per library design
    std::vector<uint8_t> XRSiz = {1, 1, 1, 1};  // No horizontal subsampling
    std::vector<uint8_t> YRSiz = {1, 1, 1, 1};  // No vertical subsampling
    std::vector<uint8_t> Ssiz = {7, 7, 7, 7};   // 8-bit components (7+1=8)
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 4, 7, 1, 16, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_rgba_image(640, 480, NVIMGCODEC_SAMPLING_444), info);
}

// Test for grayscale image with enumCS=17
TEST_F(JPEG2KParserPluginTest, Grayscale_EnumCS17) {
    // Create synthetic grayscale image with enumCS=17
    std::vector<uint8_t> XRSiz = {1};  // No subsampling
    std::vector<uint8_t> YRSiz = {1};
    std::vector<uint8_t> Ssiz = {7};   // 8-bit (7+1=8)
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 1, 7, 1, 17, XRSiz, YRSiz, Ssiz);  // enumCS=17 for grayscale
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_grayscale_image(640, 480), info);
}

// Test for sYCC color space with 4:2:0 subsampling
TEST_F(JPEG2KParserPluginTest, sYCC_420_Subsampling) {
    // Create synthetic sYCC image with 4:2:0 subsampling
    // Per spec (enumCS=18), sYCC uses YCbCr color space
    std::vector<uint8_t> XRSiz = {1, 2, 2};  // Y full, CbCr subsampled horizontally  
    std::vector<uint8_t> YRSiz = {1, 2, 2};  // Y full, CbCr subsampled vertically
    std::vector<uint8_t> Ssiz = {7, 7, 7};   // 8-bit components
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 7, 1, 18, XRSiz, YRSiz, Ssiz);  // enumCS=18 for sYCC
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_yuv_image(640, 480, NVIMGCODEC_SAMPLING_420), info);
}

// Test for sYCC color space with 4:2:2 subsampling
TEST_F(JPEG2KParserPluginTest, sYCC_422_Subsampling) {
    // Create synthetic sYCC image with 4:2:2 subsampling
    std::vector<uint8_t> XRSiz = {1, 2, 2};  // Y full, UV subsampled horizontally
    std::vector<uint8_t> YRSiz = {1, 1, 1};  // No vertical subsampling
    std::vector<uint8_t> Ssiz = {7, 7, 7};   // 8-bit components
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 7, 1, 18, XRSiz, YRSiz, Ssiz);  // enumCS=18 for sYCC
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_yuv_image(640, 480, NVIMGCODEC_SAMPLING_422), info);
}

// Test for sYCC color space with 4:4:4 (no subsampling)
TEST_F(JPEG2KParserPluginTest, sYCC_444_NoSubsampling) {
    // Create synthetic sYCC image with 4:4:4 (no subsampling)
    std::vector<uint8_t> XRSiz = {1, 1, 1};  // No subsampling
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};   // 8-bit components
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 7, 1, 18, XRSiz, YRSiz, Ssiz);  // enumCS=18 for sYCC
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    expect_eq(expected_yuv_image(640, 480, NVIMGCODEC_SAMPLING_444), info);
}

// Test for RGBA with 420 subsampling pattern (based on first 3 components)
TEST_F(JPEG2KParserPluginTest, RGBA_420_SubsamplingDetection) {
    // Create RGBA image where first 3 components form 420 pattern
    // Per JPEG2000 spec, XRSiz/YRSiz are just sampling factors; no mandatory alpha pattern
    // Library classifies based on first 3 components (color/luma+chroma)
    std::vector<uint8_t> XRSiz = {1, 2, 2, 1};  // RGB: 420 pattern, Alpha: full res (spec-compliant)
    std::vector<uint8_t> YRSiz = {1, 2, 2, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7, 7};
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 4, 7, 1, 16, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should detect 420 based on first 3 components
    EXPECT_EQ(NVIMGCODEC_SAMPLING_420, info.chroma_subsampling);
    EXPECT_EQ(4, info.num_planes);
    // Verify plane dimensions for all planes
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(480, info.plane_info[0].height);
    EXPECT_EQ(320, info.plane_info[1].width);  // Subsampled
    EXPECT_EQ(240, info.plane_info[1].height); // Subsampled
    EXPECT_EQ(320, info.plane_info[2].width);  // Subsampled
    EXPECT_EQ(240, info.plane_info[2].height); // Subsampled
    EXPECT_EQ(640, info.plane_info[3].width);  // Alpha full res
    EXPECT_EQ(480, info.plane_info[3].height);
}

// Test for RGBA with 422 subsampling pattern (spec-compliant)
TEST_F(JPEG2KParserPluginTest, RGBA_422_SubsamplingDetection) {
    // RGBA with 422 pattern on color channels, alpha at different resolution
    // This is valid per JPEG2000 spec - alpha can have independent sampling
    std::vector<uint8_t> XRSiz = {1, 2, 2, 1};  // RGB: 422 pattern, Alpha: full res
    std::vector<uint8_t> YRSiz = {1, 1, 1, 2};  // Alpha vertically subsampled (allowed by spec)
    std::vector<uint8_t> Ssiz = {7, 7, 7, 7};
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 4, 7, 1, 16, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should detect 422 based on first 3 components (library policy)
    EXPECT_EQ(NVIMGCODEC_SAMPLING_422, info.chroma_subsampling);
    EXPECT_EQ(4, info.num_planes);
}

// Test for mixed subsampling pattern not matching common standards
TEST_F(JPEG2KParserPluginTest, MixedSubsamplingPattern_Unsupported) {
    // Create a pattern that isn't 420, 422, or 444
    // Note: JPEG2000 spec allows arbitrary XRSiz/YRSiz, but library only recognizes common patterns
    std::vector<uint8_t> XRSiz = {1, 3, 2};  // Non-standard pattern (spec-compliant but uncommon)
    std::vector<uint8_t> YRSiz = {1, 1, 2};  // Mixed vertical subsampling
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Library marks as unsupported (policy: only recognize 420/422/444/none)
    EXPECT_EQ(NVIMGCODEC_SAMPLING_UNSUPPORTED, info.chroma_subsampling);
}

// Test parsing raw J2K codestream (without JP2 wrapper)
TEST_F(JPEG2KParserPluginTest, RawJ2KCodestream_RGBA_444) {
    // Test that the parser can handle bare J2K codestreams without JP2 container
    // This ensures XRSiz/YRSiz subsampling logic works on raw codestream path
    std::vector<uint8_t> XRSiz = {1, 1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7, 7};
    
    std::vector<uint8_t> buffer;
    synthetic_jp2::create_raw_j2k_codestream(buffer, 640, 480, 4, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Without JP2 header, color_spec will be UNKNOWN
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_UNKNOWN, info.color_spec);
    EXPECT_EQ(4, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_444, info.chroma_subsampling);
    // Verify all planes have full resolution (444 subsampling)
    for (int p = 0; p < 4; p++) {
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(480, info.plane_info[p].height);
    }
}

// Test parsing raw J2K codestream with 420 subsampling
TEST_F(JPEG2KParserPluginTest, RawJ2KCodestream_YUV_420) {
    // Test 420 subsampling detection on raw codestream
    std::vector<uint8_t> XRSiz = {1, 2, 2};
    std::vector<uint8_t> YRSiz = {1, 2, 2};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    std::vector<uint8_t> buffer;
    synthetic_jp2::create_raw_j2k_codestream(buffer, 640, 480, 3, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_UNKNOWN, info.color_spec);  // No JP2 header
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_420, info.chroma_subsampling);
    // Check subsampled plane dimensions
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(480, info.plane_info[0].height);
    EXPECT_EQ(320, info.plane_info[1].width);  // Subsampled
    EXPECT_EQ(240, info.plane_info[1].height); // Subsampled
}

// Test for unknown/unsupported color space enumCS
TEST_F(JPEG2KParserPluginTest, UnknownColorSpace) {
    // Test that unknown enumCS values are handled gracefully (marked as unsupported)
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 7, 1, 999, XRSiz, YRSiz, Ssiz);  // enumCS=999 (unknown)
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should parse successfully but mark color_spec as unsupported
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_UNSUPPORTED, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_NONE, info.chroma_subsampling);
    // Verify geometry and plane layout are still correct
    for (int p = 0; p < 3; p++) {
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(480, info.plane_info[p].height);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
        EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[p].sample_type);
        EXPECT_EQ(8, info.plane_info[p].precision);
    }
}

// Test for ICC profile (method=2 in colr box)
// Tests fix: "do not read invalid enumCS field for colr method 2"
TEST_F(JPEG2KParserPluginTest, ICCProfile_Method2) {
    // Test that ICC profile color specification is marked as unsupported
    // and that the parser does NOT try to read an enumCS field (which doesn't exist for method=2)
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 7, 2, 0, XRSiz, YRSiz, Ssiz);  // method=2 for ICC profile
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should parse successfully but mark color_spec as unsupported
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_UNSUPPORTED, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
    // Verify geometry is still correct even with unsupported color
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(480, info.plane_info[0].height);
    EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    EXPECT_EQ(8, info.plane_info[0].precision);
}

// Test for box with LBox=0 (extends to end of file)
TEST_F(JPEG2KParserPluginTest, BoxSizeZero_ExtendsToEOF) {
    // Test that a jp2c box with LBox=0 is correctly handled as "extends to end of file"
    // This is valid per ISO/IEC 15444-1:2019, Annex I.4
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    auto buffer = synthetic_jp2::create_jp2_with_box_size_zero(
        640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should parse successfully
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(480, info.plane_info[0].height);
}

// Test for multiple colr boxes (first should be used, subsequent ignored)
TEST_F(JPEG2KParserPluginTest, MultipleColrBoxes_FirstUsed) {
    // Per ISO/IEC 15444-1, if multiple colr boxes exist, the first should be used
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    // Create JP2 with first colr=sRGB (16), second colr=grayscale (17)
    auto buffer = synthetic_jp2::create_jp2_with_multiple_colr(
        640, 480, 3, 7, 16, 17, XRSiz, YRSiz, Ssiz);
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should use first colr box (sRGB), not second (grayscale)
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLEFORMAT_P_RGB, info.sample_format);
    EXPECT_EQ(3, info.num_planes);
}

// Test for Part-2 extensions with RSiz=0x0003
TEST_F(JPEG2KParserPluginTest, Part2Extensions_RSiz0x0003) {
    // Test that files with RSiz > 0x0002 (indicating Part-2 or vendor extensions)
    // are handled with appropriate warnings but still parsed successfully
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    auto buffer = synthetic_jp2::create_jp2_with_part2_rsiz(
        640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz, 0x0003);  // RSiz=3 (Part-2)
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should parse successfully (parser logs warning but continues)
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(3, info.num_planes);
    // Verify plane dimensions are still parsed correctly with Part-2 RSiz
    for (int p = 0; p < 3; p++) {
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(480, info.plane_info[p].height);
    }
}

// Test for Part-2 extensions with high RSiz value
TEST_F(JPEG2KParserPluginTest, Part2Extensions_RSizHigh) {
    // Test with a higher RSiz value (vendor-specific extensions)
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    
    auto buffer = synthetic_jp2::create_jp2_with_part2_rsiz(
        640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz, 0x8000);  // High RSiz (vendor)
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Should still parse basic structure even with unknown RSiz
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_NONE, info.chroma_subsampling);
    // Verify plane dimensions are still parsed correctly with unknown RSiz
    for (int p = 0; p < 3; p++) {
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(480, info.plane_info[p].height);
    }
}

// Test for invalid image header box size (negative test)
TEST_F(JPEG2KParserPluginTest, InvalidImageHeaderBoxSize) {
    // Create a JP2 file with incorrectly sized ihdr box
    // Manually constructed to precisely control the invalid size (20 instead of 22)
    std::vector<uint8_t> data;
    
    // JP2 signature
    synthetic_jp2::create_jp2_signature(data);
    
    // File type box
    synthetic_jp2::create_ftyp_box(data);
    
    synthetic_jp2::write_box_header(data, 28, "jp2h");

    // JP2 header with invalid ihdr size (wrong size = 20 instead of 22)
    synthetic_jp2::write_box_header(data, 20, "ihdr");

    synthetic_jp2::write_be32(data, 480);  // height
    synthetic_jp2::write_be32(data, 640);  // width
    synthetic_jp2::write_be16(data, 3);    // num_components
    data.push_back(7);  // bpc
    
    LoadImageFromHostMemory(instance_, stream_handle_, data.data(), data.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    
    // Should reject due to invalid ihdr box size
    ASSERT_NE(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// Test for components with different bit depths (BPCC box would be needed in real file)
TEST_F(JPEG2KParserPluginTest, DifferentBitDepthsPerComponent) {
    // Test file with components at different precisions (relies on Ssiz from codestream)
    // In JP2, ihdr would use bits_per_component=0xFF to signal "see codestream"
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 11, 15};  // 8-bit, 12-bit, 16-bit components
    
    auto buffer = synthetic_jp2::create_complete_jp2(
        640, 480, 3, 0xFF, 1, 16, XRSiz, YRSiz, Ssiz);  // bpc=0xFF means different depths
    
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    
    // Verify each plane has correct precision from Ssiz
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(NVIMGCODEC_SAMPLING_NONE, info.chroma_subsampling);  // Guard against subsampling logic regression
    
    EXPECT_EQ(8, info.plane_info[0].precision);   // Ssiz[0]=7 -> 8-bit
    EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, info.plane_info[0].sample_type);
    EXPECT_EQ(12, info.plane_info[1].precision);  // Ssiz[1]=11 -> 12-bit
    EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, info.plane_info[1].sample_type);
    EXPECT_EQ(16, info.plane_info[2].precision);  // Ssiz[2]=15 -> 16-bit
    EXPECT_EQ(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, info.plane_info[2].sample_type);
    
    // Verify geometry is still correct with mixed bit depths
    for (int p = 0; p < 3; p++) {
        EXPECT_EQ(640, info.plane_info[p].width);
        EXPECT_EQ(480, info.plane_info[p].height);
        EXPECT_EQ(1, info.plane_info[p].num_channels);
    }
}


}  // namespace test
}  // namespace nvimgcodec
