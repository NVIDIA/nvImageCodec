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
#include <optional>
#include <string>
#include <string_view>
#include <filesystem>
#include <fstream>
#include <utility>
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

// Write a 64-bit big-endian integer
inline void write_be64(std::vector<uint8_t>& data, uint64_t value) {
    for (int shift = 56; shift >= 0; shift -= 8)
        data.push_back((value >> shift) & 0xFF);
}

// Write a box header (size + type)
inline void write_box_header(std::vector<uint8_t>& data, uint32_t size, std::string_view type) {
    assert(type.size() == 4 && "Box type must be exactly 4 bytes");
    write_be32(data, size);
    data.insert(data.end(), type.begin(), type.end());
}

// Write an extended-length box header: LBox=1 signals that the real 64-bit length lives in the
// XLBox field following TBox (ISO/IEC 15444-1:2019, Annex I.4). `size` is the total box length,
// its 16-byte header included.
inline void write_extended_box_header(std::vector<uint8_t>& data, uint64_t size, std::string_view type) {
    assert(type.size() == 4 && "Box type must be exactly 4 bytes");
    write_be32(data, 1);
    data.insert(data.end(), type.begin(), type.end());
    write_be64(data, size);
}

constexpr uint64_t extended_box_header_size = 16;

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

// Create image header box. The content is a fixed 14 bytes; `extended` wraps it in an
// LBox=1 + XLBox header instead of the ordinary eight-byte one.
inline void create_ihdr_box(std::vector<uint8_t>& data, uint32_t height, uint32_t width, 
                            uint16_t num_components, uint8_t bpc, bool extended = false) {
    constexpr uint32_t ihdr_content_size = 14;
    if (extended)
        write_extended_box_header(data, extended_box_header_size + ihdr_content_size, "ihdr");
    else
        write_box_header(data, 8 + ihdr_content_size, "ihdr");
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

// Create raw J2K codestream with SIZ marker (without JP2 wrapper).
// When tile_width/tile_height are omitted, XTSiz/YTSiz default to the image size (single tile).
// Pass explicit values (including 0) to exercise custom or malformed SIZ tile geometry.
// x_offset/y_offset set XOSiz/YOSiz and tile_x_offset/tile_y_offset set XTOSiz/YTOSiz so tests can
// exercise non-zero reference-grid origins. lsiz_override forces a specific Lsiz to exercise the
// A-1 (Lsiz == 38 + 3 * CSiz) validation independently of the component count actually written.
inline void create_raw_j2k_codestream(std::vector<uint8_t>& data, uint32_t width, uint32_t height,
                                      uint16_t num_components,
                                      const std::vector<uint8_t>& XRSiz,
                                      const std::vector<uint8_t>& YRSiz,
                                      const std::vector<uint8_t>& Ssiz,
                                      uint16_t RSiz = 0,
                                      std::optional<uint32_t> tile_width = std::nullopt,
                                      std::optional<uint32_t> tile_height = std::nullopt,
                                      uint32_t x_offset = 0, uint32_t y_offset = 0,
                                      uint32_t tile_x_offset = 0, uint32_t tile_y_offset = 0,
                                      std::optional<uint16_t> lsiz_override = std::nullopt) {
    // SOC marker (0xFF4F)
    write_be16(data, 0xFF4F);
    
    // SIZ marker (0xFF51)
    write_be16(data, 0xFF51);
    
    // Calculate Lsiz (marker segment length)
    uint16_t Lsiz = lsiz_override.value_or(static_cast<uint16_t>(38 + 3 * num_components));
    write_be16(data, Lsiz);
    
    // RSiz (capabilities)
    write_be16(data, RSiz); // Defaults to 0 (No profile)
    
    // XSiz, YSiz (reference grid size)
    write_be32(data, width);
    write_be32(data, height);
    
    // XOSiz, YOSiz (image offset)
    write_be32(data, x_offset);
    write_be32(data, y_offset);
    
    // XTSiz, YTSiz (tile size)
    write_be32(data, tile_width.value_or(width));
    write_be32(data, tile_height.value_or(height));
    
    // XTOSiz, YTOSiz (tile offset)
    write_be32(data, tile_x_offset);
    write_be32(data, tile_y_offset);
    
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

// Create a JP2 file with a top-level extended-length box (LBox=1 + XLBox) sitting between the
// jp2h header and the codestream, the layout geospatial JP2 files use to carry bulky XML
// metadata in an `asoc` box. xlbox_override forces a specific XLBox value so invalid lengths
// can be exercised independently of the payload actually written.
inline std::vector<uint8_t> create_jp2_with_extended_box(uint32_t width, uint32_t height,
                                                         uint16_t num_components, uint8_t bpc,
                                                         uint8_t colr_method, uint32_t enumCS,
                                                         const std::vector<uint8_t>& XRSiz,
                                                         const std::vector<uint8_t>& YRSiz,
                                                         const std::vector<uint8_t>& Ssiz,
                                                         size_t asoc_payload_size = 32,
                                                         std::optional<uint64_t> xlbox_override = std::nullopt) {
    std::vector<uint8_t> data;
    create_jp2_signature(data);
    create_ftyp_box(data);
    create_jp2h_box(data, height, width, num_components, bpc, colr_method, enumCS);

    // Extended-length box carrying opaque metadata the parser is expected to skip
    const uint64_t asoc_size = extended_box_header_size + asoc_payload_size;
    write_extended_box_header(data, xlbox_override.value_or(asoc_size), "asoc");
    data.insert(data.end(), asoc_payload_size, 0xAB);

    std::vector<uint8_t> codestream;
    create_raw_j2k_codestream(codestream, width, height, num_components, XRSiz, YRSiz, Ssiz);
    write_box_header(data, 8 + codestream.size(), "jp2c");
    data.insert(data.end(), codestream.begin(), codestream.end());

    return data;
}

// Create a JP2 file whose jp2h superbox uses an extended-length header and holds
// extended-length sub-boxes, so the superbox content size, the sub-box skip and the fixed-size
// ihdr check all have to account for the larger header.
inline std::vector<uint8_t> create_jp2_with_extended_jp2h(uint32_t width, uint32_t height,
                                                          uint16_t num_components, uint8_t bpc,
                                                          uint8_t colr_method, uint32_t enumCS,
                                                          const std::vector<uint8_t>& XRSiz,
                                                          const std::vector<uint8_t>& YRSiz,
                                                          const std::vector<uint8_t>& Ssiz) {
    std::vector<uint8_t> data;
    create_jp2_signature(data);
    create_ftyp_box(data);

    std::vector<uint8_t> sub_boxes;
    create_ihdr_box(sub_boxes, height, width, num_components, bpc, /*extended=*/true);
    constexpr size_t xml_payload_size = 8;
    write_extended_box_header(sub_boxes, extended_box_header_size + xml_payload_size, "xml ");
    sub_boxes.insert(sub_boxes.end(), xml_payload_size, 0xCD);
    create_colr_box(sub_boxes, colr_method, enumCS);

    write_extended_box_header(data, extended_box_header_size + sub_boxes.size(), "jp2h");
    data.insert(data.end(), sub_boxes.begin(), sub_boxes.end());

    std::vector<uint8_t> codestream;
    create_raw_j2k_codestream(codestream, width, height, num_components, XRSiz, YRSiz, Ssiz);
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

TEST_F(JPEG2KParserPluginTest, XRSiz_YRSiz_is_0)
{
    if (!std::filesystem::is_directory(resources_dir + "/jpeg2k/corrupted"))
        GTEST_SKIP() << "jpeg2k/corrupted samples are not present in " << resources_dir;
    auto buffer = read_file(resources_dir + "/jpeg2k/corrupted/XRSiz_YRSiz_is_0.jp2");
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// A JP2 file whose ihdr image-header dimensions disagree with the codestream SIZ marker must
// be rejected. Otherwise the malformed file can flow through to the OpenCV/openjpeg fallback
// decoder and crash the process (SIGSEGV in opj_jp2_read_header).
TEST_F(JPEG2KParserPluginTest, IhdrHeightMismatchSiz)
{
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    // 'i','h','d','r' followed by height=475 (0x000001db) and width=640 (0x00000280) -
    // the original ihdr box of cat-1046544_640.jp2 (RGB, 640x475).
    std::vector<uint8_t> original_ihdr = {
        0x69, 0x68, 0x64, 0x72,
        0x00, 0x00, 0x01, 0xdb,
        0x00, 0x00, 0x02, 0x80};
    // Same box but with height patched to 4096 (0x00001000); the codestream SIZ marker
    // further down in the file still reports 640x475 - this reproduces the malformed case.
    std::vector<uint8_t> mutated_ihdr = {
        0x69, 0x68, 0x64, 0x72,
        0x00, 0x00, 0x10, 0x00,
        0x00, 0x00, 0x02, 0x80};
    buffer = replace(buffer, original_ihdr, mutated_ihdr);
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

TEST_F(JPEG2KParserPluginTest, IhdrWidthMismatchSiz)
{
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    std::vector<uint8_t> original_ihdr = {
        0x69, 0x68, 0x64, 0x72,
        0x00, 0x00, 0x01, 0xdb,
        0x00, 0x00, 0x02, 0x80};
    // Width patched to 4096 (0x00001000); height is left at 475.
    std::vector<uint8_t> mutated_ihdr = {
        0x69, 0x68, 0x64, 0x72,
        0x00, 0x00, 0x01, 0xdb,
        0x00, 0x00, 0x10, 0x00};
    buffer = replace(buffer, original_ihdr, mutated_ihdr);
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// A bare J2K codestream (no JP2 ihdr) whose SIZ marker reports XOSiz > XSiz must be rejected.
// Without an explicit underflow guard, XSiz - XOSiz wraps in uint32_t and propagates into
// div_up() that sets plane geometry, producing nonsense dimensions for downstream allocators.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_InvalidSizOffsets)
{
    auto buffer = read_file(resources_dir + "/jpeg2k/cat-1046544_640.jp2");
    // Strip the JP2 wrapper down to the SOC marker, matching Uint8_CodeStreamOnly.
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
    // Now patch the SIZ marker tail XSiz/YSiz/XOSiz/YOSiz so XOSiz > XSiz.
    // Original SIZ layout for cat-1046544_640: XSiz=640 (0x00000280), YSiz=475 (0x000001db),
    // XOSiz=0, YOSiz=0. Replace XOSiz with 0x000003e8 (1000) which exceeds XSiz.
    std::vector<uint8_t> original_siz_tail = {
        0x00, 0x00, 0x02, 0x80,
        0x00, 0x00, 0x01, 0xdb,
        0x00, 0x00, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00};
    std::vector<uint8_t> mutated_siz_tail = {
        0x00, 0x00, 0x02, 0x80,
        0x00, 0x00, 0x01, 0xdb,
        0x00, 0x00, 0x03, 0xe8,
        0x00, 0x00, 0x00, 0x00};
    buffer = replace(buffer, original_siz_tail, mutated_siz_tail);
    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// A bare J2K codestream whose SIZ marker reports a zero tile dimension (with XRSiz=YRSiz=1, so
// the existing XRSiz/YRSiz zero-divisor guard does not fire) must be rejected. Without this check,
// attaching TileGeometryInfo makes tile_height/tile_width zero and div_up(plane_dim, 0) raises
// SIGFPE when computing num_tiles_x/num_tiles_y. The guard uses ||, so cover both the
// symmetric (XTSiz=YTSiz=0) and asymmetric (only one dimension zero) cases.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_ZeroTileSize)
{
    const std::vector<uint8_t> XRSiz = {1};
    const std::vector<uint8_t> YRSiz = {1};
    const std::vector<uint8_t> Ssiz = {7};  // 8-bit
    const std::pair<uint32_t, uint32_t> zero_tile_cases[] = {{0, 0}, {0, 1}, {1, 0}};
    for (const auto& [tile_width, tile_height] : zero_tile_cases) {
        SCOPED_TRACE("XTSiz=" + std::to_string(tile_width) + " YTSiz=" + std::to_string(tile_height));
        std::vector<uint8_t> buffer;
        synthetic_jp2::create_raw_j2k_codestream(
            buffer, 64, 64, 1, XRSiz, YRSiz, Ssiz, /*RSiz=*/0, tile_width, tile_height);

        LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

        // Reproduce the public CodeStream path: TileGeometryInfo is always attached, which is
        // what previously crashed on this input.
        nvimgcodecTileGeometryInfo_t tile_info{
            NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_info};
        EXPECT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        stream_handle_ = nullptr;
    }
}

// A tile dimension near UINT32_MAX combined with valid sub-sampling used to overflow the
// numerator in div_up((XTSiz + XRSiz[0] - 1), XRSiz[0]), silently yielding a zero derived tile
// size and a subsequent SIGFPE in div_up(plane_dim, 0). With overflow-safe ceil division the
// derived tile size stays non-zero and the codestream parses without crashing.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_MaxTileSizeNoOverflow)
{
    const std::vector<uint8_t> XRSiz = {2};
    const std::vector<uint8_t> YRSiz = {2};
    const std::vector<uint8_t> Ssiz = {7};  // 8-bit
    std::vector<uint8_t> buffer;
    synthetic_jp2::create_raw_j2k_codestream(
        buffer, 64, 64, 1, XRSiz, YRSiz, Ssiz, /*RSiz=*/0,
        /*tile_width=*/0xFFFFFFFFu, /*tile_height=*/0xFFFFFFFFu);

    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

    nvimgcodecTileGeometryInfo_t tile_info{
        NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_info};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    // The (over-sized) tile covers the whole sub-sampled plane: derived tile size must be
    // non-zero and resolve to a single tile.
    EXPECT_GT(tile_info.tile_width, 0u);
    EXPECT_GT(tile_info.tile_height, 0u);
    EXPECT_EQ(1u, tile_info.num_tiles_x);
    EXPECT_EQ(1u, tile_info.num_tiles_y);
}

// A SIZ marker declaring CSiz=0 (while Lsiz still claims a component is present) skips the
// per-component XRSiz/YRSiz validation loop entirely. The public tile-geometry path then uses
// XRSiz[0]/YRSiz[0] as divisors - zero on a fresh parser (SIGFPE) or stale on a reused one. A
// zero component count must be rejected before the loop.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_ZeroComponentCount)
{
    const std::vector<uint8_t> XRSiz = {1};
    const std::vector<uint8_t> YRSiz = {1};
    const std::vector<uint8_t> Ssiz = {7};  // 8-bit
    std::vector<uint8_t> buffer;
    synthetic_jp2::create_raw_j2k_codestream(buffer, 64, 64, 1, XRSiz, YRSiz, Ssiz);
    // Patch the CSiz field (2-byte big-endian, immediately after YTOSiz) from 1 to 0 without
    // touching Lsiz, reproducing a malformed SIZ whose declared component count is zero.
    // Layout: SOC(2) SIZ(2) Lsiz(2) RSiz(2) XSiz(4) YSiz(4) XOSiz(4) YOSiz(4) XTSiz(4) YTSiz(4) XTOSiz(4) YTOSiz(4) CSiz(2)
    constexpr size_t csiz_offset = 2 + 2 + 2 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 4 + 4;  // = 40
    ASSERT_GE(buffer.size(), csiz_offset + 2);
    buffer[csiz_offset] = 0x00;
    buffer[csiz_offset + 1] = 0x00;

    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

    nvimgcodecTileGeometryInfo_t tile_info{
        NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_info};
    ASSERT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
}

// ISO/IEC 15444-1 equation B-5 defines the tile count on the reference grid as
// ceil((XSiz - XTOSiz) / XTSiz), independent of component sub-sampling. Deriving it from the
// cropped component-0 plane (ceil((XSiz - XOSiz) / XTSiz)) ignores the image origin and
// undercounts tiles when XOSiz != 0. Both a non-zero-origin case and a zero-origin control must
// report the same count.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_TileCountFromReferenceGrid)
{
    const std::vector<uint8_t> XRSiz = {1};
    const std::vector<uint8_t> YRSiz = {1};
    const std::vector<uint8_t> Ssiz = {7};  // 8-bit
    // XSiz=YSiz=65, XTSiz=YTSiz=16, XTOSiz=YTOSiz=0 => ceil(65/16) = 5 tiles each way.
    // With the buggy cropped-plane formula the non-zero-origin case would report ceil(60/16) = 4.
    const std::pair<uint32_t, uint32_t> origin_cases[] = {{5, 5}, {0, 0}};
    for (const auto& [x_origin, y_origin] : origin_cases) {
        SCOPED_TRACE("XOSiz=" + std::to_string(x_origin) + " YOSiz=" + std::to_string(y_origin));
        std::vector<uint8_t> buffer;
        synthetic_jp2::create_raw_j2k_codestream(
            buffer, 65, 65, 1, XRSiz, YRSiz, Ssiz, /*RSiz=*/0, /*tile_width=*/16, /*tile_height=*/16,
            /*x_offset=*/x_origin, /*y_offset=*/y_origin, /*tile_x_offset=*/0, /*tile_y_offset=*/0);

        LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

        nvimgcodecTileGeometryInfo_t tile_info{
            NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_info};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
        EXPECT_EQ(5u, tile_info.num_tiles_x);
        EXPECT_EQ(5u, tile_info.num_tiles_y);

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        stream_handle_ = nullptr;
    }
}

// ISO/IEC 15444-1 clause B.3 equations B-3/B-4 constrain the tile grid origin:
//   B-3: XTOSiz <= XOSiz (and Y equivalent) - the tile grid may not start after the image origin;
//   B-4: XTSiz + XTOSiz > XOSiz (i.e. XTSiz > XOSiz - XTOSiz) - the first tile must intersect the
//        image. Violations either wrap the unsigned tile-offset subtraction or produce an empty
//        first tile, so they must be rejected. Exact boundaries are exercised here.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_InvalidTileGridOrigin)
{
    const std::vector<uint8_t> XRSiz = {1};
    const std::vector<uint8_t> YRSiz = {1};
    const std::vector<uint8_t> Ssiz = {7};  // 8-bit
    struct Case {
        const char* name;
        uint32_t x_offset, tile_x_offset, tile_width;
        nvimgcodecStatus_t expected;
    };
    // Y dimensions are held valid (YOSiz=YTOSiz=0, YTSiz=16) so only the X constraint varies.
    const Case cases[] = {
        // B-3 boundary: tile origin exactly at the image origin is valid; one past it is rejected.
        {"XTOSiz==XOSiz (valid)", 10, 10, 16, NVIMGCODEC_STATUS_SUCCESS},
        {"XTOSiz==XOSiz+1 (reject)", 10, 11, 16, NVIMGCODEC_STATUS_BAD_CODESTREAM},
        // B-4 boundary: tile size equal to the origin gap leaves an empty first tile (reject);
        // gap+1 makes the first tile intersect the image (valid).
        {"XTSiz==XOSiz-XTOSiz (reject)", 10, 0, 10, NVIMGCODEC_STATUS_BAD_CODESTREAM},
        {"XTSiz==XOSiz-XTOSiz+1 (valid)", 10, 0, 11, NVIMGCODEC_STATUS_SUCCESS},
    };
    for (const auto& c : cases) {
        SCOPED_TRACE(c.name);
        std::vector<uint8_t> buffer;
        synthetic_jp2::create_raw_j2k_codestream(
            buffer, 64, 64, 1, XRSiz, YRSiz, Ssiz, /*RSiz=*/0, /*tile_width=*/c.tile_width, /*tile_height=*/16,
            /*x_offset=*/c.x_offset, /*y_offset=*/0, /*tile_x_offset=*/c.tile_x_offset, /*tile_y_offset=*/0);

        LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

        nvimgcodecTileGeometryInfo_t tile_info{
            NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO, sizeof(nvimgcodecTileGeometryInfo_t), nullptr};
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &tile_info};
        EXPECT_EQ(c.expected, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        stream_handle_ = nullptr;
    }
}

// ISO/IEC 15444-1 equation A-1 fixes the SIZ marker segment length at exactly Lsiz = 38 + 3 * CSiz
// (one Ssiz/XRSiz/YRSiz triplet per component). A too-short Lsiz would let the parser read
// component bytes past the declared segment; a too-long Lsiz is a malformed declaration. Only the
// exact value must be accepted. CSiz=2 => expected Lsiz=44; check 43, 44, 45.
TEST_F(JPEG2KParserPluginTest, CodeStreamOnly_SizMarkerSizeMismatch)
{
    const std::vector<uint8_t> XRSiz = {1, 1};
    const std::vector<uint8_t> YRSiz = {1, 1};
    const std::vector<uint8_t> Ssiz = {7, 7};  // 8-bit
    struct Case {
        uint16_t lsiz;
        nvimgcodecStatus_t expected;
    };
    const Case cases[] = {
        {43, NVIMGCODEC_STATUS_BAD_CODESTREAM},  // expected - 1
        {44, NVIMGCODEC_STATUS_SUCCESS},         // exact (38 + 3 * 2)
        {45, NVIMGCODEC_STATUS_BAD_CODESTREAM},  // expected + 1
    };
    for (const auto& c : cases) {
        SCOPED_TRACE("Lsiz=" + std::to_string(c.lsiz));
        std::vector<uint8_t> buffer;
        synthetic_jp2::create_raw_j2k_codestream(
            buffer, 64, 64, /*num_components=*/2, XRSiz, YRSiz, Ssiz, /*RSiz=*/0,
            /*tile_width=*/std::nullopt, /*tile_height=*/std::nullopt,
            /*x_offset=*/0, /*y_offset=*/0, /*tile_x_offset=*/0, /*tile_y_offset=*/0,
            /*lsiz_override=*/c.lsiz);

        LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());

        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        EXPECT_EQ(c.expected, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        stream_handle_ = nullptr;
    }
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

// ISO/IEC 15444-1:2019 Annex I.4 lets a box set LBox=1 and carry its real length in the 64-bit
// XLBox field that follows. Such a box must be skipped like any other, not rejected because the
// sentinel 1 looks smaller than a box header. Geospatial JP2 files hit this with a top-level
// extended-length asoc box holding XML metadata.
TEST_F(JPEG2KParserPluginTest, ExtendedLengthBox_Skipped) {
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};

    auto buffer = synthetic_jp2::create_jp2_with_extended_box(
        640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz);

    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(480, info.plane_info[0].height);
}

// An extended-length header is also legal on a superbox and on its sub-boxes, where the content
// size and the skip length must be measured against the 16-byte header rather than the usual 8.
// The ihdr sub-box is extended here too: its content is a fixed 14 bytes, so a check against the
// total box length would reject the valid 30-byte extended form.
TEST_F(JPEG2KParserPluginTest, ExtendedLengthBox_SuperboxAndSubBox) {
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};

    auto buffer = synthetic_jp2::create_jp2_with_extended_jp2h(
        640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz);

    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

    EXPECT_EQ(NVIMGCODEC_COLORSPEC_SRGB, info.color_spec);
    EXPECT_EQ(3, info.num_planes);
    EXPECT_EQ(640, info.plane_info[0].width);
    EXPECT_EQ(480, info.plane_info[0].height);
}

// XLBox holds the total length of an extended box, so anything below the 16-byte extended header
// is malformed - including 0, which carries the "extends to the end of the containing structure"
// meaning only when it appears in LBox. A length past the end of the stream must be rejected too:
// skipping it would wrap to a negative offset and seek backwards, re-reading the same boxes
// forever.
TEST_F(JPEG2KParserPluginTest, ExtendedLengthBox_InvalidLength) {
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    const uint64_t invalid_xlbox_values[] = {0, 8, 15, 0xFFFFFFFFFFFFFFF0ull};
    for (uint64_t xlbox : invalid_xlbox_values) {
        SCOPED_TRACE("XLBox=" + std::to_string(xlbox));
        auto buffer = synthetic_jp2::create_jp2_with_extended_box(
            640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz, /*asoc_payload_size=*/32, xlbox);

        LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        EXPECT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));

        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        stream_handle_ = nullptr;
    }
}

// A box whose declared length runs past the end of the stream must be rejected wherever its
// content would be skipped - including the mandatory jP /ftyp boxes, which are consumed before
// the main box loop. Skipping past the end fails without moving the stream position, so an
// unchecked skip leaves parsing to resume inside the box and read its payload as further boxes.
// Whether that produces a plausible box chain depends purely on the payload bytes, so the
// oversized ftyp is tested with a payload crafted to look like a valid box.
TEST_F(JPEG2KParserPluginTest, BoxPastEndOfStreamIsRejected) {
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    const auto valid = synthetic_jp2::create_complete_jp2(640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz);

    // Everything from the jp2h box onwards, i.e. the part of a valid file that follows ftyp.
    constexpr size_t after_ftyp = 12 + 20;
    const std::vector<uint8_t> tail(valid.begin() + after_ftyp, valid.end());

    {
        SCOPED_TRACE("extended ftyp claiming more bytes than the file holds");
        std::vector<uint8_t> data(valid.begin(), valid.begin() + 12); // jP box
        synthetic_jp2::write_extended_box_header(data, 1ull << 40, "ftyp");
        // Payload shaped as a 12-byte 'free' box: if the oversized skip were not caught, parsing
        // would resume here, step over this box and carry on to the real jp2h/jp2c.
        synthetic_jp2::write_box_header(data, 12, "free");
        data.insert(data.end(), 4, 0x00);
        data.insert(data.end(), tail.begin(), tail.end());

        LoadImageFromHostMemory(instance_, stream_handle_, data.data(), data.size());
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        EXPECT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream_handle_));
        stream_handle_ = nullptr;
    }
    {
        SCOPED_TRACE("unknown top-level box claiming more bytes than the file holds");
        std::vector<uint8_t> data(valid.begin(), valid.begin() + after_ftyp);
        synthetic_jp2::write_extended_box_header(data, 1ull << 40, "uuid");
        data.insert(data.end(), tail.begin(), tail.end());

        LoadImageFromHostMemory(instance_, stream_handle_, data.data(), data.size());
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        EXPECT_EQ(NVIMGCODEC_STATUS_BAD_CODESTREAM, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
    }
}

// A codestream box is never skipped - parsing continues into it and ends there - so it is not
// required to fit in the stream. Truncated files, such as a partial download, must still report
// the geometry carried by the SIZ marker at the start of the codestream.
TEST_F(JPEG2KParserPluginTest, TruncatedCodestreamStillParses) {
    std::vector<uint8_t> XRSiz = {1, 1, 1};
    std::vector<uint8_t> YRSiz = {1, 1, 1};
    std::vector<uint8_t> Ssiz = {7, 7, 7};
    auto buffer = synthetic_jp2::create_complete_jp2(640, 480, 3, 7, 1, 16, XRSiz, YRSiz, Ssiz);
    // Make the codestream box claim far more bytes than the file holds, as a partially downloaded
    // file does. The SIZ marker at the start of the codestream is left intact.
    size_t jp2c_size_pos = 0;
    for (size_t i = 4; i + 4 <= buffer.size(); i++) {
        if (buffer[i] == 'j' && buffer[i + 1] == 'p' && buffer[i + 2] == '2' && buffer[i + 3] == 'c') {
            jp2c_size_pos = i - 4;
            break;
        }
    }
    ASSERT_NE(0u, jp2c_size_pos);
    const uint32_t inflated = static_cast<uint32_t>(buffer.size()) * 2;
    buffer[jp2c_size_pos + 0] = (inflated >> 24) & 0xFF;
    buffer[jp2c_size_pos + 1] = (inflated >> 16) & 0xFF;
    buffer[jp2c_size_pos + 2] = (inflated >> 8) & 0xFF;
    buffer[jp2c_size_pos + 3] = inflated & 0xFF;

    LoadImageFromHostMemory(instance_, stream_handle_, buffer.data(), buffer.size());
    nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream_handle_, &info));
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
