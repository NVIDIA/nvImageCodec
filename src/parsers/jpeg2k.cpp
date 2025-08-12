/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "parsers/jpeg2k.h"
#include <nvimgcodec.h>
#include <string.h>
#include <vector>

#include "imgproc/exception.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"

// Reference: https://www.itu.int/rec/T-REC-T.800-200208-S

namespace nvimgcodec {

namespace {

const std::array<uint8_t, 12> JP2_SIGNATURE = {0x00, 0x00, 0x00, 0x0c, 0x6a, 0x50, 0x20, 0x20, 0x0d, 0x0a, 0x87, 0x0a};
const std::array<uint8_t, 2> J2K_SIGNATURE = {0xff, 0x4f};

using block_type_t = std::array<uint8_t, 4>;
const block_type_t jp2_signature = {'j', 'P', ' ', ' '};    // JPEG2000 signature
const block_type_t jp2_file_type = {'f', 't', 'y', 'p'};    // File type
const block_type_t jp2_header = {'j', 'p', '2', 'h'};       // JPEG2000 header (super box)
const block_type_t jp2_image_header = {'i', 'h', 'd', 'r'}; // Image header
const block_type_t jp2_colour_spec = {'c', 'o', 'l', 'r'};  // Color specification
const block_type_t jp2_code_stream = {'j', 'p', '2', 'c'};  // Contiguous code stream
const block_type_t jp2_url = {'u', 'r', 'l', ' '};          // Data entry URL
const block_type_t jp2_palette = {'p', 'c', 'l', 'r'};      // Palette
const block_type_t jp2_cmap = {'c', 'm', 'a', 'p'};         // Component mapping
const block_type_t jp2_cdef = {'c', 'd', 'e', 'f'};         // Channel definition
const block_type_t jp2_dtbl = {'d', 't', 'b', 'l'};         // Data reference
const block_type_t jp2_bpcc = {'b', 'p', 'c', 'c'};         // Bits per component
const block_type_t jp2_jp2 = {'j', 'p', '2', ' '};          // File type fields

enum jpeg2k_marker_t : uint16_t
{
    SOC_marker = 0xFF4F,
    SIZ_marker = 0xFF51
};

const uint8_t DIFFERENT_BITDEPTH_PER_COMPONENT = 0xFF;

bool ReadBoxHeader(block_type_t& block_type, uint32_t& block_size, nvimgcodecIoStreamDesc_t* io_stream)
{
    block_size = ReadValueBE<uint32_t>(io_stream);
    block_type = ReadValue<block_type_t>(io_stream);
    return true;
}

void SkipBox(nvimgcodecIoStreamDesc_t* io_stream, block_type_t expected_block, const char* box_description)
{
    auto block_size = ReadValueBE<uint32_t>(io_stream);
    auto block_type = ReadValue<block_type_t>(io_stream);
    if (block_type != expected_block)
        throw std::runtime_error(std::string("Failed to read ") + std::string(box_description));
    io_stream->skip(io_stream->instance, block_size - sizeof(block_size) - sizeof(block_type));
}

template <typename T, typename V>
constexpr inline T DivUp(T x, V d)
{
    return (x + d - 1) / d;
}

nvimgcodecSampleDataType_t BitsPerComponentToType(uint8_t bits_per_component)
{
    auto sign_component = bits_per_component >> 7;
    bits_per_component = bits_per_component & 0x7f;
    bits_per_component += 1;
    auto sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    if (bits_per_component <= 16 && bits_per_component > 8) {
        sample_type = sign_component ? NVIMGCODEC_SAMPLE_DATA_TYPE_INT16 : NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    } else if (bits_per_component <= 8) {
        sample_type = sign_component ? NVIMGCODEC_SAMPLE_DATA_TYPE_INT8 : NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    }
    return sample_type;
}

nvimgcodecChromaSubsampling_t XRSizYRSizToSubsampling(uint8_t CSiz, const uint8_t* XRSiz, const uint8_t* YRSiz)
{
    if (CSiz == 3 || CSiz == 4) {
        if ((XRSiz[0] == 1) && (XRSiz[1] == 2) && (XRSiz[2] == 2) && (YRSiz[0] == 1) && (YRSiz[1] == 2) && (YRSiz[2] == 2)) {
            return NVIMGCODEC_SAMPLING_420;
        } else if ((XRSiz[0] == 1) && (XRSiz[1] == 2) && (XRSiz[2] == 2) && (YRSiz[0] == 1) && (YRSiz[1] == 1) && (YRSiz[2] == 1)) {
            return NVIMGCODEC_SAMPLING_422;
        } else if ((XRSiz[0] == 1) && (XRSiz[1] == 1) && (XRSiz[2] == 1) && (YRSiz[0] == 1) && (YRSiz[1] == 1) && (YRSiz[2] == 1)) {
            return NVIMGCODEC_SAMPLING_444;
        } else {
            return NVIMGCODEC_SAMPLING_UNSUPPORTED;
        }
    } else {
        for (uint8_t i = 0; i < CSiz; i++) {
            if ((XRSiz[0] != 1) || (XRSiz[1] != 1) || (XRSiz[2] != 1) || (YRSiz[0] != 1) || (YRSiz[1] != 1) || (YRSiz[2] != 1))
                return NVIMGCODEC_SAMPLING_UNSUPPORTED;
        }
        return NVIMGCODEC_SAMPLING_NONE;
    }
}

nvimgcodecStatus_t GetCodeStreamInfoImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    if (codestream_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    strcpy(codestream_info->codec_name, "jpeg2k");
    codestream_info->num_images = 1;

    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace

JPEG2KParserPlugin::JPEG2KParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "jpeg2k", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_codestream_info, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* JPEG2KParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t JPEG2KParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t bitstream_size = 0;
    io_stream->size(io_stream->instance, &bitstream_size);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    *result = 0;

    std::array<uint8_t, 12> bitstream_start;
    size_t read_nbytes = 0;
    io_stream->read(io_stream->instance, &read_nbytes, bitstream_start.data(), bitstream_start.size());
    if (read_nbytes < bitstream_start.size())
        return NVIMGCODEC_STATUS_SUCCESS;

    if (!memcmp(bitstream_start.data(), JP2_SIGNATURE.data(), JP2_SIGNATURE.size()))
        *result = 1;
    else if (!memcmp(bitstream_start.data(), J2K_SIGNATURE.data(), J2K_SIGNATURE.size()))
        *result = 1;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t JPEG2KParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<JPEG2KParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

JPEG2KParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_destroy");
}

nvimgcodecStatus_t JPEG2KParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new JPEG2KParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create jpeg2k parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t JPEG2KParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<JPEG2KParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::parseJP2(nvimgcodecIoStreamDesc_t* io_stream)
{
    uint32_t block_size;
    block_type_t block_type;
    SkipBox(io_stream, jp2_signature, "JPEG2K signature");
    SkipBox(io_stream, jp2_file_type, "JPEG2K file type");
    while (ReadBoxHeader(block_type, block_size, io_stream)) {
        if (block_type == jp2_header) { // superbox
            auto remaining_bytes = block_size - sizeof(block_size) - sizeof(block_type);
            while (remaining_bytes > 0) {
                ReadBoxHeader(block_type, block_size, io_stream);
                if (block_type == jp2_image_header) { // Ref. I.5.3.1 Image Header box
                    if (block_size != 22) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid JPEG2K image header");
                        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                    }
                    height = ReadValueBE<uint32_t>(io_stream);
                    width = ReadValueBE<uint32_t>(io_stream);
                    num_components = ReadValueBE<uint16_t>(io_stream);

                    if (num_components > NVIMGCODEC_MAX_NUM_PLANES) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Too many components " << num_components);
                        return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                    }

                    bits_per_component = ReadValueBE<uint8_t>(io_stream);
                    io_stream->skip(io_stream->instance, sizeof(uint8_t)); // compression_type
                    io_stream->skip(io_stream->instance, sizeof(uint8_t)); // color_space_unknown
                    io_stream->skip(io_stream->instance, sizeof(uint8_t)); // IPR
                } else if (block_type == jp2_colour_spec && color_spec == NVIMGCODEC_COLORSPEC_UNKNOWN) {
                    auto method = ReadValueBE<uint8_t>(io_stream);
                    io_stream->skip(io_stream->instance, sizeof(int8_t)); // precedence
                    io_stream->skip(io_stream->instance, sizeof(int8_t)); // colourspace approximation
                    auto enumCS = ReadValueBE<uint32_t>(io_stream);
                    if (method == 1) {
                        switch (enumCS) {
                        case 16: // sRGB
                            color_spec = NVIMGCODEC_COLORSPEC_SRGB;
                            break;
                        case 17: // Greyscale
                            color_spec = NVIMGCODEC_COLORSPEC_GRAY;
                            break;
                        case 18: // sYCC
                            color_spec = NVIMGCODEC_COLORSPEC_SYCC;
                            break;
                        default:
                            color_spec = NVIMGCODEC_COLORSPEC_UNSUPPORTED;
                            break;
                        }
                    } else if (method == 2) {
                        color_spec = NVIMGCODEC_COLORSPEC_UNSUPPORTED;
                    }
                } else {
                    io_stream->skip(io_stream->instance, block_size - sizeof(block_size) - sizeof(block_type));
                }
                remaining_bytes -= block_size;
            }
        } else if (block_type == jp2_code_stream) {
            return parseCodeStream(io_stream); // parsing ends here
        }
    }
    return NVIMGCODEC_STATUS_BAD_CODESTREAM; //  didn't parse codestream
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::parseCodeStream(nvimgcodecIoStreamDesc_t* io_stream)
{
    auto marker = ReadValueBE<uint16_t>(io_stream);
    if (marker != SOC_marker) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "SOC marker not found");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
    // SOC should be followed by SIZ. Figure A.3
    marker = ReadValueBE<uint16_t>(io_stream);
    if (marker != SIZ_marker) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "SIZ marker not found");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    auto marker_size = ReadValueBE<uint16_t>(io_stream);
    if (marker_size < 41 || marker_size > 49190) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid SIZ marker size");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    io_stream->skip(io_stream->instance, sizeof(uint16_t)); // RSiz
    XSiz = ReadValueBE<uint32_t>(io_stream);
    YSiz = ReadValueBE<uint32_t>(io_stream);
    XOSiz = ReadValueBE<uint32_t>(io_stream);
    YOSiz = ReadValueBE<uint32_t>(io_stream);
    XTSiz = ReadValueBE<uint32_t>(io_stream);
    YTSiz = ReadValueBE<uint32_t>(io_stream);
    XTOSiz = ReadValueBE<uint32_t>(io_stream);
    YTOSiz = ReadValueBE<uint32_t>(io_stream);
    CSiz = ReadValueBE<uint16_t>(io_stream);

    // CSiz in table A.9, minimum of 1 and Max of 16384
    if (CSiz > NVIMGCODEC_MAX_NUM_PLANES) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Too many components " << num_components);
        return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
    }

    for (int i = 0; i < CSiz; i++) {
        Ssiz[i] = ReadValueBE<uint8_t>(io_stream);
        XRSiz[i] = ReadValue<uint8_t>(io_stream);
        YRSiz[i] = ReadValue<uint8_t>(io_stream);
        if (bits_per_component != DIFFERENT_BITDEPTH_PER_COMPONENT && Ssiz[i] != bits_per_component) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "SSiz is expected to match BPC from image header box");
            return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_get_codestream_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(codestream_info);
        return GetCodeStreamInfoImpl(plugin_id_, framework_, codestream_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream info from jpeg2k stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "jpeg2k_parser_get_image_info");
    try {
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        num_components = 0;
        height = 0xFFFFFFFF, width = 0xFFFFFFFF;
        bits_per_component = DIFFERENT_BITDEPTH_PER_COMPONENT;
        color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
        XSiz = 0;
        YSiz = 0;
        XOSiz = 0;
        YOSiz = 0;
        XTSiz = 0;
        YTSiz = 0;
        XTOSiz = 0;
        YTOSiz = 0;
        CSiz = 0;

        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t bitstream_size = 0;
        io_stream->size(io_stream->instance, &bitstream_size);
        if (bitstream_size < 12) {
            return NVIMGCODEC_STATUS_SUCCESS;
        }
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        if (image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected structure type");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        strcpy(image_info->codec_name, "jpeg2k");
        std::array<uint8_t, 12> bitstream_start;
        size_t read_nbytes = 0;
        io_stream->read(io_stream->instance, &read_nbytes, bitstream_start.data(), bitstream_start.size());
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        if (read_nbytes < bitstream_start.size())
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;

        nvimgcodecStatus_t status = NVIMGCODEC_STATUS_BAD_CODESTREAM;
        if (!memcmp(bitstream_start.data(), JP2_SIGNATURE.data(), JP2_SIGNATURE.size()))
            status = parseJP2(io_stream);
        else if (!memcmp(bitstream_start.data(), J2K_SIGNATURE.data(), J2K_SIGNATURE.size()))
            status = parseCodeStream(io_stream);

        if (status != NVIMGCODEC_STATUS_SUCCESS)
            return status;

        num_components = num_components > 0 ? num_components : CSiz;
        if (CSiz != num_components) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected number of components in main header versus image header box");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        image_info->sample_format = num_components > 1 ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : NVIMGCODEC_SAMPLEFORMAT_P_Y;
        image_info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
        image_info->chroma_subsampling = XRSizYRSizToSubsampling(CSiz, &XRSiz[0], &YRSiz[0]);
        image_info->color_spec = color_spec;
        image_info->num_planes = num_components;
        for (int p = 0; p < num_components; p++) {
            image_info->plane_info[p].height = DivUp(YSiz - YOSiz, YRSiz[p]);
            image_info->plane_info[p].width = DivUp(XSiz - XOSiz, XRSiz[p]);
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = BitsPerComponentToType(Ssiz[p]);
            image_info->plane_info[p].precision = (Ssiz[p] & 0x7F) + 1;
        }

        nvimgcodecTileGeometryInfo_t* tile_geometry_info = reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(image_info->struct_next);
        while (tile_geometry_info && tile_geometry_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO)
            tile_geometry_info = reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry_info->struct_next);
        if (tile_geometry_info && tile_geometry_info->struct_type == NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO) {
            tile_geometry_info->tile_height = DivUp(YTSiz - YTOSiz, YRSiz[0]);
            tile_geometry_info->tile_width = DivUp(XTSiz - XTOSiz, XRSiz[0]);
            tile_geometry_info->num_tiles_y = DivUp(image_info->plane_info[0].height, tile_geometry_info->tile_height);
            tile_geometry_info->num_tiles_x = DivUp(image_info->plane_info[0].width, tile_geometry_info->tile_width);
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from jpeg2k stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::static_get_codestream_info(
    nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->getCodeStreamInfo(codestream_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER; 
    }
}

nvimgcodecStatus_t JPEG2KParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<JPEG2KParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class Jpeg2kParserExtension
{
  public:
    explicit Jpeg2kParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , jpeg2k_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, jpeg2k_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~Jpeg2kParserExtension() { framework_->unregisterParser(framework_->instance, jpeg2k_parser_plugin_.getParserDesc()); }

    static nvimgcodecStatus_t jpeg2k_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "jpeg2k_parser_ext", "jpeg2k_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new Jpeg2kParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

    static nvimgcodecStatus_t jpeg2k_parser_extension_destroy(nvimgcodecExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcodec::Jpeg2kParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "jpeg2k_parser_ext", "jpeg2k_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    JPEG2KParserPlugin jpeg2k_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t jpeg2k_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "jpeg2k_parser_extension",
    NVIMGCODEC_VER, 
    NVIMGCODEC_VER,

    Jpeg2kParserExtension::jpeg2k_parser_extension_create,
    Jpeg2kParserExtension::jpeg2k_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_jpeg2k_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = jpeg2k_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec