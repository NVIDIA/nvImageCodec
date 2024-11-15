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

#include "parsers/png.h"
#include <nvimgcodec.h>
#include <string.h>
#include <vector>

#include "imgproc/exception.h"
#include "exif_orientation.h"
#include "log_ext.h"
#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcodec {

namespace {

// https://www.w3.org/TR/2003/REC-PNG-20031110

enum ColorType : uint8_t
{
    PNG_COLOR_TYPE_GRAY = 0,
    PNG_COLOR_TYPE_RGB = 2,
    PNG_COLOR_TYPE_PALETTE = 3,
    PNG_COLOR_TYPE_GRAY_ALPHA = 4,
    PNG_COLOR_TYPE_RGBA = 6
};

struct IhdrChunk
{
    uint32_t width;
    uint32_t height;
    uint8_t color_type;
    // Some fields were ommited.

    int GetNumberOfChannels(bool include_alpha)
    {
        switch (color_type) {
        case PNG_COLOR_TYPE_GRAY:
            return 1;
        case PNG_COLOR_TYPE_GRAY_ALPHA:
            return 1 + include_alpha;
        case PNG_COLOR_TYPE_RGB:
        case PNG_COLOR_TYPE_PALETTE: // 1 byte but it's converted to 3-channel BGR by OpenCV
            return 3;
        case PNG_COLOR_TYPE_RGBA:
            return 3 + include_alpha;
        default:
            throw std::runtime_error("color type not supported");
        }
    }
};

using chunk_type_field_t = std::array<uint8_t, 4>;
static constexpr chunk_type_field_t IHDR_TAG{'I', 'H', 'D', 'R'};
static constexpr chunk_type_field_t EXIF_TAG{'e', 'X', 'I', 'f'};
static constexpr chunk_type_field_t IEND_TAG{'I', 'E', 'N', 'D'};

using png_signature_t = std::array<uint8_t, 8>;
static constexpr png_signature_t PNG_SIGNATURE = {137, 80, 78, 71, 13, 10, 26, 10};

nvimgcodecStatus_t GetImageInfoImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{

    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t io_stream_length;
    io_stream->size(io_stream->instance, &io_stream_length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    size_t read_nbytes = 0;
    png_signature_t signature;
    io_stream->read(io_stream->instance, &read_nbytes, &signature[0], signature.size());
    if (read_nbytes != sizeof(png_signature_t) || signature != PNG_SIGNATURE) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected signature");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    // IHDR Chunk:
    // IHDR chunk length(4 bytes): 0x00 0x00 0x00 0x0D
    // IHDR chunk type(Identifies chunk type to be IHDR): 0x49 0x48 0x44 0x52
    // Image width in pixels(variable 4): xx xx xx xx
    // Image height in pixels(variable 4): xx xx xx xx
    // Flags in the chunk(variable 5 bytes): xx xx xx xx xx
    // CRC checksum(variable 4 bytes): xx xx xx xx

    uint32_t length = ReadValueBE<uint32_t>(io_stream);
    if (length != (4 + 4 + 5)) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected length");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    auto chunk_type = ReadValue<chunk_type_field_t>(io_stream);
    if (chunk_type != IHDR_TAG) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Missing IHDR tag");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    if (image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    strcpy(image_info->codec_name, "png");
    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    image_info->plane_info[0].width = ReadValueBE<uint32_t>(io_stream);
    image_info->plane_info[0].height = ReadValueBE<uint32_t>(io_stream);
    uint8_t bitdepth = ReadValueBE<uint8_t>(io_stream);
    nvimgcodecSampleDataType_t sample_type;
    switch (bitdepth) {
    case 1:
    case 2:
    case 4:
    case 8:
        sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        break;
    case 16:
        sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        break;
    default:
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected bitdepth: " << bitdepth);
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
    // see Table 11.1
    auto color_type = static_cast<ColorType>(ReadValueBE<uint8_t>(io_stream));
    switch (color_type) {
    case PNG_COLOR_TYPE_GRAY:
        image_info->num_planes = 1;
        break;
    case PNG_COLOR_TYPE_RGB:
        image_info->num_planes = 3;
        if (bitdepth != 8 && bitdepth != 16) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected bitdepth for RGB color type: " << bitdepth);
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        break;
    case PNG_COLOR_TYPE_PALETTE:
        image_info->num_planes = 3;
        if (bitdepth == 16) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected bitdepth for palette color type: " << bitdepth);
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        break;
    case PNG_COLOR_TYPE_GRAY_ALPHA:
        image_info->num_planes = 2;
        if (bitdepth != 8 && bitdepth != 16) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected bitdepth for Gray with alpha color type: " << bitdepth);
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        break;
    case PNG_COLOR_TYPE_RGBA:
        image_info->num_planes = 4;
        if (bitdepth != 8 && bitdepth != 16) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected bitdepth for RGBA color type: " << bitdepth);
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        break;
    default:
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected color type: " << color_type);
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
    image_info->sample_format = image_info->num_planes >= 3 ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : NVIMGCODEC_SAMPLEFORMAT_P_Y;
    image_info->color_spec = image_info->num_planes >= 3 ? NVIMGCODEC_COLORSPEC_SRGB : NVIMGCODEC_COLORSPEC_GRAY;

    for (size_t p = 0; p < image_info->num_planes; p++) {
        image_info->plane_info[p].height = image_info->plane_info[0].height;
        image_info->plane_info[p].width = image_info->plane_info[0].width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = sample_type;
        image_info->plane_info[p].precision = bitdepth;
    }

    io_stream->skip(io_stream->instance, 3 + 4); // Skip the other fields and the CRC checksum.
    image_info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    while (true) {
        uint32_t chunk_length = ReadValueBE<uint32_t>(io_stream);
        auto chunk_type = ReadValue<chunk_type_field_t>(io_stream);

        if (chunk_type == IEND_TAG)
            break;

        if (chunk_type == EXIF_TAG) {
            std::vector<uint8_t> chunk(chunk_length);
            size_t read_chunk_nbytes;
            io_stream->read(io_stream->instance, &read_chunk_nbytes, &chunk[0], chunk_length);
            if (read_chunk_nbytes != chunk_length) {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected end of stream");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }

            cv::ExifReader reader;
            if (reader.parseExif(chunk.data(), chunk.size())) {
                auto entry = reader.getTag(cv::ORIENTATION);
                if (entry.tag != cv::INVALID_TAG) {
                    image_info->orientation = FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
                }
            }
            io_stream->skip(io_stream->instance, 4);                // 4 bytes of CRC
        } else {
            io_stream->skip(io_stream->instance, chunk_length + 4); // + 4 bytes of CRC
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // end namespace

PNGParserPlugin::PNGParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "png", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* PNGParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t PNGParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "png_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t read_nbytes = 0;
    png_signature_t signature;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &read_nbytes, &signature[0], signature.size());
    if (read_nbytes == sizeof(png_signature_t) && signature == PNG_SIGNATURE) {
        *result = 1;
    } else {
        *result = 0;
    }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNGParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<PNGParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

PNGParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "png_parser_destroy");
}

nvimgcodecStatus_t PNGParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "png_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new PNGParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create png parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNGParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<PNGParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNGParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNGParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNGParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "png_parser_get_image_info");
    try {
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        return GetImageInfoImpl(plugin_id_, framework_, image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from png stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNGParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNGParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class PngParserExtension
{
  public:
    explicit PngParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , png_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, png_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~PngParserExtension() { framework_->unregisterParser(framework_->instance, png_parser_plugin_.getParserDesc()); }

    static nvimgcodecStatus_t png_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "png_parser_ext", "png_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new PngParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

    static nvimgcodecStatus_t png_parser_extension_destroy(nvimgcodecExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcodec::PngParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "png_parser_ext", "png_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    PNGParserPlugin png_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t png_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "png_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_EXT_API_VER,

    PngParserExtension::png_parser_extension_create,
    PngParserExtension::png_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_png_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = png_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec