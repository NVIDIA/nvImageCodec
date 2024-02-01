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

#include "parsers/bmp.h"
#include <nvimgcodec.h>
#include <string.h>
#include <vector>

#include "exception.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcodec {

namespace {

enum BmpCompressionType
{
    BMP_COMPRESSION_RGB = 0,
    BMP_COMPRESSION_RLE8 = 1,
    BMP_COMPRESSION_RLE4 = 2,
    BMP_COMPRESSION_BITFIELDS = 3
};

struct BitmapCoreHeader
{
    uint32_t header_size;
    uint16_t width, heigth, planes, bpp;
};
static_assert(sizeof(BitmapCoreHeader) == 12);

struct BitmapInfoHeader
{
    int32_t header_size;
    int32_t width, heigth;
    uint16_t planes, bpp;
    uint32_t compression, image_size;
    int32_t x_pixels_per_meter, y_pixels_per_meter;
    uint32_t colors_used, colors_important;
};
static_assert(sizeof(BitmapInfoHeader) == 40);

static bool is_color_palette(nvimgcodecIoStreamDesc_t* io_stream, size_t ncolors, int palette_entry_size)
{
    std::vector<uint8_t> entry;
    entry.resize(palette_entry_size);
    for (size_t i = 0; i < ncolors; i++) {
        size_t output_size;
        io_stream->read(io_stream->instance, &output_size, entry.data(), palette_entry_size);

        const auto b = entry[0], g = entry[1], r = entry[2]; // a = p[3];
        if (b != g || b != r)
            return true;
    }
    return false;
}

static int number_of_channels(
    nvimgcodecIoStreamDesc_t* io_stream, int bpp, int compression_type, size_t ncolors = 0, int palette_entry_size = 0)
{
    if (compression_type == BMP_COMPRESSION_RGB || compression_type == BMP_COMPRESSION_RLE8) {
        if (bpp <= 8 && ncolors <= static_cast<unsigned int>(1u << bpp)) {
            return is_color_palette(io_stream, ncolors, palette_entry_size) ? 3 : 1;
        } else if (bpp == 24) {
            return 3;
        } else if (bpp == 32) {
            return 4;
        }
    } else if (compression_type == BMP_COMPRESSION_BITFIELDS) {
        if (bpp == 16) {
            return 3;
        } else if (bpp == 32) {
            return 4;
        }
    }
    return 0;
}

} // namespace

BMPParserPlugin::BMPParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "bmp", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* BMPParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t BMPParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "bmp_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    constexpr size_t min_bmp_stream_size = 18u;
    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    if (length < min_bmp_stream_size) {
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    std::array<uint8_t, 2> signature;
    size_t output_size = 0;
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    io_stream->read(io_stream->instance, &output_size, signature.data(), signature.size());
    if (output_size != signature.size()) {
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    *result = signature[0] == 'B' && signature[1] == 'M';
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t BMPParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<BMPParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

BMPParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "bmp_parser_destroy");
}

nvimgcodecStatus_t BMPParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "bmp_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new BMPParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create bmp parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t BMPParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<BMPParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t BMPParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<BMPParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t BMPParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    // https://en.wikipedia.org/wiki/BMP_file_format#DIB_header_(bitmap_information_header)
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "bmp_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t length;
        io_stream->size(io_stream->instance, &length);
        if (length < 18u) {
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        if (image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected structure type");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        strcpy(image_info->codec_name, "bmp");
        constexpr size_t header_start = 14;
        io_stream->seek(io_stream->instance, header_start, SEEK_SET);
        uint32_t header_size = ReadValueLE<uint32_t>(io_stream);
        // we'll read it again - it's part of the header struct
        io_stream->seek(io_stream->instance, header_start, SEEK_SET);

        int bpp = 0;
        int compression_type = BMP_COMPRESSION_RGB;
        int ncolors = 0;
        int palette_entry_size = 0;
        ptrdiff_t palette_start = 0;

        if (length >= 26 && header_size == 12) {
            BitmapCoreHeader header = ReadValue<BitmapCoreHeader>(io_stream);
            image_info->plane_info[0].width = header.width;
            image_info->plane_info[0].height = header.heigth;
            bpp = header.bpp;
            if (bpp <= 8) {
                io_stream->tell(io_stream->instance, &palette_start);
                palette_entry_size = 3;
                ncolors = 1u << bpp;
            }
        } else if (length >= 50 && header_size >= 40) {
            BitmapInfoHeader header = ReadValue<BitmapInfoHeader>(io_stream);
            io_stream->skip(io_stream->instance, header_size - sizeof(header)); // Skip the ignored part of header
            image_info->plane_info[0].width = abs(header.width);
            image_info->plane_info[0].height = abs(header.heigth);
            bpp = header.bpp;
            compression_type = header.compression;
            ncolors = header.colors_used;
            if (bpp <= 8) {
                io_stream->tell(io_stream->instance, &palette_start);
                palette_entry_size = 4;
                ncolors = ncolors == 0 ? 1u << bpp : ncolors;
            }
        } else {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected length of a BMP header");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }

        // sanity check
        if (palette_start != 0) { // this silences a warning about unused variable
            assert(palette_start + (ncolors * palette_entry_size) <= static_cast<int>(length));
        }

        image_info->num_planes = number_of_channels(io_stream, bpp, compression_type, ncolors, palette_entry_size);
        for (size_t p = 0; p < image_info->num_planes; p++) {
            image_info->plane_info[p].height = image_info->plane_info[0].height;
            image_info->plane_info[p].width = image_info->plane_info[0].width;
            image_info->plane_info[p].num_channels = 1;
            image_info->plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8; // TODO(janton) always?
        }
        if (image_info->num_planes == 1) {
            image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
            image_info->color_spec = NVIMGCODEC_COLORSPEC_GRAY;
        } else {
            image_info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
            image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        }
        image_info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
        image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;

        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from bmp stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t BMPParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<BMPParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER; 
    }
}

class BmpParserExtension
{
  public:
    explicit BmpParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , bmp_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, bmp_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~BmpParserExtension() { framework_->unregisterParser(framework_->instance, bmp_parser_plugin_.getParserDesc()); }

    static nvimgcodecStatus_t bmp_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "bmp_parser_ext", "bmp_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new BmpParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

    static nvimgcodecStatus_t bmp_parser_extension_destroy(nvimgcodecExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcodec::BmpParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "bmp_parser_ext", "bmp_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    BMPParserPlugin bmp_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t bmp_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "bmp_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_EXT_API_VER,

    BmpParserExtension::bmp_parser_extension_create,
    BmpParserExtension::bmp_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_bmp_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = bmp_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec