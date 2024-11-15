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

#include "parsers/tiff.h"
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

constexpr int ENTRY_SIZE = 12;

enum TiffTag : uint16_t
{
    WIDTH_TAG = 256,
    HEIGHT_TAG = 257,
    PHOTOMETRIC_INTERPRETATION_TAG = 262,
    ORIENTATION_TAG = 274,
    SAMPLESPERPIXEL_TAG = 277,
    ROWS_PER_STRIP_TAG = 278,
    BITSPERSAMPLE_TAG = 258,
    TILE_WIDTH = 322,
    TILE_LENGTH = 323,
    SAMPLE_FORMAT_TAG = 339
};

enum TiffSampleFormat : uint16_t
{
    TIFF_SAMPLEFORMAT_UNINITIALIZED = 0,
    TIFF_SAMPLEFORMAT_UINT          = 1,
    TIFF_SAMPLEFORMAT_INT           = 2,
    TIFF_SAMPLEFORMAT_IEEEFP        = 3,
    TIFF_SAMPLEFORMAT_UNDEFINED     = 4
};

enum TiffDataType : uint16_t
{
    TYPE_WORD = 3,
    TYPE_DWORD = 4
};

constexpr int PHOTOMETRIC_PALETTE = 3;

using tiff_magic_t = std::array<uint8_t, 4>;
constexpr tiff_magic_t le_header = {'I', 'I', 42, 0}, be_header = {'M', 'M', 0, 42};

template <typename T, bool is_little_endian>
T TiffRead(nvimgcodecIoStreamDesc_t* io_stream)
{
    if constexpr (is_little_endian) {
        return ReadValueLE<T>(io_stream);
    } else {
        return ReadValueBE<T>(io_stream);
    }
}

nvimgcodecSampleDataType_t convert_to_sample_type(
    uint16_t bitdepth, bool sample_format_read, TiffSampleFormat sample_format)
{
    //Convert sample_format to internal sample_type
    if (!sample_format_read || sample_format == TIFF_SAMPLEFORMAT_UNDEFINED) {
        sample_format = TIFF_SAMPLEFORMAT_UINT; // default according to standard
    }

    // TODO: Do we have decoders for all bitdepths?
    switch (sample_format) {
    case TIFF_SAMPLEFORMAT_UINT:
        if (bitdepth <= 8) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        } else if (bitdepth <= 16) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
        } else if (bitdepth <= 32) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT32;
        } else if (bitdepth <= 64) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_UINT64;
        }
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    case TIFF_SAMPLEFORMAT_INT:
        if (bitdepth <= 8) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT8;
        } else if (bitdepth <= 16) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT16;
        } else if (bitdepth <= 32) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT32;
        } else if (bitdepth <= 64) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_INT64;
        }
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    case TIFF_SAMPLEFORMAT_IEEEFP:
        if (bitdepth == 32) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32;
        } else if (bitdepth == 16) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16;
        } else if (bitdepth == 64) {
            return NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64;
        }
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    default:
        return NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED;
    }
}

template <typename T, typename V>
constexpr inline T DivUp(T x, V d)
{
    return (x + d - 1) / d;
}

template <bool is_little_endian>
nvimgcodecStatus_t GetInfoImpl(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecImageInfo_t* info, nvimgcodecIoStreamDesc_t* io_stream)
{
    io_stream->seek(io_stream->instance, 4, SEEK_SET);
    const auto ifd_offset = TiffRead<uint32_t, is_little_endian>(io_stream);
    io_stream->seek(io_stream->instance, ifd_offset, SEEK_SET);
    const auto entry_count = TiffRead<uint16_t, is_little_endian>(io_stream);

    strcpy(info->codec_name, "tiff");
    info->color_spec = NVIMGCODEC_COLORSPEC_UNKNOWN;
    info->chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};

    bool width_read = false, height_read = false, samples_per_px_read = false, palette_read = false,
        bitdepth_read = false, sample_format_read = false;
    uint32_t width = 0, height = 0, nchannels = 0;
    std::array<uint16_t, NVIMGCODEC_MAX_NUM_PLANES> bitdepth = {};
    std::array<TiffSampleFormat, NVIMGCODEC_MAX_NUM_PLANES> sample_format = {};

    bool tile_width_read = false, tile_height_read = false, rows_per_strip_read = false;
    uint32_t strile_width = 0, strile_height = 0;

    for (int entry_idx = 0; entry_idx < entry_count; entry_idx++) {
        const auto entry_offset = ifd_offset + sizeof(uint16_t) + entry_idx * ENTRY_SIZE;
        io_stream->seek(io_stream->instance, entry_offset, SEEK_SET);
        const auto tag_id = TiffRead<uint16_t, is_little_endian>(io_stream);
        const auto value_type = TiffRead<uint16_t, is_little_endian>(io_stream);
        const auto value_count = TiffRead<uint32_t, is_little_endian>(io_stream);
        if (tag_id == BITSPERSAMPLE_TAG || tag_id == SAMPLE_FORMAT_TAG) {
            if (value_count > 1) {
                uint32_t value_offset = TiffRead<uint32_t, is_little_endian>(io_stream);
                io_stream->seek(io_stream->instance, value_offset, SEEK_SET);
            }

            if (tag_id == BITSPERSAMPLE_TAG) {
                if (value_type != TYPE_WORD) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Bits per sample tag should have SHORT type.");
                    return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                }

                if (value_count > NVIMGCODEC_MAX_NUM_PLANES) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                        "Couldn't read TIFF with more than " << NVIMGCODEC_MAX_NUM_PLANES << " components. Got " << value_count
                                                            << "values for bits per sample tag."
                    );
                    return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                }

                for (size_t i = 0; i < value_count; i++) {
                    bitdepth[i] = TiffRead<uint16_t, is_little_endian>(io_stream);
                }

                bitdepth_read = true;
            } else if (tag_id == SAMPLE_FORMAT_TAG) {
                if (value_type != TYPE_WORD) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Sample format tag should have SHORT type.");
                    return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                }

                if (value_count > NVIMGCODEC_MAX_NUM_PLANES) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                        "Couldn't read TIFF with more than " << NVIMGCODEC_MAX_NUM_PLANES << " components. Got " << value_count
                                                            << "values for sample format tag."
                    );
                    return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                }

                for (size_t i = 0; i < value_count; ++i) {
                    sample_format[i] = static_cast<TiffSampleFormat>(TiffRead<uint16_t, is_little_endian>(io_stream));
                }
                sample_format_read = true;
            }
        } else if (tag_id == WIDTH_TAG || tag_id == HEIGHT_TAG || tag_id == SAMPLESPERPIXEL_TAG || tag_id == ORIENTATION_TAG ||
                   tag_id == PHOTOMETRIC_INTERPRETATION_TAG || tag_id == TILE_WIDTH || tag_id == TILE_LENGTH || tag_id == ROWS_PER_STRIP_TAG) {
            if (value_count != 1) {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected value count");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }

            uint32_t value;
            if (value_type == TYPE_WORD) {
                value = TiffRead<uint16_t, is_little_endian>(io_stream);
            } else if (value_type == TYPE_DWORD) {
                value = TiffRead<uint32_t, is_little_endian>(io_stream);
            } else {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Couldn't read TIFF tag, type should be SHORT or LONG but is not.");
                return NVIMGCODEC_STATUS_BAD_CODESTREAM;
            }

            if (tag_id == WIDTH_TAG) {
                width = value;
                width_read = true;
            } else if (tag_id == HEIGHT_TAG) {
                height = value;
                height_read = true;
            } else if (tag_id == ORIENTATION_TAG) {
                info->orientation = FromExifOrientation(static_cast<ExifOrientation>(value));
            } else if (tag_id == SAMPLESPERPIXEL_TAG && !palette_read) {
                // If the palette is present, the SAMPLESPERPIXEL tag is always set to 1, so it does not
                // indicate the actual number of channels. That's why we ignore it for palette images.
                nchannels = value;
                samples_per_px_read = true;
                if (nchannels > NVIMGCODEC_MAX_NUM_PLANES) {
                    NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                        "Couldn't read TIFF with more than " << NVIMGCODEC_MAX_NUM_PLANES << " components. Got " << nchannels
                                                             << " value for samples per pixel tag."
                    );
                    return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
                }
            } else if (tag_id == PHOTOMETRIC_INTERPRETATION_TAG && value == PHOTOMETRIC_PALETTE) {
                nchannels = 3;
                palette_read = true;
            } else if (tag_id == TILE_LENGTH) {
                strile_height = value;
                tile_height_read = true;
            } else if (tag_id == TILE_WIDTH) {
                strile_width = value;
                tile_width_read = true;
            } else if (tag_id == ROWS_PER_STRIP_TAG) {
                strile_height = value;
                rows_per_strip_read = true;
            }
        }

        if (width_read && height_read && palette_read && bitdepth_read && sample_format_read
            && ((tile_width_read && tile_height_read) || rows_per_strip_read)
        ) {
            break;
        }
    }

    if (!width_read || !height_read || !bitdepth_read || (!samples_per_px_read && !palette_read)) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Couldn't read TIFF image required fields (dims, bitdepth or number of channels).");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    if (tile_width_read != tile_height_read) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Both tile width and height should be present.");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    if (tile_width_read && rows_per_strip_read) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Image should have either tiles or strips, not both.");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    // In Palette, bitdepth is specified as key size, but we want RBG bitdepth which is always 16
    if (palette_read) {
        for (uint32_t i = 0; i < nchannels; ++i) {
            bitdepth[i] = 16;
        }
    }

    info->num_planes = nchannels;
    for (size_t p = 0; p < info->num_planes; p++) {
        info->plane_info[p].height = height;
        info->plane_info[p].width = width;
        info->plane_info[p].num_channels = 1;
        info->plane_info[p].sample_type = convert_to_sample_type(
            bitdepth[p], sample_format_read, sample_format[p]
        );
        if (info->plane_info[p].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UNSUPPORTED) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                "Unsupported sample format " << sample_format[p]<< " with bitdepth "
                << bitdepth[p] << " for channel " << p
            );
            return NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED;
        }
        info->plane_info[p].precision = bitdepth[p];
    }

    if (nchannels == 1){
        info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_Y;
    }
    else {
        info->sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
    }

    if (rows_per_strip_read) {
        strile_width = width;
        strile_height = std::min(strile_height, height);
    } else if (!tile_height_read) {
        strile_width = width;
        strile_height = height;
    }

    // nvimgcodecTileGeometryInfo_t stores just tile sizes and num tiles
    nvimgcodecTileGeometryInfo_t* tile_geometry_info = reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(info->struct_next);
    while (tile_geometry_info && tile_geometry_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO){
        tile_geometry_info = reinterpret_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry_info->struct_next);
    }
    if (tile_geometry_info && tile_geometry_info->struct_type == NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO) {
        tile_geometry_info->tile_height = strile_height;
        tile_geometry_info->tile_width = strile_width;
        tile_geometry_info->num_tiles_y = DivUp(height, strile_height);
        tile_geometry_info->num_tiles_x = DivUp(width, strile_width);
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace

TIFFParserPlugin::TIFFParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "tiff", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* TIFFParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t TIFFParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t length;
        io_stream->size(io_stream->instance, &length);
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        if (length < 4) {
            *result = 0;
            return NVIMGCODEC_STATUS_SUCCESS;
        }
        tiff_magic_t header = ReadValue<tiff_magic_t>(io_stream);
        *result = header == le_header || header == be_header;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<TIFFParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

TIFFParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_destroy");
}

nvimgcodecStatus_t TIFFParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new TIFFParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create tiff parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<TIFFParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "tiff_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        size_t length;
        io_stream->size(io_stream->instance, &length);
        io_stream->seek(io_stream->instance, 0, SEEK_SET);

        tiff_magic_t header = ReadValue<tiff_magic_t>(io_stream);
        nvimgcodecStatus_t ret = NVIMGCODEC_STATUS_SUCCESS;
        if (header == le_header) {
            ret = GetInfoImpl<true>(plugin_id_, framework_, image_info, io_stream);
        } else if (header == be_header) {
            ret = GetInfoImpl<false>(plugin_id_, framework_, image_info, io_stream);
        } else {
            // should not happen (because canParse returned result==true)
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Logic error");
            ret = NVIMGCODEC_STATUS_INTERNAL_ERROR;
        }
        return ret;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from tiff stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t TIFFParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<TIFFParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class TiffParserExtension
{
  public:
    explicit TiffParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , tiff_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, tiff_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~TiffParserExtension() { framework_->unregisterParser(framework_->instance, tiff_parser_plugin_.getParserDesc()); }

    static nvimgcodecStatus_t tiff_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        try {
            CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "tiff_parser_ext", "tiff_parser_extension_create");
            CHECK_NULL(extension)
            *extension = reinterpret_cast<nvimgcodecExtension_t>(new TiffParserExtension(framework));
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t tiff_parser_extension_destroy(nvimgcodecExtension_t extension)
    {
        try {
            CHECK_NULL(extension)
            auto ext_handle = reinterpret_cast<nvimgcodec::TiffParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "tiff_parser_ext", "tiff_parser_extension_destroy");
            delete ext_handle;
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    TIFFParserPlugin tiff_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t tiff_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,
   
    NULL,
    "tiff_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_EXT_API_VER,

    TiffParserExtension::tiff_parser_extension_create,
    TiffParserExtension::tiff_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_tiff_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = tiff_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec