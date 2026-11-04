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

#include "parsers/pnm.h"
#include "parsers/codec_file_extensions.h"
#include <nvimgcodec.h>
#include <string.h>
#include <vector>
#include <limits>

#include "imgproc/exception.h"
#include "exif_orientation.h"
#include "log_ext.h"

#include "parsers/byte_io.h"
#include "parsers/exif.h"

namespace nvimgcodec {

namespace {

// comments can appear in the middle of tokens, and the newline at the
// end is part of the comment, not counted as whitespace
// http://netpbm.sourceforge.net/doc/pbm.html
size_t SkipComment(nvimgcodecIoStreamDesc_t* io_stream)
{
    char c;
    size_t skipped = 0;
    do {
        c = ReadValue<char>(io_stream);
        skipped++;
    } while (c != '\n');
    return skipped;
}

void SkipSpaces(nvimgcodecIoStreamDesc_t* io_stream)
{
    ptrdiff_t pos;
    io_stream->tell(io_stream->instance, &pos);
    while (true) {
        char c = ReadValue<char>(io_stream);
        pos++;
        if (c == '#')
            pos += SkipComment(io_stream);
        else if (!isspace(c))
            break;
    }
    // return the nonspace byte to the stream
    io_stream->seek(io_stream->instance, pos - 1, SEEK_SET);
}

int ParseInt(nvimgcodecIoStreamDesc_t* io_stream)
{
    ptrdiff_t pos;
    io_stream->tell(io_stream->instance, &pos);
    int int_value = 0;
    while (true) {
        char c = ReadValue<char>(io_stream);
        pos++;
        if (isdigit(c)) {
            if (int_value > (std::numeric_limits<int>::max() - 9) / 10) {
                throw std::overflow_error(
                    "Exceed int range when parsing pnm codestream. Already read: " + std::to_string(int_value) + ", next digit is: " + c
                );
            }
            int_value = int_value * 10 + (c - '0');
        }
        else if (c == '#')
            pos += SkipComment(io_stream);
        else
            break;
    }
    // return the nondigit byte to the stream
    io_stream->seek(io_stream->instance, pos - 1, SEEK_SET);
    return int_value;
}

nvimgcodecStatus_t GetImageInfoImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t io_stream_length;
    io_stream->size(io_stream->instance, &io_stream_length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    if (image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    strcpy(image_info->codec_name, "pnm");
    // http://netpbm.sourceforge.net/doc/ppm.html

    if (io_stream_length < 3) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected end of stream");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    std::array<uint8_t, 3> header = ReadValue<std::array<uint8_t, 3>>(io_stream);
    bool is_pnm = header[0] == 'P' && header[1] >= '1' && header[1] <= '6' && isspace(header[2]);
    if (!is_pnm) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected header");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
    // formats "P3" and "P6" are RGB color, all other formats are bitmaps or greymaps
    uint32_t nchannels = (header[1] == '3' || header[1] == '6') ? 3 : 1;

    // P1/P4 are bitmaps: they carry no max value field, so parsing one would consume pixel data.
    const bool is_bitmap = header[1] == '1' || header[1] == '4';
    uint32_t width, height, max_value = 1;
    try {
        SkipSpaces(io_stream);
        width = ParseInt(io_stream);
        SkipSpaces(io_stream);
        height = ParseInt(io_stream);
        if (!is_bitmap) {
            SkipSpaces(io_stream);
            max_value = ParseInt(io_stream);
        }
    } catch ( std::overflow_error const& ex) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, ex.what());
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    if (width == 0 || height == 0) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "PNM with zero width or height");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
    if (max_value == 0 || max_value > 65535) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Invalid PNM max value: " << max_value);
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    // The dimensions are unbounded ASCII integers, so a 30-byte header can claim a 2^31 x 2^31
    // image and have the core allocate a buffer for it. Cross-check the declared geometry against
    // the bytes the stream actually holds. This is only a lower bound - it must never reject a
    // well-formed file, so each format uses the smallest encoding the spec permits.
    {
        // ParseInt seeks back onto the non-digit byte that terminated the last header token, so that
        // separator is still ahead of us and is not raster data.
        ptrdiff_t header_end = 0;
        io_stream->tell(io_stream->instance, &header_end);
        size_t raster_offset = header_end >= 0 ? static_cast<size_t>(header_end) + 1 : io_stream_length;
        size_t remaining = io_stream_length > raster_offset ? io_stream_length - raster_offset : 0;

        constexpr uint64_t kSaturated = std::numeric_limits<uint64_t>::max();
        auto saturating_mul = [kSaturated](uint64_t a, uint64_t b) -> uint64_t {
            return (a != 0 && b > kSaturated / a) ? kSaturated : a * b;
        };

        // width and height are each bounded by INT_MAX, so the pixel count itself cannot overflow.
        uint64_t npixels = static_cast<uint64_t>(width) * height;
        uint64_t nsamples = saturating_mul(npixels, nchannels);

        uint64_t min_bytes;
        if (header[1] == '4') {
            // Raw bitmap: 8 pixels packed per byte.
            min_bytes = (npixels + 7) / 8;
        } else if (header[1] == '1') {
            // Plain bitmap: one '0'/'1' character per sample. Whitespace between them is optional,
            // so the samples may run together.
            min_bytes = nsamples;
        } else if (header[1] >= '4') {
            // Raw graymap/pixmap: a sample is two bytes once max_value exceeds 255.
            min_bytes = saturating_mul(nsamples, max_value > 255 ? 2 : 1);
        } else {
            // Plain graymap/pixmap: samples are decimal numbers, so each needs at least one digit
            // and consecutive samples must be separated - but the last one needs no trailing
            // separator, giving 2n-1 rather than 2n.
            min_bytes = nsamples > kSaturated / 2 ? kSaturated : (nsamples == 0 ? 0 : 2 * nsamples - 1);
        }

        if (min_bytes > remaining) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id,
                "PNM declares " << width << "x" << height << " (needs at least " << min_bytes << " bytes of pixel data) but only "
                                << remaining << " bytes remain in the stream");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
    }

    image_info->sample_format = nchannels >= 3 ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : NVIMGCODEC_SAMPLEFORMAT_P_Y;
    image_info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    image_info->chroma_subsampling = nchannels == 1 ? NVIMGCODEC_SAMPLING_GRAY : NVIMGCODEC_SAMPLING_NONE;
    image_info->color_spec = nchannels >= 3 ? NVIMGCODEC_COLORSPEC_SRGB : NVIMGCODEC_COLORSPEC_GRAY;
    image_info->num_planes = nchannels;
    for (size_t p = 0; p < nchannels; p++) {
        image_info->plane_info[p].height = height;
        image_info->plane_info[p].width = width;
        image_info->plane_info[p].num_channels = 1;
        // Per the PNM spec a sample occupies two bytes whenever max_value exceeds 255, not only at
        // exactly 65535 - reporting UINT8 for e.g. max_value 1000 would misdescribe the raster.
        image_info->plane_info[p].sample_type = max_value > 255 ? NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16 : NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        image_info->plane_info[p].precision = max_value > 255 ? 16 : 8;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}


nvimgcodecStatus_t GetCodeStreamInfoImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    if (codestream_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    strcpy(codestream_info->codec_name, "pnm");
    codestream_info->num_images = 1;

    // Calculate bitstream size - for PNM, it's the entire file size
    code_stream->io_stream->size(code_stream->io_stream->instance, &(codestream_info->size));

    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace

PNMParserPlugin::PNMParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "pnm", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_codestream_info, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* PNMParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t PNMParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);
    if (length < 3) {
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    std::array<uint8_t, 3> header = ReadValue<std::array<uint8_t, 3>>(io_stream);
    *result = header[0] == 'P' && header[1] >= '1' && header[1] <= '6' && isspace(header[2]);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNMParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<PNMParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

PNMParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_parser_destroy");
}

nvimgcodecStatus_t PNMParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new PNMParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create pnm parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNMParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<PNMParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNMParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNMParserPlugin::Parser::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_parser_get_codestream_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(codestream_info);
        return GetCodeStreamInfoImpl(plugin_id_, framework_, codestream_info, code_stream);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream info from pnm stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t PNMParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        return GetImageInfoImpl(plugin_id_, framework_, image_info, code_stream);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from pnm stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t PNMParserPlugin::Parser::static_get_codestream_info(
    nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->getCodeStreamInfo(codestream_info, code_stream);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER; 
    }
}

nvimgcodecStatus_t PNMParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class PnmParserExtension
{
  public:
    explicit PnmParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , pnm_parser_plugin_(framework)        
    {
        framework->registerParser(framework->instance, pnm_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
        registerCodecFileExtensions(framework, PNMParserPlugin::codecName(), PNMParserPlugin::fileExtensions());
    }
    ~PnmParserExtension()
    {
        unregisterCodecFileExtensions(framework_, PNMParserPlugin::codecName(), PNMParserPlugin::fileExtensions());
        framework_->unregisterParser(framework_->instance, pnm_parser_plugin_.getParserDesc());
    }

    static nvimgcodecStatus_t pnm_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "pnm_parser_ext", "pnm_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new PnmParserExtension(framework));
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

    static nvimgcodecStatus_t pnm_parser_extension_destroy(nvimgcodecExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcodec::PnmParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "pnm_parser_ext", "pnm_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    PNMParserPlugin pnm_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t pnm_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "pnm_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER,

    PnmParserExtension::pnm_parser_extension_create,
    PnmParserExtension::pnm_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_pnm_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = pnm_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec