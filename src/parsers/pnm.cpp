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
#include <nvimgcodec.h>
#include <string.h>
#include <vector>

#include "exception.h"
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
        if (isdigit(c))
            int_value = int_value * 10 + (c - '0');
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

    SkipSpaces(io_stream);
    uint32_t width = ParseInt(io_stream);
    SkipSpaces(io_stream);
    uint32_t height = ParseInt(io_stream);

    image_info->sample_format = nchannels >= 3 ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : NVIMGCODEC_SAMPLEFORMAT_P_Y;
    image_info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info->num_planes = nchannels;
    for (size_t p = 0; p < nchannels; p++) {
        image_info->plane_info[p].height = height;
        image_info->plane_info[p].width = width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace

PNMParserPlugin::PNMParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "pnm", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_image_info}
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
    } catch (const std::runtime_error& e) {
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
    } catch (const std::runtime_error& e) {
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
    } catch (const std::runtime_error& e) {
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
    } catch (const std::runtime_error& e) {
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
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t PNMParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        return GetImageInfoImpl(plugin_id_, framework_, image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from png stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t PNMParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<PNMParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
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
    }
    ~PnmParserExtension() { framework_->unregisterParser(framework_->instance, pnm_parser_plugin_.getParserDesc()); }

    static nvimgcodecStatus_t pnm_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "pnm_parser_ext", "pnm_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new PnmParserExtension(framework));
    } catch (const std::runtime_error& e) {
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
    } catch (const std::runtime_error& e) {
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
    NVIMGCODEC_EXT_API_VER,

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