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

#include "parsers/webp.h"
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

// Specific bits in WebpExtendedHeader::layout_mask
static constexpr uint8_t EXTENDED_LAYOUT_RESERVED = 1 << 0;
static constexpr uint8_t EXTENDED_LAYOUT_ANIMATION = 1 << 1;
static constexpr uint8_t EXTENDED_LAYOUT_XMP_METADATA = 1 << 2;
static constexpr uint8_t EXTENDED_LAYOUT_EXIF_METADATA = 1 << 3;
static constexpr uint8_t EXTENDED_LAYOUT_ALPHA = 1 << 4;
static constexpr uint8_t EXTENDED_LAYOUT_ICC_PROFILE = 1 << 5;

using chunk_type_t = std::array<uint8_t, 4>;
static constexpr chunk_type_t RIFF_TAG = {'R', 'I', 'F', 'F'};
static constexpr chunk_type_t WEBP_TAG = {'W', 'E', 'B', 'P'};
static constexpr chunk_type_t VP8_TAG = {'V', 'P', '8', ' '};  // lossy
static constexpr chunk_type_t VP8L_TAG = {'V', 'P', '8', 'L'}; // lossless
static constexpr chunk_type_t VP8X_TAG = {'V', 'P', '8', 'X'}; // extended
static constexpr chunk_type_t EXIF_TAG = {'E', 'X', 'I', 'F'}; // EXIF

nvimgcodecStatus_t GetCodeStreamInfoImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    if (codestream_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected structure type");
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    strcpy(codestream_info->codec_name, "webp");
    codestream_info->num_images = 1;

    return NVIMGCODEC_STATUS_SUCCESS;
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
    strcpy(image_info->codec_name, "webp");

    if (io_stream_length < (4 + 4 + 4)) { // RIFF + file size + WEBP
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected end of stream");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    // https://developers.google.com/speed/webp/docs/riff_container#webp_file_header
    auto riff = ReadValue<chunk_type_t>(io_stream);
    if (riff != RIFF_TAG) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected RIFF tag");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // file_size
    auto webp = ReadValue<chunk_type_t>(io_stream);
    if (webp != WEBP_TAG) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected WEBP tag");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    auto chunk_type = ReadValue<chunk_type_t>(io_stream);
    auto chunk_size = ReadValueLE<uint32_t>(io_stream);
    uint32_t width = 0, height = 0, nchannels = 3;
    bool alpha = false;
    nvimgcodecOrientation_t orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    const uint16_t mask = (1 << 14) - 1;
    if (chunk_type == VP8_TAG) {                 // lossy format
        io_stream->skip(io_stream->instance, 3); // frame_tag
        const std::array<uint8_t, 3> expected_sync_code{0x9D, 0x01, 0x2A};
        auto sync_code = ReadValue<std::array<uint8_t, 3>>(io_stream);
        if (sync_code != expected_sync_code) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected VP8 sync code");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        width = ReadValueLE<uint16_t>(io_stream) & mask;
        height = ReadValueLE<uint16_t>(io_stream) & mask;
    } else if (chunk_type == VP8L_TAG) { // lossless format
        auto signature_byte = ReadValue<uint8_t>(io_stream);
        const uint8_t expected_signature_byte = 0x2F;
        if (signature_byte != expected_signature_byte) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected VP8L signature byte");
            return NVIMGCODEC_STATUS_BAD_CODESTREAM;
        }
        auto features = ReadValueLE<uint32_t>(io_stream);
        // VP8L shape information are packed inside the features field
        width = (features & mask) + 1;
        height = ((features >> 14) & mask) + 1;
        alpha = features & (1 << (2 * 14));
    } else if (chunk_type == VP8X_TAG) { // extended format
        ptrdiff_t curr, end_of_chunk_pos;
        io_stream->tell(io_stream->instance, &curr);
        end_of_chunk_pos = curr + chunk_size;

        auto layout_mask = ReadValue<uint8_t>(io_stream);
        io_stream->skip(io_stream->instance, 3); // reserved
        // Both dimensions are encoded with 24 bits, as (width - 1) i (height - 1) respectively
        width = ReadValueLE<uint32_t, 3>(io_stream) + 1;
        height = ReadValueLE<uint32_t, 3>(io_stream) + 1;
        alpha = layout_mask & EXTENDED_LAYOUT_ALPHA;
        io_stream->seek(io_stream->instance, end_of_chunk_pos, SEEK_SET);
        if (layout_mask & EXTENDED_LAYOUT_EXIF_METADATA) {
            bool exif_parsed = false;
            while (!exif_parsed) {
                chunk_type = ReadValue<chunk_type_t>(io_stream);
                chunk_size = ReadValueLE<uint32_t>(io_stream);
                io_stream->tell(io_stream->instance, &curr);
                end_of_chunk_pos = curr + chunk_size;
                if (chunk_type == EXIF_TAG) {
                    // Parse the chunk data into the orientation
                    std::vector<uint8_t> buffer(chunk_size);
                    size_t read_nbytes = 0;
                    io_stream->read(io_stream->instance, &read_nbytes, buffer.data(), buffer.size());
                    if (read_nbytes != chunk_size) {
                        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected end of stream");
                        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
                    }
                    cv::ExifReader reader;
                    reader.parseExif(buffer.data(), buffer.size());
                    const auto entry = reader.getTag(cv::ORIENTATION);
                    if (entry.tag != cv::INVALID_TAG) {
                        orientation = FromExifOrientation(static_cast<ExifOrientation>(entry.field_u16));
                    }
                    exif_parsed = true;
                }
                io_stream->seek(io_stream->instance, end_of_chunk_pos, SEEK_SET);
            }
        }
    } else {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unexpected chunk type");
        return NVIMGCODEC_STATUS_BAD_CODESTREAM;
    }

    nchannels += static_cast<int>(alpha);

    image_info->sample_format = nchannels >= 3 ? NVIMGCODEC_SAMPLEFORMAT_P_RGB : NVIMGCODEC_SAMPLEFORMAT_P_Y;
    image_info->orientation = {NVIMGCODEC_STRUCTURE_TYPE_ORIENTATION, sizeof(nvimgcodecOrientation_t), nullptr, 0, false, false};
    image_info->chroma_subsampling = NVIMGCODEC_SAMPLING_NONE;
    image_info->color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info->num_planes = nchannels;
    image_info->orientation = orientation;
    for (size_t p = 0; p < nchannels; p++) {
        image_info->plane_info[p].height = height;
        image_info->plane_info[p].width = width;
        image_info->plane_info[p].num_channels = 1;
        image_info->plane_info[p].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        image_info->plane_info[p].precision = 8;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace

WebpParserPlugin::WebpParserPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : framework_(framework)
    , parser_desc_{NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC, sizeof(nvimgcodecParserDesc_t), nullptr, this, plugin_id_, "webp", static_can_parse, static_create,
          Parser::static_destroy, Parser::static_get_codestream_info, Parser::static_get_image_info}
{
}

nvimgcodecParserDesc_t* WebpParserPlugin::getParserDesc()
{
    return &parser_desc_;
}

nvimgcodecStatus_t WebpParserPlugin::canParse(int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "webp_parser_can_parse");
        CHECK_NULL(result);
        CHECK_NULL(code_stream);
    nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
    size_t length;
    io_stream->size(io_stream->instance, &length);
    io_stream->seek(io_stream->instance, 0, SEEK_SET);

    if (length < (4 + 4 + 4)) { // RIFF + file size + WEBP
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    // https://developers.google.com/speed/webp/docs/riff_container#webp_file_header
    auto riff = ReadValue<chunk_type_t>(io_stream);
    if (riff != RIFF_TAG) {
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }
    io_stream->skip(io_stream->instance, sizeof(uint32_t)); // file_size
    auto webp = ReadValue<chunk_type_t>(io_stream);
    if (webp != WEBP_TAG) {
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    auto chunk_type = ReadValue<chunk_type_t>(io_stream);
    if (chunk_type != VP8_TAG && chunk_type != VP8L_TAG && chunk_type != VP8X_TAG) {
        *result = 0;
        return NVIMGCODEC_STATUS_SUCCESS;
    }
        *result = 1;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if code stream can be parsed - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t WebpParserPlugin::static_can_parse(void* instance, int* result, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<WebpParserPlugin*>(instance);
        return handle->canParse(result, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

WebpParserPlugin::Parser::Parser(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework)
    : plugin_id_(plugin_id)
    , framework_(framework)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "webp_parser_destroy");
}

nvimgcodecStatus_t WebpParserPlugin::create(nvimgcodecParser_t* parser)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "webp_parser_create");
        CHECK_NULL(parser);
        *parser = reinterpret_cast<nvimgcodecParser_t>(new WebpParserPlugin::Parser(plugin_id_, framework_));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create webp parser - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t WebpParserPlugin::static_create(void* instance, nvimgcodecParser_t* parser)
{
    try {
        CHECK_NULL(instance);
        auto handle = reinterpret_cast<WebpParserPlugin*>(instance);
        handle->create(parser);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t WebpParserPlugin::Parser::static_destroy(nvimgcodecParser_t parser)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<WebpParserPlugin::Parser*>(parser);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t WebpParserPlugin::Parser::getCodeStreamInfo(nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "webp_parser_get_codestream_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(codestream_info);
        return GetCodeStreamInfoImpl(plugin_id_, framework_, codestream_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream info from webp stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t WebpParserPlugin::Parser::getImageInfo(nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "webp_parser_get_image_info");
        CHECK_NULL(code_stream);
        CHECK_NULL(image_info);
        return GetImageInfoImpl(plugin_id_, framework_, image_info, code_stream);
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image info from webp stream - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t WebpParserPlugin::Parser::static_get_codestream_info(
    nvimgcodecParser_t parser, nvimgcodecCodeStreamInfo_t* codestream_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<WebpParserPlugin::Parser*>(parser);
        return handle->getCodeStreamInfo(codestream_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER; 
    }
}

nvimgcodecStatus_t WebpParserPlugin::Parser::static_get_image_info(
    nvimgcodecParser_t parser, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t* code_stream)
{
    try {
        CHECK_NULL(parser);
        auto handle = reinterpret_cast<WebpParserPlugin::Parser*>(parser);
        return handle->getImageInfo(image_info, code_stream);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

class WebpParserExtension
{
  public:
    explicit WebpParserExtension(const nvimgcodecFrameworkDesc_t* framework)
        : framework_(framework)
        , webp_parser_plugin_(framework)
    {
        framework->registerParser(framework->instance, webp_parser_plugin_.getParserDesc(), NVIMGCODEC_PRIORITY_NORMAL);
    }
    ~WebpParserExtension() { framework_->unregisterParser(framework_->instance, webp_parser_plugin_.getParserDesc()); }

    static nvimgcodecStatus_t webp_parser_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
{
    try {
        CHECK_NULL(framework)
            NVIMGCODEC_LOG_TRACE(framework, "webp_parser_ext", "webp_parser_extension_create");
        CHECK_NULL(extension)
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new WebpParserExtension(framework));
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

    static nvimgcodecStatus_t webp_parser_extension_destroy(nvimgcodecExtension_t extension)
{
    try {
        CHECK_NULL(extension)
        auto ext_handle = reinterpret_cast<nvimgcodec::WebpParserExtension*>(extension);
            NVIMGCODEC_LOG_TRACE(ext_handle->framework_, "webp_parser_ext", "webp_parser_extension_destroy");
        delete ext_handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    WebpParserPlugin webp_parser_plugin_;
};

// clang-format off
nvimgcodecExtensionDesc_t webp_parser_extension = {
    NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC,
    sizeof(nvimgcodecExtensionDesc_t),
    NULL,

    NULL,
    "webp_parser_extension",
    NVIMGCODEC_VER,
    NVIMGCODEC_VER,

    WebpParserExtension::webp_parser_extension_create,
    WebpParserExtension::webp_parser_extension_destroy
};
// clang-format on

nvimgcodecStatus_t get_webp_parser_extension_desc(nvimgcodecExtensionDesc_t* ext_desc)
{
    if (ext_desc == nullptr) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    if (ext_desc->struct_type != NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC) {
        return NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    *ext_desc = webp_parser_extension;
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec