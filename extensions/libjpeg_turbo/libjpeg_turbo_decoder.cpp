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

#include "libjpeg_turbo_decoder.h"
#include <cassert>
#include <cstring>
#include <future>
#include "jpeg_mem.h"
#include "log.h"
#include "nvimgcodec.h"
#undef INT32
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include "error_handling.h"

#include "imgproc/out_of_bound_roi_fill.h"

namespace libjpeg_turbo {

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        std::string options);
    ~DecoderImpl();

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const;
    static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->getMetadata(code_stream, metadata, metadata_count);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    nvimgcodecProcessingStatus_t canDecode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->canDecode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    nvimgcodecStatus_t decode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<DecoderImpl*>(decoder);
            return handle->decode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    void parseOptions(std::string options);

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;

    // Options
    bool fancy_upsampling_;
    bool fast_idct_;
};

LibjpegTurboDecoderPlugin::LibjpegTurboDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg",
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, DecoderImpl::static_destroy,  DecoderImpl::static_get_metadata, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_sample, nullptr}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* LibjpegTurboDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t DecoderImpl::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const
{
    return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecProcessingStatus_t DecoderImpl::canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        XM_CHECK_NULL(code_stream);
        nvimgcodecJpegImageInfo_t jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&jpeg_info)};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
        if (!is_jpeg) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        bool is_lossless_huffman = jpeg_info.encoding == NVIMGCODEC_JPEG_ENCODING_LOSSLESS_HUFFMAN;
        if (!is_jpeg) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        } else if (is_lossless_huffman) {
            return NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
        }

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;


        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        if (codestream_info.code_stream_view) {
            if (codestream_info.code_stream_view->image_idx != 0) {
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED;
            }

            const auto& region = codestream_info.code_stream_view->region;
            if (region.ndim == 0) {
                // no ROI, okay
            } else if (region.ndim == 2) {
                if (nvimgcodec::is_region_out_of_bounds(region, cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height)) {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Out of bounds region is not supported.");
                    status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                }
            } else {
                status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Region decoding is supported only for 2 dimensions.");
            }
        }
        
        XM_CHECK_NULL(params);

        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
            break; // supported
        default:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV || image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_YUV) {
            if (cs_image_info.color_spec != NVIMGCODEC_COLORSPEC_SYCC) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "YUV/YCC decoding is only supported if image is encoded into YCC.");
            }
            if (cs_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_444) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "YUV/YCC decoding is only supported if image is encoded without subsampling (444).");
            }
        }

        if (image_info.num_planes != 1 && image_info.num_planes != 3) {
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }
        if (image_info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }
        if (image_info.plane_info[0].num_channels != 3 && image_info.plane_info[0].num_channels != 1) {
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
        }

        // This codec doesn't apply EXIF orientation
        if (params->apply_exif_orientation && (image_info.orientation.flip_x || image_info.orientation.flip_y || image_info.orientation.rotated != 0)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }
        return status;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if libjpeg_turbo can decode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

DecoderImpl::DecoderImpl(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, std::string options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
    parseOptions(std::move(options));
}

void DecoderImpl::parseOptions(std::string options)
{
    // defaults
    fancy_upsampling_ = true;
    fast_idct_ = false;

    std::istringstream iss(options);
    std::string token;
    while (std::getline(iss, token, ' ')) {
        std::string::size_type colon = token.find(':');
        std::string::size_type equal = token.find('=');
        if (colon == std::string::npos || equal == std::string::npos || colon > equal)
            continue;
        std::string module = token.substr(0, colon);
        if (module != "" && module != "libjpeg_turbo_decoder")
            continue;
        std::string option = token.substr(colon + 1, equal - colon - 1);
        std::string value_str = token.substr(equal + 1);

        std::istringstream value(value_str);
        if (option == "fancy_upsampling") {
            value >> fancy_upsampling_;
        } else if (option == "fast_idct") {
            value >> fast_idct_;
        }
    }
}

nvimgcodecStatus_t LibjpegTurboDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libjpeg_turbo_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params, options));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create libjpeg_turbo decoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t LibjpegTurboDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<LibjpegTurboDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

DecoderImpl::~DecoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libjpeg_turbo_destroy");
}

nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder)
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::decode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    assert(code_stream->io_stream);
    void* encoded_stream_data = nullptr;
    size_t encoded_stream_data_size = 0;
    if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
        NVIMGCODEC_STATUS_SUCCESS) {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    assert(encoded_stream_data != nullptr);
    assert(encoded_stream_data_size > 0);
    try {
        libjpeg_turbo::UncompressFlags flags;

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        flags.sample_format = image_info.sample_format;
        switch (flags.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_I_YUV:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
            flags.components = 3;
            break;
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
            flags.components = 1;
            break;
        default:
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample_format: " << flags.sample_format);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        flags.dct_method = fast_idct_ ? JDCT_FASTEST : JDCT_DEFAULT;
        flags.fancy_upscaling = fancy_upsampling_;
        
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        if (codestream_info.code_stream_view) { 
            auto region = codestream_info.code_stream_view->region; 
            if (region.ndim != 0) {
                flags.crop = true;
                flags.crop_y = region.start[0];
                flags.crop_x = region.start[1];
                flags.crop_height = region.end[0] - region.start[0];
                flags.crop_width = region.end[1] - region.start[1];

                if (flags.crop_x < 0 || flags.crop_y < 0 || flags.crop_height != static_cast<int>(image_info.plane_info[0].height) ||
                    flags.crop_width != static_cast<int>(image_info.plane_info[0].width)) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Region of interest is out of bounds");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
            }
        }

        auto orig_sample_format = flags.sample_format;
        if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
            orig_sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        } else if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
            orig_sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        } else if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
            flags.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        } else if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR) {
            flags.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_BGR;
        } else if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV) {
            flags.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_YUV;
        }
        auto decoded_image = libjpeg_turbo::Uncompress(encoded_stream_data, encoded_stream_data_size, flags);
        if (decoded_image == nullptr) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        } else if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        const uint8_t* src = decoded_image.get();
        uint8_t* dst = reinterpret_cast<uint8_t*>(image_info.buffer);
        if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB ||
            orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR ||
            orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_YUV
        ) {
            const int num_channels = 3;
            uint32_t plane_size = image_info.plane_info[0].height * image_info.plane_info[0].width;
            for (uint32_t i = 0; i < image_info.plane_info[0].height * image_info.plane_info[0].width; i++) {
                *(dst + plane_size * 0 + i) = *(src + 0 + i * num_channels);
                *(dst + plane_size * 1 + i) = *(src + 1 + i * num_channels);
                *(dst + plane_size * 2 + i) = *(src + 2 + i * num_channels);
            }
        } else {
            uint32_t row_size_bytes = image_info.plane_info[0].width * flags.components * sizeof(uint8_t);
            for (uint32_t y = 0; y < image_info.plane_info[0].height; y++, dst += image_info.plane_info[0].row_stride, src += row_size_bytes) {
                std::memcpy(dst, src, row_size_bytes);
            }
        }
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace libjpeg_turbo
