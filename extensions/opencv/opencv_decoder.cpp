/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "opencv_decoder.h"
#include <cstring>
#include <future>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "imgproc/convert.h"
#include "error_handling.h"
#include "log.h"
#include "opencv_utils.h"
#include "nvimgcodec.h"

namespace opencv {

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params);
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
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
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
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;;
        }
    }

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;
};

OpenCVDecoderPlugin::OpenCVDecoderPlugin(const std::string& codec_name, const nvimgcodecFrameworkDesc_t* framework)
    : codec_name_(codec_name)
    , plugin_id_("opencv_" + codec_name_ + "_decoder")
    , decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_.c_str(), codec_name_.c_str(),
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_get_metadata, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_sample, nullptr}
    , framework_(framework)
{}

const nvimgcodecDecoderDesc_t* OpenCVDecoderPlugin::getDecoderDesc() const
{
    return &decoder_desc_;
}

nvimgcodecStatus_t DecoderImpl::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) const
{
    return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
}

nvimgcodecProcessingStatus_t DecoderImpl::canDecode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    (void) thread_idx;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        bool is_supported_format =
            strcmp(cs_image_info.codec_name, "jpeg") == 0 ||
            strcmp(cs_image_info.codec_name, "jpeg2k") == 0 ||
            strcmp(cs_image_info.codec_name, "png") == 0 ||
            strcmp(cs_image_info.codec_name, "tiff") == 0 ||
            strcmp(cs_image_info.codec_name, "bmp") == 0 ||
            strcmp(cs_image_info.codec_name, "pnm") == 0 ||
            strcmp(cs_image_info.codec_name, "webp") == 0;
        if (!is_supported_format)
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        default:
            break; // supported
        }

        if (image_info.num_planes == 3 || image_info.num_planes == 4) {
            for (size_t p = 0; p < image_info.num_planes; p++) {
                if (image_info.plane_info[p].num_channels != 1)
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }

            if (image_info.num_planes == 4) {
                if (strcmp(cs_image_info.codec_name, "jpeg") == 0 ||
                    strcmp(cs_image_info.codec_name, "pnm") == 0 ||
                    strcmp(cs_image_info.codec_name, "bmp") == 0
                ) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
                } else if (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                }
            }
        } else if (image_info.num_planes == 1) {
            if (image_info.plane_info[0].num_channels != 3 &&
                image_info.plane_info[0].num_channels != 4 &&
                image_info.plane_info[0].num_channels != 1
            ) {
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }

            if (image_info.plane_info[0].num_channels == 4) {
                if (strcmp(cs_image_info.codec_name, "jpeg") == 0 ||
                    strcmp(cs_image_info.codec_name, "pnm") == 0 ||
                    strcmp(cs_image_info.codec_name, "bmp") == 0
                ) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                } else if (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                }
            }
        } else {
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        auto sample_type = image_info.plane_info[0].sample_type;
        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != sample_type) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
        switch (sample_type) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
        case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
            break;
        default:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            break;
        }

        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        if (codestream_info.code_stream_view) {
            if (codestream_info.code_stream_view->image_idx != 0) { //TODO: Add support for multiple images
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_IMAGES_UNSUPPORTED;
            }
            auto region = codestream_info.code_stream_view->region;
            if (region.ndim > 0 && region.ndim != 2) {
                status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
            }
        }
        return status;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can decode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

DecoderImpl::DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
}

nvimgcodecStatus_t OpenCVDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_.c_str(), "opencv_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params)
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_.c_str(), framework_, exec_params));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_.c_str(), "Could not create opencv decoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t OpenCVDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<OpenCVDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

DecoderImpl::~DecoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "opencv_destroy");
}

nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
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
    (void) thread_idx;
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
        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve image information");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }
        XM_CHECK_NULL(code_stream);
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        
        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        int num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
        int flags = num_channels > 1 ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
        if (num_channels > 3)
            flags |= cv::IMREAD_UNCHANGED;
        if (!params->apply_exif_orientation)
            flags |= cv::IMREAD_IGNORE_ORIENTATION;
        if (image_info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8)
            flags |= cv::IMREAD_ANYDEPTH;

        auto decoded = cv::imdecode(
            cv::_InputArray(static_cast<const uint8_t*>(encoded_stream_data), encoded_stream_data_size), flags
        );
        if (decoded.data == nullptr) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        if (decoded.channels() != num_channels) {
            NVIMGCODEC_LOG_ERROR(
                framework_, plugin_id_,
                "OpenCV could only decode " << decoded.channels() << " out of " << num_channels << " channels."
            );
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        if (codestream_info.code_stream_view ){
            auto region = codestream_info.code_stream_view->region;
            if (region.ndim > 0) {
                int start_y = region.start[0];
                int start_x = region.start[1];
                int crop_h = region.end[0] - region.start[0];
                int crop_w = region.end[1] - region.start[1];
                if (crop_h < 0 || crop_w < 0 || start_x < 0 || start_y < 0 ||
                    (start_y + crop_h) > decoded.rows || (start_x + crop_w) > decoded.cols
                ) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Region of interest is out of bounds");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }
                cv::Rect roi(start_x, start_y, crop_w, crop_h);
                cv::Mat tmp;
                decoded(roi).copyTo(tmp);
                std::swap(tmp, decoded);
            }
        }

        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
            colorConvert(decoded, cv::COLOR_BGR2RGB); // opencv decodes as BGR layout
            break;
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
            if (num_channels == 4) {
                colorConvert(decoded, cv::COLOR_BGRA2RGBA);
            }
            else if (num_channels == 3) {
                colorConvert(decoded, cv::COLOR_BGR2RGB);
            }
            break;
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
            break;
        default:
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample_format: " << image_info.sample_format);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        auto convert_ret = convertFromCvMat(image_info, decoded);
        image->imageReady(image->instance,
            convert_ret != NVIMGCODEC_STATUS_SUCCESS ? NVIMGCODEC_PROCESSING_STATUS_FAIL : NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return convert_ret;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error: " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace opencv
