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

#include "opencv_decoder.h"
#include <cstring>
#include <future>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "imgproc/convert.h"
#include "error_handling.h"
#include "log.h"
#include "nvimgcodec.h"
#include "../utils/stream_ctx.h"
#include "../utils/parallel_exec.h"

namespace opencv {

static void color_convert(cv::Mat& img, cv::ColorConversionCodes conversion)
{
    if (img.data == nullptr || img.rows == 0 || img.cols == 0)
        throw std::runtime_error("Invalid input image");
    cv::cvtColor(img, img, conversion);
}

template <typename DestType, typename SrcType>
nvimgcodecStatus_t ConvertPlanar(DestType* destinationBuffer, uint32_t plane_stride, uint32_t row_stride_bytes, const cv::Mat& image)
{
    using nvimgcodec::ConvertSatNorm;
    std::vector<cv::Mat> planes;
    cv::split(image, planes);
    size_t height = image.size[0];
    size_t width = image.size[1];
    for (size_t ch = 0; ch < planes.size(); ++ch) {
        const cv::Mat& srcPlane = planes[ch];
        const SrcType* srcPlanePtr = srcPlane.ptr<SrcType>();
        DestType* destPlanePtr = destinationBuffer + ch * plane_stride;
        for (size_t i = 0; i < height; ++i) {
            const SrcType* srcRow = srcPlanePtr + i * width;
            DestType* destRow = reinterpret_cast<DestType*>(reinterpret_cast<uint8_t*>(destPlanePtr) + i * row_stride_bytes);
            for (size_t j = 0; j < width; ++j) {
                destRow[j] = ConvertSatNorm<DestType>(srcRow[j]);
            }
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

template <typename DestType, typename SrcType>
nvimgcodecStatus_t ConvertInterleaved(DestType* destinationBuffer, uint32_t row_stride_bytes, const cv::Mat& image)
{
    using nvimgcodec::ConvertSatNorm;
    size_t height = image.size[0];
    size_t width = image.size[1];
    size_t channels = image.channels();
    for (size_t i = 0; i < height; ++i) {
        const SrcType* srcRow = image.ptr<SrcType>() + i * width * channels;
        DestType* destRow = reinterpret_cast<DestType*>(reinterpret_cast<uint8_t*>(destinationBuffer) + i * row_stride_bytes);
        for (size_t j = 0; j < width; ++j) {
            for (size_t c = 0; c < channels; c++) {
                destRow[j * channels + c] = ConvertSatNorm<DestType>(srcRow[j * channels + c]);
            }
        }
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t ConvertPlanar(nvimgcodecImageInfo_t& info, const cv::Mat& decoded)
{

#define CaseConvertPlanar(OUT_SAMPLE_TYPE, OutType, img_info, image)                                                          \
    case OUT_SAMPLE_TYPE:                                                                                                     \
        switch (image.depth()) {                                                                                              \
        case CV_8U:                                                                                                           \
            return ConvertPlanar<OutType, uint8_t>(reinterpret_cast<OutType*>(img_info.buffer),                               \
                img_info.plane_info[0].row_stride * img_info.plane_info[0].height, img_info.plane_info[0].row_stride, image); \
        case CV_16U:                                                                                                          \
            return ConvertPlanar<OutType, uint16_t>(reinterpret_cast<OutType*>(img_info.buffer),                              \
                img_info.plane_info[0].row_stride * img_info.plane_info[0].height, img_info.plane_info[0].row_stride, image); \
        default:                                                                                                              \
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;                                                               \
        }                                                                                                                     \
        break;

    switch (info.plane_info[0].sample_type) {
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, uint8_t, info, decoded);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_INT8, int8_t, info, decoded);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, uint16_t, info, decoded);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_INT16, int16_t, info, decoded);
        CaseConvertPlanar(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, float, info, decoded);
    default:
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }
#undef CaseConvertPlanar
}

nvimgcodecStatus_t ConvertInterleaved(nvimgcodecImageInfo_t& info, const cv::Mat& decoded)
{

#define CaseConvertInterleaved(OUT_SAMPLE_TYPE, OutType, img_info, image)                               \
    case OUT_SAMPLE_TYPE:                                                                               \
        switch (image.depth()) {                                                                        \
        case CV_8U:                                                                                     \
            return ConvertInterleaved<OutType, uint8_t>(                                                \
                reinterpret_cast<OutType*>(img_info.buffer), img_info.plane_info[0].row_stride, image); \
        case CV_16U:                                                                                    \
            return ConvertInterleaved<OutType, uint16_t>(                                               \
                reinterpret_cast<OutType*>(img_info.buffer), img_info.plane_info[0].row_stride, image); \
        default:                                                                                        \
            return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;                                         \
        }                                                                                               \
        break;

    switch (info.plane_info[0].sample_type) {
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8, uint8_t, info, decoded);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_INT8, int8_t, info, decoded);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16, uint16_t, info, decoded);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_INT16, int16_t, info, decoded);
        CaseConvertInterleaved(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32, float, info, decoded);
    default:
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

#undef CaseConvertInterleaved
}

nvimgcodecStatus_t Convert(nvimgcodecImageInfo_t& info, const cv::Mat& decoded)
{
    if (info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB || info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR ||
        info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
        return ConvertPlanar(info, decoded);
    } else {
        return ConvertInterleaved(info, decoded);
    }
}

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params);
    ~DecoderImpl();

    nvimgcodecStatus_t canDecodeImpl(CodeStreamCtx& ctx);
    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    void decodeImpl(BatchItemCtx& batch_item);
    nvimgcodecStatus_t decodeBatch(
        nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
    static nvimgcodecStatus_t static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
        nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;

    CodeStreamCtxManager code_stream_mgr_;
};

OpenCVDecoderPlugin::OpenCVDecoderPlugin(const std::string& codec_name, const nvimgcodecFrameworkDesc_t* framework)
    : codec_name_(codec_name)
    , plugin_id_("opencv_" + codec_name_ + "_decoder")
    , decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_.c_str(), codec_name_.c_str(),
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, DecoderImpl::static_destroy, DecoderImpl::static_can_decode,
          DecoderImpl::static_decode_batch}
    , framework_(framework)
{}

nvimgcodecDecoderDesc_t* OpenCVDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t DecoderImpl::canDecodeImpl(CodeStreamCtx& ctx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        auto* code_stream = ctx.code_stream_;
        XM_CHECK_NULL(code_stream);

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

        for (size_t i = 0; i < ctx.size(); i++) {
            auto *status = &ctx.batch_items_[i]->processing_status;
            auto *image = ctx.batch_items_[i]->image;
            const auto *params = ctx.batch_items_[i]->params;

            XM_CHECK_NULL(status);
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(params);

            *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            if (!is_supported_format) {
                *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                continue;
            }

            nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            image->getImageInfo(image->instance, &info);

            switch (info.sample_format) {
            case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
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

            if (info.num_planes > 1) {
                for (size_t p = 0; p < info.num_planes; p++) {
                    if (info.plane_info[p].num_channels != 1)
                        *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
            } else if (info.num_planes == 1) {
                if (info.plane_info[0].num_channels != 3 && info.plane_info[0].num_channels != 4 && info.plane_info[0].num_channels != 1)
                    *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            } else {
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            auto sample_type = info.plane_info[0].sample_type;
            for (size_t p = 1; p < info.num_planes; p++) {
                if (info.plane_info[p].sample_type != sample_type) {
                    *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }
            }
            switch (sample_type) {
            case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8:
            case NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16:
                break;
            default:
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                break;
            }
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can decode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t DecoderImpl::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "opencv_can_decode");
        nvtx3::scoped_range marker{"opencv_can_decode"};
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        // Groups samples belonging to the same stream
        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);

        auto task = [](int tid, int sample_idx, void* context) {
            auto this_ptr = reinterpret_cast<DecoderImpl*>(context);
            this_ptr->canDecodeImpl(*this_ptr->code_stream_mgr_[sample_idx]);
        };
        BlockParallelExec(this, task, code_stream_mgr_.size(), exec_params_);
        for (int i = 0; i < batch_size; i++) {
            status[i] = code_stream_mgr_.get_batch_item(i).processing_status;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can decode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
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
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<OpenCVDecoderPlugin*>(instance);
        handle->create(decoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
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

void DecoderImpl::decodeImpl(BatchItemCtx& batch_item)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "decode #"+ std::to_string(batch_item.index));
    nvtx3::scoped_range marker{"opencv decode " + std::to_string(batch_item.index)};
    auto* image = batch_item.image;
    const auto* params = batch_item.params;

    try {

        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        if (info.region.ndim != 0 && info.region.ndim != 2) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid region of interest");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        if (info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        int num_channels = std::max(info.num_planes, info.plane_info[0].num_channels);
        int flags = num_channels > 1 ? cv::IMREAD_COLOR : cv::IMREAD_GRAYSCALE;
        if (num_channels > 3)
            flags |= cv::IMREAD_UNCHANGED;
        if (!params->apply_exif_orientation)
            flags |= cv::IMREAD_IGNORE_ORIENTATION;
        if (info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8)
            flags |= cv::IMREAD_ANYDEPTH;

        const uint8_t* encoded_data = static_cast<const uint8_t*>(batch_item.code_stream_ctx->encoded_stream_data_);
        size_t encoded_length = batch_item.code_stream_ctx->encoded_stream_data_size_;
        auto decoded = cv::imdecode(cv::_InputArray(encoded_data, encoded_length), flags);
        if (decoded.data == nullptr) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        if (info.region.ndim == 2) {
            int start_y = info.region.start[0];
            int start_x = info.region.start[1];
            int crop_h = info.region.end[0] - info.region.start[0];
            int crop_w = info.region.end[1] - info.region.start[1];
            if (crop_h < 0 || crop_w < 0 || start_x < 0 || start_y < 0 || (start_y + crop_h) > decoded.rows ||
                (start_x + crop_w) > decoded.cols) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Region of interest is out of bounds");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                return;
            }
            cv::Rect roi(start_x, start_y, crop_w, crop_h);
            cv::Mat tmp;
            decoded(roi).copyTo(tmp);
            std::swap(tmp, decoded);
        }

        switch (info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
            color_convert(decoded, cv::COLOR_BGR2RGB); // opencv decodes as BGR layout
            break;
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
            if (num_channels == 4)
                color_convert(decoded, cv::COLOR_BGRA2RGBA);
            else if (num_channels == 3)
                color_convert(decoded, cv::COLOR_BGR2RGB);
            break;
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
            break;
        default:
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample_format: " << info.sample_format);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        auto convert_ret = Convert(info, decoded);
        image->imageReady(image->instance, 
            convert_ret != NVIMGCODEC_STATUS_SUCCESS ?
            NVIMGCODEC_PROCESSING_STATUS_FAIL :
            NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error: " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }
}

nvimgcodecStatus_t DecoderImpl::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);
        for (size_t i = 0; i < code_stream_mgr_.size(); i++) {
            code_stream_mgr_[i]->load();
        }

        auto executor = exec_params_->executor;
        auto task = [](int tid, int sample_idx, void* context) -> void {
            auto* this_ptr = reinterpret_cast<DecoderImpl*>(context);
            auto& batch_item = this_ptr->code_stream_mgr_.get_batch_item(sample_idx);
            this_ptr->decodeImpl(batch_item);
        };
        if (batch_size == 1) {
            task(0, 0, this);
        } else {
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
                executor->launch(executor->instance, NVIMGCODEC_DEVICE_CPU_ONLY, sample_idx, this, task);
            }
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t DecoderImpl::static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

} // namespace opencv
