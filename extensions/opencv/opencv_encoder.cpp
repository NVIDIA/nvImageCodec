/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define NOMINMAX
#include "opencv_encoder.h"
#include "opencv_utils.h"
#include <cstring>
#include <future>
#include <nvtx3/nvtx3.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include "imgproc/convert.h"
#include "error_handling.h"
#include "log.h"
#include "nvimgcodec.h"

namespace opencv {

struct EncoderImpl
{
    EncoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params);
    ~EncoderImpl();

    nvimgcodecProcessingStatus_t canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
        const nvimgcodecEncodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(encoder);
            auto handle = reinterpret_cast<EncoderImpl*>(encoder);
            return handle->canEncode(code_stream, image, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    nvimgcodecStatus_t encode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
        const nvimgcodecEncodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_encode_sample(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(encoder);
            auto handle = reinterpret_cast<EncoderImpl*>(encoder);
            return handle->encode(code_stream, image, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
    }

    static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;
};

OpenCVEncoderPlugin::OpenCVEncoderPlugin(const std::string& codec_name, const nvimgcodecFrameworkDesc_t* framework)
    : codec_name_(codec_name)
    , plugin_id_("opencv_" + codec_name_ + "_encoder")
    , encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_.c_str(), codec_name_.c_str(),
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, EncoderImpl::static_destroy, EncoderImpl::static_can_encode,
          EncoderImpl::static_encode_sample}
    , framework_(framework)
{}

const nvimgcodecEncoderDesc_t* OpenCVEncoderPlugin::getEncoderDesc() const
{
    return &encoder_desc_;
}

nvimgcodecProcessingStatus_t EncoderImpl::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image, 
    const nvimgcodecEncodeParams_t* params, int thread_idx)
{
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "opencv_can_encode");
        nvtx3::scoped_range marker{"opencv_can_encode"};
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);
        XM_CHECK_NULL(image);

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        std::string codec_name = cs_image_info.codec_name;
        bool is_supported_format =
            codec_name == "jpeg" ||
            codec_name == "jpeg2k" ||
            codec_name == "png" ||
            codec_name == "tiff" ||
            codec_name == "bmp" ||
            codec_name == "pnm" ||
            codec_name == "webp";

        if (!is_supported_format)
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;

        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        if(info.num_planes == 3 || info.num_planes == 4) {
            for(size_t p = 0; p < info.num_planes; p++) {
                if (info.plane_info[p].num_channels != 1)
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
            if (info.num_planes == 4) {
                if (codec_name  == "jpeg" || codec_name  == "pnm") {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
                } else if (info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                }
            }
        } else if (info.num_planes == 1) {
            if (info.plane_info[0].num_channels != 3 &&
                info.plane_info[0].num_channels != 4 &&
                info.plane_info[0].num_channels != 1
            ) {
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }

            if (info.plane_info[0].num_channels == 4) {
                if (codec_name  == "jpeg" ||  codec_name  == "pnm") {
                    status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                } else if (info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
                }
            }
        } else {
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        auto sample_type = info.plane_info[0].sample_type;
        for (size_t p = 1; p < info.num_planes; p++) {
            if (info.plane_info[p].sample_type != sample_type) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }

        if (sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
            if (sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
                if (codec_name != "jpeg2k" &&
                    codec_name != "png" &&
                    codec_name != "tiff" &&
                    codec_name != "pnm"
                ) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }
            } else {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }

    } catch(const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if opencv can encode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return status;
}

EncoderImpl::EncoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{}

EncoderImpl::~EncoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "opencv_destroy");
}

nvimgcodecStatus_t OpenCVEncoderPlugin::create(
    nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_.c_str(), "opencv_create");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(exec_params)
        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new EncoderImpl(plugin_id_.c_str(), framework_, exec_params));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_.c_str(), "Could not create opencv encoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t OpenCVEncoderPlugin::static_create(
    void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<OpenCVEncoderPlugin*>(instance);
        handle->create(encoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t EncoderImpl::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<EncoderImpl*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}


nvimgcodecStatus_t EncoderImpl::encode(const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image,
    const nvimgcodecEncodeParams_t* params, int thread_idx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "opencv encode");
        XM_CHECK_NULL(image);

        nvimgcodecImageInfo_t source_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &source_image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        nvimgcodecJpegImageInfo_t out_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), 0};
        nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), &out_jpeg_image_info};
        ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        int opencv_type;
        ret = getOpencvDataType(&opencv_type, source_image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported data format: " << source_image_info.plane_info[0].sample_type);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        cv::Mat opencv_image(source_image_info.plane_info[0].height, source_image_info.plane_info[0].width, opencv_type);

        ret = convertToCvMat(source_image_info, opencv_image);
        if(ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        int num_channels = std::max(source_image_info.num_planes, source_image_info.plane_info[0].num_channels);
        switch (source_image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
            colorConvert(opencv_image, cv::COLOR_RGB2BGR);
            break;
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
            if (num_channels == 4)
                colorConvert(opencv_image, cv::COLOR_RGBA2BGRA);
            else if (num_channels == 3)
                colorConvert(opencv_image, cv::COLOR_RGB2BGR);
            break;
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
            break;
        case NVIMGCODEC_SAMPLEFORMAT_P_YUV:
            // There is colorConvert(opencv_image, cv::COLOR_YUV2BGR);
            // but there are SDTV with BT.470 and HDTV with BT.709. Which one is used by opencv and which one we expect?
            // break;
            [[fallthrough]];
        default:
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample_format: " << source_image_info.sample_format);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED;
        }

        std::string target_codec = out_image_info.codec_name;
        std::string extension = std::string(".") + target_codec;
        if (extension == ".jpeg") {
            extension = ".jpg";
        }
        if (extension == ".jpeg2k") {
            extension = ".jp2";
        }

        if (target_codec != "jpeg") {
            if (source_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported chroma subsampling: " << source_image_info.chroma_subsampling);
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED);
                return NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED;
            }
        }

        std::vector<int> encode_params;
        // std::vector<int> encode_params = {IMWRITE_JPEG_QUALITY,5}; // For instance

        if (target_codec == "jpeg") {
            if (out_jpeg_image_info.encoding != NVIMGCODEC_JPEG_ENCODING_UNKNOWN) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, " - encoding: " << out_jpeg_image_info.encoding);

                // OpenCV can only perform one of these two algorithms
                if (out_jpeg_image_info.encoding != NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT && out_jpeg_image_info.encoding != NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported jpeg encoding: " << out_jpeg_image_info.encoding);
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED);
                    return NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED;
                }

                encode_params.push_back(cv::IMWRITE_JPEG_PROGRESSIVE);
                encode_params.push_back(out_jpeg_image_info.encoding == NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN);
            }

            const nvimgcodecJpegEncodeParams_t *jpeg_encode_params = getJpegEncodeParams(params);
            if (jpeg_encode_params) {
                encode_params.push_back(cv::IMWRITE_JPEG_OPTIMIZE);
                encode_params.push_back(jpeg_encode_params->optimized_huffman);
            }

            // Default jpeg subsampling is 420 in OpenCV
            if (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_420) {
                encode_params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR);
                switch (out_image_info.chroma_subsampling) {
                    case NVIMGCODEC_SAMPLING_411:
                        encode_params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR_411);
                        break;
                    case NVIMGCODEC_SAMPLING_422:
                        encode_params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR_422);
                        break;
                    case NVIMGCODEC_SAMPLING_440:
                        encode_params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR_440);
                        break;
                    case NVIMGCODEC_SAMPLING_444:
                        encode_params.push_back(cv::IMWRITE_JPEG_SAMPLING_FACTOR_444);
                        break;
                    default:
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported jpeg chroma subsampling: " << out_image_info.chroma_subsampling);
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED);
                        return NVIMGCODEC_STATUS_EXTENSION_CODESTREAM_UNSUPPORTED;
                }
            }

            encode_params.push_back(cv::IMWRITE_JPEG_QUALITY);
            encode_params.push_back(params->quality);
        }

        if (target_codec == "jpeg2k") {
            const nvimgcodecJpeg2kEncodeParams_t *jpeg2k_encode_params = getJpeg2kEncodeParams(params);

            int compression_ratio = params->quality * 10;
            if (compression_ratio == 0) {
                // OpenCV uses different default compression ratio depending which implementation is used.
                // We fix it to one common ratio.
                compression_ratio = 500;
            }

            if (jpeg2k_encode_params) {
                if (jpeg2k_encode_params->irreversible == 0) {
                    compression_ratio = 1000;
                }

                if (jpeg2k_encode_params->code_block_h != 64)
                {
                    NVIMGCODEC_LOG_WARNING(
                        framework_, plugin_id_, "nvimgcodecJpeg2kEncodeParams_t.code_block_h is not applicable "
                        "to OpenCV encoder, set it to 64 to disable this warning."
                    );
                }

                if (jpeg2k_encode_params->code_block_w != 64)
                {
                    NVIMGCODEC_LOG_WARNING(
                        framework_, plugin_id_, "nvimgcodecJpeg2kEncodeParams_t.code_block_w is not applicable "
                        "to OpenCV encoder, set it to 64 to disable this warning."
                    );
                }

                if (jpeg2k_encode_params->num_resolutions != 6)
                {
                    NVIMGCODEC_LOG_WARNING(
                        framework_, plugin_id_, "nvimgcodecJpeg2kEncodeParams_t.num_resolutions is not applicable "
                        "to OpenCV encoder, set it to 6 to disable this warning."
                    );
                }

                if (jpeg2k_encode_params->prog_order != NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL)
                {
                    NVIMGCODEC_LOG_WARNING(
                        framework_, plugin_id_, "nvimgcodecJpeg2kEncodeParams_t.prog_order is not applicable "
                        "to OpenCV encoder, set it to NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL to disable this warning."
                    );
                }

                if (jpeg2k_encode_params->stream_type != NVIMGCODEC_JPEG2K_STREAM_JP2)
                {
                    NVIMGCODEC_LOG_WARNING(
                        framework_, plugin_id_, "nvimgcodecJpeg2kEncodeParams_t.stream_type is not applicable "
                        "to OpenCV encoder, set it to NVIMGCODEC_JPEG2K_STREAM_JP2 to disable this warning."
                    );
                }
            }

            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "Using IMWRITE_JPEG2000_COMPRESSION_X1000 = " << compression_ratio);
            encode_params.push_back(cv::IMWRITE_JPEG2000_COMPRESSION_X1000);
            encode_params.push_back(compression_ratio);

            if (params->target_psnr != 50 && params->target_psnr != 0) {
                NVIMGCODEC_LOG_WARNING(
                    framework_, plugin_id_,
                    "nvimgcodecEncodeParams_t.target_psnr is not applicable to OpenCV encoder, set it to 50 or 0"
                    "to disable this warning. Use nvimgcodecEncodeParams_t.quality instead."
                );
            }
        }
        
        if (target_codec == "webp") {
            encode_params.push_back(cv::IMWRITE_WEBP_QUALITY);
            encode_params.push_back(params->quality);
        }

        // Some additional opencv flags can be passed: https://docs.opencv.org/3.4/d8/d6a/group__imgcodecs__flags.html
        // but are currently impossible to feed (e.g., for PNG)

        std::vector<uchar> encoded;
        if (!cv::imencode(extension, opencv_image, encoded, encode_params)) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Error when encoding with OpenCV.");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXTENSION_EXECUTION_FAILED;
        }

        auto io_stream = code_stream->io_stream;
        ret = io_stream->reserve(io_stream->instance, encoded.size());
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        size_t written_size;
        ret = io_stream->write(io_stream->instance, &written_size, encoded.data(), encoded.size());
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }
        if (written_size != encoded.size()) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }

        ret = io_stream->flush(io_stream->instance);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return ret;
        }

        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode using opencv: " << e.what());
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace opencv
