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

#include <nvimgcodec.h>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "encoder.h"
#include "error_handling.h"
#include "log.h"

namespace nvpnm {

struct EncodeState
{
    struct Sample
    {
        nvimgcodecCodeStreamDesc_t* code_stream;
        nvimgcodecImageDesc_t* image;
        const nvimgcodecEncodeParams_t* params;
    };
    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    std::vector<Sample> samples_;
};

struct EncoderImpl
{
    EncoderImpl(
        const char* id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options);
    ~EncoderImpl();

    static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder);

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

    nvimgcodecStatus_t encode(
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx);
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

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    std::unique_ptr<EncodeState> encode_state_batch_;
    const nvimgcodecExecutionParams_t* exec_params_;
    std::string options_;
};

template <typename D, int SAMPLE_FORMAT = NVIMGCODEC_SAMPLEFORMAT_P_RGB>
static int write_pnm(nvimgcodecIoStreamDesc_t* io_stream, const D* chanR, size_t pitchR, const D* chanG, size_t pitchG, const D* chanB,
    size_t pitchB, const D* chanA, size_t pitchA, size_t width, size_t height, int num_components, uint8_t precision)
{
    size_t written_size;
    int red = 0;
    int green = 0;
    int blue = 0;
    int alpha = 0;
    std::stringstream ss{};
    if (num_components == 4) {
        ss << "P7\n";
        ss << "#nvImageCodec\n";
        ss << "WIDTH " << width << "\n";
        ss << "HEIGHT " << height << "\n";
        ss << "DEPTH " << num_components << "\n";
        ss << "MAXVAL " << (1 << precision) - 1 << "\n";
        ss << "TUPLTYPE RGB_ALPHA\n";
        ss << "ENDHDR\n";
    } else if (num_components == 1) {
        ss << "P5\n";
        ss << "#nvImageCodec\n";
        ss << width << " " << height << "\n";
        ss << (1 << precision) - 1 << "\n";
    } else {
        ss << "P6\n";
        ss << "#nvImageCodec\n";
        ss << width << " " << height << "\n";
        ss << (1 << precision) - 1 << "\n";
    }
    std::string header = ss.str();
    size_t length = header.size() + (precision / 8) * num_components * height * width;
    io_stream->reserve(io_stream->instance, length);
    io_stream->write(io_stream->instance, &written_size, static_cast<void*>(header.data()), header.size());

    for (size_t y = 0; y < height; y++) {
        for (size_t x = 0; x < width; x++) {
            if (SAMPLE_FORMAT == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
                red = chanR[y * pitchR + x];
                if (num_components > 1) {
                    green = chanG[y * pitchG + x];
                    blue = chanB[y * pitchB + x];
                    if (num_components == 4) {
                        alpha = chanA[y * pitchA + x];
                    }
                }
            } else if (SAMPLE_FORMAT == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
                red = chanR[(y * pitchR + 3 * x)];
                if (num_components > 1) {
                    green = chanR[(y * pitchR + 3 * x) + 1];
                    blue = chanR[(y * pitchR + 3 * x) + 2];
                    if (num_components == 4) {
                        alpha = chanR[y * pitchR + x];
                    }
                }
            }
            if (precision == 8) {
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red));
                if (num_components > 1) {
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha));
                    }
                }
            } else {
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red >> 8));
                io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(red & 0xFF));
                if (num_components > 1) {
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green >> 8));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(green & 0xFF));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue >> 8));
                    io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(blue & 0xFF));
                    if (num_components == 4) {
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha >> 8));
                        io_stream->putc(io_stream->instance, &written_size, static_cast<unsigned char>(alpha & 0xFF));
                    }
                }
            }
        }
    }
    io_stream->flush(io_stream->instance);
    return 0;
}

NvPnmEncoderPlugin::NvPnmEncoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_, "pnm",
          NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create, EncoderImpl::static_destroy, EncoderImpl::static_can_encode,
          EncoderImpl::static_encode_sample}
    , framework_(framework)
{
}

nvimgcodecEncoderDesc_t* NvPnmEncoderPlugin::getEncoderDesc()
{
    return &encoder_desc_;
}

nvimgcodecStatus_t NvPnmEncoderPlugin::create(
    nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_create_encoder");
        XM_CHECK_NULL(encoder);
        XM_CHECK_NULL(exec_params);
        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new EncoderImpl(plugin_id_, framework_, exec_params, options));
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create pnm encoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t NvPnmEncoderPlugin::static_create(
    void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    auto handle = reinterpret_cast<NvPnmEncoderPlugin*>(instance);
    return handle->create(encoder, exec_params, options);
}

nvimgcodecStatus_t EncoderImpl::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<EncoderImpl*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

EncoderImpl::EncoderImpl(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
    , options_(options)
{
    encode_state_batch_ = std::make_unique<EncodeState>();
    encode_state_batch_->plugin_id_ = plugin_id_;
    encode_state_batch_->framework_ = framework_;
}

EncoderImpl::~EncoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_destroy_encoder");
}

nvimgcodecProcessingStatus_t EncoderImpl::canEncode(const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
{
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_can_encode");
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        ret = code_stream->getImageInfo(code_stream->instance, &out_image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }

        if (strcmp(out_image_info.codec_name, "pnm") != 0) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "cannot encode because it is not pnm codec but " << out_image_info.codec_name);
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        if (params->quality_type != NVIMGCODEC_QUALITY_TYPE_DEFAULT && params->quality_type != NVIMGCODEC_QUALITY_TYPE_LOSSLESS) {
            status |= NVIMGCODEC_PROCESSING_STATUS_QUALITY_TYPE_UNSUPPORTED;
        }

        if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB) {
            status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        if ((image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
        if (((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.num_planes != 3)) ||
            ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) && (image_info.num_planes != 1))) {
            status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            if (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 &&
                (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }

            if (((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.plane_info[p].num_channels != 1)) ||
                ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) && (image_info.plane_info[p].num_channels != 3))) {
                status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }
        }

        // Check for tile geometry - PNM encoder does not support tiling
        nvimgcodecTileGeometryInfo_t* tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(params->struct_next);
        while (tile_geometry && tile_geometry->struct_type != NVIMGCODEC_STRUCTURE_TYPE_TILE_GEOMETRY_INFO)
            tile_geometry = static_cast<nvimgcodecTileGeometryInfo_t*>(tile_geometry->struct_next);
        if (tile_geometry) {
            if (tile_geometry->num_tiles_x != 0 || tile_geometry->num_tiles_y != 0 ||
                tile_geometry->tile_width != 0 || tile_geometry->tile_height != 0) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Tiling is not supported with PNM encoder.");
                status |= NVIMGCODEC_PROCESSING_STATUS_TILING_UNSUPPORTED;
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if pnm can encode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return status;
}

nvimgcodecStatus_t EncoderImpl::encode(const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t* params, int thread_idx)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_encode");
        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }
        
        unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        if (NVIMGCODEC_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
            write_pnm<unsigned char, NVIMGCODEC_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer,
                image_info.plane_info[0].row_stride, NULL, 0, NULL, 0, NULL, 0, image_info.plane_info[0].width,
                image_info.plane_info[0].height, image_info.plane_info[0].num_channels,
                image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
        } else if (NVIMGCODEC_SAMPLEFORMAT_P_RGB == image_info.sample_format) {
            write_pnm<unsigned char>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
                host_buffer + (size_t)image_info.plane_info[0].row_stride * (size_t)image_info.plane_info[0].height,
                image_info.plane_info[1].row_stride,
                host_buffer + (size_t)image_info.plane_info[0].row_stride * (size_t)image_info.plane_info[0].height +
                    image_info.plane_info[1].row_stride * image_info.plane_info[0].height,
                image_info.plane_info[2].row_stride, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height,
                image_info.num_planes, image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
        } else {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode pnm code stream - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

} // namespace nvpnm
