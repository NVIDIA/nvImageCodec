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

template <typename D, int SAMPLE_FORMAT = NVIMGCODEC_SAMPLEFORMAT_P_RGB>
static int write_pnm(nvimgcodecIoStreamDesc_t* io_stream, const D* chanR, size_t pitchR, const D* chanG, size_t pitchG, const D* chanB,
    size_t pitchB, const D* chanA, size_t pitchA, int width, int height, int num_components, uint8_t precision)
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

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
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
    : encoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), NULL, this, plugin_id_, "pnm", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create,
          Encoder::static_destroy, Encoder::static_can_encode, Encoder::static_encode_batch}
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
        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(new NvPnmEncoderPlugin::Encoder(plugin_id_, framework_, exec_params, options));
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create pnm encoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t NvPnmEncoderPlugin::static_create(
    void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        XM_CHECK_NULL(exec_params);
        auto handle = reinterpret_cast<NvPnmEncoderPlugin*>(instance);
        return handle->create(encoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

nvimgcodecStatus_t NvPnmEncoderPlugin::Encoder::static_destroy(nvimgcodecEncoder_t encoder)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvPnmEncoderPlugin::Encoder*>(encoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvPnmEncoderPlugin::Encoder::Encoder(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
    , options_(options)
{
    encode_state_batch_ = std::make_unique<NvPnmEncoderPlugin::EncodeState>();
    encode_state_batch_->plugin_id_ = plugin_id_;
    encode_state_batch_->framework_ = framework_;
}

NvPnmEncoderPlugin::Encoder::~Encoder()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_destroy_encoder");
}

nvimgcodecStatus_t NvPnmEncoderPlugin::Encoder::canEncode(nvimgcodecProcessingStatus_t* status, nvimgcodecImageDesc_t** images,
    nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_can_encode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        auto result = status;
        auto code_stream = code_streams;
        auto image = images;
        for (int i = 0; i < batch_size; ++i, ++result, ++code_stream, ++image) {
            *result = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
            nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            (*code_stream)->getImageInfo((*code_stream)->instance, &cs_image_info);

            if (strcmp(cs_image_info.codec_name, "pnm") != 0) {
                NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "cannot encode because it is not pnm codec but " << cs_image_info.codec_name);
                *result = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                continue;
            }
            if (*result != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                continue;
            }

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            (*image)->getImageInfo((*image)->instance, &image_info);
            nvimgcodecImageInfo_t out_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            (*code_stream)->getImageInfo((*code_stream)->instance, &out_image_info);

            if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
            if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if (out_image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if ((image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB)) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }
            if (((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.num_planes != 3)) ||
                ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) && (image_info.num_planes != 1))) {
                *result |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                if (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 &&
                    (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16)) {
                    *result |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }

                if (((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.plane_info[p].num_channels != 1)) ||
                    ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) && (image_info.plane_info[p].num_channels != 3))) {
                    *result |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
            }
        }

        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if pnm can encode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvPnmEncoderPlugin::Encoder::static_can_encode(nvimgcodecEncoder_t encoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvPnmEncoderPlugin::Encoder*>(encoder);
        return handle->canEncode(status, images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

nvimgcodecProcessingStatus_t NvPnmEncoderPlugin::Encoder::encode(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework,
    nvimgcodecImageDesc_t* image, nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework, plugin_id, "pnm_encode");
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    image->getImageInfo(image->instance, &image_info);
    unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

    if (NVIMGCODEC_SAMPLEFORMAT_I_RGB == image_info.sample_format) {
        write_pnm<unsigned char, NVIMGCODEC_SAMPLEFORMAT_I_RGB>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            NULL, 0, NULL, 0, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height,
            image_info.plane_info[0].num_channels, image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
    } else if (NVIMGCODEC_SAMPLEFORMAT_P_RGB == image_info.sample_format) {
        write_pnm<unsigned char>(code_stream->io_stream, host_buffer, image_info.plane_info[0].row_stride,
            host_buffer + image_info.plane_info[0].row_stride * image_info.plane_info[0].height, image_info.plane_info[1].row_stride,
            host_buffer + +image_info.plane_info[0].row_stride * image_info.plane_info[0].height +
                image_info.plane_info[1].row_stride * image_info.plane_info[0].height,
            image_info.plane_info[2].row_stride, NULL, 0, image_info.plane_info[0].width, image_info.plane_info[0].height,
            image_info.num_planes, image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8 ? 8 : 16);
    } else {
            return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED | NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
        return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Could not encode pnm code stream - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }

    return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvPnmEncoderPlugin::Encoder::encodeBatch(
    nvimgcodecImageDesc_t** images, nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "pnm_encode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        encode_state_batch_->samples_.clear();
        encode_state_batch_->samples_.resize(batch_size);
        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "batch size - " << batch_size);
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            encode_state_batch_->samples_[sample_idx].code_stream = code_streams[sample_idx];
            encode_state_batch_->samples_[sample_idx].image = images[sample_idx];
            encode_state_batch_->samples_[sample_idx].params = params;
            }

            auto executor = exec_params_->executor;
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            executor->launch(executor->instance, NVIMGCODEC_DEVICE_CPU_ONLY, sample_idx, encode_state_batch_.get(),
                [](int tid, int sample_idx, void* context) -> void {
                    auto* encode_state = reinterpret_cast<EncodeState*>(context);
                    auto& sample = encode_state->samples_[sample_idx];
                    auto result =
                        encode(encode_state->plugin_id_, encode_state->framework_, sample.image, sample.code_stream, sample.params);
                    sample.image->imageReady(sample.image->instance, result);
                });
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not encode pnm batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t NvPnmEncoderPlugin::Encoder::static_encode_batch(nvimgcodecEncoder_t encoder, nvimgcodecImageDesc_t** images,
    nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecEncodeParams_t* params)
{
    try {
        XM_CHECK_NULL(encoder);
        auto handle = reinterpret_cast<NvPnmEncoderPlugin::Encoder*>(encoder);
        return handle->encodeBatch(images, code_streams, batch_size, params);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
}

} // namespace nvpnm
