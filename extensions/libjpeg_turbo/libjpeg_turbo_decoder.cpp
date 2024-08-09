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
#include "nvimgcodec.h"

#include <cstring>
#include <future>
#include "jpeg_mem.h"
#include "log.h"
#undef INT32
#include <iostream>
#include <nvtx3/nvtx3.hpp>
#include "error_handling.h"
#include "../utils/stream_ctx.h"
#include "../utils/parallel_exec.h"

namespace libjpeg_turbo {
struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        std::string options);
    ~DecoderImpl();

    nvimgcodecStatus_t canDecodeImpl(CodeStreamCtx& ctx);
    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
    
    void decodeImpl(BatchItemCtx&);
    nvimgcodecStatus_t decodeBatch(
        nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
    static nvimgcodecStatus_t static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
        nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    void parseOptions(std::string options);

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;

    CodeStreamCtxManager code_stream_mgr_;

    // Options
    bool fancy_upsampling_;
    bool fast_idct_;
};

LibjpegTurboDecoderPlugin::LibjpegTurboDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create,
          DecoderImpl::static_destroy, DecoderImpl::static_can_decode, DecoderImpl::static_decode_batch}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* LibjpegTurboDecoderPlugin::getDecoderDesc()
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

        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;

        for (size_t i = 0; i < ctx.size(); i++) {
            auto& batch_item = *ctx.batch_items_[i];
            auto *status = &batch_item.processing_status;
            auto *image = batch_item.image;
            const auto *params = batch_item.params;

            XM_CHECK_NULL(status);
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(params);

            *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

            if (!is_jpeg) {
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
            case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
            case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
            case NVIMGCODEC_SAMPLEFORMAT_P_Y:
            case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
            case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
            default:
                break; // supported
            }

            if (info.num_planes != 1 && info.num_planes != 3) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }
            if (info.plane_info[0].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
            if (info.plane_info[0].num_channels != 3 && info.plane_info[0].num_channels != 1) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
            }

            // This codec doesn't apply EXIF orientation
            if (params->apply_exif_orientation && (info.orientation.flip_x || info.orientation.flip_y || info.orientation.rotated != 0)) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
            }
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if libjpeg_turbo can decode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
}

nvimgcodecStatus_t DecoderImpl::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");
        nvtx3::scoped_range marker{"libjpeg_turbo_can_decode"};
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);

        // Groups samples belonging to the same stream
        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);

        auto task = [](int tid, int stream_idx, void* context) {
            auto this_ptr = reinterpret_cast<DecoderImpl*>(context);
            this_ptr->canDecodeImpl(*this_ptr->code_stream_mgr_[stream_idx]);
        };
        BlockParallelExec(this, task, code_stream_mgr_.size(), exec_params_);
        for (int i = 0; i < batch_size; i++) {
            status[i] = code_stream_mgr_.get_batch_item(i).processing_status;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if libjpeg_turbo can decode - " << e.what());
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
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<LibjpegTurboDecoderPlugin*>(instance);
        handle->create(decoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
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


void DecoderImpl::decodeImpl(BatchItemCtx& batch_item)
{
    nvtx3::scoped_range marker{"libjpeg_turbo decode " + std::to_string(batch_item.index)};
    CodeStreamCtx *ctx = batch_item.code_stream_ctx;
    auto *image = batch_item.image;
    try {
        libjpeg_turbo::UncompressFlags flags;
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        flags.sample_format = info.sample_format;
        switch (flags.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED:
            flags.components = 3;
            break;
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
            flags.components = 1;
            break;
        default:
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unsupported sample_format: " << flags.sample_format);
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
            return;
        }
        flags.dct_method = fast_idct_ ? JDCT_FASTEST : JDCT_DEFAULT;
        flags.fancy_upscaling = fancy_upsampling_;

        if (info.region.ndim != 0 && info.region.ndim != 2) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Invalid region of interest");
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
            return;
        }
        if (info.region.ndim == 2) {
            flags.crop = true;
            flags.crop_y = info.region.start[0];
            flags.crop_x = info.region.start[1];
            flags.crop_height = info.region.end[0] - info.region.start[0];
            flags.crop_width = info.region.end[1] - info.region.start[1];

            if (flags.crop_x < 0 || flags.crop_y < 0 || flags.crop_height != static_cast<int>(info.plane_info[0].height) ||
                flags.crop_width != static_cast<int>(info.plane_info[0].width)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Region of interest is out of bounds");
                image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED);
                return;
            }
        }

        assert(ctx->encoded_stream_data_ != nullptr);
        assert(ctx->encoded_stream_data_size_ > 0);

        auto orig_sample_format = flags.sample_format;
        if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED) {
            orig_sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        } else if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) {
            orig_sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        }

        if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
            flags.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        } else if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR) {
            flags.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_BGR;
        }
        auto decoded_image = libjpeg_turbo::Uncompress(ctx->encoded_stream_data_, ctx->encoded_stream_data_size_, flags);
        if (decoded_image == nullptr) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        } else if (info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        const uint8_t* src = decoded_image.get();
        uint8_t* dst = reinterpret_cast<uint8_t*>(info.buffer);
        if (orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB || orig_sample_format == NVIMGCODEC_SAMPLEFORMAT_P_BGR) {
            const int num_channels = 3;
            uint32_t plane_size = info.plane_info[0].height * info.plane_info[0].width;
            for (uint32_t i = 0; i < info.plane_info[0].height * info.plane_info[0].width; i++) {
                *(dst + plane_size * 0 + i) = *(src + 0 + i * num_channels);
                *(dst + plane_size * 1 + i) = *(src + 1 + i * num_channels);
                *(dst + plane_size * 2 + i) = *(src + 2 + i * num_channels);
            }
        } else {
            uint32_t row_size_bytes = info.plane_info[0].width * flags.components * sizeof(uint8_t);
            for (uint32_t y = 0; y < info.plane_info[0].height; y++, dst += info.plane_info[0].row_stride, src += row_size_bytes) {
                std::memcpy(dst, src, row_size_bytes);
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return;
    }
    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
}

nvimgcodecStatus_t DecoderImpl::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libjpeg_turbo_decode_batch");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }

        auto executor = exec_params_->executor;
        XM_CHECK_NULL(executor)

        code_stream_mgr_.feedSamples(code_streams, images, batch_size, params);
        for (size_t i = 0; i < code_stream_mgr_.size(); i++) {
            code_stream_mgr_[i]->load();
        }

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
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
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

} // namespace libjpeg_turbo
