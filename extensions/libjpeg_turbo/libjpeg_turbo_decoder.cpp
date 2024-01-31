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

namespace libjpeg_turbo {

struct DecodeState
{
    DecodeState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, int num_threads)
        : plugin_id_(plugin_id)
        , framework_(framework)
        , per_thread_(num_threads)
    {
    }
    ~DecodeState() = default;

    struct PerThreadResources
    {
        std::vector<uint8_t> buffer;
    };

    struct Sample
    {
        nvimgcodecCodeStreamDesc_t* code_stream;
        nvimgcodecImageDesc_t* image;
        const nvimgcodecDecodeParams_t* params;
    };

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    std::vector<PerThreadResources> per_thread_;
    std::vector<Sample> samples_;

    // Options
    bool fancy_upsampling_;
    bool fast_idct_;
};

struct DecoderImpl
{
    DecoderImpl(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        std::string options);
    ~DecoderImpl();

    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image,
        const nvimgcodecDecodeParams_t* params);
    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);
    static nvimgcodecProcessingStatus_t decode(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework,
        nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params,
        std::vector<uint8_t>& buffer, bool fancy_upsampling = true, bool fast_idct = false);
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
    std::unique_ptr<DecodeState> decode_state_batch_;

    struct CanDecodeCtx
    {
        DecoderImpl* this_ptr;
        nvimgcodecProcessingStatus_t* status;
        nvimgcodecCodeStreamDesc_t** code_streams;
        nvimgcodecImageDesc_t** images;
        const nvimgcodecDecodeParams_t* params;
        int num_samples;
        int num_blocks;
        std::vector<std::promise<void>> promise;
    };
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

nvimgcodecStatus_t DecoderImpl::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
    nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libjpeg_turbo_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);
        *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg") != 0) {
            *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            return NVIMGCODEC_STATUS_SUCCESS;
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
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libjpeg_turbo_can_decode");
        nvtx3::scoped_range marker{"libjpeg_turbo_can_decode"};
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images);
        XM_CHECK_NULL(params);
        auto executor = exec_params_->executor;
        int num_threads = executor->getNumThreads(executor->instance);

        if (batch_size < (num_threads + 1)) {  // not worth parallelizing
            for (int i = 0; i < batch_size; i++)
                canDecode(&status[i], code_streams[i], images[i], params);
        } else {
            int num_blocks = num_threads + 1;  // the last block is processed in the current thread
            CanDecodeCtx canDecodeCtx{this, status, code_streams, images, params, batch_size, num_blocks};
            canDecodeCtx.promise.resize(num_threads);
            std::vector<std::future<void>> fut;
            fut.reserve(num_threads);
            for (auto& pr : canDecodeCtx.promise)
                fut.push_back(pr.get_future());
            auto task = [](int tid, int block_idx, void* context) -> void {
                auto* ctx = reinterpret_cast<CanDecodeCtx*>(context);
                int64_t i_start = ctx->num_samples * block_idx / ctx->num_blocks;
                int64_t i_end = ctx->num_samples * (block_idx + 1) / ctx->num_blocks;
                for (int i = i_start; i < i_end; i++) {
                    ctx->this_ptr->canDecode(&ctx->status[i], ctx->code_streams[i], ctx->images[i], ctx->params);
                }
                if (block_idx < static_cast<int>(ctx->promise.size()))
                    ctx->promise[block_idx].set_value();
            };
            int block_idx = 0;
            for (; block_idx < num_threads; ++block_idx) {
                executor->launch(executor->instance, exec_params_->device_id, block_idx, &canDecodeCtx, task);
            }
            task(-1, block_idx, &canDecodeCtx);

            // wait for it to finish
            for (auto& f : fut)
                f.wait();
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
    int num_threads = exec_params_->executor->getNumThreads(exec_params_->executor->instance);
    decode_state_batch_ = std::make_unique<DecodeState>(plugin_id_, framework_, num_threads);

    parseOptions(std::move(options));
}

void DecoderImpl::parseOptions(std::string options)
{
    // defaults
    decode_state_batch_->fancy_upsampling_ = true;
    decode_state_batch_->fast_idct_ = false;

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
            value >> decode_state_batch_->fancy_upsampling_;
        } else if (option == "fast_idct") {
            value >> decode_state_batch_->fast_idct_;
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

nvimgcodecProcessingStatus_t DecoderImpl::decode(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework,
    nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params,
    std::vector<uint8_t>& buffer, bool fancy_upsampling, bool fast_idct)
{
    try {
        libjpeg_turbo::UncompressFlags flags;
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;

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
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Unsupported sample_format: " << flags.sample_format);
            return NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }
        flags.dct_method = fast_idct ? JDCT_FASTEST : JDCT_DEFAULT;
        flags.fancy_upscaling = fancy_upsampling;

        if (info.region.ndim != 0 && info.region.ndim != 2) {
            NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Invalid region of interest");
            return NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
        }
        if (info.region.ndim == 2) {
            flags.crop = true;
            flags.crop_y = info.region.start[0];
            flags.crop_x = info.region.start[1];
            flags.crop_height = info.region.end[0] - info.region.start[0];
            flags.crop_width = info.region.end[1] - info.region.start[1];

            if (flags.crop_x < 0 || flags.crop_y < 0 || flags.crop_height != static_cast<int>(info.plane_info[0].height) ||
                flags.crop_width != static_cast<int>(info.plane_info[0].width)) {
                NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Region of interest is out of bounds");
                return NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
            }
        }

        auto io_stream = code_stream->io_stream;
        size_t data_size;
        ret = io_stream->size(io_stream->instance, &data_size);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        void* ptr;
        ret = io_stream->map(io_stream->instance, &ptr, 0, data_size);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
        auto auto_unmap =
            std::shared_ptr<void>(ptr, [io_stream, data_size](void* addr) { io_stream->unmap(io_stream->instance, addr, data_size); });

        const uint8_t* encoded_data = static_cast<const uint8_t*>(ptr);
        if (!ptr && data_size > 0) {
            buffer.resize(data_size);
            size_t read_nbytes = 0;
            io_stream->seek(io_stream->instance, 0, SEEK_SET);
            ret = io_stream->read(io_stream->instance, &read_nbytes, buffer.data(), buffer.size());
            if (ret != NVIMGCODEC_STATUS_SUCCESS)
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            if (read_nbytes != buffer.size()) {
                return NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED;
            }
            encoded_data = buffer.data();
        }

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
        auto decoded_image = libjpeg_turbo::Uncompress(encoded_data, data_size, flags);
        if (decoded_image == nullptr) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        } else if (info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
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
        NVIMGCODEC_LOG_ERROR(framework, plugin_id, "Could not decode jpeg code stream - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
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
        decode_state_batch_->samples_.resize(batch_size);
        for (int i = 0; i < batch_size; i++) {
            decode_state_batch_->samples_[i].code_stream = code_streams[i];
            decode_state_batch_->samples_[i].image = images[i];
            decode_state_batch_->samples_[i].params = params;
        }

        auto task = [](int tid, int sample_idx, void* context) -> void {
            nvtx3::scoped_range marker{"libjpeg_turbo decode " + std::to_string(sample_idx)};
            auto* decode_state = reinterpret_cast<DecodeState*>(context);
            auto& sample = decode_state->samples_[sample_idx];
            auto& thread_resources = decode_state->per_thread_[tid];
            auto& plugin_id = decode_state->plugin_id_;
            auto& framework = decode_state->framework_;
            auto result = decode(plugin_id, framework, sample.code_stream, sample.image, sample.params, thread_resources.buffer,
                decode_state->fancy_upsampling_, decode_state->fast_idct_);
            sample.image->imageReady(sample.image->instance, result);
        };
        if (batch_size == 1) {
            task(0, 0, decode_state_batch_.get());
        } else {
            auto executor = exec_params_->executor;
            for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
                executor->launch(executor->instance, NVIMGCODEC_DEVICE_CPU_ONLY, sample_idx, decode_state_batch_.get(), task);
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
