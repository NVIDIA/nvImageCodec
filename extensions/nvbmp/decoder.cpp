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

#include <cuda_runtime_api.h>
#include <nvimgcodec.h>
#include <cstring>
#include <memory>
#include <vector>

#include <nvtx3/nvtx3.hpp>

#include "decoder.h"
#include "error_handling.h"
#include "log.h"
#include "../utils/stream_ctx.h"
#include "../utils/parallel_exec.h"

namespace nvbmp {

struct DecoderImpl
{
    DecoderImpl(
        const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params);
    ~DecoderImpl();

    nvimgcodecStatus_t canDecodeImpl(CodeStreamCtx& ctx);
    nvimgcodecStatus_t canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
        nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params);

    void decodeImpl(BatchItemCtx& batch_item, int tid);
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

    struct PerThreadResources {
        std::vector<uint8_t> buffer;
    };
    std::vector<PerThreadResources> per_thread_;
    CodeStreamCtxManager code_stream_mgr_;
};

NvBmpDecoderPlugin::NvBmpDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "bmp", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, static_create,
          DecoderImpl::static_destroy, DecoderImpl::static_can_decode, DecoderImpl::static_decode_batch}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvBmpDecoderPlugin::getDecoderDesc()
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

        bool is_bmp = strcmp(cs_image_info.codec_name, "bmp") == 0;

        for (size_t i = 0; i < ctx.size(); i++) {
            auto *status = &ctx.batch_items_[i]->processing_status;
            auto *image = ctx.batch_items_[i]->image;
            const auto *params = ctx.batch_items_[i]->params;

            XM_CHECK_NULL(status);
            XM_CHECK_NULL(image);
            XM_CHECK_NULL(params);

            *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;

            if (!is_bmp) {
                *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                continue;
            }

            if (params->enable_roi) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
            }

            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            image->getImageInfo(image->instance, &image_info);
            if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
            }
            if (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
            }
            if ((image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_RGB)) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            }
            if ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.num_planes != 3)) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            if ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) && (image_info.num_planes != 1)) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }

            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                if (image_info.plane_info[p].sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                    *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
                }

                if ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) && (image_info.plane_info[p].num_channels != 1)) {
                    *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }

                if ((image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) && (image_info.plane_info[p].num_channels != 3)) {
                    *status |= NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvbmp can decode - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t DecoderImpl::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "libtiff_can_decode");
        nvtx3::scoped_range marker{"nvbmp_can_decode"};
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
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvbmp can decode - " << e.what());
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
    auto  executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);
    per_thread_.resize(num_threads);
}

nvimgcodecStatus_t NvBmpDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvbmp_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new DecoderImpl(plugin_id_, framework_, exec_params));
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvbmp decoder - " << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvBmpDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        auto handle = reinterpret_cast<NvBmpDecoderPlugin*>(instance);
        handle->create(decoder, exec_params, options);
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

DecoderImpl::~DecoderImpl()
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvbmp_destroy");
}

nvimgcodecStatus_t DecoderImpl::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        auto handle = reinterpret_cast<DecoderImpl*>(decoder);
        delete handle;
    } catch (const std::runtime_error& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    return NVIMGCODEC_STATUS_SUCCESS;
}



void DecoderImpl::decodeImpl(BatchItemCtx& batch_item, int tid)
{
    NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "decode");
    nvtx3::scoped_range marker{"nvbmp decode " + std::to_string(batch_item.index)};
    auto *image = batch_item.image;
    auto *io_stream = batch_item.code_stream_ctx->code_stream_->io_stream;
    try {
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            return;
        }

        size_t size = 0;
        size_t output_size = 0;
        io_stream->size(io_stream->instance, &size);
        auto& buffer = per_thread_[tid].buffer;
        buffer.resize(size);
        static constexpr int kHeaderStart = 14;
        io_stream->seek(io_stream->instance, kHeaderStart, SEEK_SET);
        uint32_t header_size;
        io_stream->read(io_stream->instance, &output_size, &header_size, sizeof(header_size));
        io_stream->seek(io_stream->instance, 0, SEEK_SET);
        io_stream->read(io_stream->instance, &output_size, &buffer[0], size);
        if (output_size != size) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return;
        }

        unsigned char* host_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_P_RGB) {
            for (size_t p = 0; p < image_info.num_planes; p++) {
                for (size_t y = 0; y < image_info.plane_info[p].height; y++) {
                    for (size_t x = 0; x < image_info.plane_info[p].width; x++) {
                        host_buffer[(image_info.num_planes - p - 1) * image_info.plane_info[p].height * image_info.plane_info[p].width +
                                    (image_info.plane_info[p].height - y - 1) * image_info.plane_info[p].width + x] =
                            buffer[kHeaderStart + header_size + image_info.num_planes * (y * image_info.plane_info[p].width + x) + p];
                    }
                }
            }
        } else if (image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_RGB) {
            for (size_t c = 0; c < image_info.plane_info[0].num_channels; c++) {
                for (size_t y = 0; y < image_info.plane_info[0].height; y++) {
                    for (size_t x = 0; x < image_info.plane_info[0].width; x++) {
                        auto src_idx = kHeaderStart + header_size +
                                       image_info.plane_info[0].num_channels * (y * image_info.plane_info[0].width + x) + c;
                        auto dst_idx = (image_info.plane_info[0].height - y - 1) * image_info.plane_info[0].width *
                                           image_info.plane_info[0].num_channels +
                                       x * image_info.plane_info[0].num_channels + (image_info.plane_info[0].num_channels - c - 1);
                        host_buffer[dst_idx] = buffer[src_idx];
                    }
                }
            }
        } else {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED);
            return;
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode bmp code stream - " << e.what());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return;
    }
    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
    return;
}

nvimgcodecStatus_t DecoderImpl::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvbmp_decode_batch");
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

        // The io stream can be used directly by libtiff, so we make sure that only one thread access a given io stream
        // by grouping samples per code stream
        auto task = [](int tid, int stream_idx, void* context) -> void {
            auto* this_ptr = reinterpret_cast<DecoderImpl*>(context);
            auto stream_ctx = this_ptr->code_stream_mgr_[stream_idx];
            for (auto* batch_item : stream_ctx->batch_items_) {
                this_ptr->decodeImpl(*batch_item, tid);
            }
        };

        if (code_stream_mgr_.size() == 1) {
            task(0, 0, this);
        } else {
            auto executor = exec_params_->executor;
            for (size_t stream_idx = 0; stream_idx < code_stream_mgr_.size(); stream_idx++) {
                executor->launch(executor->instance, NVIMGCODEC_DEVICE_CPU_ONLY, stream_idx, this, task);
            }
        }
    } catch (const std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode bmp batch - " << e.what());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
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

} // namespace nvbmp
