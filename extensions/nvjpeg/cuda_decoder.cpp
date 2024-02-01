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

#include "cuda_decoder.h"
#include <library_types.h>
#include <nvimgcodec.h>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <vector>
#include "errors_handling.h"
#include "log.h"
#include "nvjpeg_utils.h"
#include "type_convert.h"

#if WITH_DYNAMIC_NVJPEG_ENABLED
    #include "dynlink/dynlink_nvjpeg.h"
#else
    #define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

NvJpegCudaDecoderPlugin::NvJpegCudaDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
          static_create, Decoder::static_destroy, Decoder::static_can_decode, Decoder::static_decode_batch}
    , framework_(framework)
{}

nvimgcodecDecoderDesc_t* NvJpegCudaDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t* code_stream,
    nvimgcodecImageDesc_t* image, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_can_decode");
        XM_CHECK_NULL(status);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(params);
        *status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        nvimgcodecJpegImageInfo_t jpeg_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&jpeg_info)};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        if (strcmp(cs_image_info.codec_name, "jpeg") != 0) {
            *status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        nvimgcodecJpegImageInfo_t* jpeg_image_info = static_cast<nvimgcodecJpegImageInfo_t*>(cs_image_info.struct_next);
        while (jpeg_image_info && jpeg_image_info->struct_type != NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO)
            jpeg_image_info = static_cast<nvimgcodecJpegImageInfo_t*>(jpeg_image_info->struct_next);
        if (jpeg_image_info) {
            static const std::set<nvimgcodecJpegEncoding_t> supported_encoding{NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT,
                NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN};
            if (supported_encoding.find(jpeg_image_info->encoding) == supported_encoding.end()) {
                *status = NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                return NVIMGCODEC_STATUS_SUCCESS;
            }
        }

        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &image_info);
        static const std::set<nvimgcodecColorSpec_t> supported_color_space{NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB,
            NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC, NVIMGCODEC_COLORSPEC_CMYK, NVIMGCODEC_COLORSPEC_YCCK};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            *status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
            NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY,
            NVIMGCODEC_SAMPLING_410V};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }
        static const std::set<nvimgcodecSampleFormat_t> supported_sample_format{
            NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB,
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_BGR,
            NVIMGCODEC_SAMPLEFORMAT_I_BGR,
            NVIMGCODEC_SAMPLEFORMAT_P_Y,
            NVIMGCODEC_SAMPLEFORMAT_P_YUV,
        };
        if (supported_sample_format.find(image_info.sample_format) == supported_sample_format.end()) {
            *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                *status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg can decode - " << e.info());
        *status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::canDecode(nvimgcodecProcessingStatus_t* status, nvimgcodecCodeStreamDesc_t** code_streams,
    nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_can_decode");
        nvtx3::scoped_range marker{"nvjpeg_can_decode"};
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
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg can decode - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::static_can_decode(nvimgcodecDecoder_t decoder, nvimgcodecProcessingStatus_t* status,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        auto handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        return handle->canDecode(status, code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}

void NvJpegCudaDecoderPlugin::Decoder::parseOptions(const char* options)
{
    gpu_hybrid_huffman_threshold_ = DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD;
    std::istringstream iss(options ? options : "");
    std::string token;
    while (std::getline(iss, token, ' ')) {
        std::string::size_type colon = token.find(':');
        std::string::size_type equal = token.find('=');
        if (colon == std::string::npos || equal == std::string::npos || colon > equal)
            continue;
        std::string module = token.substr(0, colon);
        if (module != "" && module != "nvjpeg_cuda_decoder")
            continue;
        std::string option = token.substr(colon + 1, equal - colon - 1);
        std::string value_str = token.substr(equal + 1);

        std::istringstream value(value_str);
        if (option == "hybrid_huffman_threshold") {
            value >> gpu_hybrid_huffman_threshold_;
        }
    }
}

NvJpegCudaDecoderPlugin::Decoder::Decoder(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
{
    bool use_nvjpeg_create_ex_v2 = false;
    if (nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
        if (exec_params_->device_allocator && exec_params_->device_allocator->device_malloc &&
            exec_params_->device_allocator->device_free) {
            device_allocator_.dev_ctx = exec_params_->device_allocator->device_ctx;
            device_allocator_.dev_malloc = exec_params_->device_allocator->device_malloc;
            device_allocator_.dev_free = exec_params_->device_allocator->device_free;
            if (exec_params_->device_allocator->device_mem_padding > 0)
                device_mem_padding_ = exec_params_->device_allocator->device_mem_padding;
        }

        if (exec_params_->pinned_allocator && exec_params_->pinned_allocator->pinned_malloc &&
            exec_params_->pinned_allocator->pinned_free) {
            pinned_allocator_.pinned_ctx = exec_params_->pinned_allocator->pinned_ctx;
            pinned_allocator_.pinned_malloc = exec_params_->pinned_allocator->pinned_malloc;
            pinned_allocator_.pinned_free = exec_params_->pinned_allocator->pinned_free;
            if (exec_params_->pinned_allocator->pinned_mem_padding > 0)
                pinned_mem_padding_ = exec_params_->pinned_allocator->pinned_mem_padding;
        }
        use_nvjpeg_create_ex_v2 =
            device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free;
    }

    unsigned int nvjpeg_flags = get_nvjpeg_flags("nvjpeg_cuda_decoder", options);
    parseOptions(options);

    if (use_nvjpeg_create_ex_v2) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, nvjpeg_flags, &handle_));
    }

    if (exec_params_->device_allocator && (exec_params_->device_allocator->device_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(exec_params_->device_allocator->device_mem_padding, handle_));
    }
    if (exec_params_->pinned_allocator && (exec_params_->pinned_allocator->pinned_mem_padding != 0)) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(exec_params_->pinned_allocator->pinned_mem_padding, handle_));
    }

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance);

    decode_state_batch_ = std::make_unique<NvJpegCudaDecoderPlugin::DecodeState>(plugin_id_, framework_, handle_, &device_allocator_,
        device_mem_padding_, &pinned_allocator_, pinned_mem_padding_, num_threads, gpu_hybrid_huffman_threshold_);
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        if (exec_params->device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;

        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new NvJpegCudaDecoderPlugin::Decoder(plugin_id_, framework_, exec_params, options));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder - " << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        XM_CHECK_NULL(instance);
        NvJpegCudaDecoderPlugin* handle = reinterpret_cast<NvJpegCudaDecoderPlugin*>(instance);
        handle->create(decoder, exec_params, options);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvJpegCudaDecoderPlugin::Decoder::~Decoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_destroy");
        decode_state_batch_.reset();
        if (handle_)
            XM_NVJPEG_D_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.info());
    }
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegCudaDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        delete handle;
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

NvJpegCudaDecoderPlugin::DecodeState::DecodeState(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, nvjpegHandle_t handle,
    nvjpegDevAllocatorV2_t* device_allocator, size_t device_mem_padding, nvjpegPinnedAllocatorV2_t* pinned_allocator,
    size_t pinned_mem_padding, int num_threads, size_t gpu_hybrid_huffman_threshold)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , handle_(handle)
    , device_allocator_(device_allocator)
    , device_mem_padding_(device_mem_padding)
    , pinned_allocator_(pinned_allocator)
    , pinned_mem_padding_(pinned_mem_padding)
    , gpu_hybrid_huffman_threshold_(gpu_hybrid_huffman_threshold)
{
    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        const int npages = res.pages_.size();

        XM_CHECK_CUDA(cudaStreamCreateWithFlags(&res.stream_, cudaStreamNonBlocking));
        XM_CHECK_CUDA(cudaEventCreate(&res.event_));

        for (int page_idx = 0; page_idx < npages; page_idx++) {
            auto& p = res.pages_[page_idx];
            for (auto backend : {NVJPEG_BACKEND_HYBRID, NVJPEG_BACKEND_GPU_HYBRID}) {
                auto& decoder = p.decoder_data[backend].decoder;
                auto& state = p.decoder_data[backend].state;
                XM_CHECK_NVJPEG(nvjpegDecoderCreate(handle_, backend, &decoder));
                XM_CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder, &state));
            }
            if (pinned_allocator_ && pinned_allocator_->pinned_malloc && pinned_allocator_->pinned_free) {
                XM_CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(handle, pinned_allocator_, &p.pinned_buffer_));
#if NVJPEG_BUFFER_RESIZE_API
                if (pinned_mem_padding_ > 0 && nvjpegIsSymbolAvailable("nvjpegBufferPinnedResize")) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                        "Preallocating pinned buffer (thread#" << i << " page#" << page_idx << ") size=" << pinned_mem_padding_);
                    XM_CHECK_NVJPEG(nvjpegBufferPinnedResize(p.pinned_buffer_, pinned_mem_padding_, res.stream_));
                }
#endif
            } else {
                XM_CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle, nullptr, &p.pinned_buffer_));
            }
            XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &p.parse_state_.nvjpeg_stream_));
        }
        if (device_allocator_ && device_allocator_->dev_malloc && device_allocator_->dev_free) {
            XM_CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(handle, device_allocator_, &res.device_buffer_));
#if NVJPEG_BUFFER_RESIZE_API
        if (pinned_mem_padding_ > 0 && nvjpegIsSymbolAvailable("nvjpegBufferDeviceResize")) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Preallocating device buffer (thread#" << i << ") size=" << device_mem_padding_);
            XM_CHECK_NVJPEG(nvjpegBufferDeviceResize(res.device_buffer_, device_mem_padding_, res.stream_));
        }
#endif
        } else {
            XM_CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle, nullptr, &res.device_buffer_));
        }
    }
}

NvJpegCudaDecoderPlugin::DecodeState::~DecodeState()
{
    for (auto& res : per_thread_) {
        const int npages = res.pages_.size();
        for (int page_idx = 0; page_idx < npages; page_idx++) {
            auto& p = res.pages_[page_idx];
            if (p.parse_state_.nvjpeg_stream_) {
                XM_NVJPEG_D_LOG_DESTROY(nvjpegJpegStreamDestroy(p.parse_state_.nvjpeg_stream_));
            }
            if (p.pinned_buffer_) {
                XM_NVJPEG_D_LOG_DESTROY(nvjpegBufferPinnedDestroy(p.pinned_buffer_));
            }
            for (auto& decoder_data : p.decoder_data) {
                if (decoder_data.state) {
                    XM_NVJPEG_D_LOG_DESTROY(nvjpegJpegStateDestroy(decoder_data.state));
                }
                if (decoder_data.decoder) {
                    XM_NVJPEG_D_LOG_DESTROY(nvjpegDecoderDestroy(decoder_data.decoder));
                }
            }
        }
        if (res.device_buffer_) {
            XM_NVJPEG_D_LOG_DESTROY(nvjpegBufferDeviceDestroy(res.device_buffer_));
        }
        if (res.event_) {
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
        }
        if (res.stream_) {
            XM_CUDA_LOG_DESTROY(cudaStreamDestroy(res.stream_));
        }
    }
    per_thread_.clear();
    samples_.clear();
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::decode(int sample_idx, bool immediate)
{
    auto task = [](int tid, int sample_idx, void* context) -> void {
        nvtx3::scoped_range marker{"nvjpeg_cuda decode " + std::to_string(sample_idx)};
        auto* decode_state = reinterpret_cast<NvJpegCudaDecoderPlugin::DecodeState*>(context);
        nvimgcodecCodeStreamDesc_t* code_stream = decode_state->samples_[sample_idx].code_stream;
        nvimgcodecIoStreamDesc_t* io_stream = code_stream->io_stream;
        nvimgcodecImageDesc_t* image = decode_state->samples_[sample_idx].image;
        const nvimgcodecDecodeParams_t* params = decode_state->samples_[sample_idx].params;
        auto& handle = decode_state->handle_;
        auto& framework_ = decode_state->framework_;
        auto& plugin_id_ = decode_state->plugin_id_;
        auto& t = decode_state->per_thread_[tid];
        t.current_page_idx = (t.current_page_idx + 1) % 2;
        int page_idx = t.current_page_idx;
        auto& p = t.pages_[page_idx];
        try {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            image->getImageInfo(image->instance, &image_info);
            unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

            nvjpegDecodeParams_t nvjpeg_params_;
            XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle, &nvjpeg_params_));
            std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params(
                nvjpeg_params_, &nvjpegDecodeParamsDestroy);
            nvjpegOutputFormat_t nvjpeg_format = nvimgcodec_to_nvjpeg_format(image_info.sample_format);
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetOutputFormat(nvjpeg_params.get(), nvjpeg_format));
            int allow_cmyk = (image_info.color_spec != NVIMGCODEC_COLORSPEC_UNCHANGED) &&
                             (image_info.color_spec != NVIMGCODEC_COLORSPEC_CMYK) && ((image_info.color_spec != NVIMGCODEC_COLORSPEC_YCCK));
            XM_CHECK_NVJPEG(nvjpegDecodeParamsSetAllowCMYK(nvjpeg_params.get(), allow_cmyk));

            if (params->apply_exif_orientation) {
                nvjpegExifOrientation_t orientation = nvimgcodec_to_nvjpeg_orientation(image_info.orientation);

                // This is a workaround for a known bug in nvjpeg.
                if (!nvjpeg_at_least(12, 2, 0)) {
                    if (orientation == NVJPEG_ORIENTATION_ROTATE_90)
                        orientation = NVJPEG_ORIENTATION_ROTATE_270;
                    else if (orientation == NVJPEG_ORIENTATION_ROTATE_270)
                        orientation = NVJPEG_ORIENTATION_ROTATE_90;
                }

                if (orientation == NVJPEG_ORIENTATION_UNKNOWN) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                    return;
                }

                if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                    if (!nvjpegIsSymbolAvailable("nvjpegDecodeParamsSetExifOrientation")) {
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                        return;
                    }
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Setting up EXIF orientation " << orientation);
                    XM_CHECK_NVJPEG(nvjpegDecodeParamsSetExifOrientation(nvjpeg_params.get(), orientation));
                }
            }

            if (params->enable_roi && image_info.region.ndim > 0) {
                auto region = image_info.region;
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                    "Setting up ROI :" << region.start[0] << ", " << region.start[1] << ", " << region.end[0] << ", " << region.end[1]);
                auto roi_width = region.end[1] - region.start[1];
                auto roi_height = region.end[0] - region.start[0];
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), region.start[1], region.start[0], roi_width, roi_height));
            } else {
                XM_CHECK_NVJPEG(nvjpegDecodeParamsSetROI(nvjpeg_params.get(), 0, 0, -1, -1));
            }

            size_t encoded_stream_data_size = 0;
            io_stream->size(io_stream->instance, &encoded_stream_data_size);
            void* encoded_stream_data = nullptr;
            void* mapped_encoded_stream_data = nullptr;
            io_stream->map(io_stream->instance, &mapped_encoded_stream_data, 0, encoded_stream_data_size);
            if (!mapped_encoded_stream_data) {
                if (p.parse_state_.buffer_.size() != encoded_stream_data_size) {
                    p.parse_state_.buffer_.resize(encoded_stream_data_size);
                    io_stream->seek(io_stream->instance, 0, SEEK_SET);
                    size_t read_nbytes = 0;
                    io_stream->read(io_stream->instance, &read_nbytes, &p.parse_state_.buffer_[0], encoded_stream_data_size);
                    if (read_nbytes != encoded_stream_data_size) {
                        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected end-of-stream");
                        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                        return;
                    }
                }
                encoded_stream_data = &p.parse_state_.buffer_[0];
            } else {
                encoded_stream_data = mapped_encoded_stream_data;
            }
            {
                nvtx3::scoped_range marker{"nvjpegJpegStreamParse"};
                XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle, static_cast<const unsigned char*>(encoded_stream_data),
                    encoded_stream_data_size, false, false, p.parse_state_.nvjpeg_stream_));
            }
            if (!mapped_encoded_stream_data) {
                io_stream->unmap(io_stream->instance, encoded_stream_data, encoded_stream_data_size);
            }
            nvjpegJpegEncoding_t jpeg_encoding;
            nvjpegJpegStreamGetJpegEncoding(p.parse_state_.nvjpeg_stream_, &jpeg_encoding);

            int is_gpu_hybrid_supported = -1;                    // zero means is supported
            if (jpeg_encoding == NVJPEG_ENCODING_BASELINE_DCT) { //gpu hybrid is not supported for progressive
                XM_CHECK_NVJPEG(nvjpegDecoderJpegSupported(p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID].decoder, p.parse_state_.nvjpeg_stream_,
                    nvjpeg_params.get(), &is_gpu_hybrid_supported));
            }

            bool is_gpu_hybrid =
                (image_info.plane_info[0].height * image_info.plane_info[0].width) > decode_state->gpu_hybrid_huffman_threshold_ &&
                is_gpu_hybrid_supported == 0;
            auto& decoder_data = is_gpu_hybrid ? p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID] : p.decoder_data[NVJPEG_BACKEND_HYBRID];
            auto& decoder = decoder_data.decoder;
            auto& state = decoder_data.state;

            XM_CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(state, p.pinned_buffer_));

            {
                nvtx3::scoped_range marker{"nvjpegDecodeJpegHost (is_gpu_hybrid=" + std::to_string(is_gpu_hybrid) + ")"};
                XM_CHECK_NVJPEG(nvjpegDecodeJpegHost(handle, decoder, state, nvjpeg_params.get(), p.parse_state_.nvjpeg_stream_));
            }

            nvjpegImage_t nvjpeg_image;
            unsigned char* ptr = device_buffer;
            for (uint32_t c = 0; c < image_info.num_planes; ++c) {
                nvjpeg_image.channel[c] = ptr;
                nvjpeg_image.pitch[c] = image_info.plane_info[c].row_stride;
                ptr += nvjpeg_image.pitch[c] * image_info.plane_info[c].height;
            }
            // Waits for GPU stage from previous iteration (on this thread)
            XM_CHECK_CUDA(cudaEventSynchronize(t.event_));

            XM_CHECK_NVJPEG(nvjpegStateAttachDeviceBuffer(state, t.device_buffer_));

            XM_CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle, decoder, state, p.parse_state_.nvjpeg_stream_, t.stream_));

            {
                nvtx3::scoped_range marker{"nvjpegDecodeJpegDevice"};
                XM_CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle, decoder, state, &nvjpeg_image, t.stream_));
            }

            // this captures the state of t.stream_ in the cuda event t.event_
            XM_CHECK_CUDA(cudaEventRecord(t.event_, t.stream_));
            // this is so that any post processing on image waits for t.event_ i.e. decoding to finish,
            // without this the post-processing tasks such as encoding, would not know that decoding has finished on this
            // particular image
            XM_CHECK_CUDA(cudaStreamWaitEvent(image_info.cuda_stream, t.event_));

            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        } catch (const NvJpegException& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.info());
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
    };

    if (immediate) {
        task(0, sample_idx, decode_state_batch_.get());
    } else {
        auto executor = exec_params_->executor;
        executor->launch(executor->instance, exec_params_->device_id, sample_idx, decode_state_batch_.get(), std::move(task));
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::decodeBatch(
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        nvtx3::scoped_range marker{"nvjpeg cuda decodeBatch"};
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_decode_batch, " << batch_size << " samples");
        XM_CHECK_NULL(code_streams);
        XM_CHECK_NULL(images)
        XM_CHECK_NULL(params)
        if (batch_size < 1) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Batch size lower than 1");
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        decode_state_batch_->samples_.clear();
        for (int sample_idx = 0; sample_idx < batch_size; sample_idx++) {
            decode_state_batch_->samples_.push_back(
                NvJpegCudaDecoderPlugin::DecodeState::Sample{code_streams[sample_idx], images[sample_idx], params});
        }

        nvjpegDecodeParams_t nvjpeg_params;
        XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &nvjpeg_params));
        std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params_raii(
            nvjpeg_params, &nvjpegDecodeParamsDestroy);

        int nsamples = decode_state_batch_->samples_.size();
        bool immediate = nsamples == 1; //  if single image, do not use executor
        for (int i = 0; i < nsamples; i++)
            this->decode(i, immediate);
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg batch - " << e.info());
        for (int i = 0; i < batch_size; ++i) {
            images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::Decoder::static_decode_batch(nvimgcodecDecoder_t decoder,
    nvimgcodecCodeStreamDesc_t** code_streams, nvimgcodecImageDesc_t** images, int batch_size, const nvimgcodecDecodeParams_t* params)
{
    try {
        XM_CHECK_NULL(decoder);
        NvJpegCudaDecoderPlugin::Decoder* handle = reinterpret_cast<NvJpegCudaDecoderPlugin::Decoder*>(decoder);
        return handle->decodeBatch(code_streams, images, batch_size, params);
    } catch (const NvJpegException& e) {
        return e.nvimgcodecStatus();
    }
}
} // namespace nvjpeg
