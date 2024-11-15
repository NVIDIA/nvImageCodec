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
#define NOMINMAX
#include "cuda_decoder.h"
#include <library_types.h>
#include <nvimgcodec.h>
#include <cassert>
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

#define DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD 1000u * 1000u

#if WITH_DYNAMIC_NVJPEG_ENABLED
    #include "dynlink/dynlink_nvjpeg.h"
#else
    #define nvjpegIsSymbolAvailable(T) (true)
#endif

namespace nvjpeg {

struct Decoder
{
    Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        const char* options = nullptr);
    ~Decoder();

    nvimgcodecProcessingStatus_t canDecode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->canDecode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
    }

    nvimgcodecStatus_t decode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecStatus_t static_decode_sample(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->decode(image, code_stream, params, thread_idx);
        } catch (const std::runtime_error& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            ;
        }
    }

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder)
    {
        try {
            XM_CHECK_NULL(decoder);
            Decoder* handle = reinterpret_cast<Decoder*>(decoder);
            delete handle;
        } catch (const NvJpegException& e) {
            return e.nvimgcodecStatus();
        }
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    void parseOptions(const char* options);

    const char* plugin_id_;
    nvjpegHandle_t handle_;
    nvjpegDevAllocatorV2_t device_allocator_;
    nvjpegPinnedAllocatorV2_t pinned_allocator_;
    const nvimgcodecFrameworkDesc_t* framework_;

    struct ParseState
    {
        std::optional<uint64_t> parsed_stream_id_;
        nvjpegJpegStream_t nvjpeg_stream_;
    };

    struct DecoderData
    {
        nvjpegJpegDecoder_t decoder = nullptr;
        nvjpegJpegState_t state = nullptr;
    };

    // Set of resources per-thread.
    // Some of them are double-buffered, so that we can simultaneously decode
    // the host part of the next sample, while the GPU part of the previous
    // is still consuming the data from the previous iteration pinned buffer.
    struct PerThreadResources
    {

        // double-buffered
        struct Page
        {
            // indexing via nvjpegBackend_t (NVJPEG_BACKEND_GPU_HYBRID and NVJPEG_BACKEND_HYBRID)
            std::array<DecoderData, 3> decoder_data;
            nvjpegBufferPinned_t pinned_buffer_;
            ParseState parse_state_;
        };
        std::array<Page, 2> pages_;
        int current_page_idx = 0;
        cudaEvent_t event_;
        nvjpegBufferDevice_t device_buffer_;
    };
    std::vector<PerThreadResources> per_thread_;

    const nvimgcodecExecutionParams_t* exec_params_;
    size_t gpu_hybrid_huffman_threshold_ = DEFAULT_GPU_HYBRID_HUFFMAN_THRESHOLD;
    bool preallocate_buffers_ = true;
    std::optional<size_t> device_mem_padding_;
    std::optional<size_t> pinned_mem_padding_;
};

NvJpegCudaDecoderPlugin::NvJpegCudaDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), NULL, this, plugin_id_, "jpeg",
          NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, static_create, Decoder::static_destroy, Decoder::static_can_decode,
          Decoder::static_decode_sample, nullptr, nullptr}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvJpegCudaDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

void Decoder::parseOptions(const char* options)
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
        } else if (option == "device_memory_padding") {
            size_t padding_value = 0;
            value >> padding_value;
            device_mem_padding_ = padding_value;
        } else if (option == "host_memory_padding") {
            size_t padding_value = 0;
            value >> padding_value;
            pinned_mem_padding_ = padding_value;
        } else if (option == "preallocate_buffers") {
            value >> preallocate_buffers_;
        }
    }
}

Decoder::Decoder(
    const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : plugin_id_(plugin_id)
    , device_allocator_{nullptr, nullptr, nullptr}
    , pinned_allocator_{nullptr, nullptr, nullptr}
    , framework_(framework)
    , exec_params_(exec_params)
{
    parseOptions(options);

    if (!device_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->device_allocator->device_mem_padding != 0)
        device_mem_padding_ = exec_params_->device_allocator->device_mem_padding;
    if (!pinned_mem_padding_.has_value() && exec_params_->device_allocator && exec_params_->pinned_allocator->pinned_mem_padding != 0)
        pinned_mem_padding_ = exec_params_->pinned_allocator->pinned_mem_padding;

    bool use_nvjpeg_create_ex_v2 = false;
    if (nvjpegIsSymbolAvailable("nvjpegCreateExV2")) {
        if (exec_params_->device_allocator && exec_params_->device_allocator->device_malloc &&
            exec_params_->device_allocator->device_free) {
            device_allocator_.dev_ctx = exec_params_->device_allocator->device_ctx;
            device_allocator_.dev_malloc = exec_params_->device_allocator->device_malloc;
            device_allocator_.dev_free = exec_params_->device_allocator->device_free;
        } else {
            device_mem_padding_ = 0;
        }

        if (exec_params_->pinned_allocator && exec_params_->pinned_allocator->pinned_malloc &&
            exec_params_->pinned_allocator->pinned_free) {
            pinned_allocator_.pinned_ctx = exec_params_->pinned_allocator->pinned_ctx;
            pinned_allocator_.pinned_malloc = exec_params_->pinned_allocator->pinned_malloc;
            pinned_allocator_.pinned_free = exec_params_->pinned_allocator->pinned_free;
        } else {
            device_mem_padding_ = 0;
        }
        use_nvjpeg_create_ex_v2 =
            device_allocator_.dev_malloc && device_allocator_.dev_free && pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free;
    }

    unsigned int nvjpeg_flags = get_nvjpeg_flags("nvjpeg_cuda_decoder", options);

    if (use_nvjpeg_create_ex_v2) {
        XM_CHECK_NVJPEG(nvjpegCreateExV2(NVJPEG_BACKEND_DEFAULT, &device_allocator_, &pinned_allocator_, nvjpeg_flags, &handle_));
    } else {
        XM_CHECK_NVJPEG(nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, nullptr, nullptr, nvjpeg_flags, &handle_));
    }

    if (device_mem_padding_.has_value() && device_mem_padding_.value() > 0) {
        XM_CHECK_NVJPEG(nvjpegSetDeviceMemoryPadding(device_mem_padding_.value(), handle_));
    }
    if (pinned_mem_padding_.has_value() && pinned_mem_padding_.value() > 0) {
        XM_CHECK_NVJPEG(nvjpegSetPinnedMemoryPadding(pinned_mem_padding_.value(), handle_));
    }

    auto executor = exec_params_->executor;
    int num_threads = executor->getNumThreads(executor->instance) + 1; // +1 for the caller thread, which can also run decoding

    per_thread_.reserve(num_threads);
    for (int i = 0; i < num_threads; i++) {
        per_thread_.emplace_back();
        auto& res = per_thread_.back();
        const int npages = res.pages_.size();

        XM_CHECK_CUDA(cudaEventCreate(&res.event_));

        for (int page_idx = 0; page_idx < npages; page_idx++) {
            auto& p = res.pages_[page_idx];
            for (auto backend : {NVJPEG_BACKEND_HYBRID, NVJPEG_BACKEND_GPU_HYBRID}) {
                auto& decoder = p.decoder_data[backend].decoder;
                auto& state = p.decoder_data[backend].state;
                XM_CHECK_NVJPEG(nvjpegDecoderCreate(handle_, backend, &decoder));
                XM_CHECK_NVJPEG(nvjpegDecoderStateCreate(handle_, decoder, &state));
            }
            if (pinned_allocator_.pinned_malloc && pinned_allocator_.pinned_free) {
                XM_CHECK_NVJPEG(nvjpegBufferPinnedCreateV2(handle_, &pinned_allocator_, &p.pinned_buffer_));
#if NVJPEG_BUFFER_RESIZE_API
                if (preallocate_buffers_ && pinned_mem_padding_ > 0) {
                    if (nvjpegIsSymbolAvailable("nvjpegBufferPinnedResize")) {
                        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_,
                            "Preallocating pinned buffer (thread#" << i << " page#" << page_idx
                                                                   << ") size=" << pinned_mem_padding_.value());
                        XM_CHECK_NVJPEG(nvjpegBufferPinnedResize(p.pinned_buffer_, pinned_mem_padding_.value(), res.stream_));
                    } else {
                        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "nvjpegBufferPinnedResize not available. Skip preallocation");
                    }
                }
#endif
            } else {
                XM_CHECK_NVJPEG(nvjpegBufferPinnedCreate(handle_, nullptr, &p.pinned_buffer_));
            }
            XM_CHECK_NVJPEG(nvjpegJpegStreamCreate(handle_, &p.parse_state_.nvjpeg_stream_));
        }
        if (device_allocator_.dev_malloc && device_allocator_.dev_free) {
            XM_CHECK_NVJPEG(nvjpegBufferDeviceCreateV2(handle_, &device_allocator_, &res.device_buffer_));
#if NVJPEG_BUFFER_RESIZE_API
            if (preallocate_buffers_ && pinned_mem_padding_ > 0) {
                if (nvjpegIsSymbolAvailable("nvjpegBufferDeviceResize")) {
                    NVIMGCODEC_LOG_DEBUG(
                        framework_, plugin_id_, "Preallocating device buffer (thread#" << i << ") size=" << device_mem_padding_.value());
                    XM_CHECK_NVJPEG(nvjpegBufferDeviceResize(res.device_buffer_, device_mem_padding_.value(), res.stream_));
                } else {
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "nvjpegBufferDeviceResize not available. Skip preallocation");
                }
            }
#endif
        } else {
            XM_CHECK_NVJPEG(nvjpegBufferDeviceCreate(handle_, nullptr, &res.device_buffer_));
        }
    }
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

        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new Decoder(plugin_id_, framework_, exec_params, options));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvjpeg decoder:" << e.info());
        return e.nvimgcodecStatus();
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvJpegCudaDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    NvJpegCudaDecoderPlugin* handle = reinterpret_cast<NvJpegCudaDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

Decoder::~Decoder()
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvjpeg_destroy");

        for (auto& res : per_thread_) {
            const int npages = res.pages_.size();
            for (int page_idx = 0; page_idx < npages; page_idx++) {
                auto& p = res.pages_[page_idx];
                if (p.parse_state_.nvjpeg_stream_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegJpegStreamDestroy(p.parse_state_.nvjpeg_stream_));
                }
                if (p.pinned_buffer_) {
                    XM_NVJPEG_LOG_DESTROY(nvjpegBufferPinnedDestroy(p.pinned_buffer_));
                }
                for (auto& decoder_data : p.decoder_data) {
                    if (decoder_data.state) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegJpegStateDestroy(decoder_data.state));
                    }
                    if (decoder_data.decoder) {
                        XM_NVJPEG_LOG_DESTROY(nvjpegDecoderDestroy(decoder_data.decoder));
                    }
                }
            }
            if (res.device_buffer_) {
                XM_NVJPEG_LOG_DESTROY(nvjpegBufferDeviceDestroy(res.device_buffer_));
            }
            if (res.event_) {
                XM_CUDA_LOG_DESTROY(cudaEventDestroy(res.event_));
            }
        }
        per_thread_.clear();

        if (handle_)
            XM_NVJPEG_LOG_DESTROY(nvjpegDestroy(handle_));
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not properly destroy nvjpeg decoder - " << e.info());
    }
}

nvimgcodecProcessingStatus_t Decoder::canDecode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int tid)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode ");
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        nvimgcodecJpegImageInfo_t cs_jpeg_image_info{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr};
        nvimgcodecImageInfo_t cs_image_info{
            NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), static_cast<void*>(&cs_jpeg_image_info)};

        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        bool is_jpeg = strcmp(cs_image_info.codec_name, "jpeg") == 0;
        if (!is_jpeg) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        static const std::set<nvimgcodecJpegEncoding_t> encodings_{NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT,
            NVIMGCODEC_JPEG_ENCODING_EXTENDED_SEQUENTIAL_DCT_HUFFMAN, NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN};
        bool supported_encoding = encodings_.find(cs_jpeg_image_info.encoding) != encodings_.end();
        if (!supported_encoding) {
            return NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
        }

        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        static const std::set<nvimgcodecColorSpec_t> supported_color_space{NVIMGCODEC_COLORSPEC_UNCHANGED, NVIMGCODEC_COLORSPEC_SRGB,
            NVIMGCODEC_COLORSPEC_GRAY, NVIMGCODEC_COLORSPEC_SYCC, NVIMGCODEC_COLORSPEC_CMYK, NVIMGCODEC_COLORSPEC_YCCK};
        if (supported_color_space.find(image_info.color_spec) == supported_color_space.end()) {
            status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }

        static const std::set<nvimgcodecChromaSubsampling_t> supported_css{NVIMGCODEC_SAMPLING_444, NVIMGCODEC_SAMPLING_422,
            NVIMGCODEC_SAMPLING_420, NVIMGCODEC_SAMPLING_440, NVIMGCODEC_SAMPLING_411, NVIMGCODEC_SAMPLING_410, NVIMGCODEC_SAMPLING_GRAY,
            NVIMGCODEC_SAMPLING_410V};
        if (supported_css.find(image_info.chroma_subsampling) == supported_css.end()) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
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
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
        }

        for (uint32_t p = 0; p < image_info.num_planes; ++p) {
            auto sample_type = image_info.plane_info[p].sample_type;
            if (sample_type != NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            }
        }
        return status;
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvjpeg can decode - " << e.info());
    }
    return NVIMGCODEC_PROCESSING_STATUS_FAIL;
}

nvimgcodecStatus_t Decoder::decode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    assert(code_stream->io_stream);
    void* encoded_stream_data_raw = nullptr;
    uint8_t* encoded_stream_data = nullptr;
    size_t encoded_stream_data_size = 0;
    if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data_raw, 0, encoded_stream_data_size) !=
        NVIMGCODEC_STATUS_SUCCESS) {
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    encoded_stream_data = static_cast<uint8_t*>(encoded_stream_data_raw);
    assert(encoded_stream_data != nullptr);
    assert(encoded_stream_data_size > 0);

    PerThreadResources& t = per_thread_[thread_idx];
    t.current_page_idx = (t.current_page_idx + 1) % 2;
    int page_idx = t.current_page_idx;
    auto& p = t.pages_[page_idx];

    try {
        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED);
            return ret;
        }

        unsigned char* device_buffer = reinterpret_cast<unsigned char*>(image_info.buffer);

        nvjpegDecodeParams_t nvjpeg_params_;
        XM_CHECK_NVJPEG(nvjpegDecodeParamsCreate(handle_, &nvjpeg_params_));
        std::unique_ptr<std::remove_pointer<nvjpegDecodeParams_t>::type, decltype(&nvjpegDecodeParamsDestroy)> nvjpeg_params(
            nvjpeg_params_, &nvjpegDecodeParamsDestroy);
        int num_channels = std::max(image_info.num_planes, image_info.plane_info[0].num_channels);
        auto sample_format = num_channels == 1 ? NVIMGCODEC_SAMPLEFORMAT_P_Y : image_info.sample_format;
        nvjpegOutputFormat_t nvjpeg_format = nvimgcodec_to_nvjpeg_format(sample_format);
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
                return NVIMGCODEC_STATUS_EXECUTION_FAILED;
            }

            if (orientation != NVJPEG_ORIENTATION_NORMAL) {
                if (!nvjpegIsSymbolAvailable("nvjpegDecodeParamsSetExifOrientation")) {
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
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

        {
            nvtx3::scoped_range marker{"nvjpegJpegStreamParse"};
            XM_CHECK_NVJPEG(nvjpegJpegStreamParse(handle_, encoded_stream_data, encoded_stream_data_size,
                false, false, p.parse_state_.nvjpeg_stream_));
        }

        nvjpegJpegEncoding_t jpeg_encoding;
        nvjpegJpegStreamGetJpegEncoding(p.parse_state_.nvjpeg_stream_, &jpeg_encoding);

        int is_gpu_hybrid_supported = -1;                    // zero means is supported
        if (jpeg_encoding == NVJPEG_ENCODING_BASELINE_DCT) { //gpu hybrid is not supported for progressive
            XM_CHECK_NVJPEG(nvjpegDecoderJpegSupported(p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID].decoder, p.parse_state_.nvjpeg_stream_,
                nvjpeg_params.get(), &is_gpu_hybrid_supported));
        }

        bool is_gpu_hybrid = (image_info.plane_info[0].height * image_info.plane_info[0].width) > gpu_hybrid_huffman_threshold_ &&
                             is_gpu_hybrid_supported == 0;
        auto& decoder_data = is_gpu_hybrid ? p.decoder_data[NVJPEG_BACKEND_GPU_HYBRID] : p.decoder_data[NVJPEG_BACKEND_HYBRID];
        auto& decoder = decoder_data.decoder;
        auto& state = decoder_data.state;

        XM_CHECK_NVJPEG(nvjpegStateAttachPinnedBuffer(state, p.pinned_buffer_));

        {
            nvtx3::scoped_range marker{"nvjpegDecodeJpegHost (is_gpu_hybrid=" + std::to_string(is_gpu_hybrid) + ")"};
            XM_CHECK_NVJPEG(nvjpegDecodeJpegHost(handle_, decoder, state, nvjpeg_params.get(), p.parse_state_.nvjpeg_stream_));
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

        XM_CHECK_NVJPEG(nvjpegDecodeJpegTransferToDevice(handle_, decoder, state, p.parse_state_.nvjpeg_stream_, image_info.cuda_stream));

        {
            nvtx3::scoped_range marker{"nvjpegDecodeJpegDevice"};
            XM_CHECK_NVJPEG(nvjpegDecodeJpegDevice(handle_, decoder, state, &nvjpeg_image, image_info.cuda_stream));
        }
        // this captures the state of image_info.cuda_stream in the cuda event t.event_
        XM_CHECK_CUDA(cudaEventRecord(t.event_, image_info.cuda_stream));

        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const NvJpegException& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not decode jpeg code stream - " << e.info());
        image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return NVIMGCODEC_STATUS_SUCCESS;
    }
}

} // namespace nvjpeg
