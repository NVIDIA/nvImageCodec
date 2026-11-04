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
#include "cuda_decoder.h"
#include "batching_heuristics.h"
#include "metadata_extractor.h"
#include "nvtiff_utils.h"

#include <library_types.h>
#include <nvimgcodec.h>
#include <nvjpeg.h>
#include <nvtiff.h>
#include <nvtiff_version.h>
#include <nvtx3/nvtx3.hpp>
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <deque>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>
#include "error_handling.h"
#include "imgproc/convert_kernel_gpu.h"
#include "imgproc/device_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/exit_guard.h"
#include "imgproc/image_info_checks.h"
#include "imgproc/out_of_bound_roi_fill.h"
#include "imgproc/region_orientation.h"
#include "imgproc/sample_format_utils.h"
#include "imgproc/stream_device.h"
#include "imgproc/type_utils.h"
#include "log.h"
#include "utils/library_version.h"

#if WITH_DYNAMIC_NVTIFF_ENABLED
#include "dynlink/dynlink_nvtiff.h"
#endif

namespace nvtiff {


struct PreparedDecodeSample
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    nvimgcodecImageInfo_t dec_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    nvimgcodecCodeStreamInfoTiffExt_t codestream_info_tiff_ext{NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr};
    nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
    nvtiffDecodeRegion_t decode_region{};
    nvtiffImage_t nvtiff_image{};
    std::array<void*, NVIMGCODEC_MAX_NUM_PLANES> plane_data{};
    std::array<size_t, NVIMGCODEC_MAX_NUM_PLANES> plane_pitch_bytes{};
    nvtiffOutputFormat_t output_format = NVTIFF_OUTPUT_UNCHANGED_I;
    bool need_conversion = false;
    bool has_roi = false;
    bool has_oob_roi = false;
    bool skip_decode = false;
};


struct Decoder
{
    using MetadataExtractor = nvtiff::MetadataExtractor;

    Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
        const char* options = nullptr);
    ~Decoder();

    nvimgcodecStatus_t getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count);
    static nvimgcodecStatus_t static_get_metadata(nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->getMetadata(code_stream, metadata, metadata_count);
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
        }
    }

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder);
    nvimgcodecProcessingStatus_t canDecode(
        const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx);
    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
    {
        try {
            XM_CHECK_NULL(decoder);
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->canDecode(image, code_stream, params, thread_idx);
        } catch (const std::exception& e) {
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
            auto handle = reinterpret_cast<Decoder*>(decoder);
            return handle->decode(image, code_stream, params, thread_idx);
        } catch (const std::exception& e) {
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;;
        }
    }

    nvimgcodecStatus_t decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size,
        const nvimgcodecDecodeParams_t* params);
    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t** images,
        const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx);

    using Subbatch = std::vector<int>;
    std::vector<Subbatch> planSubbatches(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
        int batch_size);
    void decodeSubbatch(const Subbatch& batch, const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
        const nvimgcodecDecodeParams_t* params, int thread_idx);

    const char* plugin_id_;
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecExecutionParams_t* exec_params_;
    size_t max_subbatch_images_ = 0;
    bool trace_subbatches_ = false;

    struct PerThreadResources
    {
        const nvimgcodecFrameworkDesc_t* framework_;
        const char* plugin_id_;

        nvtiffStream_t nvtiff_stream_ = nullptr;
        nvtiffDecodeParams_t decode_params_ = nullptr;
        nvtiffDecoder_t decoder_ = nullptr;
        cudaEvent_t event_ = nullptr;
        cudaStream_t batch_stream_ = nullptr;
        std::vector<nvimgcodec::DeviceBuffer> batch_buffers_;
        size_t used_batch_buffers_ = 0;
        size_t small_batch_streak_ = 0;
        nvimgcodec::DeviceBuffer device_buffer_;
        std::optional<cudaStream_t> cuda_stream_ = std::nullopt;
        nvtiffDeviceAllocator_t device_allocator_ = {};
        nvtiffPinnedAllocator_t pinned_allocator_ = {};
        nvtiffDeviceAllocator_t* device_allocator_ptr_ = nullptr;
        nvtiffPinnedAllocator_t* pinned_allocator_ptr_ = nullptr;
        MetadataExtractor metadata_extractor_;

        // The parsed source is valid only during the current call, because the
        // native stream uses mapped bytes that the caller owns.
        // io_stream ids are process-unique (monotonic), so the id alone identifies a stream.
        std::optional<uint64_t> cached_io_stream_id_ = std::nullopt;
        nvtiffImageInfo_t iinfo_ = {};

        PerThreadResources(const nvimgcodecFrameworkDesc_t* framework, const char* plugin_id,
            const nvimgcodecExecutionParams_t* exec_params)
            : framework_(framework)
            , plugin_id_(plugin_id)
            , device_buffer_(exec_params ? exec_params->device_allocator : nullptr)
            , metadata_extractor_(framework_, plugin_id_, &nvtiff_stream_)
        {
            XM_CHECK_NVTIFF(nvtiffDecodeParamsCreate(&decode_params_));
            XM_CHECK_CUDA(cudaEventCreateWithFlags(&event_, cudaEventDisableTiming));

            if (exec_params && exec_params->device_allocator) {
                device_allocator_ = {
                    exec_params->device_allocator->device_malloc,
                    exec_params->device_allocator->device_free,
                    exec_params->device_allocator->device_ctx
                };
                device_allocator_ptr_ = &device_allocator_;
            } else {
                device_allocator_ptr_ = nullptr;
            }

            if (exec_params && exec_params->pinned_allocator) {
                pinned_allocator_ = {
                    exec_params->pinned_allocator->pinned_malloc,
                    exec_params->pinned_allocator->pinned_free,
                    exec_params->pinned_allocator->pinned_ctx
                };
                pinned_allocator_ptr_ = &pinned_allocator_;
            } else {
                pinned_allocator_ptr_ = nullptr;
            }
        }

        PerThreadResources(const PerThreadResources& other) = delete;
        PerThreadResources& operator=(const PerThreadResources& other) = delete;

        void closeStream()
        {
            if (nvtiff_stream_) {
                XM_NVTIFF_LOG_DESTROY(nvtiffStreamClose(nvtiff_stream_));
                nvtiff_stream_ = nullptr;
            }
            cached_io_stream_id_ = std::nullopt;
        }

        void finishStream(std::optional<cudaStream_t> stream = std::nullopt)
        {
            // A failed decode may throw before recording its completion event.
            if (stream) {
                XM_CUDA_LOG_DESTROY(cudaStreamSynchronize(*stream));
            } else {
                XM_CUDA_LOG_DESTROY(cudaEventSynchronize(event_));
            }
            closeStream();
        }

        ~PerThreadResources() {
            // See imgproc/exit_guard.h: at process exit, the CUDA driver or
            // the nvTIFF library may already be tearing down, so skip the
            // teardown calls below and intentionally leak instead of risking
            // a segfault.
            if (nvimgcodec::IsProcessExiting()) {
                return;
            }
            XM_CUDA_LOG_DESTROY(cudaEventDestroy(event_));
            XM_NVTIFF_LOG_DESTROY(nvtiffDecodeParamsDestroy(decode_params_));
            // The cuda_stream_.value() calls below are guarded by has_value() one
            // line above, so std::bad_optional_access is unreachable; Coverity cannot
            // prove that the guard dominates the value() reads and reports the
            // noexcept escape conservatively.
            if (cuda_stream_.has_value()) {
                // coverity[fun_call_w_exception : FALSE]
                XM_CUDA_LOG_DESTROY(cudaStreamSynchronize(cuda_stream_.value()));
                // coverity[fun_call_w_exception : FALSE]
                XM_NVTIFF_LOG_DESTROY(nvtiffDecoderDestroy(decoder_, cuda_stream_.value()));
            }
            closeStream();
            // Release stream-bound buffers before destroying the owned stream.
            batch_buffers_.clear();
            device_buffer_.d_ptr.reset();
            if (batch_stream_)
                XM_CUDA_LOG_DESTROY(cudaStreamDestroy(batch_stream_));
        }

        // Retain scratch across short fluctuations. After eight submissions using
        // at most a quarter of the cached slots, release the unused tail. Called
        // only after all worker work has completed, including idle workers.
        void retireBatchBuffers()
        {
            if (batch_buffers_.size() > 16 && used_batch_buffers_ <= batch_buffers_.size() / 4) {
                if (++small_batch_streak_ == 8) {
                    batch_buffers_.erase(batch_buffers_.begin() + used_batch_buffers_, batch_buffers_.end());
                    small_batch_streak_ = 0;
                }
            } else {
                small_batch_streak_ = 0;
            }
            used_batch_buffers_ = 0;
        }

        cudaStream_t batchStream()
        {
            if (!batch_stream_)
                XM_CHECK_CUDA(cudaStreamCreateWithFlags(&batch_stream_, cudaStreamNonBlocking));
            return batch_stream_;
        }

        nvtiffDecoder_t& decoder(cudaStream_t cuda_stream) {
            if (cuda_stream_ != cuda_stream) {
                if (cuda_stream_) {  // should not really happen but just in case
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Recreating nvtiff decoder instance");
                    XM_CHECK_CUDA(cudaStreamSynchronize(cuda_stream_.value()));
                    XM_CHECK_NVTIFF(nvtiffDecoderDestroy(decoder_, cuda_stream_.value()));
                }
                XM_CHECK_NVTIFF(nvtiffDecoderCreate(&decoder_, device_allocator_ptr_, pinned_allocator_ptr_, cuda_stream));
                cuda_stream_ = cuda_stream;
            }
            return decoder_;
        }

        nvimgcodecStatus_t ensureStreamParsed(const nvimgcodecCodeStreamDesc_t* code_stream) {
            uint64_t io_stream_id = code_stream->io_stream->id;

            // Repeated ROI views within this call need no further I/O queries.
            if (cached_io_stream_id_ == io_stream_id && nvtiff_stream_) {
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            size_t encoded_stream_data_size = 0;
            if (code_stream->io_stream->size(code_stream->io_stream->instance, &encoded_stream_data_size) != NVIMGCODEC_STATUS_SUCCESS) {
                return NVIMGCODEC_STATUS_INTERNAL_ERROR;
            }

            try {
                void* encoded_stream_data = nullptr;
                if (code_stream->io_stream->map(code_stream->io_stream->instance, &encoded_stream_data, 0, encoded_stream_data_size) !=
                    NVIMGCODEC_STATUS_SUCCESS) {
                    return NVIMGCODEC_STATUS_INTERNAL_ERROR;
                }

                closeStream();
                XM_CHECK_NVTIFF(nvtiffStreamOpen(
                    static_cast<const uint8_t*>(encoded_stream_data), encoded_stream_data_size, &nvtiff_stream_));
                cached_io_stream_id_ = io_stream_id;
                return NVIMGCODEC_STATUS_SUCCESS;
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not parse tiff data - " << e.what());
                closeStream();
                return NVIMGCODEC_STATUS_INTERNAL_ERROR;
            }
        }

        const nvtiffImageInfo_t& getImageInfo(size_t ifd_offset)
        {
            if (!nvtiff_stream_) {
                throw std::runtime_error("No parsed TIFF stream is selected");
            }
            XM_CHECK_NVTIFF(nvtiffStreamGetImageInfo(nvtiff_stream_, ifd_offset, &iinfo_));
            return iinfo_;
        }
    };

    // Sample preparation, decode, and postprocessing.
    nvimgcodecProcessingStatus_t prepareDecodeSample(PreparedDecodeSample& prepared, PerThreadResources& per_thread,
        nvimgcodec::DeviceBuffer& temp_buffer, const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecDecodeParams_t* params, std::optional<cudaStream_t> stream = std::nullopt,
        const nvimgcodecImageInfo_t* output_info = nullptr);
    nvimgcodecProcessingStatus_t decodePreparedSample(PerThreadResources& per_thread, PreparedDecodeSample& sample);
    nvimgcodecProcessingStatus_t postprocessPreparedSample(PreparedDecodeSample& sample, cudaStream_t stream);

    // Waits for the decode, then closes the parsed source. The caller can release
    // the source and the output only after this completes.
    struct SourceGuard
    {
        PerThreadResources& resources;
        std::optional<cudaStream_t> stream;
        ~SourceGuard() { resources.finishStream(stream); }
    };

    void traceBufferCache()
    {
        if (!trace_subbatches_)
            return;
        size_t slots = 0, bytes = 0;
        for (const auto& resource : per_thread_) {
            slots += resource.batch_buffers_.size();
            for (const auto& buffer : resource.batch_buffers_)
                bytes += buffer.capacity;
        }
        NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "TIFF_BUFFER_CACHE slots=" << slots << " bytes=" << bytes);
    }

    // Worker resources for thread_idx. The last slot is reserved for the planner.
    // Non-threaded user executors may use the caller index (-1).
    size_t workerIndex(int thread_idx) const
    {
        return thread_idx >= 0 && thread_idx < static_cast<int>(per_thread_.size() - 1) ? thread_idx : 0;
    }

    // Stable addresses: metadata extractors and native allocators refer to their owner.
    std::deque<PerThreadResources> per_thread_;
};

Decoder::Decoder(const char* plugin_id, const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecExecutionParams_t* exec_params,
    const char* options)
    : plugin_id_(plugin_id)
    , framework_(framework)
    , exec_params_(exec_params)
{
    (void)nvimgcodec::IsProcessExiting();  // ensure atexit handler is registered before any decoder lives
    std::string invalid_options;
    max_subbatch_images_ = parseUIntOption(options, "max_subbatch_images", &invalid_options);
    trace_subbatches_ = parseUIntOption(options, "trace_subbatches", &invalid_options) != 0;
    if (!invalid_options.empty())
        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Malformed nvTIFF option(s): " << invalid_options
            << "; effective max_subbatch_images=" << max_subbatch_images_ << ", tracing=" << (trace_subbatches_ ? "enabled" : "disabled"));
    auto executor = exec_params_->executor;
    const int num_threads = std::max(1, executor->getNumThreads(executor->instance));

    // One slot per worker thread, plus a last slot for planSubbatches.
    while (per_thread_.size() < static_cast<size_t>(num_threads + 1)) {
        per_thread_.emplace_back(framework_, plugin_id_, exec_params_);
    }
}

Decoder::~Decoder()
{
}

nvimgcodecStatus_t Decoder::static_destroy(nvimgcodecDecoder_t decoder)
{
    try {
        XM_CHECK_NULL(decoder);
        Decoder* handle = reinterpret_cast<Decoder*>(decoder);
        delete handle;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t Decoder::getMetadata(const nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecMetadata_t** metadata, int* metadata_count) 
{
    try {

        int thread_idx = 0; //TODO: Implement getting thread index from execution params
        auto& resources = per_thread_[thread_idx];
        SourceGuard source_lifetime{resources};

        // Ensure stream is parsed before extracting metadata
        auto parse_ret = resources.ensureStreamParsed(code_stream);
        if (parse_ret != NVIMGCODEC_STATUS_SUCCESS) {
            return parse_ret;
        }

        //get metadata count
        if (metadata == nullptr) {
            if (!metadata_count) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }
            *metadata_count =  resources.metadata_extractor_.getMetadataCount(code_stream); 
            return NVIMGCODEC_STATUS_SUCCESS;
        }

        if (!metadata_count || *metadata_count <= 0) {
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        }
        
        //get buffer sets
        for (int i = 0; i < *metadata_count; i++) {
            if (!metadata[i]) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }

            //Check metadata structure type and format
            if (metadata[i]->struct_type != NVIMGCODEC_STRUCTURE_TYPE_METADATA ||
                metadata[i]->struct_size != sizeof(nvimgcodecMetadata_t)) {
                return NVIMGCODEC_STATUS_INVALID_PARAMETER;
            }

            auto ret = resources.metadata_extractor_.getMetadata(code_stream, metadata[i], i);
            if (ret == NVIMGCODEC_STATUS_CODESTREAM_UNSUPPORTED) {
                // Tag not found, signal via metadata_count=0
                *metadata_count = 0;
                return NVIMGCODEC_STATUS_SUCCESS;
            }
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                return ret;
            }
         }

        return NVIMGCODEC_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        return NVIMGCODEC_STATUS_INTERNAL_ERROR;
    }
}

nvimgcodecProcessingStatus_t Decoder::canDecode(
    const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    (void) thread_idx;
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "can_decode");

        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);

        XM_CHECK_NULL(image);
        nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        auto ret = image->getImageInfo(image->instance, &image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_stream->getImageInfo(code_stream->instance, &cs_image_info);

        bool is_tiff = strcmp(cs_image_info.codec_name, "tiff") == 0;
        if (!is_tiff) {
            return NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        }

        switch(cs_image_info.plane_info[0].precision) {
        case 0:  // means full precision of the data type
        case 8:
        case 16:
        case 32:
            break; // supported
        default:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_,
                "nvTIFF extension can only decode 8, 16 or 32 bit input. Got " << cs_image_info.plane_info[0].precision << " bit.");
            break;
        }

        if (!nvimgcodec::check_planes_consistency(framework_, plugin_id_, image_info)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_TYPE_UNSUPPORTED;
        }

        switch (image_info.sample_format) {
        case NVIMGCODEC_SAMPLEFORMAT_I_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_P_BGR:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGB:
        case NVIMGCODEC_SAMPLEFORMAT_I_RGBA:
        case NVIMGCODEC_SAMPLEFORMAT_P_RGBA:
        case NVIMGCODEC_SAMPLEFORMAT_P_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_Y:
        case NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED:
            break; // supported

        default:
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLE_FORMAT_UNSUPPORTED;
            break;
        }

        // TODO: Do we care about this for decoding? I would argue that subsampling is used during encoding.
        // nvTIFF support decoding the same type of subsampled images as nvJPEG (as nvtTIFF uses it under the hood)
        // but I think this value should be NVIMGCODEC_SAMPLING_NONE even if we are decoding subsampled image.
        if ((image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_NONE) 
            && (image_info.chroma_subsampling != NVIMGCODEC_SAMPLING_GRAY)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_SAMPLING_UNSUPPORTED;
        }

        // This codec doesn't apply EXIF orientation
        if (params->apply_exif_orientation &&
            (image_info.orientation.flip_x || image_info.orientation.flip_y || image_info.orientation.rotated != 0)) {
            status |= NVIMGCODEC_PROCESSING_STATUS_ORIENTATION_UNSUPPORTED;
        }

        if (image_info.color_spec != NVIMGCODEC_COLORSPEC_SRGB &&
            image_info.color_spec != NVIMGCODEC_COLORSPEC_GRAY &&
            image_info.color_spec != NVIMGCODEC_COLORSPEC_UNKNOWN) {
            status |= NVIMGCODEC_PROCESSING_STATUS_COLOR_SPEC_UNSUPPORTED;
        }
        nvimgcodecCodeStreamInfo_t codestream_info{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), nullptr};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;

        if (codestream_info.code_stream_view) {
            const auto& region = codestream_info.code_stream_view->region;
            if (region.ndim == 0) {
                // ignore
            }
            else if (region.ndim == 2) {
                if (cs_image_info.plane_info[0].height > static_cast<uint32_t>(std::numeric_limits<int>::max()) ||
                    cs_image_info.plane_info[0].width > static_cast<uint32_t>(std::numeric_limits<int>::max())) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Image dimensions exceeds int32, nvtiff ROI decode is not supported.");
                    image->imageReady(image->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
                    return NVIMGCODEC_STATUS_EXECUTION_FAILED;
                }

                if (nvimgcodec::is_region_out_of_bounds_effective(region, image_info.orientation,
                        cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height,
                        params->apply_exif_orientation)) {
                    if (auto err_message = nvimgcodec::verify_region_fill_support(region, image_info); !err_message.empty()) {
                        status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                        NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, err_message);
                    }
                }
                uint32_t roi_height = static_cast<uint32_t>(region.end[0] - region.start[0]);
                uint32_t roi_width = static_cast<uint32_t>(region.end[1] - region.start[1]);
                if (image_info.plane_info[0].width != roi_width || image_info.plane_info[0].height != roi_height) {
                    status |= NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED;
                    NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "ROI size must match decoded image size.");
                }
            } else {
                status |= NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Region decoding is supported only for 2 dimensions.");
            }
        }
        return status;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not check if nvTIFF can decode - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

void setOutputInfoToUINT8RGB(nvimgcodecImageInfo_t& info) {
    info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    info.plane_info[0].precision = 8;
    info.plane_info[0].num_channels = 3;
    info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
}

void setOutputInfoToUINT16RGB(nvimgcodecImageInfo_t& info) {
    info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    info.plane_info[0].precision = 16;
    info.plane_info[0].num_channels = 3;
    info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
}

nvimgcodecProcessingStatus_t Decoder::prepareDecodeSample(PreparedDecodeSample& prepared, PerThreadResources& per_thread,
    nvimgcodec::DeviceBuffer& temp_buffer, const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, std::optional<cudaStream_t> stream, const nvimgcodecImageInfo_t* output_info)
{
    try {
        XM_CHECK_NULL(image);
        XM_CHECK_NULL(code_stream);
        XM_CHECK_NULL(params);
        assert(code_stream->io_stream);

        auto& image_info = prepared.image_info;
        auto ret = NVIMGCODEC_STATUS_SUCCESS;
        if (output_info)
            image_info = *output_info;
        else
            ret = image->getImageInfo(image->instance, &image_info);
        if (stream)
            image_info.cuda_stream = *stream;
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        auto& cs_image_info = prepared.cs_image_info;
        ret = code_stream->getImageInfo(code_stream->instance, &cs_image_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        if (image_info.buffer_kind != NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Unexpected buffer kind");
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        if (nvimgcodec::IsPlanar(image_info.sample_format)) {
            for (uint32_t p = 0; p < image_info.num_planes; ++p) {
                if (image_info.plane_info[p].num_channels != 1) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "In planar format, each plane should have single channel");
                    return NVIMGCODEC_PROCESSING_STATUS_NUM_CHANNELS_UNSUPPORTED;
                }
            }
        } else {
            if (image_info.num_planes != 1) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "In non planar format, image should have a single plane");
                return NVIMGCODEC_PROCESSING_STATUS_NUM_PLANES_UNSUPPORTED;
            }
        }

        for (size_t p = 1; p < image_info.num_planes; p++) {
            if (image_info.plane_info[p].sample_type != image_info.plane_info[0].sample_type) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same data type");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
            if (image_info.plane_info[p].precision != image_info.plane_info[0].precision) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same precision");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }

            if (image_info.plane_info[p].width != image_info.plane_info[0].width ||
                image_info.plane_info[p].height != image_info.plane_info[0].height) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "All components are expected to have the same dimension");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }

        auto& codestream_info = prepared.codestream_info;
        auto& codestream_info_tiff_ext = prepared.codestream_info_tiff_ext;
        codestream_info_tiff_ext = {
            NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfoTiffExt_t), nullptr};
        codestream_info = {
            NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(nvimgcodecCodeStreamInfo_t), &codestream_info_tiff_ext};
        ret = code_stream->getCodeStreamInfo(code_stream->instance, &codestream_info);
        if (ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not retrieve code stream information");
            return NVIMGCODEC_PROCESSING_STATUS_IMAGE_CORRUPTED;
        }
        auto region = nvimgcodecRegion_t{NVIMGCODEC_STRUCTURE_TYPE_REGION, sizeof(nvimgcodecRegion_t), nullptr};
        if(codestream_info.code_stream_view) {
            region = codestream_info.code_stream_view->region;
            prepared.has_roi = region.ndim != 0;
            if (prepared.has_roi) {
                prepared.has_oob_roi = nvimgcodec::is_region_out_of_bounds(region, cs_image_info.plane_info[0].width, cs_image_info.plane_info[0].height);
            }
        }

        auto& dec_image_info = prepared.dec_image_info;
        dec_image_info = cs_image_info;
        // nvTIFF decoded to interleaved, but our parser assumes planar
        if (dec_image_info.num_planes == 1) {
            // One-channel planar and interleaved Y have identical storage.
            // Avoid a needless conversion (which also rejects float32 output).
            dec_image_info.sample_format =
                image_info.sample_format == NVIMGCODEC_SAMPLEFORMAT_I_Y ? NVIMGCODEC_SAMPLEFORMAT_I_Y : NVIMGCODEC_SAMPLEFORMAT_P_Y;
        } else {
            dec_image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        }
        dec_image_info.plane_info[0].num_channels = dec_image_info.num_planes;
        dec_image_info.num_planes = 1;

        if (nvimgcodec::NumberOfChannels(dec_image_info) < nvimgcodec::NumberOfChannels(image_info)) {
            if (dec_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_P_Y && dec_image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_Y) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "When increasing number of channels, can convert only from P_Y or I_Y");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
            if (!nvimgcodec::IsRgb(image_info.sample_format) && !nvimgcodec::IsBgr(image_info.sample_format)) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "When increasing number of channels, can convert only to RGB or BGR");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }

        // All plane_info should be the same, so it is enough to check with only the first one
        if (image_info.plane_info[0].precision == 0) {
            image_info.plane_info[0].precision = image_info.plane_info[0].sample_type >> 8;
        }

        auto parse_ret = per_thread.ensureStreamParsed(code_stream);
        if (parse_ret != NVIMGCODEC_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not parse tiff data");
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        size_t ifd_offset = resolve_ifd_selection(per_thread.nvtiff_stream_, codestream_info);
        try {
            const auto& tiff_info = per_thread.getImageInfo(ifd_offset);

            const auto src_sample_type = cs_image_info.plane_info[0].sample_type;
            // nvTIFF's RGB output formats are unsigned only: nvtiffDecodeCheckSupported() refuses
            // NVTIFF_OUTPUT_RGB_I_UINT8/UINT16 for a signed integer source, which showed up as a
            // silently rejected sample. Decode such images unchanged instead and let the existing
            // need_conversion path run LaunchConvertNormKernel, exactly as the grayscale request
            // already does. Restricted to signed *integers*: IsSigned() is also true for float,
            // and nvTIFF does accept float sources for the RGB outputs.
            const bool src_is_signed_integer =
                nvimgcodec::IsSigned(src_sample_type) && nvimgcodec::IsIntegral(src_sample_type);
            const bool wants_rgb_out =
                nvimgcodec::IsRgb(image_info.sample_format) || nvimgcodec::IsBgr(image_info.sample_format);

            if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_MASK) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "NVTIFF_PHOTOMETRIC_MASK is not supported");
                return NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }

            if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_PALETTE) {
                setOutputInfoToUINT16RGB(dec_image_info);
                prepared.output_format = NVTIFF_OUTPUT_RGB_I_UINT16;
            } else if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_YCBCR) {
                if (tiff_info.compression != NVTIFF_COMPRESSION_JPEG) {
                    NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "nvTIFF support YCbCr only with JPEG compression.");
                    return NVIMGCODEC_PROCESSING_STATUS_ENCODING_UNSUPPORTED;
                }
                //nvJPEG supports only 8 bit images
                setOutputInfoToUINT8RGB(dec_image_info);
                prepared.output_format = NVTIFF_OUTPUT_RGB_I_UINT8;
            } else if (wants_rgb_out && !src_is_signed_integer &&
                       image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8) {
                setOutputInfoToUINT8RGB(dec_image_info);
                prepared.output_format = NVTIFF_OUTPUT_RGB_I_UINT8;
            } else if (wants_rgb_out && !src_is_signed_integer &&
                       image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16) {
                setOutputInfoToUINT16RGB(dec_image_info);
                prepared.output_format = NVTIFF_OUTPUT_RGB_I_UINT16;
            } else if (tiff_info.photometric_int == NVTIFF_PHOTOMETRIC_RGB && tiff_info.compression == NVTIFF_COMPRESSION_JPEG) {
                // For now nvTIFF doesn't support unchanged interleaved in that case
                setOutputInfoToUINT8RGB(dec_image_info);
                prepared.output_format = NVTIFF_OUTPUT_RGB_I_UINT8;
            } else {
                prepared.output_format = NVTIFF_OUTPUT_UNCHANGED_I;
            }
        } catch (const std::exception& e) {
            NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not get image info - " << e.what());
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }

        // dimensions of image that will be decode by nvtiff (excluding oob roi)
        size_t decoded_image_width = image_info.plane_info[0].width;
        size_t decoded_image_height = image_info.plane_info[0].height;
        int roi_y_begin = 0;
        int roi_x_begin = 0;
        int roi_y_end = cs_image_info.plane_info[0].height;
        int roi_x_end = cs_image_info.plane_info[0].width;
        if (prepared.has_roi) {
            NVIMGCODEC_LOG_DEBUG(
                framework_,
                plugin_id_,
                "Input ROI: y=[" << region.start[0] << ", " << region.end[0] << "); x=["
                                 << region.start[1] << ", " << region.end[1] << ")"
            );
            
            roi_y_begin = std::max(0, region.start[0]);
            roi_x_begin = std::max(0, region.start[1]);
            roi_y_end = std::min((int)cs_image_info.plane_info[0].height, region.end[0]);
            roi_x_end = std::min((int)cs_image_info.plane_info[0].width, region.end[1]);

            NVIMGCODEC_LOG_DEBUG(
                framework_,
                plugin_id_,
                "ROI After clipping: y=[" << roi_y_begin << ", " << roi_y_end << "); x=[" 
                                          << roi_x_begin << ", " << roi_x_end << ")"
            );

            if (roi_x_end < roi_x_begin || roi_y_end < roi_y_begin) {
                // ROI lies entirely beyond the image (clipped extent inverted) and decodes no
                // pixels; the fill cannot produce it. Reject, matching the nvjpeg / nvjpeg2k decoders.
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "ROI lies entirely beyond the image and cannot be decoded.");
                return NVIMGCODEC_PROCESSING_STATUS_ROI_UNSUPPORTED;
            }
            if (roi_x_end <= roi_x_begin || roi_y_end <= roi_y_begin) {
                NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "ROI has empty clipped intersection");
                prepared.skip_decode = true;
            }
        }
        if (prepared.skip_decode) {
            decoded_image_width = 0;
            decoded_image_height = 0;
        } else {
            decoded_image_width = roi_x_end - roi_x_begin;
            decoded_image_height = roi_y_end - roi_y_begin;
        }
        auto& decode_region = prepared.decode_region;
        decode_region.ifd_offset = ifd_offset;
        decode_region.offset_x = roi_x_begin;
        decode_region.offset_y = roi_y_begin;
        decode_region.width = decoded_image_width;
        decode_region.height = decoded_image_height;

        prepared.need_conversion =
            (dec_image_info.sample_format != image_info.sample_format && image_info.sample_format != NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED) ||
            nvimgcodec::NumberOfChannels(dec_image_info) != nvimgcodec::NumberOfChannels(image_info) ||
            dec_image_info.plane_info[0].precision != image_info.plane_info[0].precision ||
            dec_image_info.plane_info[0].sample_type != image_info.plane_info[0].sample_type;

        NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, (prepared.need_conversion ? "nvtiffDecode with conversion" : "Direct nvtiffDecode"));

        if (prepared.skip_decode) {
            prepared.need_conversion = false;
        }

        if (prepared.need_conversion) {
            //TODO: Make FP work with conversion
            if (dec_image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32 ||
                image_info.plane_info[0].sample_type == NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "FP32 conversion is not supported");
                return NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
            }

            // If needing conversion, we decode to a temporary dec_image_info, then convert
            dec_image_info.plane_info[0].width = image_info.plane_info[0].width;
            dec_image_info.plane_info[0].height = image_info.plane_info[0].height;

            const size_t BPP = nvimgcodec::TypeSize(dec_image_info.plane_info[0].sample_type);
            dec_image_info.plane_info[0].row_stride = BPP * dec_image_info.plane_info[0].width * dec_image_info.plane_info[0].num_channels;
            assert(nvimgcodec::GetBufferSize(dec_image_info) == nvimgcodec::GetImageSize(dec_image_info));
            temp_buffer.resize(nvimgcodec::GetBufferSize(dec_image_info), image_info.cuda_stream);
            dec_image_info.buffer = temp_buffer.data;
        }

        const nvimgcodecImageInfo_t& target_iinfo = prepared.need_conversion ? dec_image_info : image_info;
        assert(target_iinfo.num_planes <= NVIMGCODEC_MAX_NUM_PLANES);
        auto& nvtiff_image = prepared.nvtiff_image;
        nvtiff_image = {prepared.plane_data.data(), prepared.plane_pitch_bytes.data(), target_iinfo.num_planes};
        unsigned char* ptr = reinterpret_cast<unsigned char*>(target_iinfo.buffer);
        nvtiff_image.num_planes = target_iinfo.num_planes;
        for (uint32_t c = 0; c < target_iinfo.num_planes; ++c) {
            nvtiff_image.plane_data[c] = ptr;
            nvtiff_image.plane_pitch_bytes[c] = target_iinfo.plane_info[c].row_stride;
            ptr += nvtiff_image.plane_pitch_bytes[c] * target_iinfo.plane_info[c].height;

            // Shift the channel pointers if the ROI starts before image's top-left corner
            if (prepared.has_roi) {
                int roi_y_begin = region.start[0];
                int roi_x_begin = region.start[1];
                unsigned char* plane_data = reinterpret_cast<unsigned char*>(nvtiff_image.plane_data[c]);
                plane_data += (roi_y_begin < 0) * (-roi_y_begin) * nvtiff_image.plane_pitch_bytes[c];
                int bytes_per_sample = target_iinfo.plane_info[c].sample_type >> 11;
                plane_data += (roi_x_begin < 0) * (-roi_x_begin) * bytes_per_sample * target_iinfo.plane_info[c].num_channels;
                nvtiff_image.plane_data[c] = plane_data;
            }
        }

        return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to prepare nvTIFF decode sample - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

nvimgcodecProcessingStatus_t Decoder::postprocessPreparedSample(PreparedDecodeSample& sample, cudaStream_t stream)
{
    try {
        // sample.image_info.cuda_stream already equals `stream` (set in decodePreparedSample),
        // and both callees take the stream explicitly or read it from image_info, so we can pass
        // the sample's infos directly instead of copying ~2 KB structs just to set cuda_stream.
        if (!sample.skip_decode && sample.need_conversion) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "LaunchConvertNormKernel");
            nvimgcodec::LaunchConvertNormKernel(sample.image_info, sample.dec_image_info, stream);
        }

        if (sample.has_oob_roi) {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "fill_out_of_bounds_region");
            nvimgcodec::fill_out_of_bounds_region(
                sample.image_info,
                sample.cs_image_info.plane_info[0].width, sample.cs_image_info.plane_info[0].height,
                sample.codestream_info.code_stream_view->region
            );
        }
        return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    } catch (std::runtime_error& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not postprocess nvTIFF decode sample - " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

nvimgcodecProcessingStatus_t Decoder::decodePreparedSample(PerThreadResources& per_thread, PreparedDecodeSample& sample)
{
    cudaStream_t stream = sample.image_info.cuda_stream;

    // prepareDecodeSample() already parsed this stream into per_thread on the same
    // thread, and nothing switches the stream in between, so per_thread.nvtiff_stream_
    // is already the one we need (no re-parse).

    // Serialize against the previous decode on this thread's resources before reusing
    // decode_params_ / the temporary buffer.
    XM_CHECK_CUDA(cudaEventSynchronize(per_thread.event_));

    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    if (!sample.skip_decode) {
        XM_CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(per_thread.decode_params_, sample.output_format));
        XM_CHECK_NVTIFF(nvtiffDecodeParamsSetRegions(per_thread.decode_params_, &sample.decode_region, 1));

        auto& nvtiff_decoder = per_thread.decoder(stream);
        nvtiffStatus_t decode_check_ret =
            nvtiffDecodeCheckSupported(per_thread.nvtiff_stream_, nvtiff_decoder, per_thread.decode_params_, &sample.nvtiff_image);
        if (decode_check_ret != NVTIFF_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "nvtiffDecodeCheckSupported returned error code: " << decode_check_ret);
            status = NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
        } else {
            NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "nvtiffDecode");
            nvtiffStatus_t decode_ret =
                nvtiffDecode(per_thread.nvtiff_stream_, nvtiff_decoder, per_thread.decode_params_, &sample.nvtiff_image, stream);
            if (decode_ret != NVTIFF_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "nvtiffDecode returned error code: " << decode_ret);
                status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
            }
        }
    }

    if (status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
        status = postprocessPreparedSample(sample, stream);
    }

    // Record on every path, including decode failure: a failed nvtiffDecode may have
    // launched partial work on `stream`, so the next decode's cudaEventSynchronize must
    // wait for it before reusing decode_params_ / the temporary buffer.
    XM_CHECK_CUDA(cudaEventRecord(per_thread.event_, stream));
    return status;
}

nvimgcodecStatus_t Decoder::decode(const nvimgcodecImageDesc_t* image, const nvimgcodecCodeStreamDesc_t* code_stream,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
    try {
        auto& resources = per_thread_[workerIndex(thread_idx)];
        {
            XM_CHECK_CUDA(cudaEventSynchronize(resources.event_));
            const auto stream = resources.batchStream();
            PreparedDecodeSample prepared;
            SourceGuard lifetime{resources, stream};
            nvimgcodecImageInfo_t output{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(output), nullptr};
            if (image->getImageInfo(image->instance, &output) == NVIMGCODEC_STATUS_SUCCESS) {
                // Use the same worker stream as native batch calls. Wait for
                // work on the caller stream first.
                XM_CHECK_CUDA(cudaEventRecord(resources.event_, output.cuda_stream));
                XM_CHECK_CUDA(cudaStreamWaitEvent(stream, resources.event_, 0));
                status = prepareDecodeSample(prepared, resources, resources.device_buffer_, image, code_stream, params, stream, &output);
                if (status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
                    status = decodePreparedSample(resources, prepared);
            }
        } // Wait for the decode before reporting the image as ready.
        resources.retireBatchBuffers();
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Failed to decode with nvTIFF - " << e.what());
        status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    image->imageReady(image->instance, status);
    return status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_EXECUTION_FAILED;
}

nvimgcodecStatus_t Decoder::static_decode_batch(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t** images,
    const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size, const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    if (!decoder || !images || !code_streams || !params || batch_size <= 0)
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    try {
        auto* handle = reinterpret_cast<Decoder*>(decoder);
        // A single image needs no planning. Decode it on the calling thread.
        if (batch_size == 1) {
            auto status = handle->decode(images[0], code_streams[0], params, thread_idx);
            const size_t used = handle->workerIndex(thread_idx);
            // The framework serializes batch callbacks, and nextParallelProcessor()
            // skips batched plugins. Thus no other decode uses the worker
            // resources now, and it is safe to retire their buffers.
            for (size_t i = 0; i < handle->per_thread_.size(); ++i)
                if (i != used)
                    handle->per_thread_[i].retireBatchBuffers();
            handle->traceBufferCache();
            return status;
        }
        auto status = handle->decodeBatch(images, code_streams, batch_size, params);
        handle->traceBufferCache();
        return status;
    } catch (const std::exception&) {
        return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
}

std::vector<Decoder::Subbatch> Decoder::planSubbatches(const nvimgcodecImageDesc_t** images,
    const nvimgcodecCodeStreamDesc_t** code_streams, int batch_size)
{
    nvtx3::scoped_range marker{"nvtiff planSubbatches"};

    struct Group
    {
        std::vector<int> samples;
        nvtiffCompression_t compression{};
        bool is_hamamatsu = false;
    };
    std::map<std::vector<uint64_t>, Group> groups;
    auto& planner = per_thread_.back();
    // Visit all images of one source together. The planner keeps only one
    // parsed source, so interleaved sources would be parsed again and again.
    std::map<uint64_t, std::vector<int>> sources;
    for (int i = 0; i < batch_size; i++)
        sources[code_streams[i]->io_stream->id].push_back(i);
    XM_CHECK_CUDA(cudaEventSynchronize(planner.event_));
    for (const auto& source : sources) {
        // Many ROI views use the same IFD, so read its metadata only once.
        // The map is per source because IFD indexes of different files are unrelated.
        struct Metadata
        {
            std::vector<uint64_t> compatibility_key;
            nvtiffCompression_t compression;
            bool is_hamamatsu;
        };
        std::map<size_t, Metadata> ifds;
        for (int i : source.second) {
            try {
                if (planner.ensureStreamParsed(code_streams[i]) != NVIMGCODEC_STATUS_SUCCESS)
                    throw std::runtime_error("Cannot parse TIFF source");
                nvimgcodecCodeStreamInfoTiffExt_t tiff{NVIMGCODEC_STRUCTURE_TYPE_TIFF_CODE_STREAM_INFO, sizeof(tiff), nullptr};
                nvimgcodecCodeStreamInfo_t cs{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_INFO, sizeof(cs), &tiff};
                if (code_streams[i]->getCodeStreamInfo(code_streams[i]->instance, &cs) != NVIMGCODEC_STATUS_SUCCESS)
                    throw std::runtime_error("Cannot retrieve TIFF selection");
                const size_t ifd = resolve_ifd_selection(planner.nvtiff_stream_, cs);
                auto metadata = ifds.find(ifd);
                if (metadata == ifds.end()) {
                    auto info = planner.getImageInfo(ifd);
                    nvtiffImageGeometry_t geometry{};
                    XM_CHECK_NVTIFF(nvtiffStreamGetImageGeometry(planner.nvtiff_stream_, ifd, &geometry));
                    const bool is_hamamatsu = isHamamatsu(info.compression, [&](uint16_t tag) {
                        int present = 0;
                        XM_CHECK_NVTIFF(nvtiffStreamHasTag(planner.nvtiff_stream_, ifd, tag, &present));
                        return present != 0;
                    });
                    // All ROI views of this IFD have the same source properties.
                    std::vector<uint64_t> key{source.first, static_cast<uint64_t>(info.compression),
                        static_cast<uint64_t>(info.photometric_int), static_cast<uint64_t>(info.planar_config), info.samples_per_pixel,
                        static_cast<uint64_t>(is_hamamatsu), static_cast<uint64_t>(geometry.type), geometry.strile_width,
                        geometry.strile_height, geometry.strile_depth};
                    for (size_t c = 0; c < info.samples_per_pixel && c < MAX_NUM_SAMPLES; c++) {
                        key.push_back(info.bits_per_sample[c]);
                        key.push_back(info.sample_format[c]);
                    }
                    key.resize(key.size() + 2); // Output format and type are filled per ROI.
                    metadata = ifds.emplace(ifd, Metadata{std::move(key), info.compression, is_hamamatsu}).first;
                }
                nvimgcodecImageInfo_t output{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(output), nullptr};
                if (images[i]->getImageInfo(images[i]->instance, &output) != NVIMGCODEC_STATUS_SUCCESS)
                    throw std::runtime_error("Cannot retrieve output information");

                // Images with the same key can share a native call. The key includes
                // the source id, so images from different files never share a call.
                // nvtiffDecodeCheckSupported makes the final decision.
                auto& key = metadata->second.compatibility_key;
                key[key.size() - 2] = static_cast<uint64_t>(output.sample_format);
                key.back() = static_cast<uint64_t>(output.plane_info[0].sample_type);
                auto& group = groups[key];
                group.samples.push_back(i);
                group.compression = metadata->second.compression;
                group.is_hamamatsu = metadata->second.is_hamamatsu;
            } catch (const std::exception& e) {
                NVIMGCODEC_LOG_WARNING(framework_, plugin_id_, "Cannot group TIFF sample: " << e.what());
                images[i]->imageReady(images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
            }
        }
    }

    std::vector<std::vector<Subbatch>> grouped_batches;
    auto* executor = exec_params_->executor;
    const size_t workers = std::max(1, executor->getNumThreads(executor->instance));
    for (const auto& entry : groups) {
        const auto& group = entry.second;
        auto& batches = grouped_batches.emplace_back();
        const size_t max_images = subbatchSize(
            group.compression, group.is_hamamatsu, group.samples.size(), workers, max_subbatch_images_);
        for (size_t begin = 0; begin < group.samples.size();) {
            const size_t end = begin + std::min(max_images, group.samples.size() - begin);
            batches.emplace_back(group.samples.begin() + begin, group.samples.begin() + end);
            begin = end;
        }
    }
    // Order the subbatches so that each group starts before any group
    // gets its second subbatch.
    std::vector<Subbatch> batches;
    for (size_t part = 0;; part++) {
        const size_t before = batches.size();
        for (auto& group : grouped_batches)
            if (part < group.size())
                batches.push_back(std::move(group[part]));
        if (batches.size() == before)
            break;
    }
    return batches;
}

nvimgcodecStatus_t Decoder::decodeBatch(const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
    int batch_size, const nvimgcodecDecodeParams_t* params)
{
    // Parsed sources use input bytes that are valid only during this call.
    // All worker tasks are complete before this guard closes the sources.
    struct SourceLifetimeGuard
    {
        std::deque<PerThreadResources>& resources;
        ~SourceLifetimeGuard()
        {
            for (auto& resource : resources) {
                resource.closeStream();
                resource.retireBatchBuffers();
            }
        }
    } source_lifetime{per_thread_};
    auto batches = planSubbatches(images, code_streams, batch_size);
    auto* executor = exec_params_->executor;
    if (batches.empty())
        return NVIMGCODEC_STATUS_SUCCESS; // Individual failures were reported above.

    struct Context
    {
        Decoder* decoder;
        const std::vector<Subbatch>& batches;
        const nvimgcodecImageDesc_t** images;
        const nvimgcodecCodeStreamDesc_t** streams;
        const nvimgcodecDecodeParams_t* params;
    } context{this, batches, images, code_streams, params};
    auto task = [](int thread_idx, int sample_idx, void* ptr) {
        auto& c = *static_cast<Context*>(ptr);
        try {
            nvimgcodec::DeviceGuard guard(c.decoder->exec_params_->device_id);
            c.decoder->decodeSubbatch(c.batches[sample_idx], c.images, c.streams, c.params, thread_idx);
        } catch (const std::exception& e) {
            // Device selection and host allocations can fail before
            // decodeSubbatch starts its own error handling.
            NVIMGCODEC_LOG_ERROR(c.decoder->framework_, c.decoder->plugin_id_, "Cannot start TIFF subbatch: " << e.what());
            for (int i : c.batches[sample_idx])
                c.images[i]->imageReady(c.images[i]->instance, NVIMGCODEC_PROCESSING_STATUS_FAIL);
        }
    };
    // The executor runs the subbatches on its worker threads.
    size_t scheduled = 0;
    for (; scheduled < batches.size(); ++scheduled)
        if (executor->schedule(executor->instance, exec_params_->device_id, scheduled, &context, task) != NVIMGCODEC_STATUS_SUCCESS)
            break;
    if (scheduled) {
        const auto run_status = executor->run(executor->instance, exec_params_->device_id);
        const auto wait_status = executor->wait(executor->instance, exec_params_->device_id);
        if (run_status != NVIMGCODEC_STATUS_SUCCESS || wait_status != NVIMGCODEC_STATUS_SUCCESS)
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }
    // If scheduling failed, decode the remaining subbatches on this thread.
    for (size_t i = scheduled; i < batches.size(); ++i)
        task(0, i, &context);
    return NVIMGCODEC_STATUS_SUCCESS;
}

void Decoder::decodeSubbatch(const Subbatch& batch, const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t** code_streams,
    const nvimgcodecDecodeParams_t* params, int thread_idx)
{
    nvtx3::scoped_range marker{"nvtiff decodeSubbatch"};
    auto& resources = per_thread_[workerIndex(thread_idx)];
    const size_t count = batch.size();
    std::vector<nvimgcodecProcessingStatus_t> statuses(count, NVIMGCODEC_PROCESSING_STATUS_FAIL);
    // These structures contain pointers into themselves; construct at final size
    // and never move them while native descriptors are in use.
    std::vector<PreparedDecodeSample> prepared(count);
    cudaStream_t stream = nullptr;
    try {
        XM_CHECK_CUDA(cudaEventSynchronize(resources.event_));
        stream = resources.batchStream();
        resources.used_batch_buffers_ = std::max(resources.used_batch_buffers_, count);
        while (resources.batch_buffers_.size() < count)
            resources.batch_buffers_.emplace_back(exec_params_->device_allocator);
        std::map<nvtiffOutputFormat_t, std::vector<size_t>> formats;
        {
            nvtx3::scoped_range prepare_marker{"nvtiff prepareSubbatch"};
            std::set<cudaStream_t> producer_streams;
            for (size_t j = 0; j < count; j++) {
                const int i = batch[j];
                nvimgcodecImageInfo_t output{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(output), nullptr};
                if (images[i]->getImageInfo(images[i]->instance, &output) != NVIMGCODEC_STATUS_SUCCESS)
                    continue;
                // Wait for work on each caller stream (for example, the output
                // allocation) before this worker stream writes the output.
                if (producer_streams.insert(output.cuda_stream).second) {
                    XM_CHECK_CUDA(cudaEventRecord(resources.event_, output.cuda_stream));
                    XM_CHECK_CUDA(cudaStreamWaitEvent(stream, resources.event_, 0));
                }
                statuses[j] =
                    prepareDecodeSample(prepared[j], resources, resources.batch_buffers_[j], images[i], code_streams[i], params, stream, &output);
                if (statuses[j] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS && !prepared[j].skip_decode)
                    formats[prepared[j].output_format].push_back(j);
            }
        }
        auto& decoder = resources.decoder(stream);
        for (const auto& format : formats) {
            const auto& indices = format.second;
            std::vector<nvtiffDecodeRegion_t> regions;
            std::vector<nvtiffImage_t> outputs;
            for (size_t j : indices) {
                regions.push_back(prepared[j].decode_region);
                outputs.push_back(prepared[j].nvtiff_image);
            }
            XM_CHECK_NVTIFF(nvtiffDecodeParamsSetOutputFormat(resources.decode_params_, format.first));
            // nvTIFF can reject a group for details that the key does not include.
            // If so, split the group in two and try again. Do not send images that
            // nvTIFF supports to the CPU fallback.
            auto decode_range = [&](auto&& self, size_t begin, size_t end) -> void {
                XM_CHECK_NVTIFF(nvtiffDecodeParamsSetRegions(resources.decode_params_, regions.data() + begin, end - begin));
                nvtiffStatus_t ret;
                {
                    nvtx3::scoped_range check_marker{"nvtiffDecodeCheckSupported"};
                    ret = nvtiffDecodeCheckSupported(resources.nvtiff_stream_, decoder, resources.decode_params_, outputs.data() + begin);
                }
                auto failure = NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
                if (ret == NVTIFF_STATUS_SUCCESS) {
                    NVIMGCODEC_LOG_DEBUG(framework_, plugin_id_, "Native TIFF subbatch: " << end - begin << " outputs");
                    if (trace_subbatches_) {
                        size_t bytes = 0;
                        for (size_t k = begin; k < end; ++k)
                            bytes += nvimgcodec::GetBufferSize(prepared[indices[k]].image_info);
                        NVIMGCODEC_LOG_INFO(framework_, plugin_id_, "TIFF_NATIVE_CALL images=" << end - begin << " bytes=" << bytes);
                    }
                    {
                        nvtx3::scoped_range decode_marker{"nvtiffDecode"};
                        ret = nvtiffDecode(resources.nvtiff_stream_, decoder, resources.decode_params_, outputs.data() + begin, stream);
                    }
                    // Wait also for a failed decode before a retry uses the scratch memory.
                    {
                        nvtx3::scoped_range wait_marker{"nvtiff waitDecode"};
                        XM_CHECK_CUDA(cudaStreamSynchronize(stream));
                    }
                    if (ret == NVTIFF_STATUS_SUCCESS)
                        return;
                    failure = NVIMGCODEC_PROCESSING_STATUS_FAIL;
                }
                // A single unsupported or damaged region fails only its own image.
                if (end - begin == 1) {
                    statuses[indices[begin]] = failure;
                } else {
                    const size_t middle = begin + (end - begin) / 2;
                    self(self, begin, middle);
                    self(self, middle, end);
                }
            };
            decode_range(decode_range, 0, indices.size());
        }
        {
            nvtx3::scoped_range postprocess_marker{"nvtiff postprocessSubbatch"};
            for (size_t j = 0; j < count; j++)
                if (statuses[j] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
                    statuses[j] = postprocessPreparedSample(prepared[j], stream);
        }
        XM_CHECK_CUDA(cudaEventRecord(resources.event_, stream));
        XM_CHECK_CUDA(cudaEventSynchronize(resources.event_));
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Native TIFF subbatch failed: " << e.what());
        if (stream)
            XM_CUDA_LOG_DESTROY(cudaStreamSynchronize(stream));
        std::fill(statuses.begin(), statuses.end(), NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }
    for (size_t j = 0; j < count; j++)
        images[batch[j]]->imageReady(images[batch[j]]->instance, statuses[j]);
}

NvTiffCudaDecoderPlugin::NvTiffCudaDecoderPlugin(const nvimgcodecFrameworkDesc_t* framework)
    : decoder_desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), nullptr, this, plugin_id_, "tiff",
          NVIMGCODEC_BACKEND_KIND_GPU_ONLY, static_create, Decoder::static_destroy, Decoder::static_get_metadata,
          Decoder::static_can_decode, Decoder::static_decode_sample, Decoder::static_decode_batch}
    , framework_(framework)
{
}

nvimgcodecDecoderDesc_t* NvTiffCudaDecoderPlugin::getDecoderDesc()
{
    return &decoder_desc_;
}

nvimgcodecStatus_t NvTiffCudaDecoderPlugin::create(
    nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    try {
        NVIMGCODEC_LOG_TRACE(framework_, plugin_id_, "nvtiff_create");
        XM_CHECK_NULL(decoder);
        XM_CHECK_NULL(exec_params);
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(new Decoder(plugin_id_, framework_, exec_params, options));
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(framework_, plugin_id_, "Could not create nvtiff decoder:" << e.what());
        return NVIMGCODEC_STATUS_EXTENSION_INTERNAL_ERROR;
    }
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t NvTiffCudaDecoderPlugin::static_create(
    void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t* exec_params, const char* options)
{
    if (!instance) {
        return NVIMGCODEC_STATUS_EXTENSION_INVALID_PARAMETER;
    }

    NvTiffCudaDecoderPlugin* handle = reinterpret_cast<NvTiffCudaDecoderPlugin*>(instance);
    return handle->create(decoder, exec_params, options);
}

} // namespace nvtiff
