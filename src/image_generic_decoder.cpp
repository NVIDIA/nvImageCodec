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
#include "image_generic_decoder.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <unordered_set>
#include "decode_state_batch.h"
#include "default_executor.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iimage.h"
#include "iimage_decoder.h"
#include "iimage_decoder_factory.h"
#include "imgproc/device_guard.h"
#include "imgproc/exception.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"

namespace nvimgcodec {

void ImageGenericDecoder::canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
    const nvimgcodecDecodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format) noexcept
{
    try {
        curr_params_ = params;
        canProcess(code_streams, images, processing_status, force_format);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during canDecode: " << e.what());
        std::fill(processing_status, processing_status + code_streams.size(), NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }
}

ProcessingResultsPromise::FutureImpl ImageGenericDecoder::decode(
    const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params) noexcept
{
    try {
        NVIMGCODEC_LOG_INFO(logger_, "ImageGenericDecoder::decode num_samples=" << code_streams.size());
        curr_params_ = params;
        return process(code_streams, images);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during decode: " << e.what());
        auto promise = std::make_shared<ProcessingResultsPromise>(code_streams.size());
        for (size_t i = 0; i < code_streams.size(); i++) {
            promise->set(i, ProcessingResult::failure(NVIMGCODEC_PROCESSING_STATUS_FAIL));
        }
        return promise->getFuture();
    }
}

void ImageGenericDecoder::sortSamples()
{
    nvtx3::scoped_range marker{"sort samples"};
    assert(curr_order_.size() == code_streams_.size());
    int batch_size = code_streams_.size();
    curr_order_.clear();
    curr_order_.resize(batch_size);
    std::iota(curr_order_.begin(), curr_order_.end(), 0);

    subsampling_score_.resize(batch_size);
    area_.resize(batch_size);
    auto subsamplingScore = [](nvimgcodecChromaSubsampling_t subsampling) {
        switch (subsampling) {
        case NVIMGCODEC_SAMPLING_444:
            return 8;
        case NVIMGCODEC_SAMPLING_422:
            return 7;
        case NVIMGCODEC_SAMPLING_420:
            return 6;
        case NVIMGCODEC_SAMPLING_440:
            return 5;
        case NVIMGCODEC_SAMPLING_411:
            return 4;
        case NVIMGCODEC_SAMPLING_410:
            return 3;
        case NVIMGCODEC_SAMPLING_GRAY:
            return 2;
        case NVIMGCODEC_SAMPLING_410V:
        default:
            return 1;
        }
    };
    for (int i = 0; i < batch_size; i++) {
        nvimgcodecImageInfo_t cs_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        code_streams_[i]->getImageInfo(&cs_image_info);
        subsampling_score_[i] = subsamplingScore(cs_image_info.chroma_subsampling);
        area_[i] = cs_image_info.plane_info[0].height * cs_image_info.plane_info[0].width;
    }

    // Sort in descending order
    std::sort(curr_order_.begin(), curr_order_.end(), [&](int lhs, int rhs) {
        if (subsampling_score_[lhs] != subsampling_score_[rhs])
            return subsampling_score_[lhs] > subsampling_score_[rhs]; // descending order
        if (area_[lhs] != area_[rhs])
            return area_[lhs] > area_[rhs]; // descending order
        else
            return lhs < rhs; // ascending order
    });
}

nvimgcodecProcessingStatus_t ImageGenericDecoder::canProcessImpl(Entry& sample, ProcessorEntry* processor, int tid) noexcept
{
    NVIMGCODEC_LOG_TRACE(this->logger_, tid << ": " << processor->id_ << " canDecode #" << sample.sample_idx);
    nvimgcodecProcessingStatus_t status;
    try {
        processor->instance_->canDecode(
            sample.getImageDesc(), sample.code_stream->getCodeStreamDesc(), curr_params_, &status, tid);
        NVIMGCODEC_LOG_DEBUG(
            this->logger_, tid << ": " << processor->id_ << " canDecode #" << sample.sample_idx << " returned " << status);
        return status;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during canDecode: " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

bool ImageGenericDecoder::processImpl(Entry& sample, int tid) noexcept
{
    try {
        sample.copy_pending = allocateTempBuffers(sample);
        auto decode_ret =
            sample.processor->instance_->decode(sample.getImageDesc(), sample.code_stream->getCodeStreamDesc(), curr_params_, tid);
        assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        bool decode_success = decode_ret && sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        sample.copy_pending = sample.copy_pending && decode_success;
        if (decode_success && sample.copy_pending && tid < static_cast<int>(num_threads_cuda_)) {
            nvtx3::scoped_range marker{"copyToOutputBuffer #" + std::to_string(sample.sample_idx)};
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << sample.processor->id_ << " copyToOutputBuffer #" << sample.sample_idx);
            copyToOutputBuffer(sample.orig_image_info, sample.image_info);
            sample.copy_pending = false;
        }
        return decode_success;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during decode: " << e.what());
        sample.status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return false;
    }
}

bool ImageGenericDecoder::processBatchImpl(ProcessorEntry& processor) noexcept
{
    try {
        nvtx3::scoped_range marker{processor.id_ + " decodeBatch"};
        NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ << " decodeBatch");

        for (auto& sample : batched_processed_) {
            sample->copy_pending = allocateTempBuffers(*sample);
        }
        auto ret = processor.instance_->decodeBatch(
            batched_image_descs_.data(), batched_code_stream_descs_.data(), curr_params_, batched_processed_.size(), 0);
        if (ret != NVIMGCODEC_STATUS_SUCCESS)
            return false;
        for (auto* sample : batched_processed_) {
            if (ret == NVIMGCODEC_STATUS_SUCCESS && sample->status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                if (sample->copy_pending) {
                    NVIMGCODEC_LOG_DEBUG(logger_, sample->processor->id_ << " copyToOutputBuffer #" << sample->sample_idx);
                    nvtx3::scoped_range marker{"copyToOutputBuffer " + std::to_string(sample->sample_idx)};
                    copyToOutputBuffer(sample->orig_image_info, sample->image_info);
                    sample->copy_pending = false;
                }
            }
        }
        return true;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during decodeBatch: " << e.what());
        for (auto* sample : batched_processed_) {
            sample->status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        }
        return false;
    }
}

bool ImageGenericDecoder::allocateTempBuffers(Entry& sample)
{
    auto backend_kind = sample.processor->backend_kind_;
    auto& info = sample.image_info;
    bool h2d = sample.orig_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE &&
               backend_kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
    bool d2h =
        sample.orig_image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST && backend_kind != NVIMGCODEC_BACKEND_KIND_CPU_ONLY;

    if (!h2d && !d2h)
        return false;

    if (h2d) {
        nvtx3::scoped_range marker{"allocateTempBuffers"};
        NVIMGCODEC_LOG_DEBUG(logger_, "pinned_buffer.resize");
        sample.pinned_buffer.resize(info.buffer_size, info.cuda_stream);
        info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        info.buffer = sample.pinned_buffer.data;
        assert(info.buffer_size == sample.pinned_buffer.size);
        assert(info.cuda_stream == sample.pinned_buffer.stream);
    } else if (d2h) {
        nvtx3::scoped_range marker{"allocateTempBuffers"};
        NVIMGCODEC_LOG_DEBUG(logger_, "device_buffer.resize");
        sample.device_buffer.resize(info.buffer_size, info.cuda_stream);
        info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        info.buffer = sample.device_buffer.data;
        assert(info.buffer_size == sample.device_buffer.size);
        assert(info.cuda_stream == sample.device_buffer.stream);
    } else {
        assert(false); // should not happen
        return false;
    }
    return true;
}

void ImageGenericDecoder::copyToOutputBuffer(const nvimgcodecImageInfo_t& output_info, const nvimgcodecImageInfo_t& info)
{
    nvtx3::scoped_range marker{"copyToOutputBuffer"};
    bool d2h = output_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST &&
               info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    bool h2d = output_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE &&
               info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    assert((static_cast<int>(d2h) + static_cast<int>(h2d)) == 1);
    (void)h2d;
    assert(info.buffer_size == output_info.buffer_size);
    auto copy_direction = d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    NVIMGCODEC_LOG_DEBUG(logger_, "cudaMemcpyAsync " << (d2h ? "D2H" : "H2D") << " stream=" << info.cuda_stream);
    CHECK_CUDA(cudaMemcpyAsync(output_info.buffer, info.buffer, info.buffer_size, copy_direction, info.cuda_stream));
    if (d2h)
        CHECK_CUDA(cudaStreamSynchronize(info.cuda_stream));
}

} // namespace nvimgcodec
