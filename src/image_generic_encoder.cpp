/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "image_generic_encoder.h"
#include <imgproc/device_guard.h>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <unordered_set>
#include "default_executor.h"
#include "encode_state_batch.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iimage.h"
#include "iimage_encoder.h"
#include "iimage_encoder_factory.h"
#include "imgproc/exception.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"

namespace nvimgcodec {

void ImageGenericEncoder::canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format) noexcept
{
    try {
        curr_params_ = params;
        canProcess(code_streams, images, processing_status, force_format);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during canEncode: " << e.what());
        std::fill(processing_status, processing_status + code_streams.size(), NVIMGCODEC_PROCESSING_STATUS_FAIL);
    }
}

ProcessingResultsPromise::FutureImpl ImageGenericEncoder::encode(
    const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params) noexcept
{
    try {
        curr_params_ = params;
        return process(code_streams, images);
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during encode: " << e.what());
        auto promise = std::make_shared<ProcessingResultsPromise>(code_streams.size());
        for (size_t i = 0; i < code_streams.size(); i++) {
            promise->set(i, ProcessingResult::failure(NVIMGCODEC_PROCESSING_STATUS_FAIL));
        }
        return promise->getFuture();
    }
}

nvimgcodecProcessingStatus_t ImageGenericEncoder::canProcessImpl(Entry& sample, ProcessorEntry* processor, int tid) noexcept
{
    NVIMGCODEC_LOG_TRACE(this->logger_, tid << ": " << sample.processor->id_ << " canEncode #" << sample.sample_idx);
    nvimgcodecProcessingStatus_t status;
    try {
        sample.processor->instance_->canEncode(
            sample.code_stream->getCodeStreamDesc(), sample.getImageDesc(), curr_params_, &status, tid);
        NVIMGCODEC_LOG_DEBUG(
            this->logger_, tid << ": " << sample.processor->id_ << " canEncode #" << sample.sample_idx << " returned " << status);
        return status;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during canEncode: " << e.what());
        return NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
}

bool ImageGenericEncoder::processImpl(Entry& sample, int tid) noexcept
{
    try {
        copyToTempBuffers(sample);
        bool encode_ret = sample.processor->instance_->encode(
            sample.code_stream->getCodeStreamDesc(), sample.getImageDesc(), curr_params_, &sample.status, tid);

        assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        bool encode_success = encode_ret && sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
        return encode_success;
    } catch (const std::exception& e) {
        NVIMGCODEC_LOG_ERROR(logger_, "Exception during processImpl: " << e.what());
        sample.status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
        return false;
    }
}

void ImageGenericEncoder::sortSamples()
{
}

bool ImageGenericEncoder::processBatchImpl(ProcessorEntry& processor) noexcept
{
    return true;
}

bool ImageGenericEncoder::copyToTempBuffers(Entry& sample)
{
    nvtx3::scoped_range marker{"copyToTempBuffers"};
    auto& info = sample.image_info;
    auto& processor = sample.processor;
    auto& input_info = sample.orig_image_info;
    bool d2h = processor->backend_kind_ == NVIMGCODEC_BACKEND_KIND_CPU_ONLY &&
               input_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    bool h2d =
        processor->backend_kind_ != NVIMGCODEC_BACKEND_KIND_CPU_ONLY && input_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;

    if (!h2d && !d2h)
        return false;

    if (h2d) {
        sample.device_buffer.resize(info.buffer_size, info.cuda_stream);
        info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        info.buffer = sample.device_buffer.data;
        assert(info.buffer_size == sample.device_buffer.size);
        assert(info.cuda_stream == sample.device_buffer.stream);

    } else if (d2h) {
        sample.pinned_buffer.resize(info.buffer_size, info.cuda_stream);
        info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        info.buffer = sample.pinned_buffer.data;
        assert(info.buffer_size == sample.pinned_buffer.size);
        assert(info.cuda_stream == sample.pinned_buffer.stream);
    } else {
        assert(false); // should not happen
        return false;
    }
    assert(info.buffer_size == input_info.buffer_size);
    auto copy_direction = d2h ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
    NVIMGCODEC_LOG_DEBUG(logger_, "cudaMemcpyAsync " << (d2h ? "D2H" : "H2D") << " stream=" << info.cuda_stream);
    CHECK_CUDA(cudaMemcpyAsync(info.buffer, input_info.buffer, info.buffer_size, copy_direction, info.cuda_stream));
    if (d2h)
        CHECK_CUDA(cudaStreamSynchronize(info.cuda_stream));
    return true;
}

} // namespace nvimgcodec
