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

#pragma once

#include <nvimgcodec.h>
#include <cassert>
#include <map>
#include <vector>
#include "exception.h"
#include "icode_stream.h"
#include "iimage.h"
#include "processing_results.h"

namespace nvimgcodec {

/**
 * @brief Describes a sub-batch of work to be processed
 *
 * This object contains a (shared) promise from the original request containing the full batch
 * and a mapping of indices from this sub-batch to the full batch.
 * It also contains a subset of relevant sample views, sources, etc.
 */
template <typename T>
struct Work
{
    Work(const ProcessingResultsPromise& results, const T* params)
        : results_(std::move(results))
        , params_(std::move(params))
    {
    }

    void clear()
    {
        indices_.clear();
        code_streams_.clear();
        images_.clear();
        host_temp_buffers_.clear();
        device_temp_buffers_.clear();
        idx2orig_buffer_.clear();
    }

    int getSamplesNum() const { return indices_.size(); }

    bool empty() const { return indices_.empty(); }

    void resize(int num_samples)
    {
        indices_.resize(num_samples);
        code_streams_.resize(num_samples);
        if (!images_.empty())
            images_.resize(num_samples);
    }

    void init(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
              const std::vector<size_t>& order)
    {
        int N = images.size();
        assert(static_cast<int>(order.size()) == N || order.empty());
        indices_.reserve(N);
        images_.reserve(N);
        code_streams_.reserve(N);
        if (!order.empty()) {
            for (int i = 0; i < N; i++) {
                int sample_idx = order[i];
                indices_.push_back(sample_idx);
                images_.push_back(images[sample_idx]);
                code_streams_.push_back(code_streams[sample_idx]);
            }
        } else {
            for (int i = 0; i < N; i++) {
                indices_.push_back(i);
                images_.push_back(images[i]);
                code_streams_.push_back(code_streams[i]);
            }
        }
    }

    /**
   * @brief Moves one work entry from another work to this one
   */
    void moveEntry(Work* from, int which)
    {
        code_streams_.push_back(from->code_streams_[which]);
        indices_.push_back(from->indices_[which]);
        if (!from->images_.empty())
            images_.push_back(from->images_[which]);
        if (!from->host_temp_buffers_.empty())
            host_temp_buffers_.push_back(std::move(from->host_temp_buffers_[which]));
        if (!from->device_temp_buffers_.empty())
            device_temp_buffers_.push_back(std::move(from->device_temp_buffers_[which]));
        if (from->idx2orig_buffer_.find(which) != from->idx2orig_buffer_.end()) {
            auto entry = from->idx2orig_buffer_.extract(which);
            idx2orig_buffer_.insert(std::move(entry));
        }
    }

    /**
   * @brief Allocates temporary CPU inputs/outputs for this sub-batch
   *
   * This function is used when falling back from GPU to CPU decoder.
   */

    void alloc_host_temps()
    {
        host_temp_buffers_.clear();
        for (int i = 0, n = indices_.size(); i < n; i++) {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            images_[i]->getImageInfo(&image_info);
            void* h_pinned = nullptr;
            CHECK_CUDA(cudaMallocHost(&h_pinned, image_info.buffer_size));
            std::unique_ptr<void, decltype(&cudaFreeHost)> h_ptr(h_pinned, &cudaFreeHost);
            host_temp_buffers_.push_back(std::move(h_ptr));
        }
    }

    void alloc_device_temps()
    {
        device_temp_buffers_.clear();
        for (int i = 0, n = indices_.size(); i < n; i++) {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            images_[i]->getImageInfo(&image_info);
            void* device_buffer = nullptr;
            CHECK_CUDA(cudaMalloc(&device_buffer, image_info.buffer_size));
            std::unique_ptr<void, decltype(&cudaFree)> d_ptr(device_buffer, &cudaFree);
            device_temp_buffers_.push_back(std::move(d_ptr));
        }
    }

    void ensure_expected_buffer_for_decode_each_image(bool is_device_output)
    {
        for (size_t i = 0; i < images_.size(); ++i) {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            images_[i]->getImageInfo(&image_info);

            if (!is_device_output && image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (host_temp_buffers_.empty()) {
                    alloc_host_temps();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
                image_info.buffer = host_temp_buffers_[i].get();
                images_[i]->setImageInfo(&image_info);
            }
            if (is_device_output && image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
                if (device_temp_buffers_.empty()) {
                    alloc_device_temps();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                image_info.buffer = device_temp_buffers_[i].get();
                images_[i]->setImageInfo(&image_info);
            }
        }
    }

    void copy_buffer_if_necessary(bool is_device_output, int sub_idx, ProcessingResult* r)
    {
        auto it = idx2orig_buffer_.find(sub_idx);
        try {
            if (it != idx2orig_buffer_.end()) {
                nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                images_[sub_idx]->getImageInfo(&image_info);
                auto copy_direction = is_device_output ? cudaMemcpyDeviceToHost : cudaMemcpyHostToDevice;
                CHECK_CUDA(cudaMemcpyAsync(it->second, image_info.buffer, image_info.buffer_size, copy_direction, image_info.cuda_stream));
                image_info.buffer = it->second;
                image_info.buffer_kind = image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE
                                             ? NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST
                                             : NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                images_[sub_idx]->setImageInfo(&image_info);
                idx2orig_buffer_.erase(it);
            }
        } catch (...) {
            *r = ProcessingResult::failure(std::current_exception());
        }
    }

    void ensure_expected_buffer_for_encode_each_image(bool is_input_expected_in_device)
    {
        cudaEvent_t event;
        CHECK_CUDA(cudaEventCreate(&event));

        for (size_t i = 0; i < images_.size(); ++i) {
            nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            images_[i]->getImageInfo(&image_info);

            if (!is_input_expected_in_device && image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE) {
                if (host_temp_buffers_.empty()) {
                    this->alloc_host_temps();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer = host_temp_buffers_[i].get();
                image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
                images_[i]->setImageInfo(&image_info);

                CHECK_CUDA(cudaMemcpyAsync(
                    image_info.buffer, idx2orig_buffer_[i], image_info.buffer_size, cudaMemcpyDeviceToHost, image_info.cuda_stream));
                CHECK_CUDA(cudaEventRecord(event, image_info.cuda_stream));
            }
            if (is_input_expected_in_device && image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
                if (device_temp_buffers_.empty()) {
                    this->alloc_device_temps();
                }
                idx2orig_buffer_[i] = image_info.buffer;
                image_info.buffer = device_temp_buffers_[i].get();
                image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                images_[i]->setImageInfo(&image_info);

                CHECK_CUDA(cudaMemcpyAsync(
                    image_info.buffer, idx2orig_buffer_[i], image_info.buffer_size, cudaMemcpyHostToDevice, image_info.cuda_stream));
                CHECK_CUDA(cudaEventRecord(event, image_info.cuda_stream));
            }
            images_[i]->setImageInfo(&image_info);
        }

        CHECK_CUDA(cudaEventSynchronize(event));
        CHECK_CUDA(cudaEventDestroy(event));
    }

    void clean_after_encoding(bool is_input_expected_in_device, int sub_idx, ProcessingResult* r)
    {
        auto it = idx2orig_buffer_.find(sub_idx);
        if (it != idx2orig_buffer_.end()) {
            try {
                nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
                images_[sub_idx]->getImageInfo(&image_info);
                image_info.buffer = it->second;
                image_info.buffer_kind = image_info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE
                                             ? NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST
                                             : NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
                images_[sub_idx]->setImageInfo(&image_info);
                idx2orig_buffer_.erase(it);
            } catch (...) {
                *r = ProcessingResult::failure(std::current_exception());
            }
        }
    }

    // The original promise
    ProcessingResultsPromise results_;
    // The indices in the original request
    std::vector<int> indices_;
    std::vector<ICodeStream*> code_streams_;
    std::vector<IImage*> images_;
    std::vector<std::unique_ptr<void, decltype(&cudaFreeHost)>> host_temp_buffers_;
    std::vector<std::unique_ptr<void, decltype(&cudaFree)>> device_temp_buffers_;
    std::map<int, void*> idx2orig_buffer_;
    const T* params_;
    std::unique_ptr<Work> next_;
};

} // namespace nvimgcodec