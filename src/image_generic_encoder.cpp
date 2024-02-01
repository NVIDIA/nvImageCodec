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
#include "image_generic_encoder.h"
#include <cassert>
#include <condition_variable>
#include <memory>
#include <thread>
#include "device_guard.h"
#include "encode_state_batch.h"
#include "encoder_worker.h"
#include "exception.h"
#include "icodec_registry.h"
#include "icode_stream.h"
#include "icodec.h"
#include "iimage.h"
#include "iimage_encoder.h"
#include "iimage_encoder_factory.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"
#include "default_executor.h"

namespace nvimgcodec {

static std::unique_ptr<IExecutor> GetExecutor(const nvimgcodecExecutionParams_t* exec_params, ILogger* logger)
{
    std::unique_ptr<IExecutor> exec;
    if (exec_params->executor)
        exec = std::make_unique<UserExecutor>(exec_params->executor);
    else
        exec = std::make_unique<DefaultExecutor>(logger, exec_params->max_num_cpu_threads);
    return exec;
}

ImageGenericEncoder::ImageGenericEncoder(ILogger* logger, ICodecRegistry* codec_registry, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    : logger_(logger)
    , codec_registry_(codec_registry)
    , exec_params_(*exec_params)
    , backends_(exec_params->num_backends)
    , options_(options ? options : "")
    , executor_(std::move(GetExecutor(exec_params, logger)))
{
    if (exec_params_.device_id == NVIMGCODEC_DEVICE_CURRENT)
        CHECK_CUDA(cudaGetDevice(&exec_params_.device_id));
    auto backend = exec_params->backends;
    for (int i = 0; i < exec_params_.num_backends; ++i) {
        backends_[i] = *backend;
        ++backend;
    }
    exec_params_.backends = backends_.data();
    exec_params_.executor = executor_->getExecutorDesc();

    if (exec_params_.pre_init) {
        for (size_t codec_idx = 0; codec_idx < codec_registry_->getCodecsCount(); codec_idx++) {
            auto* codec = codec_registry_->getCodecByIndex(codec_idx);
            workers_.emplace(codec, std::make_unique<EncoderWorker>(logger_, this, &exec_params_, options_, codec, 0));
        }
    }
}

ImageGenericEncoder::~ImageGenericEncoder()
{
}

void ImageGenericEncoder::canEncode(const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams,
    const nvimgcodecEncodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format)
{
    std::map<const ICodec*, std::vector<int>> codec2indices;
    for (size_t i = 0; i < code_streams.size(); i++) {
        ICodec* codec = code_streams[i]->getCodec();
        if (!codec || codec->getEncodersNum() == 0) {
            processing_status[i] = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            continue;
        }
        auto it = codec2indices.find(codec);
        if (it == codec2indices.end())
            it = codec2indices.insert(std::pair<const ICodec*, std::vector<int>>(codec, std::vector<int>())).first;
        it->second.push_back(i);
    }

    for (auto& entry : codec2indices) {
        std::vector<ICodeStream*> codec_code_streams(entry.second.size());
        std::vector<IImage*> codec_images(entry.second.size());
        std::vector<int> org_idx(entry.second.size());
        for (size_t i = 0; i < entry.second.size(); ++i) {
            org_idx[i] = entry.second[i];
            codec_code_streams[i] = code_streams[entry.second[i]];
            codec_images[i] = images[entry.second[i]];
        }
        auto worker = getWorker(entry.first);
        while (worker && codec_code_streams.size() != 0) {
            auto encoder = worker->getEncoder();
            std::vector<bool> mask(codec_code_streams.size());
            std::vector<nvimgcodecProcessingStatus_t> status(codec_code_streams.size());
            encoder->canEncode(codec_images, codec_code_streams, params, &mask, &status);
            //filter out ready items
            int removed = 0;
            for (size_t i = 0; i < codec_code_streams.size() + removed; ++i) {
                processing_status[org_idx[i - removed]] = status[i];
                bool can_decode_with_other_format_or_params = (static_cast<unsigned int>(status[i]) & 0b11) == 0b01;
                if (status[i] == NVIMGCODEC_PROCESSING_STATUS_SUCCESS || (!force_format && can_decode_with_other_format_or_params)) {
                    codec_code_streams.erase(codec_code_streams.begin() + i - removed);
                    codec_images.erase(codec_images.begin() + i - removed);
                    org_idx.erase(org_idx.begin() + i - removed);
                    ++removed;
                }
            }
            worker = worker->getFallback();
        }
    }
}

std::unique_ptr<ProcessingResultsFuture> ImageGenericEncoder::encode(
    const std::vector<IImage*>& images, const std::vector<ICodeStream*>& code_streams, const nvimgcodecEncodeParams_t* params)
{
    int N = images.size();
    assert(static_cast<int>(code_streams.size()) == N);

    ProcessingResultsPromise results(N);
    auto future = results.getFuture();

    auto work = createNewWork(std::move(results), params);
    work->init(code_streams, images, std::vector<size_t>{});

    distributeWork(std::move(work));

    return future;
}

std::unique_ptr<Work<nvimgcodecEncodeParams_t>> ImageGenericEncoder::createNewWork(
    const ProcessingResultsPromise& results, const void* params)
{
    if (free_work_items_) {
        std::lock_guard<std::mutex> g(work_mutex_);
        if (free_work_items_) {
            auto ptr = std::move(free_work_items_);
            free_work_items_ = std::move(ptr->next_);
            ptr->results_ = std::move(results);
            ptr->params_ = reinterpret_cast<const nvimgcodecEncodeParams_t*>(params);
            return ptr;
        }
    }

    return std::make_unique<Work<nvimgcodecEncodeParams_t>>(std::move(results), reinterpret_cast<const nvimgcodecEncodeParams_t*>(params));
}

void ImageGenericEncoder::recycleWork(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work)
{
    std::lock_guard<std::mutex> g(work_mutex_);
    work->clear();
    work->next_ = std::move(free_work_items_);
    free_work_items_ = std::move(work);
}

void ImageGenericEncoder::combineWork(Work<nvimgcodecEncodeParams_t>* target, std::unique_ptr<Work<nvimgcodecEncodeParams_t>> source)
{
    //if only one has temporary CPU  storage, allocate it in the other
    if (target->host_temp_buffers_.empty() && !source->host_temp_buffers_.empty())
        target->alloc_host_temps();
    else if (!target->host_temp_buffers_.empty() && source->host_temp_buffers_.empty())
        source->alloc_host_temps();
    //if only one has temporary GPU storage, allocate it in the other
    if (target->device_temp_buffers_.empty() && !source->device_temp_buffers_.empty())
        target->alloc_device_temps();
    else if (!target->device_temp_buffers_.empty() && source->device_temp_buffers_.empty())
        source->alloc_device_temps();

    auto move_append = [](auto& dst, auto& src) {
        dst.reserve(dst.size() + src.size());
        for (auto& x : src)
            dst.emplace_back(std::move(x));
    };

    move_append(target->images_, source->images_);
    move_append(target->code_streams_, source->code_streams_);
    move_append(target->indices_, source->indices_);
    move_append(target->host_temp_buffers_, source->host_temp_buffers_);
    move_append(target->device_temp_buffers_, source->device_temp_buffers_);
    std::move(source->idx2orig_buffer_.begin(), source->idx2orig_buffer_.end(),
        std::inserter(target->idx2orig_buffer_, std::end(target->idx2orig_buffer_)));
    recycleWork(std::move(source));
}

EncoderWorker* ImageGenericEncoder::getWorker(const ICodec* codec)
{
    auto it = workers_.find(codec);
    if (it == workers_.end()) {
        it = workers_.emplace(codec, std::make_unique<EncoderWorker>(logger_, this, &exec_params_, options_, codec, 0)).first;
    }

    return it->second.get();
}

void ImageGenericEncoder::distributeWork(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work)
{
    NVIMGCODEC_LOG_TRACE(logger_, "distributeWork");
    std::map<const ICodec*, std::unique_ptr<Work<nvimgcodecEncodeParams_t>>> dist;
    for (int i = 0; i < work->getSamplesNum(); i++) {
        ICodec* codec = work->code_streams_[i]->getCodec();
        if (!codec) {
            work->results_.set(i, ProcessingResult::failure(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED));
            continue;
        }
        auto& w = dist[codec];
        if (!w)
            w = createNewWork(work->results_, work->params_);
        w->moveEntry(work.get(), i);
    }

    for (auto& [codec, w] : dist) {
        auto worker = getWorker(codec);
        worker->addWork(std::move(w));
    }
}

} // namespace nvimgcodec
