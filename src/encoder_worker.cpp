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
#include "encoder_worker.h"

#include <cassert>

#include "device_guard.h"
#include "icodec.h"
#include "iimage_encoder_factory.h"
#include "log.h"

namespace nvimgcodec {

EncoderWorker::EncoderWorker(ILogger* logger, IWorkManager<nvimgcodecEncodeParams_t>* work_manager,
    const nvimgcodecExecutionParams_t* exec_params, const std::string& options, const ICodec* codec, int index)
    : logger_(logger)
    , work_manager_(work_manager)
    , codec_(codec)
    , index_(index)
    , exec_params_(exec_params)
    , options_(options)
{
    if (exec_params_->pre_init) {
        EncoderWorker* current = this;
        do {
            current->getEncoder(); // initializes the encoder
            current = current->getFallback();
        } while (current);
    }
}

EncoderWorker::~EncoderWorker()
{
    stop();
}

EncoderWorker* EncoderWorker::getFallback()
{
    if (!fallback_) {
        int n = codec_->getEncodersNum();
        if (index_ + 1 < n) {
            fallback_ = std::make_unique<EncoderWorker>(logger_, work_manager_, exec_params_, options_, codec_, index_ + 1);
        }
    }
    return fallback_.get();
}

IImageEncoder* EncoderWorker::getEncoder()
{
    while (!encoder_ && (index_ < codec_->getEncodersNum())) {
        auto encoder_factory = codec_->getEncoderFactory(index_);
        if (encoder_factory) {
            auto backend_kind = encoder_factory->getBackendKind();
            bool backend_allowed = exec_params_->num_backends == 0;
            auto backend = exec_params_->backends;
            for (auto b = 0; b < exec_params_->num_backends; ++b) {
                backend_allowed = backend->kind == backend_kind;
                if (backend_allowed)
                    break;

                ++backend;
            }

            if (backend_allowed) {
                NVIMGCODEC_LOG_DEBUG(logger_, "createEncoder " << encoder_factory->getEncoderId());
                encoder_ = encoder_factory->createEncoder(exec_params_, options_.c_str());
                if (encoder_) {
                    encode_state_batch_ = encoder_->createEncodeStateBatch();
                    is_input_expected_in_device_ = backend_kind != NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
                }
            } else {
                index_++;
            }
        } else {
            index_++;
        }
    }
    return encoder_.get();
}

void EncoderWorker::start()
{
    std::call_once(started_, [&]() { worker_ = std::thread(&EncoderWorker::run, this); });
}

void EncoderWorker::stop()
{
    if (worker_.joinable()) {
        {
            std::lock_guard lock(mtx_);
            stop_requested_ = true;
            work_.reset();
        }
        cv_.notify_all();
        worker_.join();
        worker_ = {};
    }
}

void EncoderWorker::run()
{
    DeviceGuard dg(exec_params_->device_id);
    std::unique_lock lock(mtx_, std::defer_lock);
    while (!stop_requested_) {
        lock.lock();
        cv_.wait(lock, [&]() { return stop_requested_ || work_ != nullptr; });
        if (stop_requested_)
            break;
        assert(work_ != nullptr);
        auto w = std::move(work_);
        lock.unlock();
        processBatch(std::move(w));
    }
}

void EncoderWorker::addWork(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work)
{
    assert(work->getSamplesNum() > 0);
    {
        std::lock_guard guard(mtx_);
        assert((work->images_.size() == work->code_streams_.size()));
        if (work_) {
            work_manager_->combineWork(work_.get(), std::move(work));
            // no need to notify - a work item was already there, so it will be picked up regardless
        } else {
            work_ = std::move(work);
            cv_.notify_one();
        }
    }
    start();
}

static void move_work_to_fallback(Work<nvimgcodecEncodeParams_t>* fb, Work<nvimgcodecEncodeParams_t>* work, const std::vector<bool>& keep)
{
    int moved = 0;
    size_t n = work->code_streams_.size();
    for (size_t i = 0; i < n; i++) {
        if (keep[i]) {
            if (moved) {
                // compact
                if (!work->images_.empty())
                    work->images_[i - moved] = work->images_[i];
                if (!work->host_temp_buffers_.empty())
                    work->host_temp_buffers_[i - moved] = std::move(work->host_temp_buffers_[i]);
                if (!work->device_temp_buffers_.empty())
                    work->device_temp_buffers_[i - moved] = std::move(work->device_temp_buffers_[i]);
                if (!work->idx2orig_buffer_.empty())
                    work->idx2orig_buffer_[i - moved] = std::move(work->idx2orig_buffer_[i]);
                if (!work->code_streams_.empty())
                    work->code_streams_[i - moved] = work->code_streams_[i];
                work->indices_[i - moved] = work->indices_[i];
            }
        } else {
            if (fb)
                fb->moveEntry(work, i);
            moved++;
        }
    }
    if (moved)
        work->resize(n - moved);
}

static void filter_work(Work<nvimgcodecEncodeParams_t>* work, const std::vector<bool>& keep)
{
    move_work_to_fallback(nullptr, work, keep);
}

void EncoderWorker::processBatch(std::unique_ptr<Work<nvimgcodecEncodeParams_t>> work) noexcept
{
    NVIMGCODEC_LOG_TRACE(logger_, "processBatch");
    assert(work->getSamplesNum() > 0);
    assert(work->images_.size() == work->code_streams_.size());

    IImageEncoder* encoder = getEncoder();
    std::vector<bool> mask(work->code_streams_.size());
    std::vector<nvimgcodecProcessingStatus_t> status(work->code_streams_.size());
    if (encoder) {
        NVIMGCODEC_LOG_DEBUG(logger_, "code streams: " << work->code_streams_.size());
        encoder->canEncode(work->images_, work->code_streams_, work->params_, &mask, &status);
    } else {
        NVIMGCODEC_LOG_ERROR(logger_, "Could not create encoder");
        work->results_.setAll(ProcessingResult::failure(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED));
        return;
    }
    std::unique_ptr<Work<nvimgcodecEncodeParams_t>> fallback_work;
    auto fallback_worker = getFallback();
    if (fallback_worker) {
        fallback_work = work_manager_->createNewWork(work->results_, work->params_);
        move_work_to_fallback(fallback_work.get(), work.get(), mask);
        if (!fallback_work->empty()) {
            for (auto idx : fallback_work->indices_) {
                NVIMGCODEC_LOG_WARNING(logger_, "[" << encoder_->encoderId() << "]"
                                                 << " encode #" << idx << " fallback");
            }
            fallback_worker->addWork(std::move(fallback_work));
        }
    } else {
        filter_work(work.get(), mask);
        for (size_t i = 0; i < mask.size(); i++) {
            if (!mask[i])
                work->results_.set(work->indices_[i], ProcessingResult::failure(status[i]));
        }
    }

    if (!work->code_streams_.empty()) {
        work->ensure_expected_buffer_for_encode_each_image(is_input_expected_in_device_);
        auto future = encoder_->encode(encode_state_batch_.get(), work->images_, work->code_streams_, work->params_);

        for (;;) {
            auto indices = future->waitForNew();
            if (indices.second == 0)
                break; // if wait_new returns with an empty result, it means that everything is ready

            for (size_t i = 0; i < indices.second; ++i) {
                int sub_idx = indices.first[i];
                ProcessingResult r = future->getOne(sub_idx);
                if (r.success) {
                    work->clean_after_encoding(is_input_expected_in_device_, sub_idx, &r);
                    work->results_.set(work->indices_[sub_idx], r);
                } else { // failed to encode
                    if (fallback_worker) {
                        // if there's fallback, we don't set the result, but try to use the fallback first
                        NVIMGCODEC_LOG_WARNING(logger_, "[" << encoder_->encoderId() << "]"
                                                         << " encode #" << sub_idx << " fallback");
                        if (!fallback_work)
                            fallback_work = work_manager_->createNewWork(work->results_, work->params_);
                        fallback_work->moveEntry(work.get(), sub_idx);
                    } else {
                        // no fallback - just propagate the result to the original promise
                        work->results_.set(work->indices_[sub_idx], r);
                    }
                }
            }

            if (fallback_work && !fallback_work->empty())
                fallback_worker->addWork(std::move(fallback_work));
        }
    }
    work_manager_->recycleWork(std::move(work));
}

} // namespace nvimgcodec
