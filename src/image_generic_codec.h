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

#pragma once

#define NOMINMAX

#include <nvimgcodec.h>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <nvtx3/nvtx3.hpp>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "default_executor.h"
#include "icode_stream.h"
#include "icodec.h"
#include "icodec_registry.h"
#include "iexecutor.h"
#include "iimage.h"
#include "image.h"
#include "imgproc/device_buffer.h"
#include "imgproc/device_guard.h"
#include "imgproc/exception.h"
#include "imgproc/pinned_buffer.h"
#include "log.h"
#include "processing_results.h"
#include "user_executor.h"
#include "work.h"

namespace nvimgcodec {

class IImage;
class ICodeStream;
class ICodecRegistry;
class ICodec;
class ILogger;

template <typename ProcessorEntry>
struct SampleEntry : public IImage
{
    SampleEntry(const nvimgcodecExecutionParams_t* exec_params)
        : image_desc{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC, sizeof(nvimgcodecImageDesc_t), nullptr, this, static_get_image_info, static_image_ready}
        , pinned_buffer(exec_params)
        , device_buffer(exec_params)
    {
    }

    SampleEntry(SampleEntry<ProcessorEntry>&& oth) = default;

    void setIndex(int index) override { sample_idx = index; }
    int getIndex() override { return sample_idx; }

    void setImageInfo(const nvimgcodecImageInfo_t* new_image_info) override { image_info = *new_image_info; }
    void getImageInfo(nvimgcodecImageInfo_t* out_image_info) override { *out_image_info = image_info; }

    void setPromise(std::shared_ptr<ProcessingResultsPromise> new_promise) override { promise = new_promise; }

    std::shared_ptr<ProcessingResultsPromise> getPromise() override { return promise; }

    nvimgcodecStatus_t imageReady(nvimgcodecProcessingStatus_t processing_status)
    {
        status = processing_status;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_get_image_info(void* instance, nvimgcodecImageInfo_t* result)
    {
        assert(instance);
        SampleEntry* handle = reinterpret_cast<SampleEntry*>(instance);
        handle->getImageInfo(result);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_image_ready(void* instance, nvimgcodecProcessingStatus_t processing_status)
    {
        assert(instance);
        SampleEntry* handle = reinterpret_cast<SampleEntry*>(instance);
        handle->imageReady(processing_status);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecImageDesc_t* getImageDesc() override {
        return &image_desc;
    }

    nvimgcodecImageDesc_t image_desc;
    int sample_idx = -1;
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ICodeStream* code_stream = nullptr;
    ICodec* codec = nullptr;
    IImage* orig_image = nullptr; // image descriptor from the user
    nvimgcodecImageInfo_t orig_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ProcessorEntry* processor = nullptr;
    bool copy_pending;
    PinnedBuffer pinned_buffer;
    DeviceBuffer device_buffer;
    std::shared_ptr<ProcessingResultsPromise> promise;

    std::promise<void> setup_done_promise;
    std::shared_future<void> setup_done_future;
    std::promise<void> process_done_promise;
    std::shared_future<void> decode_done_future;

    void reset()
    {
        image_desc.instance = this;
        sample_idx = -1;
        status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        code_stream = nullptr;
        orig_image = nullptr;
        orig_image_info = nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image_info = nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        processor = nullptr;
        copy_pending = false;
        promise.reset();
        setup_done_promise = std::promise<void>{};
        setup_done_future = setup_done_promise.get_future().share();
        process_done_promise = std::promise<void>{};
        decode_done_future = process_done_promise.get_future().share();
    }
};

struct PerThread
{
    explicit PerThread()
    {
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUDA(cudaEventCreate(&event));
    }
    PerThread(const PerThread& oth) = delete;
    PerThread& operator=(const PerThread& oth) = delete;

    PerThread(PerThread&& other) noexcept :
        stream(std::exchange(other.stream, nullptr)),
        event(std::exchange(other.event, nullptr)),
        user_streams(std::move(other.user_streams)),
        last_activity(std::move(other.last_activity))
    {}

    PerThread& operator=(PerThread&& other) noexcept {
        if (this != &other) {
            if (event)
                cudaEventDestroy(event);
            if (stream)
                cudaStreamDestroy(stream);
            stream = std::exchange(other.stream, nullptr);
            event = std::exchange(other.event, nullptr);
            user_streams = std::move(other.user_streams);
            last_activity = std::move(other.last_activity);
        }
        return *this;
    }

    ~PerThread()
    {
        if (event) {
            cudaEventDestroy(event);
            event = nullptr;
        }
        if (stream) {
            cudaStreamDestroy(stream);
            stream = nullptr;
        }
    }

    cudaStream_t stream;
    cudaEvent_t event;
    std::set<cudaStream_t> user_streams;
    std::chrono::time_point<std::chrono::high_resolution_clock> last_activity;
};

// CRTP pattern
template <typename Impl, typename Factory, typename Processor>
class ImageGenericCodec
{
  public:
    struct ProcessorEntry
    {
        const Factory* factory_;
        std::unique_ptr<Processor> instance_;
        std::string id_;
        nvimgcodecBackendKind_t backend_kind_;
        nvimgcodecBackendParams_t backend_params_;
        size_t sample_count_hint_ = 0;
        std::unique_ptr<std::atomic<size_t>> sample_count_;
        ProcessorEntry* fallback_ = nullptr;
    };

    explicit ImageGenericCodec(
        ILogger* logger, ICodecRegistry* codec_registry, const nvimgcodecExecutionParams_t* exec_params, const char* options = nullptr)
        : logger_(logger)
        , codec_registry_(codec_registry)
        , exec_params_(*exec_params)
        , backends_(exec_params->num_backends)
        , options_(options ? options : "")
        , executor_(std::move(GetExecutor(exec_params, logger)))
    {
        if (exec_params_.device_id == NVIMGCODEC_DEVICE_CURRENT) {
            CHECK_CUDA(cudaGetDevice(&exec_params_.device_id));
        }

        auto backend = exec_params->backends;
        for (int i = 0; i < exec_params->num_backends; ++i) {
            backends_[i] = *backend;
            ++backend;
        }
        exec_params_.backends = backends_.data();
        exec_params_.executor = executor_->getExecutorDesc();
        num_threads_ = exec_params_.executor->getNumThreads(exec_params_.executor->instance);

        num_threads_cuda_ = num_threads_;
        for (auto& b : backends_) {
            if ((b.kind == NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU || b.kind == NVIMGCODEC_BACKEND_KIND_GPU_ONLY) &&
                b.params.load_hint < 1.0f) {
                auto candidate_num_threads_cuda = std::round(num_threads_ * b.params.load_hint);
                candidate_num_threads_cuda = candidate_num_threads_cuda < 0 ? 0 : candidate_num_threads_cuda;
                if (candidate_num_threads_cuda < num_threads_cuda_) {
                    num_threads_cuda_ = candidate_num_threads_cuda;
                    NVIMGCODEC_LOG_INFO(
                        logger_, "Setting num_threads_cuda = " << num_threads_cuda_ << " (num_threads = " << num_threads_ << ")");
                }
            }
        }
        per_thread_.resize(num_threads_ + 1);

        size_t total_num_processors = 0;
        for (size_t codec_idx = 0; codec_idx < codec_registry_->getCodecsCount(); codec_idx++) {
            auto* codec = codec_registry_->getCodecByIndex(codec_idx);
            total_num_processors += Impl::getProcessorsNum(codec);
        }
        processors_.resize(total_num_processors);

        size_t processor_count = 0;
        NVIMGCODEC_LOG_TRACE(logger_, "initializing " << codec_registry_->getCodecsCount() << " codecs");
        for (size_t codec_idx = 0; codec_idx < codec_registry_->getCodecsCount(); codec_idx++) {
            auto* codec = codec_registry_->getCodecByIndex(codec_idx);
            NVIMGCODEC_LOG_TRACE(logger_, "initializing " << codec->name());

            ProcessorEntry*& first_processor = codec_to_first_processor_[codec];
            size_t num_processors = Impl::getProcessorsNum(codec);
            NVIMGCODEC_LOG_TRACE(logger_, "initializing " << num_processors << " processors for " << codec->name());
            for (size_t i = 0; i < num_processors; i++) {
                auto* factory = Impl::getFactory(codec, i);
                if (!factory) {
                    continue;
                }
                NVIMGCODEC_LOG_TRACE(logger_, "initializing " << Impl::getId(factory));

                auto backend_kind = factory->getBackendKind();
                auto backend_params = nvimgcodecBackendParams_t{NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t),
                    nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED};
                auto backend = exec_params_.backends;

                bool backend_allowed = true;
                if (exec_params_.num_backends != 0) {
                    backend_allowed = false;
                    for (auto b = 0; b < exec_params_.num_backends; ++b) {
                        if (backend[b].kind == backend_kind) {
                            backend_allowed = true;
                            backend_params = backend[b].params;
                            break;
                        }
                    }
                }
                if (!backend_allowed) {
                    NVIMGCODEC_LOG_DEBUG(logger_, "backend " << backend_kind << " not allowed. Skipping " << Impl::getId(factory));
                    continue;
                }

                assert(processor_count < total_num_processors);
                auto& processor = processors_[processor_count++];
                processor.backend_kind_ = backend_kind;
                processor.backend_params_ = backend_params;
                processor.id_ = Impl::getId(factory);
                processor.factory_ = factory;
                processor.sample_count_ = std::make_unique<std::atomic<size_t>>(0);
                if (exec_params->pre_init) {
                    NVIMGCODEC_LOG_INFO(logger_, "create " << processor.id_ << " load_hint " << processor.backend_params_.load_hint
                                                           << " load_hint_policy " << processor.backend_params_.load_hint_policy);
                    processor.instance_ = Impl::createInstance(processor.factory_, &exec_params_, options_.c_str());
                    if (Impl::hasBatchedAPI(processor.instance_.get()))
                        batched_processors_.insert(&processor);
                }
                if (!first_processor) {
                    first_processor = &processor;
                } else {
                    auto* curr_processor = first_processor;
                    while (curr_processor->fallback_) {
                        curr_processor = curr_processor->fallback_;
                    }
                    curr_processor->fallback_ = &processor;
                }

                NVIMGCODEC_LOG_TRACE(logger_, "initializing " << Impl::getId(factory) << " DONE");
            }
        }
        assert(processor_count <= total_num_processors);
        processors_.resize(processor_count); // actual number of processors instantiated
        for (auto& processor : processors_) {
            if (processor.fallback_ == nullptr &&
                (processor.backend_params_.load_hint < 1.0f ||
                    processor.backend_params_.load_hint_policy == NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME)) {
                NVIMGCODEC_LOG_INFO(
                    logger_, processor.id_ << ": adjusting load hint to 1.0f (non-adaptive), as there's no fallback processor available");
                processor.backend_params_.load_hint = 1.0f;
                processor.backend_params_.load_hint_policy = NVIMGCODEC_LOAD_HINT_POLICY_FIXED;
            }
        }
    }

    virtual ~ImageGenericCodec()
    {
        // deallocate temp buffers before we destroy thread resources
        samples_.clear();
        // destroy processors
        batched_processors_.clear();
        processors_.clear();
        // now destroy thread resources
        per_thread_.clear();
    }

  protected:
    ILogger* logger_;
    ICodecRegistry* codec_registry_;

    std::vector<ICodeStream*> code_streams_;
    std::vector<IImage*> images_;

    int num_samples_ = 0;
    std::vector<SampleEntry<ProcessorEntry>> samples_;
    std::atomic<int> atomic_idx_{0};
    std::atomic<int> atomic_idx2_{0};

    float adaptive_delta_ = 0.1f;
    std::vector<ProcessorEntry> processors_;
    std::set<ProcessorEntry*> batched_processors_;
    std::unordered_map<const ICodec*, ProcessorEntry*> codec_to_first_processor_;

    std::vector<const nvimgcodecCodeStreamDesc_t*> batched_code_stream_descs_;
    std::vector<const nvimgcodecImageDesc_t*> batched_image_descs_;
    std::vector<SampleEntry<ProcessorEntry>*> batched_processed_;

    std::vector<PerThread> per_thread_;
    size_t num_threads_;
    size_t num_threads_cuda_ = 0;

    std::vector<int> curr_order_;
    std::shared_ptr<ProcessingResultsPromise> curr_promise_;

    const nvimgcodecDecodeParams_t* curr_params_;
    nvimgcodecExecutionParams_t exec_params_;
    std::vector<nvimgcodecBackend_t> backends_;
    std::string options_;
    std::unique_ptr<IExecutor> executor_;

    std::chrono::time_point<std::chrono::high_resolution_clock> start_iteration_time_ =
        std::chrono::time_point<std::chrono::high_resolution_clock>::min();
    std::chrono::time_point<std::chrono::high_resolution_clock> last_activity_main_thread_ =
        std::chrono::time_point<std::chrono::high_resolution_clock>::min();

    void preSync(SampleEntry<ProcessorEntry>& sample, int tid)
    {
        auto& t = per_thread_[tid];
        auto user_cuda_stream = sample.orig_image_info.cuda_stream;
        if (t.user_streams.find(user_cuda_stream) == t.user_streams.end()) {
            if (!exec_params_.skip_pre_sync) {
                nvtx3::scoped_range marker{"sync"};
                NVIMGCODEC_LOG_TRACE(logger_, "cudaEventRecord(" << t.event << ", " << user_cuda_stream << ")");
                CHECK_CUDA(cudaEventRecord(t.event, user_cuda_stream));
                NVIMGCODEC_LOG_TRACE(logger_, "cudaStreamWaitEvent(" << t.stream << ", " << t.event << ")");
                CHECK_CUDA(cudaStreamWaitEvent(t.stream, t.event));
            }
            t.user_streams.insert(user_cuda_stream);
        }
    }

    void postSync(int tid)
    {
        auto& t = per_thread_[tid];
        if (!t.user_streams.empty()) {
            nvtx3::scoped_range marker{"sync"};
            for (auto user_cuda_stream : t.user_streams) {
                // this captures the state of t.stream in the cuda event t.event
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": cudaEventRecord(" << t.event << ", " << t.stream << ")");
                CHECK_CUDA(cudaEventRecord(t.event, t.stream));
                // this is so that any post processing on image waits for t.event_ i.e. decoding to finish,
                // without this the post-processing tasks such as encoding, would not know that decoding has finished on this
                // particular image
                NVIMGCODEC_LOG_TRACE(logger_, tid << ": cudaStreamWaitEvent(" << user_cuda_stream << ", " << t.event << ")");
                CHECK_CUDA(cudaStreamWaitEvent(user_cuda_stream, t.event));
            }
        }
    }

    // Each thread can run this in parallel, returns when all samples have been assigned to the preferred processor
    void cooperativeSetup(int tid)
    {
        int ordered_sample_idx = atomic_idx_.load();
        if (ordered_sample_idx >= num_samples_)
            return;
        nvtx3::scoped_range marker{"cooperativeSetup"};
        NVIMGCODEC_LOG_TRACE(logger_, tid << ": cooperativeSetup start");
        while ((ordered_sample_idx = atomic_idx_.fetch_add(1)) < num_samples_) {
            int sample_idx = curr_order_[ordered_sample_idx];
            NVIMGCODEC_LOG_TRACE(logger_, tid << ": cooperativeSetup #" << sample_idx);
            auto& sample = samples_[sample_idx];
            sample.sample_idx = sample_idx;
            sample.code_stream = code_streams_[sample_idx];
            sample.orig_image = images_[sample_idx];
            sample.orig_image_info = nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            sample.orig_image->getImageInfo(&sample.orig_image_info);
            sample.image_info = sample.orig_image_info;
            sample.promise = sample.orig_image->getPromise();
            sample.codec = sample.code_stream->getCodec();
            auto it = codec_to_first_processor_.find(sample.codec);
            assert(it != codec_to_first_processor_.end());
            sample.processor = it->second;
            sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;

            if (!sample.codec || !sample.processor) {
                sample.processor = nullptr;
                sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": failure #" << sample_idx << " : NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED");
                curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
                sample.setup_done_promise.set_value();
                continue;
            }

            for (; sample.processor; sample.processor = sample.processor->fallback_) {
                assert(sample.processor->instance_);
                sample.status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                bool ret = static_cast<Impl*>(this)->canProcessImpl(sample, tid);
                if (ret) {
                    if (sample.processor->fallback_ &&
                        sample.processor->sample_count_->fetch_add(1) >= sample.processor->sample_count_hint_) {
                        NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << sample.processor->id_ << " reached max sample count. Fallback sample #"
                                                          << sample.sample_idx << " to " << sample.processor->fallback_->id_);
                        continue;
                    }
                    break;
                }
            }
            if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": Assign sample #" << sample_idx << " to " << sample.processor->id_);
            } else {
                NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << (sample.processor ? sample.processor->id_ : "") << " set failure #"
                                                  << sample_idx << " status=" << sample.status);
                curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
            }
            NVIMGCODEC_LOG_TRACE(logger_, tid << ": Set promise #" << sample_idx);
            sample.setup_done_promise.set_value();
        }
        NVIMGCODEC_LOG_TRACE(logger_, tid << ": cooperativeSetup done");
    }

    void initState(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images)
    {
        int N = images.size();
        assert(static_cast<int>(code_streams.size()) == N);

        curr_promise_ = std::make_shared<ProcessingResultsPromise>(N);

        code_streams_ = code_streams;
        images_ = images;

        num_samples_ = N;
        samples_.reserve(N);
        while (samples_.size() < static_cast<size_t>(N))
            samples_.emplace_back(&exec_params_);
        while (samples_.size() > static_cast<size_t>(N))
            samples_.pop_back();
        for (auto& sample : samples_) {
            sample.reset();
        }

        assert(code_streams.size() == static_cast<size_t>(num_samples_));

        curr_order_.resize(num_samples_);
        std::iota(curr_order_.begin(), curr_order_.end(), 0);

        for (int sample_idx : curr_order_) {
            auto& sample = samples_[sample_idx];
            sample.sample_idx = sample_idx;
            sample.code_stream = code_streams_[sample_idx];
            sample.orig_image = images_[sample_idx];
            sample.orig_image_info = nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            sample.orig_image->getImageInfo(&sample.orig_image_info);
            sample.image_info = sample.orig_image_info;
            sample.promise = sample.orig_image->getPromise();
            sample.codec = sample.code_stream->getCodec();
            if (!exec_params_.pre_init) {
                if (!sample.codec)
                    continue;
                auto it = codec_to_first_processor_.find(sample.codec);
                assert(it != codec_to_first_processor_.end());
                auto* processor = sample.processor = it->second;
                while (processor) {
                    if (processor->instance_)
                        break;

                    NVIMGCODEC_LOG_INFO(logger_, "create " << processor->id_ << " load_hint " << processor->backend_params_.load_hint
                                                           << " load_hint_policy " << processor->backend_params_.load_hint_policy);
                    processor->instance_ = Impl::createInstance(processor->factory_, &exec_params_, options_.c_str());
                    if (Impl::hasBatchedAPI(processor->instance_.get()))
                        batched_processors_.insert(processor);
                    processor = processor->fallback_;
                }
            }
        }
        for (auto& processor : processors_) {
            processor.sample_count_hint_ = num_samples_;
            processor.sample_count_->store(0);
        }

        atomic_idx_ = 0;
        atomic_idx2_ = 0;
    }

    void canProcess(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
        nvimgcodecProcessingStatus_t* processing_status, int force_format)
    {
        initState(code_streams, images);
        for (int sample_idx : curr_order_) {
            auto& sample = samples_[sample_idx];

            processing_status[sample_idx] = sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            if (!sample.codec) {
                continue;
            }
            while (sample.processor) {
                sample.status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                static_cast<Impl*>(this)->canProcessImpl(sample, 0);
                bool can_decode_with_other_format_or_params = (static_cast<unsigned int>(sample.status) & 0b11) == 0b01;
                if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS || (!force_format && can_decode_with_other_format_or_params)) {
                    break;
                }
                sample.processor = sample.processor->fallback_;
            }
            processing_status[sample_idx] = sample.status;
        }
    }

    ProcessorEntry* nextParallelProcessor(ProcessorEntry* processor)
    {
        if (processor && Impl::hasBatchedAPI(processor->instance_.get()))
            return nextParallelProcessor(processor->fallback_);
        else
            return processor;
    }

    void parallelProcessLoop(int tid)
    {
        auto& t = per_thread_[tid];
        t.user_streams.clear();
        NVIMGCODEC_LOG_TRACE(logger_, tid << ": parallelProcessLoop");

        if (atomic_idx2_.load() < num_samples_) {
            int ordered_sample_idx = 0;
            while ((ordered_sample_idx = atomic_idx2_.fetch_add(1)) < num_samples_) {
                int sample_idx = curr_order_[ordered_sample_idx];
                if (curr_promise_->isSet(sample_idx))
                    continue;
                auto& sample = samples_[sample_idx];

                // there is a chance that the setup hasn't completed yet for this sample, wait a bit.
                sample.setup_done_future.wait();

                // Maybe it was already set by the setup stage
                if (curr_promise_->isSet(sample_idx))
                    continue;

                if (Impl::hasBatchedAPI(sample.processor->instance_.get()))
                    continue;

                auto sample_idx_str = std::to_string(sample_idx);
                nvtx3::scoped_range marker{Impl::process_name() + " #" + sample_idx_str};

                sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
                sample.image_info.cuda_stream = t.stream;

                preSync(sample, tid);

                bool failed = false;
                for (auto*& processor = sample.processor; processor; processor = nextParallelProcessor(processor->fallback_)) {
                    if (failed) {
                        NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << processor->id_ << " canDecode #" << sample.sample_idx);
                        assert(processor->instance_);

                        sample.status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                        assert(!Impl::hasBatchedAPI(processor->instance_.get()));
 
                        failed = !static_cast<Impl*>(this)->canProcessImpl(sample, tid);
                        if (failed)
                            continue;
                    }

                    nvtx3::scoped_range marker{processor->id_ + " decode #" + sample_idx_str};
                    NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << processor->id_ << " decode #" << sample_idx);

                    assert(!curr_promise_->isSet(sample_idx));
                    sample.status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;

                    failed = !static_cast<Impl*>(this)->processImpl(sample, tid);
                    if (!failed)
                        break;
                }

                if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << sample.processor->id_ << " set success #" << sample_idx << " done");
                    curr_promise_->set(sample_idx, ProcessingResult::success());
                } else {
                    assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
                    NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << (sample.processor ? sample.processor->id_ : "") << " set failure #"
                                                      << sample_idx << " status=" << sample.status);
                    curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
                }
                sample.process_done_promise.set_value();
            }
            postSync(tid);
        }

        if (num_threads_cuda_ < num_threads_ && tid < static_cast<int>(num_threads_cuda_)) {
            static_cast<Impl*>(this)->postSyncCudaThreads(tid);
        }
        NVIMGCODEC_LOG_TRACE(logger_, tid << ": leaving");
    }

    ProcessingResultsPromise::FutureImpl process(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images)
    {
        NVIMGCODEC_LOG_INFO(logger_, Impl::process_name() << " num_samples=" << code_streams.size());

        auto executor = executor_->getExecutorDesc();
        assert(static_cast<int>(per_thread_.size()) >= executor->getNumThreads(executor->instance));

        NVIMGCODEC_LOG_TRACE(logger_, "wait for previous iteration");
        executor->wait(executor->instance, exec_params_.device_id);

        std::chrono::milliseconds last_iter_duration_tp(0), last_iter_duration_main(0);
        if (start_iteration_time_ > std::chrono::time_point<std::chrono::high_resolution_clock>::min()) {
            if (!per_thread_.empty()) {
                auto last_activity = per_thread_[0].last_activity;
                for (size_t i = 1; i < per_thread_.size(); i++) {
                    if (per_thread_[i].last_activity > last_activity)
                        last_activity = per_thread_[i].last_activity;
                }
                last_iter_duration_tp = std::chrono::duration_cast<std::chrono::milliseconds>(last_activity - start_iteration_time_);
            }
            last_iter_duration_main =
                std::chrono::duration_cast<std::chrono::milliseconds>(last_activity_main_thread_ - start_iteration_time_);

            NVIMGCODEC_LOG_INFO(
                logger_, "Last iter time thread pool : " << last_iter_duration_tp.count()
                                                         << "ms, Last iter time main thread : " << last_iter_duration_main.count() << "ms");
        }
        start_iteration_time_ = std::chrono::high_resolution_clock::now();

        initState(code_streams, images);

        assert(curr_promise_);
        auto future = curr_promise_->getFuture();

        static_cast<Impl*>(this)->sortSamples();

        adjustBatchSizes(last_iter_duration_main, last_iter_duration_tp);

        auto parallel_process_task = [](int tid, int sample_idx, void* context) -> void {
            auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
            this_ptr->per_thread_[tid].last_activity = std::chrono::high_resolution_clock::now();
            this_ptr->cooperativeSetup(tid);
            this_ptr->parallelProcessLoop(tid);
            this_ptr->per_thread_[tid].last_activity = std::chrono::high_resolution_clock::now();
        };

        if (num_samples_ <= 1 || num_threads_ <= 1) {
            parallel_process_task(0, 0, this);
        } else {
            {
                nvtx3::scoped_range marker{"schedule"};
                NVIMGCODEC_LOG_TRACE(logger_, "schedule work");
                for (size_t task_idx = 0; task_idx < num_threads_; task_idx++) {
                    NVIMGCODEC_LOG_TRACE(logger_, "schedule task #" << task_idx);
                    executor->schedule(executor->instance, exec_params_.device_id, task_idx, this, parallel_process_task);
                }
                NVIMGCODEC_LOG_TRACE(logger_, "run");
                executor->run(executor->instance, exec_params_.device_id);
            }

            NVIMGCODEC_LOG_TRACE(logger_, "cooperativeSetup");
            cooperativeSetup(num_threads_); // one past the last thread idx
        }

        batchProcess();

        last_activity_main_thread_ = std::chrono::high_resolution_clock::now();
        return future;
    }

    void batchProcess()
    {
        NVIMGCODEC_LOG_TRACE(logger_, "batchProcess");
        auto executor = executor_->getExecutorDesc();
        // Batch processors run on the main thread
        NVIMGCODEC_LOG_TRACE(logger_, "batched_processors_ size=" << batched_processors_.size());
        for (auto* processor_ptr : batched_processors_) {
            auto& processor = *processor_ptr;
            assert(processor.instance_ && Impl::hasBatchedAPI(processor.instance_.get()));
            NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ + " start");
            batched_code_stream_descs_.clear();
            batched_image_descs_.clear();
            batched_processed_.clear();
            for (int sample_idx : curr_order_) {
                auto& sample = samples_[sample_idx];
                assert(sample_idx == sample.sample_idx);
                NVIMGCODEC_LOG_TRACE(logger_, "Wait future #" << sample.sample_idx);
                sample.setup_done_future.wait();
                NVIMGCODEC_LOG_TRACE(logger_, "Wait future #" << sample.sample_idx << " DONE");
                if (sample.processor == processor_ptr && sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    batched_processed_.push_back(&sample);
                    batched_code_stream_descs_.push_back(sample.code_stream->getCodeStreamDesc());
                    batched_image_descs_.push_back(sample.getImageDesc());
                }
            }

            if (batched_code_stream_descs_.empty()) {
                NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ + ": no samples for this processor. Skip");
                continue;
            }

            auto ret = static_cast<Impl*>(this)->processBatchImpl(processor);
            size_t fallback_count = 0;
            for (auto* sample : batched_processed_) {
                if (ret && sample->status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ << " set success #" << sample->sample_idx);
                    curr_promise_->set(sample->sample_idx, ProcessingResult::success());
                    sample->process_done_promise.set_value();
                } else if (!sample->processor->fallback_) {
                    NVIMGCODEC_LOG_INFO(logger_, processor.id_ + " set failure #" << sample->sample_idx << " status=" << sample->status);
                    curr_promise_->set(sample->sample_idx, ProcessingResult::failure(sample->status));
                    sample->process_done_promise.set_value();
                } else {
                    NVIMGCODEC_LOG_INFO(
                        logger_, processor.id_ << " failed #" << sample->sample_idx << ". Trying next: " << processor.fallback_->id_);
                    sample->processor = sample->processor->fallback_;
                    auto parallel_process_task = [](int tid, int sample_idx, void* context) -> void {
                        auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
                        this_ptr->per_thread_[tid].last_activity = std::chrono::high_resolution_clock::now();
                        this_ptr->cooperativeSetup(tid);
                        this_ptr->parallelProcessLoop(tid);
                        this_ptr->per_thread_[tid].last_activity = std::chrono::high_resolution_clock::now();
                    };
                    executor->schedule(executor->instance, exec_params_.device_id, sample->sample_idx, this, parallel_process_task);
                    fallback_count++;
                }
            }
            if (fallback_count > 0)
                executor->run(executor->instance, exec_params_.device_id);

            NVIMGCODEC_LOG_DEBUG(logger_, processor.id_ + " DONE");
        }
        NVIMGCODEC_LOG_TRACE(logger_, "batchProcess DONE");
    }

    static std::unique_ptr<IExecutor> GetExecutor(const nvimgcodecExecutionParams_t* exec_params, ILogger* logger)
    {
        if (exec_params->executor)
            return std::make_unique<UserExecutor>(exec_params->executor);
        else
            return std::make_unique<DefaultExecutor>(logger, exec_params->max_num_cpu_threads);
    }

    void adjustBatchSizes(std::chrono::milliseconds last_iter_duration_main, std::chrono::milliseconds last_iter_duration_tp)
    {
        bool adaptive_load_update_done = false;
        float multiplier = 1.0f;
        auto adaptive_load_param_update = [&]() {
            if (adaptive_load_update_done)
                return;
            const float decay_rate = 0.1f;
            const float target = 0.005f; // 0.5% increments at the end
            adaptive_delta_ = adaptive_delta_ + decay_rate * (target - adaptive_delta_);
            if ((last_iter_duration_main + std::chrono::milliseconds(2)) < last_iter_duration_tp) {
                multiplier = 1.0f;
            } else if ((last_iter_duration_tp + std::chrono::milliseconds(2)) < last_iter_duration_main) {
                multiplier = -1.0f;
            } else {
                multiplier = 0.0f;
            }
        };

        auto adjustBatchSize = [](float load_hint, int max_batch_size, int preferred_mini_batch) {
            if (load_hint == 0.0f)
                return 0;
            auto batch_size = static_cast<int>(std::round(load_hint * max_batch_size));
            if (preferred_mini_batch > 0) {
                int tail = batch_size % preferred_mini_batch;
                if (tail > 0) {
                    batch_size = batch_size + preferred_mini_batch - tail;
                }
                if (batch_size > max_batch_size) {
                    batch_size = max_batch_size;
                }
            }
            return batch_size;
        };

        for (auto* processor : batched_processors_) {
            if (!processor->instance_ || !Impl::hasBatchedAPI(processor->instance_.get()) || !processor->fallback_) {
                processor->sample_count_hint_ = num_samples_;
            } else if (processor->backend_params_.load_hint_policy == NVIMGCODEC_LOAD_HINT_POLICY_ADAPTIVE_MINIMIZE_IDLE_TIME) {
                if (!adaptive_load_update_done) {
                    adaptive_load_param_update();
                    adaptive_load_update_done = true;
                }
                auto new_load_hint = std::clamp(processor->backend_params_.load_hint + multiplier * adaptive_delta_, 0.0f, 1.0f);
                NVIMGCODEC_LOG_INFO(logger_,
                    processor->id_ << " adjust load hint from " << processor->backend_params_.load_hint << " to " << new_load_hint);
                processor->backend_params_.load_hint = new_load_hint;
            }
            int minibatch = Impl::getMiniBatchSize(processor->instance_.get());
            processor->sample_count_hint_ = adjustBatchSize(processor->backend_params_.load_hint, num_samples_, minibatch);
            NVIMGCODEC_LOG_INFO(
                logger_, processor->id_ << " adjust batch size to " << processor->sample_count_hint_ << " mini_bs=" << minibatch);
        }
    }
};

} // namespace nvimgcodec
