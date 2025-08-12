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
    {
    }

    SampleEntry(SampleEntry<ProcessorEntry>&& oth) = default;

    void setIndex(int index) override { sample_idx = index; }
    int getIndex() override { return sample_idx; }

    void setImageInfo(const nvimgcodecImageInfo_t* new_image_info) override { image_info = *new_image_info; }
    void getImageInfo(nvimgcodecImageInfo_t* out_image_info) override { *out_image_info = image_info; }

    void setPromise(std::shared_ptr<ProcessingResultsPromise> new_promise) override { /** not used */ }
    std::shared_ptr<ProcessingResultsPromise> getPromise() override { /** not used */ return nullptr; }

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

    nvimgcodecImageDesc_t image_desc{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC, sizeof(nvimgcodecImageDesc_t), nullptr};
    int sample_idx = -1;
    nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    nvimgcodecProcessingStatus_t fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ICodeStream* code_stream = nullptr;
    ICodec* codec = nullptr;
    IImage* orig_image = nullptr; // image descriptor from the user
    nvimgcodecImageInfo_t orig_image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    ProcessorEntry* processor = nullptr;
    bool should_copy = false;
};

struct PerThread
{
    explicit PerThread(int device_id)
    {
        if (device_id == NVIMGCODEC_DEVICE_CPU_ONLY) {
            stream = nullptr;
            event = nullptr;
        } else {
            CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            CHECK_CUDA(cudaEventCreate(&event));
        }
        pinned_buffers.resize(2);
        device_buffers.resize(2);
    }
    PerThread(const PerThread& oth) = delete;
    PerThread& operator=(const PerThread& oth) = delete;

    PerThread(PerThread&& other) noexcept
        : stream(std::exchange(other.stream, nullptr))
        , event(std::exchange(other.event, nullptr))
        , user_streams(std::move(other.user_streams))
        , pinned_buffers(std::move(other.pinned_buffers))
        , device_buffers(std::move(other.device_buffers))
    {
    }

    PerThread& operator=(PerThread&& other) noexcept {
        if (this != &other) {
            if (event) {
                LOG_CUDA_ERROR(cudaEventDestroy(event));
            }

            if (stream) {
                LOG_CUDA_ERROR(cudaStreamDestroy(stream));
            }

            stream = std::exchange(other.stream, nullptr);
            event = std::exchange(other.event, nullptr);
            user_streams = std::move(other.user_streams);
            pinned_buffers = std::move(other.pinned_buffers);
            device_buffers = std::move(other.device_buffers);
        }
        return *this;
    }

    ~PerThread()
    {
        pinned_buffers.clear();
        device_buffers.clear();
        if (event) {
            LOG_CUDA_ERROR(cudaEventDestroy(event));
            event = nullptr;
        }
        if (stream) {
            LOG_CUDA_ERROR(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }

    cudaStream_t stream;
    cudaEvent_t event;
    std::vector<std::pair<int, nvimgcodecProcessingStatus_t>> processed_samples;
    std::set<cudaStream_t> user_streams;   // reusable temporary buffers
    std::vector<PinnedBuffer> pinned_buffers;
    std::vector<DeviceBuffer> device_buffers;
    size_t pinned_buffer_idx = 0;
    size_t device_buffer_idx = 0;
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
        size_t assigned_batch_size_ = 0;
        size_t sample_count_ = 0;
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

        per_thread_.reserve(num_threads_ + 1);
        for (size_t i = 0; i < num_threads_ + 1; i++) {
            per_thread_.emplace_back(exec_params_.device_id);
        }

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
                processor.sample_count_ = 0;

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

    float adaptive_delta_ = 0.1f;
    std::vector<ProcessorEntry> processors_;
    std::set<ProcessorEntry*> batched_processors_;
    std::unordered_map<const ICodec*, ProcessorEntry*> codec_to_first_processor_;

    std::vector<const nvimgcodecCodeStreamDesc_t*> batched_code_stream_descs_;
    std::vector<const nvimgcodecImageDesc_t*> batched_image_descs_;
    std::vector<SampleEntry<ProcessorEntry>*> batched_processed_;

    std::vector<PerThread> per_thread_;

    size_t num_threads_;

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
    std::chrono::time_point<std::chrono::high_resolution_clock> last_activity_ =
        std::chrono::time_point<std::chrono::high_resolution_clock>::min();

    void preSync(SampleEntry<ProcessorEntry>& sample, int tid)
    {
        if (exec_params_.device_id == NVIMGCODEC_DEVICE_CPU_ONLY)
            return;
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
        NVIMGCODEC_LOG_DEBUG(logger_, tid << ": postSync");
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
            t.user_streams.clear();
        }
        if (!t.processed_samples.empty()) {
            curr_promise_->set(t.processed_samples);
            t.processed_samples.clear();
        }
        NVIMGCODEC_LOG_DEBUG(logger_, tid << ": postSync (" << t.processed_samples.size() << " samples) DONE");
    }

    void initSample(int sample_idx) {
        auto& sample = samples_[sample_idx];
        sample.image_desc.instance = &sample;
        sample.should_copy = false;
        sample.sample_idx = sample_idx;
        sample.code_stream = code_streams_[sample_idx];
        sample.orig_image = images_[sample_idx];
        sample.orig_image_info = nvimgcodecImageInfo_t{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        sample.orig_image->getImageInfo(&sample.orig_image_info);
        sample.image_info = sample.orig_image_info;
        sample.codec = sample.code_stream->getCodec();
    }

    void setupSample(int sample_idx, int tid)
    {
        NVIMGCODEC_LOG_TRACE(logger_, tid << ": setupSample #" << sample_idx);
        // If we know the processors were already initialized, we can init the sample in parallel here
        // otherwise it needs to be done earlier
        if (exec_params_.pre_init)
            initSample(sample_idx);

        auto& sample = samples_[sample_idx];

        auto it = codec_to_first_processor_.find(sample.codec);
        assert(it != codec_to_first_processor_.end());
        sample.processor = it->second;
        sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
        sample.fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;

        if (!sample.codec || !sample.processor) {
            sample.processor = nullptr;
            sample.status = NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED;
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": failure #" << sample_idx << " : NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED");
            curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
            return;
        }

        for (; sample.processor; sample.processor = sample.processor->fallback_) {
            sample.fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
            sample.status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor, tid);
            if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                // there's a chance we'd like to fallback. Querying the next decoder just in case.
                if (static_cast<int>(sample.processor->sample_count_hint_) < num_samples_ &&
                    sample_idx >= static_cast<int>(sample.processor->sample_count_hint_) && sample.processor->fallback_) {
                    sample.fallback_status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor->fallback_, tid);
                }
                break;
            }
        }
        if (sample.status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            curr_promise_->set(sample_idx, ProcessingResult::failure(sample.status));
        }
    }

    void cooperativeSetup(int tid) {
        nvtx3::scoped_range marker{"cooperativeSetup"};
        int ordered_sample_idx;
        while ((ordered_sample_idx = atomic_idx_.fetch_add(1)) < num_samples_) {
            int sample_idx = curr_order_[ordered_sample_idx];
            setupSample(sample_idx, tid);
        }
    }

    void completeSetup() {
        nvtx3::scoped_range marker{"completeSetup"};
        for (int sample_idx : curr_order_) {
            auto& sample = samples_[sample_idx];
            if (sample.status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                NVIMGCODEC_LOG_INFO(logger_, "Sample #" << sample.sample_idx << " can't be processed");
                continue;
            }
            auto* first_processor = sample.processor;
            while (sample.processor->sample_count_ >= sample.processor->sample_count_hint_ && sample.processor->fallback_) {
                if (sample.fallback_status == NVIMGCODEC_PROCESSING_STATUS_UNKNOWN) {
                    sample.fallback_status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor->fallback_, 0);
                }
                assert(sample.fallback_status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
                if (sample.fallback_status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
                    NVIMGCODEC_LOG_DEBUG(logger_, "Sample #" << sample.sample_idx << ", moving to next processor: " << sample.processor->fallback_->id_);
                    sample.processor = sample.processor->fallback_;
                    sample.status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
                    sample.fallback_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
                } else {
                    NVIMGCODEC_LOG_DEBUG(logger_, "Sample #" << sample.sample_idx << ", using first processor: " << first_processor->id_);
                    sample.processor = first_processor;
                    break;  // stop search
                }
            }
            assert(sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            NVIMGCODEC_LOG_DEBUG(logger_, "Sample #" << sample.sample_idx << " assigned to " << sample.processor->id_);
            sample.processor->sample_count_++;
        }
    }

    ProcessorEntry* initProcessorsAndGetFirstForCodec(const ICodec* codec) {
        auto it = codec_to_first_processor_.find(codec);
        assert(it != codec_to_first_processor_.end());
        auto* firstProcessor = it->second;
        auto* processor = firstProcessor;
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
        return firstProcessor;
    }

    void initState(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images)
    {
        int N = images.size();
        assert(static_cast<int>(code_streams.size()) == N);

        curr_promise_ = std::make_shared<ProcessingResultsPromise>(N);
        code_streams_ = code_streams;
        images_ = images;

        num_samples_ = N;
        while (static_cast<int>(samples_.size()) < N)
            samples_.emplace_back(&exec_params_);
        while (static_cast<int>(samples_.size()) > N)
            samples_.pop_back();

        assert(code_streams.size() == static_cast<size_t>(num_samples_));
        curr_order_.resize(num_samples_);
        std::iota(curr_order_.begin(), curr_order_.end(), 0);

        if (!exec_params_.pre_init) {
            for (int sample_idx : curr_order_) {
                initSample(sample_idx);
                auto& sample = samples_[sample_idx];
                if (!sample.codec)
                    continue;
                sample.processor = initProcessorsAndGetFirstForCodec(sample.codec);
            }
        }

        for (auto& processor : processors_) {
            processor.sample_count_hint_ = num_samples_;
            processor.sample_count_ = 0;
        }
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
                sample.status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor, 0);
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

    bool shouldProcessSingleImage(int sample_idx) {
        auto& sample = samples_[sample_idx];
        if (curr_promise_->isSet(sample_idx))
            return false;
        else if (Impl::hasBatchedAPI(sample.processor->instance_.get()))
            return false;
        else
            return true;
    }

    void processSample(int sample_idx, int tid)
    {
        auto& t = per_thread_[tid];
        NVIMGCODEC_LOG_TRACE(logger_, tid << ": processSample");

        auto& sample = samples_[sample_idx];
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

                assert(!Impl::hasBatchedAPI(processor->instance_.get()));
                sample.status = static_cast<Impl*>(this)->canProcessImpl(sample, sample.processor, tid);
                failed = (sample.status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
                if (failed)
                    continue;
            }

            nvtx3::scoped_range marker{processor->id_ + " decode #" + sample_idx_str};
            NVIMGCODEC_LOG_INFO(logger_, tid << ": " << processor->id_ << " decode #" << sample_idx);

            assert(!curr_promise_->isSet(sample_idx));
            sample.status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
            failed = !static_cast<Impl*>(this)->processImpl(sample, tid);
            if (!failed)
                break;
        }

        if (sample.status == NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << sample.processor->id_ << " set success #" << sample_idx << " done");
        } else {
            assert(sample.status != NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
            NVIMGCODEC_LOG_DEBUG(logger_, tid << ": " << (sample.processor ? sample.processor->id_ : "") << " set failure #" << sample_idx
                                              << " status=" << sample.status);
        }
        t.processed_samples.emplace_back(sample_idx, sample.status);
        NVIMGCODEC_LOG_DEBUG(logger_, tid << ": processed_samples size=" << t.processed_samples.size());

        NVIMGCODEC_LOG_TRACE(logger_, tid << ": processSample DONE");
    }

    ProcessingResultsPromise::FutureImpl process(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images)
    {
        NVIMGCODEC_LOG_INFO(logger_, Impl::process_name() << " num_samples=" << code_streams.size());
        auto executor = executor_->getExecutorDesc();

        NVIMGCODEC_LOG_TRACE(logger_, "waiting for previous tasks to finish");
        executor->wait(executor->instance, exec_params_.device_id);
        NVIMGCODEC_LOG_TRACE(logger_, "previous tasks finished");

        assert(static_cast<int>(per_thread_.size()) >= executor->getNumThreads(executor->instance));
        std::chrono::milliseconds last_iter_duration(0), last_iter_duration_main(0);
        if (start_iteration_time_ > std::chrono::time_point<std::chrono::high_resolution_clock>::min()) {
            last_iter_duration = std::chrono::duration_cast<std::chrono::milliseconds>(last_activity_ - start_iteration_time_);
            last_iter_duration_main =
                std::chrono::duration_cast<std::chrono::milliseconds>(last_activity_main_thread_ - start_iteration_time_);
            NVIMGCODEC_LOG_INFO(logger_, "Last iter time : " << last_iter_duration.count() << "ms, Last iter time main thread : "
                                                             << last_iter_duration_main.count() << "ms");
        }
        start_iteration_time_ = std::chrono::high_resolution_clock::now();

        initState(code_streams, images);

        assert(curr_promise_);
        auto future = curr_promise_->getFuture();

        static_cast<Impl*>(this)->sortSamples();

        adjustBatchSizes(last_iter_duration_main, last_iter_duration);

        for (auto& t : per_thread_) {
            t.user_streams.clear();
            t.processed_samples.clear();
        }

        int single_image_api_count = 0;

        if (num_samples_ <= 1 || num_threads_ <= 1) {
            for (int sample_idx : curr_order_) {
                setupSample(sample_idx, 0);
            }
            completeSetup();
            for (int sample_idx : curr_order_) {
                if (shouldProcessSingleImage(sample_idx)) {
                    processSample(sample_idx, 0);
                    single_image_api_count++;
                }
            }
            postSync(0);
        } else {
            {
                atomic_idx_.store(0);
                auto setup_task = [](int tid, int sample_idx, void* context) -> void {
                    auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
                    this_ptr->cooperativeSetup(tid);
                };
                for (size_t task_idx = 0; task_idx < num_threads_; task_idx++) {
                    executor->schedule(executor->instance, exec_params_.device_id, task_idx, this, setup_task);
                }
                executor->run(executor->instance, exec_params_.device_id);
                cooperativeSetup(num_threads_);  // one past the last thread idx
                executor->wait(executor->instance, exec_params_.device_id);
                completeSetup();
            }
            {
                nvtx3::scoped_range marker{"schedule parallel"};
                atomic_idx_.store(0);
                for (int i = 0; i < num_samples_; i++) {
                    if (shouldProcessSingleImage(i)) {
                        single_image_api_count++;
                    }
                }
                auto process_task = [](int tid, int task_idx, void* context) -> void {
                    auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
                    int ordered_sample_idx;
                    while ((ordered_sample_idx = this_ptr->atomic_idx_.fetch_add(1)) < this_ptr->num_samples_) {
                        int sample_idx = this_ptr->curr_order_[ordered_sample_idx];
                        if (this_ptr->shouldProcessSingleImage(sample_idx))
                            this_ptr->processSample(sample_idx, tid);
                    }
                    this_ptr->postSync(tid);
                };
                for (int task_idx = 0; task_idx < std::min<int>(num_threads_, single_image_api_count); task_idx++)
                    executor->schedule(executor->instance, exec_params_.device_id, task_idx, this, process_task);
                if (single_image_api_count > 0)
                    executor->run(executor->instance, exec_params_.device_id);
            }
        }

        batchProcess();
        last_activity_main_thread_ = std::chrono::high_resolution_clock::now();
        last_activity_ = std::chrono::high_resolution_clock::now();
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
                } else if (!sample->processor->fallback_) {
                    NVIMGCODEC_LOG_INFO(logger_, processor.id_ + " set failure #" << sample->sample_idx << " status=" << sample->status);
                    curr_promise_->set(sample->sample_idx, ProcessingResult::failure(sample->status));
                } else {
                    NVIMGCODEC_LOG_INFO(
                        logger_, processor.id_ << " failed #" << sample->sample_idx << ". Trying next: " << processor.fallback_->id_);
                    sample->processor = sample->processor->fallback_;
                    auto parallel_process_task = [](int tid, int sample_idx, void* context) -> void {
                        auto* this_ptr = reinterpret_cast<ImageGenericCodec<Impl, Factory, Processor>*>(context);
                        this_ptr->setupSample(sample_idx, tid);
                        assert(this_ptr->shouldProcessSingleImage(sample_idx));
                        this_ptr->processSample(sample_idx, tid);
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

    void adjustBatchSizes(std::chrono::milliseconds last_iter_duration_main, std::chrono::milliseconds last_iter_duration)
    {
        bool adaptive_load_update_done = false;
        float multiplier = 1.0f;
        auto adaptive_load_param_update = [&]() {
            if (adaptive_load_update_done)
                return;
            const float decay_rate = 0.1f;
            const float target = 0.005f; // 0.5% increments at the end
            adaptive_delta_ = adaptive_delta_ + decay_rate * (target - adaptive_delta_);
            if ((last_iter_duration_main + std::chrono::milliseconds(2)) < last_iter_duration) {
                multiplier = 1.0f;
            } else if ((last_iter_duration + std::chrono::milliseconds(2)) < last_iter_duration_main) {
                multiplier = -1.0f;
            } else {
                multiplier = 0.0f;
            }
        };

        auto selectBatchSize = [](float load_hint, int max_batch_size, int preferred_mini_batch) {
            if (load_hint == 0.0f)
                return 0;
            auto batch_size = static_cast<int>(std::round(load_hint * max_batch_size));
            if (preferred_mini_batch > 0) {
                int tail = batch_size % preferred_mini_batch;
                if (tail > 0) {
                    batch_size = batch_size + preferred_mini_batch - tail;
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


            if ((processor->instance_ != nullptr) && (processor->assigned_batch_size_ == 0)) {
                int minibatch = Impl::getMiniBatchSize(processor->instance_.get());
                processor->assigned_batch_size_ =
                    selectBatchSize(processor->backend_params_.load_hint, num_samples_, minibatch);
                NVIMGCODEC_LOG_INFO(
                    logger_, processor->id_ << " selecting max batch size to " << processor->assigned_batch_size_ << " mini_bs=" << minibatch);
            } else {
                NVIMGCODEC_LOG_INFO(
                    logger_, processor->id_ << " Using previous max batch size of " << processor->assigned_batch_size_);
            }
            processor->sample_count_hint_ = std::min<size_t>(processor->assigned_batch_size_, num_samples_);
        }
    }
};

} // namespace nvimgcodec
