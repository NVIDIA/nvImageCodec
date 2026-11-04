/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "parsers/parser_test_utils.h"

namespace nvimgcodec { namespace test {

// Minimal 1x1 and 2x1 24-bit BMPs, recognized by the builtin bmp parser.
inline constexpr uint8_t kOnePixelBmp[] = {0x42, 0x4D, 0x1E, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00,
    0x00, 0x00, 0x01, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00};
inline constexpr uint8_t kTwoPixelBmp[] = {0x42, 0x4D, 0x22, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x1A, 0x00, 0x00, 0x00, 0x0C, 0x00,
    0x00, 0x00, 0x02, 0x00, 0x01, 0x00, 0x01, 0x00, 0x18, 0x00, 0x00, 0x00, 0xFF, 0x00, 0xFF, 0x00, 0x00, 0x00};

constexpr uint8_t kUntouched = 0xCD; // output contents before decoding
constexpr uint8_t kDecoded = 0x42;   // what a fake decoder writes to the outputs it decodes
// What a fake encoder writes to the code streams it encodes.
inline constexpr uint8_t kEncoded[] = {'F', 'A', 'K', 'E'};

// Which decode entry points a fake decoder implements.
enum class DecoderApi
{
    SingleAndBatch, // like nvjpeg_hw_decoder
    BatchOnly,      // like nvjpeg_lossless_decoder (decode == NULL)
    SingleOnly
};

inline const char* ToString(DecoderApi api)
{
    switch (api) {
    case DecoderApi::SingleAndBatch:
        return "SingleAndBatch";
    case DecoderApi::BatchOnly:
        return "BatchOnly";
    case DecoderApi::SingleOnly:
        return "SingleOnly";
    }
    return "?";
}

// How a fake decoder or encoder answers canDecode() / canEncode().
enum class CanProcess
{
    Accept,
    HostBufferOnly, // refuses an image whose buffer is not in host memory, as a CPU plugin may
    Throw
};

inline nvimgcodecProcessingStatus_t AnswerCanProcess(CanProcess behavior, const nvimgcodecImageDesc_t* image)
{
    switch (behavior) {
    case CanProcess::Accept:
        return NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    case CanProcess::HostBufferOnly: {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &info);
        return info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS
                                                                             : NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }
    case CanProcess::Throw:
        throw std::runtime_error("fake plugin failed to check the image");
    }
    return NVIMGCODEC_PROCESSING_STATUS_FAIL;
}

// A decoder for the "bmp" codec. Set its behavior before the nvImageCodec decoder that uses it is created.
// Every callback but create() gets the handle that create() returned, so the framework must never call them
// for a decoder whose create() failed; a NULL handle fails the test.
class FakeDecoder
{
  public:
    FakeDecoder(const char* id, nvimgcodecBackendKind_t backend_kind, DecoderApi api)
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(nvimgcodecDecoderDesc_t), nullptr, this, id, "bmp", backend_kind,
              static_create, static_destroy, static_get_metadata, static_can_decode, api == DecoderApi::BatchOnly ? nullptr : static_decode,
              api == DecoderApi::SingleOnly ? nullptr : static_decode_batch,
              api == DecoderApi::SingleOnly ? nullptr : static_get_mini_batch_size}
    {
    }

    nvimgcodecDecoderDesc_t* getDecoderDesc() { return &desc_; }

    bool create_fails = false; // like nvjpeg_hw_decoder when nvjpegJpegStateCreate fails
    bool decode_fails = false;
    CanProcess can_decode = CanProcess::Accept;
    uint32_t max_width = std::numeric_limits<uint32_t>::max(); // wider images are refused
    std::chrono::milliseconds decode_delay{0};                 // added to every single-image decode()

    std::atomic<int> create_calls{0};
    std::atomic<int> decode_calls{0}; // single-image decode() only
    std::atomic<int> decode_batch_calls{0};

  private:
    static FakeDecoder* fromHandle(nvimgcodecDecoder_t decoder)
    {
        if (!decoder)
            ADD_FAILURE() << "decoder callback called with a NULL handle";
        return reinterpret_cast<FakeDecoder*>(decoder);
    }

    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t*, const char*)
    {
        auto* self = static_cast<FakeDecoder*>(instance);
        self->create_calls++;
        if (self->create_fails)
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        *decoder = reinterpret_cast<nvimgcodecDecoder_t>(self);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_destroy(nvimgcodecDecoder_t decoder)
    {
        return fromHandle(decoder) ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    static nvimgcodecStatus_t static_get_metadata(
        nvimgcodecDecoder_t decoder, const nvimgcodecCodeStreamDesc_t*, nvimgcodecMetadata_t**, int*)
    {
        fromHandle(decoder);
        return NVIMGCODEC_STATUS_IMPLEMENTATION_UNSUPPORTED;
    }

    static nvimgcodecProcessingStatus_t static_can_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t*, const nvimgcodecDecodeParams_t*, int)
    {
        auto* self = fromHandle(decoder);
        if (!self)
            return NVIMGCODEC_PROCESSING_STATUS_FAIL;
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &info);
        if (info.plane_info[0].width > self->max_width)
            return NVIMGCODEC_PROCESSING_STATUS_RESOLUTION_UNSUPPORTED;
        return AnswerCanProcess(self->can_decode, image);
    }

    static nvimgcodecStatus_t static_decode(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t* image,
        const nvimgcodecCodeStreamDesc_t*, const nvimgcodecDecodeParams_t*, int)
    {
        auto* self = fromHandle(decoder);
        if (self) {
            self->decode_calls++;
            std::this_thread::sleep_for(self->decode_delay);
        }
        return finish(self, image) ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }

    static nvimgcodecStatus_t static_decode_batch(nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t** images,
        const nvimgcodecCodeStreamDesc_t**, int batch_size, const nvimgcodecDecodeParams_t*, int)
    {
        auto* self = fromHandle(decoder);
        if (self)
            self->decode_batch_calls++;
        bool all_decoded = true;
        for (int i = 0; i < batch_size; ++i)
            all_decoded = finish(self, images[i]) && all_decoded;
        return all_decoded ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }

    static nvimgcodecStatus_t static_get_mini_batch_size(nvimgcodecDecoder_t decoder, int* batch_size)
    {
        fromHandle(decoder);
        *batch_size = 1;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    // Writes kDecoded to the image's 8-bit output and reports it as decoded, or reports it as failed. A GPU
    // decoder with a host output gets the framework's device buffer, which is copied to the output on the
    // image's CUDA stream afterwards.
    static bool finish(const FakeDecoder* self, const nvimgcodecImageDesc_t* image)
    {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        image->getImageInfo(image->instance, &info);
        bool decoded = self && !self->decode_fails;
        auto* plane_ptr = static_cast<uint8_t*>(info.buffer);
        for (uint32_t p = 0; decoded && p < info.num_planes; ++p) {
            const auto& plane = info.plane_info[p];
            const size_t row_bytes = static_cast<size_t>(plane.width) * plane.num_channels;
            if (info.buffer_kind == NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST) {
                for (uint32_t y = 0; y < plane.height; ++y)
                    memset(plane_ptr + y * plane.row_stride, kDecoded, row_bytes);
            } else if (cudaMemset2DAsync(plane_ptr, plane.row_stride, kDecoded, row_bytes, plane.height, info.cuda_stream) != cudaSuccess) {
                ADD_FAILURE() << self->desc_.id << " could not write its device output";
                decoded = false;
            }
            plane_ptr += plane.row_stride * plane.height;
        }
        image->imageReady(image->instance, decoded ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS : NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return decoded;
    }

    nvimgcodecDecoderDesc_t desc_;
};

// An encoder for the "bmp" codec, with the same rules as FakeDecoder. It writes kEncoded to the code
// stream of each image it encodes.
class FakeEncoder
{
  public:
    FakeEncoder(const char* id, nvimgcodecBackendKind_t backend_kind)
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC, sizeof(nvimgcodecEncoderDesc_t), nullptr, this, id, "bmp", backend_kind,
              static_create, static_destroy, static_can_encode, static_encode}
    {
    }

    nvimgcodecEncoderDesc_t* getEncoderDesc() { return &desc_; }

    bool create_fails = false;
    CanProcess can_encode = CanProcess::Accept;

    std::atomic<int> create_calls{0};
    std::atomic<int> encode_calls{0};

  private:
    static FakeEncoder* fromHandle(nvimgcodecEncoder_t encoder)
    {
        if (!encoder)
            ADD_FAILURE() << "encoder callback called with a NULL handle";
        return reinterpret_cast<FakeEncoder*>(encoder);
    }

    static nvimgcodecStatus_t static_create(void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t*, const char*)
    {
        auto* self = static_cast<FakeEncoder*>(instance);
        self->create_calls++;
        if (self->create_fails)
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        *encoder = reinterpret_cast<nvimgcodecEncoder_t>(self);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_destroy(nvimgcodecEncoder_t encoder)
    {
        return fromHandle(encoder) ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_INVALID_PARAMETER;
    }

    static nvimgcodecProcessingStatus_t static_can_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t*,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t*, int)
    {
        auto* self = fromHandle(encoder);
        return self ? AnswerCanProcess(self->can_encode, image) : NVIMGCODEC_PROCESSING_STATUS_FAIL;
    }

    static nvimgcodecStatus_t static_encode(nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t* code_stream,
        const nvimgcodecImageDesc_t* image, const nvimgcodecEncodeParams_t*, int)
    {
        auto* self = fromHandle(encoder);
        if (self)
            self->encode_calls++;
        const bool encoded = self && write(code_stream);
        image->imageReady(image->instance, encoded ? NVIMGCODEC_PROCESSING_STATUS_SUCCESS : NVIMGCODEC_PROCESSING_STATUS_FAIL);
        return encoded ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_EXECUTION_FAILED;
    }

    static bool write(const nvimgcodecCodeStreamDesc_t* code_stream)
    {
        auto* io_stream = code_stream->io_stream;
        std::vector<uint8_t> bytes(std::begin(kEncoded), std::end(kEncoded));
        size_t written = 0;
        return io_stream->reserve(io_stream->instance, bytes.size()) == NVIMGCODEC_STATUS_SUCCESS &&
               io_stream->seek(io_stream->instance, 0, SEEK_SET) == NVIMGCODEC_STATUS_SUCCESS &&
               io_stream->write(io_stream->instance, &written, bytes.data(), bytes.size()) == NVIMGCODEC_STATUS_SUCCESS &&
               written == bytes.size() && io_stream->flush(io_stream->instance) == NVIMGCODEC_STATUS_SUCCESS;
    }

    nvimgcodecEncoderDesc_t desc_;
};

// Registers fake decoders and encoders. Lower priority values are tried first.
class FakePluginsExtension
{
  public:
    FakePluginsExtension()
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), nullptr, this, "fake_plugins_ext",
              NVIMGCODEC_VER, NVIMGCODEC_VER, static_extension_create, static_extension_destroy}
    {
    }

    void addDecoder(FakeDecoder* decoder, nvimgcodecPriority_t priority) { decoders_.emplace_back(decoder, priority); }
    void addEncoder(FakeEncoder* encoder, nvimgcodecPriority_t priority) { encoders_.emplace_back(encoder, priority); }
    nvimgcodecExtensionDesc_t* getExtensionDesc() { return &desc_; }

  private:
    struct Extension
    {
        Extension(const nvimgcodecFrameworkDesc_t* framework, const FakePluginsExtension* plugins)
            : framework_(framework)
            , plugins_(plugins)
        {
            for (auto& [decoder, priority] : plugins_->decoders_)
                framework_->registerDecoder(framework_->instance, decoder->getDecoderDesc(), priority);
            for (auto& [encoder, priority] : plugins_->encoders_)
                framework_->registerEncoder(framework_->instance, encoder->getEncoderDesc(), priority);
        }
        ~Extension()
        {
            for (auto& [decoder, priority] : plugins_->decoders_)
                framework_->unregisterDecoder(framework_->instance, decoder->getDecoderDesc());
            for (auto& [encoder, priority] : plugins_->encoders_)
                framework_->unregisterEncoder(framework_->instance, encoder->getEncoderDesc());
        }
        const nvimgcodecFrameworkDesc_t* framework_;
        const FakePluginsExtension* plugins_;
    };

    static nvimgcodecStatus_t static_extension_create(
        void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        *extension = reinterpret_cast<nvimgcodecExtension_t>(new Extension(framework, static_cast<FakePluginsExtension*>(instance)));
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_extension_destroy(nvimgcodecExtension_t extension)
    {
        delete reinterpret_cast<Extension*>(extension);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    std::vector<std::pair<FakeDecoder*, nvimgcodecPriority_t>> decoders_;
    std::vector<std::pair<FakeEncoder*, nvimgcodecPriority_t>> encoders_;
    nvimgcodecExtensionDesc_t desc_;
};

// A single-threaded executor that runs the scheduled tasks within run(). Like an executor that enforces
// the documented rule that wait() may only be called after run(), it fails a wait() without one, and
// counts it.
class FakeExecutor
{
  public:
    FakeExecutor()
        : desc_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, sizeof(nvimgcodecExecutorDesc_t), nullptr, this, static_schedule, static_run,
              static_wait, static_get_num_threads}
    {
    }

    nvimgcodecExecutorDesc_t* getExecutorDesc() { return &desc_; }

    bool wait_fails = false; // a wait() after run() fails too, and leaves the work outstanding
    int waits_without_run = 0;

  private:
    using Task = void (*)(int thread_id, int sample_idx, void* task_context);

    static nvimgcodecStatus_t static_schedule(void* instance, int, int sample_idx, void* task_context, Task task)
    {
        static_cast<FakeExecutor*>(instance)->tasks_.push_back({sample_idx, task_context, task});
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_run(void* instance, int)
    {
        auto* self = static_cast<FakeExecutor*>(instance);
        self->ran_ = true;
        auto tasks = std::move(self->tasks_);
        self->tasks_.clear();
        for (const auto& [sample_idx, task_context, task] : tasks)
            task(0, sample_idx, task_context);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t static_wait(void* instance, int)
    {
        auto* self = static_cast<FakeExecutor*>(instance);
        if (!self->ran_) {
            self->waits_without_run++;
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        }
        if (self->wait_fails)
            return NVIMGCODEC_STATUS_EXECUTION_FAILED;
        self->ran_ = false;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static int static_get_num_threads(void*) { return 1; }

    struct ScheduledTask
    {
        int sample_idx;
        void* task_context;
        Task task;
    };
    std::vector<ScheduledTask> tasks_;
    bool ran_ = false;
    nvimgcodecExecutorDesc_t desc_;
};

// Owns the fakes, so they outlive the nvImageCodec objects that call back into them. Derived fixtures destroy
// their decoders, encoders, images and code streams before calling FakePluginsTest::TearDown().
class FakePluginsTest : public ::testing::Test
{
  protected:
    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t create_info{
            NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), 0};
        create_info.load_builtin_modules = 1; // builtin bmp parser
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &create_info));
    }

    void TearDown() override
    {
        if (extension_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension_));
        }
        if (instance_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
        }
    }

    FakeDecoder& AddDecoder(const char* id, nvimgcodecBackendKind_t backend_kind, DecoderApi api, nvimgcodecPriority_t priority)
    {
        decoders_.push_back(std::make_unique<FakeDecoder>(id, backend_kind, api));
        extension_factory_.addDecoder(decoders_.back().get(), priority);
        return *decoders_.back();
    }

    FakeEncoder& AddEncoder(const char* id, nvimgcodecBackendKind_t backend_kind, nvimgcodecPriority_t priority)
    {
        encoders_.push_back(std::make_unique<FakeEncoder>(id, backend_kind));
        extension_factory_.addEncoder(encoders_.back().get(), priority);
        return *encoders_.back();
    }

    // Registers the fakes added so far.
    void RegisterPlugins()
    {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extension_, extension_factory_.getExtensionDesc()));
    }

    // Modelled on DALI's: the HW backend gets a fractional load hint, so a HW decoder is expected to share each
    // batch with the decoders after it. With a custom executor, num_threads is ignored.
    nvimgcodecExecutionParams_t MakeExecParams(bool pre_init, int num_threads, nvimgcodecExecutorDesc_t* executor = nullptr)
    {
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), 0};
        exec_params.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params.max_num_cpu_threads = num_threads;
        exec_params.executor = executor;
        exec_params.backends = backends_.data();
        exec_params.num_backends = static_cast<int>(backends_.size());
        exec_params.pre_init = pre_init ? 1 : 0;
        return exec_params;
    }

    nvimgcodecInstance_t instance_ = nullptr;
    FakeExecutor executor_; // for tests that pass a custom executor

  private:
    static nvimgcodecBackend_t MakeBackend(nvimgcodecBackendKind_t kind, float load_hint)
    {
        return {NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr, kind,
            {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, load_hint,
                NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
    }

    std::vector<std::unique_ptr<FakeDecoder>> decoders_;
    std::vector<std::unique_ptr<FakeEncoder>> encoders_;
    FakePluginsExtension extension_factory_;
    nvimgcodecExtension_t extension_ = nullptr;
    std::vector<nvimgcodecBackend_t> backends_ = {MakeBackend(NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, 0.9f),
        MakeBackend(NVIMGCODEC_BACKEND_KIND_GPU_ONLY, 1.0f), MakeBackend(NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, 1.0f),
        MakeBackend(NVIMGCODEC_BACKEND_KIND_CPU_ONLY, 1.0f)};
};

// Decodes BMPs into 8-bit host outputs with fake decoders.
class FakeDecoderTest : public FakePluginsTest
{
  protected:
    void TearDown() override
    {
        for (auto image : images_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
        }
        for (auto stream : streams_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
        }
        if (decoder_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        }
        FakePluginsTest::TearDown();
    }

    // Registers the fakes and creates the decoder.
    void CreateDecoder(bool pre_init, int num_threads, nvimgcodecExecutorDesc_t* executor = nullptr)
    {
        ASSERT_NO_FATAL_FAILURE(RegisterPlugins());
        auto exec_params = MakeExecParams(pre_init, num_threads, executor);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance_, &decoder_, &exec_params, nullptr));
    }

    // Adds `count` images to decode from `bmp`.
    void AddImages(int count, const uint8_t* bmp = kOnePixelBmp, size_t bmp_size = sizeof(kOnePixelBmp))
    {
        for (int i = 0; i < count; ++i) {
            nvimgcodecCodeStream_t stream = nullptr;
            LoadImageFromHostMemory(instance_, stream, bmp, bmp_size);
            ASSERT_NE(nullptr, stream);
            streams_.push_back(stream);

            nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream, &info));
            size_t size = 0;
            for (uint32_t p = 0; p < info.num_planes; ++p) {
                auto& plane = info.plane_info[p];
                plane.row_stride = static_cast<size_t>(plane.width) * plane.num_channels;
                size += plane.row_stride * plane.height;
            }
            ASSERT_GT(size, 0u);
            auto& output = outputs_.emplace_back(size);
            info.buffer = output.data();
            info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
            nvimgcodecImage_t image = nullptr;
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &info));
            images_.push_back(image);
        }
    }

    // Sets every output byte to kUntouched, then decodes all images.
    std::vector<nvimgcodecProcessingStatus_t> Decode()
    {
        for (auto& output : outputs_)
            std::fill(output.begin(), output.end(), kUntouched);
        std::vector<nvimgcodecProcessingStatus_t> statuses(images_.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        nvimgcodecFuture_t future = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderDecode(decoder_, streams_.data(), images_.data(), static_cast<int>(images_.size()), &params, &future));
        if (!future)
            return statuses;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future));
        size_t status_size = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
        EXPECT_EQ(images_.size(), status_size);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future));
        return statuses;
    }

    std::vector<nvimgcodecProcessingStatus_t> CanDecode()
    {
        std::vector<nvimgcodecProcessingStatus_t> statuses(images_.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), 0};
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(decoder_, streams_.data(), images_.data(),
                                                 static_cast<int>(images_.size()), &params, statuses.data(), 1));
        return statuses;
    }

    bool OutputIs(int image_idx, uint8_t value) const
    {
        const auto& output = outputs_[image_idx];
        return std::all_of(output.begin(), output.end(), [value](uint8_t byte) { return byte == value; });
    }

    nvimgcodecDecoder_t decoder_ = nullptr;
    std::vector<nvimgcodecCodeStream_t> streams_;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<std::vector<uint8_t>> outputs_;
};

}} // namespace nvimgcodec::test
