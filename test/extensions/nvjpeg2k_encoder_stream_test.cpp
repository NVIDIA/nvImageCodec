/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <algorithm>
#include <chrono>
#include <cstring>
#include <memory>
#include <random>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <extensions/nvjpeg2k/cuda_encoder.h>

#include "nvjpeg2k_ext_test_common.h"

namespace nvimgcodec { namespace test {
namespace {

struct EncoderOutputBuffer
{
    ~EncoderOutputBuffer()
    {
        if (pinned_data)
            (void)cudaFreeHost(pinned_data);
    }

    static unsigned char* Resize(void* ctx, size_t bytes)
    {
        auto& buffer = *static_cast<EncoderOutputBuffer*>(ctx);
        if (!buffer.pinned_data) {
            buffer.pageable_data.resize(bytes);
            buffer.size = bytes;
            return buffer.pageable_data.data();
        }
        if (bytes > buffer.capacity)
            throw std::length_error("Pinned test output capacity exceeded");
        buffer.size = bytes;
        return buffer.pinned_data;
    }

    unsigned char* data() { return pinned_data ? pinned_data : pageable_data.data(); }

    unsigned char* pinned_data = nullptr;
    size_t capacity = 0;
    size_t size = 0;
    std::vector<unsigned char> pageable_data;
};

} // namespace

class NvJpeg2kExtEncoderStreamTest : public NvJpeg2kExtTestBase, public testing::Test
{
  protected:
    void SetUp() override
    {
        NvJpeg2kExtTestBase::SetUp();
        ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
        image_info_.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        image_info_.sample_format = NVIMGCODEC_SAMPLEFORMAT_P_RGB;
        image_info_.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        image_info_.plane_info[0].width = 128;
        image_info_.plane_info[0].height = 128;
        image_info_.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        image_info_.plane_info[0].precision = 8;
        PrepareImageForPlanarFormat();
        std::mt19937 rng(20261005);
        for (auto& value : image_buffer_)
            value = static_cast<unsigned char>(rng());
        zeros_.resize(image_buffer_.size(), 0);
        ASSERT_EQ(cudaSuccess, cudaMalloc(&device_buffer_, image_buffer_.size()));
        image_info_.buffer = device_buffer_;
        image_info_.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
        image_info_.cuda_stream = stream_;
        strcpy(image_info_.codec_name, "jpeg2k");
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &in_image_, &image_info_));

        backend_.kind = NVIMGCODEC_BACKEND_KIND_GPU_ONLY;
        backend_.params = {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr,
            1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED};
        exec_params_.device_id = NVIMGCODEC_DEVICE_CURRENT;
        exec_params_.max_num_cpu_threads = 1;
        exec_params_.num_backends = 1;
        exec_params_.backends = &backend_;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params_, nullptr));

        jpeg2k_params_.stream_type = NVIMGCODEC_JPEG2K_STREAM_J2K;
        jpeg2k_params_.prog_order = NVIMGCODEC_JPEG2K_PROG_ORDER_LRCP;
        jpeg2k_params_.num_resolutions = 2;
        jpeg2k_params_.code_block_w = 32;
        jpeg2k_params_.code_block_h = 32;
        jpeg2k_params_.mct_mode = 1;
        params_.quality_type = NVIMGCODEC_QUALITY_TYPE_LOSSLESS;
    }

    void TearDown() override
    {
        if (stream_) {
            EXPECT_EQ(cudaSuccess, cudaStreamSynchronize(stream_));
        }
        if (encoder_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        NvJpeg2kExtTestBase::TearDown();
        if (device_buffer_) {
            EXPECT_EQ(cudaSuccess, cudaFree(device_buffer_));
        }
        if (stream_) {
            EXPECT_EQ(cudaSuccess, cudaStreamDestroy(stream_));
        }
    }

    void Encode(nvimgcodecCodeStream_t code_stream, const std::vector<unsigned char>& input,
        nvimgcodecProcessingStatus_t expected_status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS)
    {
        ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(device_buffer_, input.data(), input.size(), cudaMemcpyHostToDevice, stream_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderEncode(encoder_, &in_image_, &code_stream, 1, &params_, &future_));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future_));
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t count = 0;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future_, &status, &count));
        EXPECT_EQ(1, count);
        EXPECT_EQ(expected_status, status);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future_));
        future_ = nullptr;
    }

    nvimgcodecEncoder_t encoder_ = nullptr;
    cudaStream_t stream_ = nullptr;
    void* device_buffer_ = nullptr;
    std::vector<unsigned char> zeros_;
    nvimgcodecBackend_t backend_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr};
    nvimgcodecExecutionParams_t exec_params_{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
    nvimgcodecJpeg2kEncodeParams_t jpeg2k_params_{
        NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), nullptr};
    nvimgcodecEncodeParams_t params_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), &jpeg2k_params_};
    EncoderOutputBuffer output_;
    EncoderOutputBuffer reference_;
};

class NvJpeg2kExtEncoderStreamReuseTest : public NvJpeg2kExtEncoderStreamTest,
                                        public testing::WithParamInterface<std::tuple<bool, bool>>
{};

TEST_P(NvJpeg2kExtEncoderStreamReuseTest, PersistentNativeStreamTracksExactSizeAndBytes)
{
    jpeg2k_params_.ht = std::get<0>(GetParam());
    if (std::get<1>(GetParam())) {
        output_.capacity = 1 << 20;
        ASSERT_EQ(cudaSuccess, cudaMallocHost(reinterpret_cast<void**>(&output_.pinned_data), output_.capacity));
        cudaPointerAttributes attributes{};
        ASSERT_EQ(cudaSuccess, cudaPointerGetAttributes(&attributes, output_.pinned_data));
        ASSERT_EQ(cudaMemoryTypeHost, attributes.type);
    }

    std::vector<std::vector<unsigned char>> expected;
    for (const auto* input : {&image_buffer_, &zeros_}) {
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateToHostMem(instance_, &in_code_stream_, &reference_, EncoderOutputBuffer::Resize, &image_info_));
        ASSERT_NO_FATAL_FAILURE(Encode(in_code_stream_, *input));
        expected.emplace_back(reference_.data(), reference_.data() + reference_.size);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(in_code_stream_));
        in_code_stream_ = nullptr;
    }
    ASSERT_GT(expected[0].size(), expected[1].size());
    if (output_.pinned_data) {
        ASSERT_LE(expected[0].size(), output_.capacity);
    }

    // Create once: unlike Python CodeStream.reuse(), this preserves the underlying
    // MemIoStream and its previous logical size across all encode calls.
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, &output_, EncoderOutputBuffer::Resize, &image_info_));
    for (int iteration = 0; iteration < 16; ++iteration) {
        SCOPED_TRACE(iteration);
        size_t index = iteration % 2;
        ASSERT_NO_FATAL_FAILURE(Encode(out_code_stream_, index ? zeros_ : image_buffer_));
        ASSERT_EQ(expected[index].size(), output_.size);
        EXPECT_EQ(0, memcmp(expected[index].data(), output_.data(), output_.size));
    }
}

INSTANTIATE_TEST_SUITE_P(Lossless, NvJpeg2kExtEncoderStreamReuseTest, testing::Combine(testing::Bool(), testing::Bool()));

TEST_F(NvJpeg2kExtEncoderStreamTest, PinnedOutputCapacityFailureIsReported)
{
    output_.capacity = 1;
    ASSERT_EQ(cudaSuccess, cudaMallocHost(reinterpret_cast<void**>(&output_.pinned_data), output_.capacity));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecCodeStreamCreateToHostMem(instance_, &out_code_stream_, &output_, EncoderOutputBuffer::Resize, &image_info_));

    ASSERT_NO_FATAL_FAILURE(Encode(out_code_stream_, image_buffer_, NVIMGCODEC_PROCESSING_STATUS_FAIL));
    EXPECT_EQ(0u, output_.size);
}

TEST_F(NvJpeg2kExtEncoderStreamTest, ReserveFailureFinishesGpuWorkBeforeImageReady)
{
    nvimgcodecFrameworkDesc_t framework{NVIMGCODEC_STRUCTURE_TYPE_FRAMEWORK_DESC, sizeof(nvimgcodecFrameworkDesc_t), nullptr};
    framework.log = [](void*, nvimgcodecDebugMessageSeverity_t, nvimgcodecDebugMessageCategory_t, const nvimgcodecDebugMessageData_t*) {
        return NVIMGCODEC_STATUS_SUCCESS;
    };
    nvimgcodecExecutorDesc_t executor{NVIMGCODEC_STRUCTURE_TYPE_EXECUTOR_DESC, sizeof(nvimgcodecExecutorDesc_t), nullptr};
    executor.getNumThreads = [](void*) { return 1; };
    auto plugin_exec_params = exec_params_;
    plugin_exec_params.executor = &executor;
    nvjpeg2k::NvJpeg2kEncoderPlugin plugin(&framework);
    auto* desc = plugin.getEncoderDesc();
    nvimgcodecEncoder_t plugin_encoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, desc->create(desc->instance, &plugin_encoder, &plugin_exec_params, ""));
    std::unique_ptr<std::remove_pointer<nvimgcodecEncoder_t>::type, decltype(desc->destroy)> encoder_guard(plugin_encoder, desc->destroy);

    struct FailureContext
    {
        nvimgcodecImageInfo_t image_info;
        cudaEvent_t event = nullptr;
        bool reserve_called = false;
        int ready_count = 0;
        cudaError_t event_status_at_ready = cudaErrorNotReady;
        nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        ~FailureContext()
        {
            (void)cudaStreamSynchronize(image_info.cuda_stream);
            if (event)
                (void)cudaEventDestroy(event);
        }
    } context{image_info_};
    ASSERT_EQ(cudaSuccess, cudaEventCreateWithFlags(&context.event, cudaEventDisableTiming));
    ASSERT_EQ(cudaSuccess, cudaMemcpyAsync(device_buffer_, image_buffer_.data(), image_buffer_.size(), cudaMemcpyHostToDevice, stream_));

    nvimgcodecIoStreamDesc_t io{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr};
    io.instance = &context;
    io.reserve = [](void* ctx, size_t) {
        auto& context = *static_cast<FailureContext*>(ctx);
        context.reserve_called = true;
        // Queue observable work after encode, then fail before map/retrieval.
        // imageReady must not run while this stream work is still pending.
        EXPECT_EQ(cudaSuccess, cudaLaunchHostFunc(context.image_info.cuda_stream,
            [](void*) { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }, nullptr));
        EXPECT_EQ(cudaSuccess, cudaEventRecord(context.event, context.image_info.cuda_stream));
        return NVIMGCODEC_STATUS_ALLOCATOR_FAILURE;
    };
    auto get_info = [](void* ctx, nvimgcodecImageInfo_t* info) {
        *info = static_cast<FailureContext*>(ctx)->image_info;
        return NVIMGCODEC_STATUS_SUCCESS;
    };
    nvimgcodecCodeStreamDesc_t code_stream{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr};
    code_stream.instance = &context;
    code_stream.io_stream = &io;
    code_stream.getImageInfo = get_info;
    nvimgcodecImageDesc_t image{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_DESC, sizeof(nvimgcodecImageDesc_t), nullptr};
    image.instance = &context;
    image.getImageInfo = get_info;
    image.imageReady = [](void* ctx, nvimgcodecProcessingStatus_t status) {
        auto& context = *static_cast<FailureContext*>(ctx);
        ++context.ready_count;
        context.processing_status = status;
        context.event_status_at_ready = cudaEventQuery(context.event);
        return NVIMGCODEC_STATUS_SUCCESS;
    };
    ASSERT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, desc->canEncode(plugin_encoder, &code_stream, &image, &params_, 0));
    EXPECT_EQ(NVIMGCODEC_STATUS_EXECUTION_FAILED, desc->encode(plugin_encoder, &code_stream, &image, &params_, 0));
    EXPECT_TRUE(context.reserve_called);
    EXPECT_EQ(1, context.ready_count);
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, context.processing_status);
    EXPECT_EQ(cudaSuccess, context.event_status_at_ready);
}

}} // namespace nvimgcodec::test
