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

#include <algorithm>
#include <cstring>
#include <iterator>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "fake_plugins.h"

namespace nvimgcodec { namespace test {

namespace {

constexpr int kBatchSize = 4;
// The second call checks that an encoder that failed to create is not created again.
constexpr int kCalls = 2;

// Encodes 1x1 RGB host images into "bmp" code streams in host memory with fake encoders.
class EncoderFallbackTest : public FakePluginsTest, public ::testing::WithParamInterface<bool /*pre_init*/>
{
  protected:
    void TearDown() override
    {
        for (auto stream : streams_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
        }
        for (auto image : images_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
        }
        if (encoder_) {
            EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder_));
        }
        FakePluginsTest::TearDown();
    }

    bool preInit() const { return GetParam(); }

    // Registers the fakes, then creates the encoder and the images to encode.
    void CreateEncoder(nvimgcodecExecutorDesc_t* executor = nullptr)
    {
        ASSERT_NO_FATAL_FAILURE(RegisterPlugins());
        auto exec_params = MakeExecParams(preInit(), 1, executor);
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance_, &encoder_, &exec_params, nullptr));

        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
        info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling = NVIMGCODEC_SAMPLING_444;
        info.num_planes = 1;
        info.plane_info[0].width = 1;
        info.plane_info[0].height = 1;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].row_stride = 3;
        info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].precision = 8;
        info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        info.buffer = pixels_;
        std::strcpy(info.codec_name, "bmp");

        outputs_.resize(kBatchSize);
        for (int i = 0; i < kBatchSize; ++i) {
            nvimgcodecImage_t image = nullptr;
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &image, &info));
            images_.push_back(image);
            nvimgcodecCodeStream_t stream = nullptr;
            ASSERT_EQ(
                NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance_, &stream, &outputs_[i], ResizeOutput, &info));
            streams_.push_back(stream);
        }
    }

    // Sets every byte already written to the outputs to kUntouched, then encodes all images.
    std::vector<nvimgcodecProcessingStatus_t> Encode()
    {
        for (auto& output : outputs_)
            std::fill(output.begin(), output.end(), kUntouched);
        std::vector<nvimgcodecProcessingStatus_t> statuses(images_.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        nvimgcodecEncodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        nvimgcodecFuture_t future = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecEncoderEncode(encoder_, images_.data(), streams_.data(), static_cast<int>(images_.size()), &params, &future));
        if (!future)
            return statuses;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future));
        size_t status_size = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &status_size));
        EXPECT_EQ(images_.size(), status_size);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(future));
        return statuses;
    }

    std::vector<nvimgcodecProcessingStatus_t> CanEncode()
    {
        std::vector<nvimgcodecProcessingStatus_t> statuses(images_.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        nvimgcodecEncodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), 0};
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCanEncode(encoder_, images_.data(), streams_.data(),
                                                 static_cast<int>(images_.size()), &params, statuses.data(), 1));
        return statuses;
    }

    const std::vector<unsigned char>& Output(int image_idx) const { return outputs_[image_idx]; }

  private:
    static unsigned char* ResizeOutput(void* ctx, size_t size)
    {
        auto* output = static_cast<std::vector<unsigned char>*>(ctx);
        output->resize(size);
        return output->data();
    }

    uint8_t pixels_[3] = {0x10, 0x20, 0x30};
    nvimgcodecEncoder_t encoder_ = nullptr;
    std::vector<nvimgcodecImage_t> images_;
    std::vector<nvimgcodecCodeStream_t> streams_;
    std::vector<std::vector<unsigned char>> outputs_;
};

} // namespace

// An encoder whose create() fails must be removed from the fallback chain, and the next encoder must take
// every image.
TEST_P(EncoderFallbackTest, FallbackSkipsEncoderThatFailedToCreate)
{
    auto& never_created = AddEncoder("never_created_encoder", NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_PRIORITY_VERY_HIGH);
    never_created.create_fails = true;
    auto& working = AddEncoder("working_encoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, NVIMGCODEC_PRIORITY_LOW);
    ASSERT_NO_FATAL_FAILURE(CreateEncoder());
    EXPECT_EQ(preInit() ? 1 : 0, never_created.create_calls.load());

    const std::vector<unsigned char> encoded(std::begin(kEncoded), std::end(kEncoded));
    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto can_encode = CanEncode();
        auto statuses = Encode();
        for (int i = 0; i < kBatchSize; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, can_encode[i]) << "image #" << i;
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << "image #" << i;
            EXPECT_EQ(encoded, Output(i)) << "image #" << i << " was not encoded by the fallback encoder";
        }
    }

    EXPECT_EQ(1, never_created.create_calls.load());
    EXPECT_EQ(kCalls * kBatchSize, working.encode_calls.load());
}

// Without any other encoder for the codec, images must be reported as failed.
TEST_P(EncoderFallbackTest, NoFallbackReportsFailure)
{
    auto& never_created = AddEncoder("never_created_encoder", NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_PRIORITY_VERY_HIGH);
    never_created.create_fails = true;
    ASSERT_NO_FATAL_FAILURE(CreateEncoder());

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto can_encode = CanEncode();
        auto statuses = Encode();
        for (int i = 0; i < kBatchSize; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED, can_encode[i]) << "image #" << i;
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED, statuses[i]) << "image #" << i;
        }
    }

    EXPECT_EQ(1, never_created.create_calls.load());
}

// The executor contract allows wait() only after run(), so CanEncode must not wait when no tasks have been
// started.
TEST_P(EncoderFallbackTest, CanEncodeWaitsOnlyForStartedTasks)
{
    AddEncoder("working_encoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, NVIMGCODEC_PRIORITY_NORMAL);
    ASSERT_NO_FATAL_FAILURE(CreateEncoder(executor_.getExecutorDesc()));

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        for (auto status : CanEncode()) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
        }
    }
    EXPECT_EQ(0, executor_.waits_without_run) << "CanEncode waited before any tasks were started";
}

// The same for Encode. A single-threaded encode starts no tasks, so it never has anything to wait for.
TEST_P(EncoderFallbackTest, EncodeWaitsOnlyForStartedTasks)
{
    AddEncoder("working_encoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, NVIMGCODEC_PRIORITY_NORMAL);
    ASSERT_NO_FATAL_FAILURE(CreateEncoder(executor_.getExecutorDesc()));

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        for (auto status : Encode()) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
        }
    }
    EXPECT_EQ(0, executor_.waits_without_run) << "Encode waited before any tasks were started";
}

// A canEncode() that throws refuses the image.
TEST_P(EncoderFallbackTest, ThrowingCanEncodeReportsFailure)
{
    auto& throwing = AddEncoder("throwing_encoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, NVIMGCODEC_PRIORITY_NORMAL);
    throwing.can_encode = CanProcess::Throw;
    ASSERT_NO_FATAL_FAILURE(CreateEncoder());

    auto can_encode = CanEncode();
    auto statuses = Encode();
    for (int i = 0; i < kBatchSize; ++i) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, can_encode[i]) << "image #" << i;
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, statuses[i]) << "image #" << i;
    }
    EXPECT_EQ(0, throwing.encode_calls.load());
}

INSTANTIATE_TEST_SUITE_P(ENCODE_CREATE_FAILURE_FALLBACK, EncoderFallbackTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "PreInit" : "LazyInit"; });

}} // namespace nvimgcodec::test
