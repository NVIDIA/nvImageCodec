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

#include <chrono>
#include <string>
#include <tuple>

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "fake_plugins.h"

namespace nvimgcodec { namespace test {

namespace {

// Large enough for the HW decoder's 0.9 load hint to leave a share of the batch to the next decoder.
constexpr int kBatchSize = 16;
// The second call checks that a decoder that failed to create is not created again.
constexpr int kCalls = 2;

class DecoderFallbackTest : public FakeDecoderTest, public ::testing::WithParamInterface<bool /*pre_init*/>
{
  protected:
    bool preInit() const { return GetParam(); }
};

using BatchFailureParams = std::tuple<DecoderApi, bool /*pre_init*/>;

class DecoderBatchFailureTest : public FakeDecoderTest, public ::testing::WithParamInterface<BatchFailureParams>
{};

class DecoderBatchFailureWithThreadsTest : public FakeDecoderTest
{};

} // namespace

// A decoder whose create() fails (e.g. nvjpeg_hw_decoder when nvjpegJpegStateCreate fails) must be removed
// from the fallback chain, and the next decoder must take every image.
TEST_P(DecoderFallbackTest, FallbackSkipsDecoderThatFailedToCreate)
{
    auto& never_created =
        AddDecoder("never_created_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    never_created.create_fails = true;
    auto& working = AddDecoder("working_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_LOW);
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 1));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));
    EXPECT_EQ(preInit() ? 1 : 0, never_created.create_calls.load());

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto can_decode = CanDecode();
        auto statuses = Decode();
        for (int i = 0; i < kBatchSize; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, can_decode[i]) << "image #" << i;
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << "image #" << i;
            EXPECT_TRUE(OutputIs(i, kDecoded)) << "image #" << i << " was not decoded by the fallback decoder";
        }
    }

    EXPECT_EQ(1, never_created.create_calls.load());
    EXPECT_EQ(kCalls * kBatchSize, working.decode_calls.load());
}

// The same with the decoder that fails to create in the middle of the chain, so that it is unlinked from
// the previous decoder's fallback_ (or, with pre_init, never linked). The first decoder's 0.9 load hint
// leaves part of each batch to the decoders after it.
TEST_P(DecoderFallbackTest, FallbackSkipsDecoderThatFailedToCreateInTheMiddle)
{
    auto& first =
        AddDecoder("working_batch_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    auto& never_created =
        AddDecoder("never_created_decoder", NVIMGCODEC_BACKEND_KIND_GPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_NORMAL);
    never_created.create_fails = true;
    auto& last = AddDecoder("working_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_LOW);
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 1));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto statuses = Decode();
        for (int i = 0; i < kBatchSize; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << "image #" << i;
            EXPECT_TRUE(OutputIs(i, kDecoded)) << "image #" << i << " was not decoded";
        }
    }

    EXPECT_EQ(1, never_created.create_calls.load());
    EXPECT_GT(first.decode_batch_calls.load(), 0);
    EXPECT_GT(last.decode_calls.load(), 0) << "no image reached the decoder after the one that failed to create";
}

// Without any other decoder for the codec, images must be reported as failed, not as decoded by a decoder
// that does not exist.
TEST_P(DecoderFallbackTest, NoFallbackReportsFailure)
{
    auto& never_created =
        AddDecoder("never_created_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    never_created.create_fails = true;
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 1));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto can_decode = CanDecode();
        auto statuses = Decode();
        for (int i = 0; i < kBatchSize; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED, can_decode[i]) << "image #" << i;
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_CODEC_UNSUPPORTED, statuses[i]) << "image #" << i;
            EXPECT_TRUE(OutputIs(i, kUntouched)) << "image #" << i;
        }
    }

    EXPECT_EQ(1, never_created.create_calls.load());
}

// A decode's future can be ready while its worker threads are still finishing, so a CanDecode call right
// after it must not change the state those threads use. Without a decoder for the codec, they would then
// crash on the missing one. The race doesn't show on every call, hence the repetitions.
TEST_P(DecoderFallbackTest, CanDecodeRightAfterDecodeWithSeveralThreads)
{
    auto& never_created =
        AddDecoder("never_created_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    never_created.create_fails = true;
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 4));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    for (int call = 0; call < 200; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        for (auto status : Decode()) {
            EXPECT_NE(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
        }
        for (auto status : CanDecode()) {
            EXPECT_NE(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
        }
    }
}

// CanDecode must wait for the tasks of a previous decode, but the executor contract allows wait() only
// after run(), so it must not wait when no tasks have been started. When the wait fails, the query must
// fail rather than change the state those tasks may still use, and wait again next time.
TEST_P(DecoderFallbackTest, CanDecodeWaitsOnlyForStartedTasks)
{
    // The failed batch is handed to the fallback decoder in tasks, so a decode starts some.
    auto& failing =
        AddDecoder("failing_batch_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    failing.decode_fails = true;
    AddDecoder("working_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_LOW);
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 1, executor_.getExecutorDesc()));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    for (auto status : CanDecode()) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    }
    EXPECT_EQ(0, executor_.waits_without_run) << "CanDecode waited before any tasks were started";

    for (auto status : Decode()) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    }
    executor_.wait_fails = true;
    for (auto status : CanDecode()) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, status);
    }
    executor_.wait_fails = false;
    for (auto status : CanDecode()) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    }
    EXPECT_EQ(0, executor_.waits_without_run) << "a call waited without tasks to wait for";
}

// The same for Decode, which waits for the tasks of the previous decode before reusing its state.
TEST_P(DecoderFallbackTest, DecodeWaitsOnlyForStartedTasks)
{
    // The failed batch is handed to the fallback decoder in tasks, so each decode starts some.
    auto& failing =
        AddDecoder("failing_batch_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    failing.decode_fails = true;
    AddDecoder("working_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_LOW);
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 1, executor_.getExecutorDesc()));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    for (auto status : Decode()) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    }
    executor_.wait_fails = true;
    auto statuses = Decode();
    for (int i = 0; i < kBatchSize; ++i) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, statuses[i]) << "image #" << i;
        EXPECT_TRUE(OutputIs(i, kUntouched)) << "image #" << i;
    }
    executor_.wait_fails = false;
    for (auto status : Decode()) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status);
    }
    EXPECT_EQ(0, executor_.waits_without_run) << "Decode waited without tasks to wait for";
}

// A canDecode() that throws refuses the image.
TEST_P(DecoderFallbackTest, ThrowingCanDecodeReportsFailure)
{
    auto& throwing = AddDecoder("throwing_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_NORMAL);
    throwing.can_decode = CanProcess::Throw;
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(preInit(), 1));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    auto can_decode = CanDecode();
    auto statuses = Decode();
    for (int i = 0; i < kBatchSize; ++i) {
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, can_decode[i]) << "image #" << i;
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_FAIL, statuses[i]) << "image #" << i;
        EXPECT_TRUE(OutputIs(i, kUntouched)) << "image #" << i;
    }
    EXPECT_EQ(0, throwing.decode_calls.load());
}

INSTANTIATE_TEST_SUITE_P(DECODE_CREATE_FAILURE_FALLBACK, DecoderFallbackTest, ::testing::Bool(),
    [](const ::testing::TestParamInfo<bool>& info) { return info.param ? "PreInit" : "LazyInit"; });

// When a batch decoder fails, its images must go to the next decoder that doesn't decode in batches, without
// being offered to the failed decoder again. The fallback refuses device buffers, so it must be asked about
// the caller's host output rather than the batch decoder's temporary device buffer.
TEST_P(DecoderBatchFailureTest, BatchFailureContinuesWithNextDecoder)
{
    const auto [api, pre_init] = GetParam();
    auto& failing = AddDecoder("failing_batch_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, api, NVIMGCODEC_PRIORITY_VERY_HIGH);
    failing.decode_fails = true;
    auto& working = AddDecoder("working_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_LOW);
    working.can_decode = CanProcess::HostBufferOnly;
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(pre_init, 1));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto statuses = Decode();
        for (int i = 0; i < kBatchSize; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << "image #" << i;
            EXPECT_TRUE(OutputIs(i, kDecoded)) << "image #" << i << " was not decoded by the fallback decoder";
        }
    }

    EXPECT_GT(failing.decode_batch_calls.load(), 0);
    EXPECT_EQ(0, failing.decode_calls.load()) << "an image was offered again to the decoder whose batch decode failed";
    EXPECT_EQ(kCalls * kBatchSize, working.decode_calls.load());
}

INSTANTIATE_TEST_SUITE_P(DECODE_BATCH_FAILURE_FALLBACK, DecoderBatchFailureTest,
    ::testing::Combine(::testing::Values(DecoderApi::SingleAndBatch, DecoderApi::BatchOnly), ::testing::Bool()),
    [](const ::testing::TestParamInfo<BatchFailureParams>& info) {
        return std::string(ToString(std::get<0>(info.param))) + (std::get<1>(info.param) ? "_PreInit" : "_LazyInit");
    });

// While the images of a failed batch are handed to the fallback decoder, the worker threads must leave them
// alone, or they are decoded twice. The batch decoder refuses the 2x1 images, which are sorted first, so the
// worker threads are still busy decoding them, slowly, when the batch of 1x1 images fails.
TEST_F(DecoderBatchFailureWithThreadsTest, WorkerThreadsLeaveFailedBatchToFallback)
{
    constexpr int kThreads = 4;
    auto& failing =
        AddDecoder("failing_batch_decoder", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY, DecoderApi::SingleAndBatch, NVIMGCODEC_PRIORITY_VERY_HIGH);
    failing.decode_fails = true;
    failing.max_width = 1;
    auto& working = AddDecoder("working_decoder", NVIMGCODEC_BACKEND_KIND_CPU_ONLY, DecoderApi::SingleOnly, NVIMGCODEC_PRIORITY_LOW);
    working.decode_delay = std::chrono::milliseconds(20);
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(true, kThreads));
    ASSERT_NO_FATAL_FAILURE(AddImages(kThreads, kTwoPixelBmp, sizeof(kTwoPixelBmp)));
    ASSERT_NO_FATAL_FAILURE(AddImages(kBatchSize));
    const int num_images = kThreads + kBatchSize;

    for (int call = 0; call < kCalls; ++call) {
        SCOPED_TRACE("call " + std::to_string(call));
        auto statuses = Decode();
        for (int i = 0; i < num_images; ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << "image #" << i;
            EXPECT_TRUE(OutputIs(i, kDecoded)) << "image #" << i << " was not decoded by the fallback decoder";
        }
    }

    EXPECT_GT(failing.decode_batch_calls.load(), 0);
    EXPECT_EQ(kCalls * num_images, working.decode_calls.load()) << "an image was decoded more than once";
}

}} // namespace nvimgcodec::test
