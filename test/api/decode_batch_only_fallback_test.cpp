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

#include <gtest/gtest.h>
#include <nvimgcodec.h>

#include "fake_plugins.h"

namespace nvimgcodec { namespace test {

namespace {

class NvImageCodecsDecodeBatchOnlyFallbackTest : public FakeDecoderTest
{};

} // namespace

// Regression test: a decoder that implements only the batched API (decode == NULL, like the
// nvjpeg lossless decoder) must not cause a NULL function-pointer call when a decodeBatch
// failure routes the sample down the single-sample fallback path. Before the fix this
// segfaulted (call to 0x0 from ImageDecoder::decode); the guard was only an assert(), which is
// stripped in release builds.
TEST_F(NvImageCodecsDecodeBatchOnlyFallbackTest, BatchOnlyDecoderFailureDoesNotCallNullDecode)
{
    // batch_only_a is tried first; batch_only_b is its fallback (both batch-only, and both fail).
    auto& first = AddDecoder("batch_only_a", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, DecoderApi::BatchOnly, NVIMGCODEC_PRIORITY_HIGH);
    auto& second = AddDecoder("batch_only_b", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU, DecoderApi::BatchOnly, NVIMGCODEC_PRIORITY_NORMAL);
    first.decode_fails = true;
    second.decode_fails = true;
    ASSERT_NO_FATAL_FAILURE(CreateDecoder(false, 1));
    ASSERT_NO_FATAL_FAILURE(AddImages(1));

    auto statuses = Decode();
    // Every decoder fails, so the sample must report failure, gracefully rather than by crashing.
    EXPECT_NE(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[0]);
    EXPECT_EQ(1, first.decode_batch_calls.load());
}

}} // namespace nvimgcodec::test
