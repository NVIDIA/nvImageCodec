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
#include <string>
#include "imgproc/type_utils.h"

namespace nvimgcodec {
namespace test {

// The fix short-circuits every non-integral type through !IsIntegral(), so the invariants below
// have to hold for all three float types -- not just FLOAT32. Parameterising over them keeps a
// regression in the IsIntegral() whitelist (e.g. a float type accidentally listed as integral)
// from slipping through.
class TypeUtilsFloat : public ::testing::TestWithParam<nvimgcodecSampleDataType_t>
{
  public:
    static std::string Name(const ::testing::TestParamInfo<nvimgcodecSampleDataType_t>& info)
    {
        switch (info.param) {
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16:
            return "FLOAT16";
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32:
            return "FLOAT32";
        case NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64:
            return "FLOAT64";
        default:
            return "UNKNOWN";
        }
    }
};

INSTANTIATE_TEST_SUITE_P(FloatTypes, TypeUtilsFloat,
    ::testing::Values(NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16, NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32,
        NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64),
    TypeUtilsFloat::Name);

// Every float type must be reported as non-integral; the short-circuits below all key off this.
TEST_P(TypeUtilsFloat, IsNotIntegral)
{
    EXPECT_FALSE(IsIntegral(GetParam()));
}

// Floating point samples are normalised to [0, 1] -- see the ConvertNorm contract, where
// ConvertNorm<float, uint8_t>(255) == 1.0f. There is no sub-range packing to undo, so no
// precision can require dynamic range scaling of a float plane.
TEST_P(TypeUtilsFloat, NoDynamicRangeScalingForFloat)
{
    EXPECT_FALSE(NeedDynamicRangeScaling(0, GetParam()));
    EXPECT_FALSE(NeedDynamicRangeScaling(8, GetParam()));
    EXPECT_FALSE(NeedDynamicRangeScaling(32, GetParam()));
}

// A float plane already spans its full range, so it never needs rescaling on its own account.
TEST_P(TypeUtilsFloat, DynamicRangeMultiplierForFloatIsUnity)
{
    EXPECT_DOUBLE_EQ(1.0, DynamicRangeMultiplier(0, GetParam()));
    EXPECT_DOUBLE_EQ(1.0, DynamicRangeMultiplier(8, GetParam()));
}

// The multiplier LaunchConvertNormKernelImpl() applies is
//     DynamicRangeMultiplier(out, in) / MaxValue(in_dtype)
// and the kernel evaluates ConvertNorm<Out, float>(multiplier * sample). Decoding 12-bit
// JPEG 2000 / TIFF content to a float output must therefore map the largest 12-bit value onto 1.0.
TEST_P(TypeUtilsFloat, TwelveBitUint16ToFloatMapsMaxOntoOne)
{
    const auto u16 = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16;
    const auto flt = GetParam();
    ASSERT_TRUE(NeedDynamicRangeScaling(0, flt, 12, u16));
    const double multiplier = DynamicRangeMultiplier(0, flt, 12, u16) / MaxValue(u16);
    EXPECT_NEAR(1.0, 4095.0 * multiplier, 1e-6);
}

// Full-width uint16 to float needs no multiplier at all: the data spans the type and the float
// side is normalised, so PassthroughKernel's ConvertNorm<float, uint16_t> does the work (65535 -> 1.0).
TEST_P(TypeUtilsFloat, FullWidthUint16ToFloatNeedsNoScaling)
{
    EXPECT_FALSE(NeedDynamicRangeScaling(0, GetParam(), 16, NVIMGCODEC_SAMPLE_DATA_TYPE_UINT16));
}

// float -> same float type must short-circuit before MagnitudeBits() is consulted.
TEST_P(TypeUtilsFloat, FloatToFloatNeedsNoScaling)
{
    const auto flt = GetParam();
    EXPECT_FALSE(NeedDynamicRangeScaling(0, flt, 0, flt));
    EXPECT_FALSE(NeedDynamicRangeScaling(0, flt, 8, flt));
}

// float -> uint8 is the reverse direction and must also be reachable without asserting.
TEST_P(TypeUtilsFloat, FloatToUint8IsWellDefined)
{
    const auto flt = GetParam();
    const auto u8 = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    EXPECT_FALSE(NeedDynamicRangeScaling(0, u8, 0, flt));
    EXPECT_DOUBLE_EQ(1.0, DynamicRangeMultiplier(0, flt));
}

// Conversions between two different float types must not reach the integer helpers either.
TEST_P(TypeUtilsFloat, FloatToOtherFloatNeedsNoScaling)
{
    const auto flt = GetParam();
    for (auto other : {NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT16, NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT32,
             NVIMGCODEC_SAMPLE_DATA_TYPE_FLOAT64}) {
        EXPECT_FALSE(NeedDynamicRangeScaling(0, other, 0, flt)) << "from " << flt << " to " << other;
        EXPECT_DOUBLE_EQ(1.0, DynamicRangeMultiplier(0, other, 0, flt)) << "from " << flt << " to " << other;
    }
}

} // namespace test
} // namespace nvimgcodec
