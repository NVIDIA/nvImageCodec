/*
 * SPDX-FileCopyrightText: Copyright (c) 2019-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "imgproc/convert.h"
#include "imgproc/convert_test_static.h"
#include "imgproc/math_util.h"

namespace nvimgcodec {

TEST(ConvertSat, float2int) {
  for (int exp = -10; exp < 100; exp++) {
    for (float sig = -256; sig <= 256; sig++) {
      float f = ldexpf(sig, exp);
      float integral;
      float fract = modff(f, &integral);
      if (fract == 0.5f || fract == -0.5f)
        continue;
      double rounded = roundf(f);
      int64_t clamped = clamp<double>(rounded, -128, 127);
      ASSERT_EQ(ConvertSat<int8_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, 0, 255);
      ASSERT_EQ(ConvertSat<uint8_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, -0x8000, 0x7fff);
      ASSERT_EQ(ConvertSat<int16_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, 0, 0xffff);
      ASSERT_EQ(ConvertSat<uint16_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, int32_t(~0x7fffffff), 0x7fffffff);
      ASSERT_EQ(ConvertSat<int32_t>(f), clamped) << " with f = " << f;
      clamped = clamp<double>(rounded, 0, 0xffffffffu);
      ASSERT_EQ(ConvertSat<uint32_t>(f), clamped) << " with f = " << f;
    }
  }
}

TEST(ConvertSat, int2int) {
  EXPECT_EQ((ConvertSat<uint8_t, int8_t>(42)), 42);
  EXPECT_EQ((ConvertSat<uint8_t, int8_t>(-42)), 0);
  EXPECT_EQ((ConvertSat<int8_t, uint8_t>(200)), 127);
  EXPECT_EQ((ConvertSat<int32_t, uint64_t>(0x101234567ull)), 0x7fffffff);
  EXPECT_EQ((ConvertSat<int64_t, uint64_t>(0xc123456701234567ull)), 0x7fffffffffffffffll);
  EXPECT_EQ((ConvertSat<int32_t, int64_t>(-0x101234567ull)), int32_t(~0x7fffffff));
}

TEST(ConvertNorm, int2int) {
  EXPECT_EQ((ConvertNorm<uint8_t, uint8_t>(0)), 0);
  EXPECT_EQ((ConvertNorm<uint8_t, int8_t>(127)), 255);
}

TEST(ConvertNorm, float2int) {
  EXPECT_EQ(ConvertNorm<uint8_t>(0.0f), 0);
  EXPECT_EQ(ConvertNorm<uint8_t>(0.499f), 127);
  EXPECT_EQ(ConvertNorm<uint8_t>(1.0f), 255);
  EXPECT_EQ(ConvertNorm<int8_t>(1.0f), 127);
  EXPECT_EQ(ConvertNorm<int8_t>(0.499f), 63);
  EXPECT_EQ(ConvertNorm<int8_t>(-1.0f), -127);


  EXPECT_EQ(ConvertNorm<uint16_t>(0.0f), 0);
  EXPECT_EQ(ConvertNorm<uint16_t>(1.0f), 0xffff);
  EXPECT_EQ(ConvertNorm<int16_t>(1.0f), 0x7fff);
  EXPECT_EQ(ConvertNorm<int16_t>(-1.0f), -0x7fff);
}

TEST(ConvertSatNorm, float2int) {
  EXPECT_EQ(ConvertSatNorm<uint8_t>(2.0f), 255);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(0.499f), 127);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(-2.0f), 0);
  EXPECT_EQ(ConvertSatNorm<int8_t>(2.0f), 127);
  EXPECT_EQ(ConvertSatNorm<int8_t>(0.499f), 63);
  EXPECT_EQ(ConvertSatNorm<int8_t>(-2.0f), -128);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(0.4f/255), 0);
  EXPECT_EQ(ConvertSatNorm<uint8_t>(0.6f/255), 1);

  EXPECT_EQ(ConvertSatNorm<int16_t>(2.0f), 0x7fff);
  EXPECT_EQ(ConvertSatNorm<int16_t>(-2.0f), -0x8000);
}

TEST(ConvertNorm, int2float) {
  EXPECT_EQ((ConvertNorm<float, uint8_t>(255)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, uint8_t>(127)), 1.0f*127/255, 1e-7f);
  EXPECT_EQ((ConvertNorm<float, int8_t>(127)), 1.0f);
  EXPECT_NEAR((ConvertNorm<float, int8_t>(64)), 1.0f*64/127, 1e-7f);
}

TEST(ConvertNorm, int16uint8) {
  EXPECT_EQ((ConvertNorm<uint8_t, int16_t>(-2000)), 120);
  EXPECT_EQ((ConvertNorm<uint8_t, int16_t>(2278)), 136);
  EXPECT_EQ((ConvertNorm<uint8_t, int16_t>(32767)), 255);
  EXPECT_EQ((ConvertNorm<uint8_t, int16_t>(-32767)), 0);
}

TEST(ConvertNorm, uint8int16) {
  EXPECT_EQ((ConvertNorm<int16_t, uint8_t>(255)), 32767);
  EXPECT_EQ((ConvertNorm<int16_t, uint8_t>(0)), -32767);
}

TEST(ConvertNorm, uint16int8) {
  EXPECT_EQ((ConvertNorm<int8_t, uint16_t>(0)), -127);
  EXPECT_EQ((ConvertNorm<int8_t, uint16_t>(65535)), 127);
}

TEST(ConvertNorm, int8uint16) {
  EXPECT_EQ((ConvertNorm<uint16_t, int8_t>(-127)), 0);
  EXPECT_EQ((ConvertNorm<uint16_t, int8_t>(127)), 65535);
}


}  // namespace nvimgcodec