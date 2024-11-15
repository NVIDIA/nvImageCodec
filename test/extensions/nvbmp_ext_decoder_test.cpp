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

#include <extensions/nvbmp/nvbmp_ext.h>
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/bmp.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "nvimgcodec_tests.h"

using ::testing::Combine;
using ::testing::Values;

namespace nvimgcodec { namespace test {

class NvbmpExtDecoderTest : public CommonExtDecoderTestWithPathAndFormat
{
  public:
    void SetUp() override
    {
        CommonExtDecoderTestWithPathAndFormat::SetUp();

        nvimgcodecExtensionDesc_t bmp_parser_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_bmp_parser_extension_desc(&bmp_parser_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &bmp_parser_extension_desc));

        nvimgcodecExtensionDesc_t nvbmp_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_nvbmp_extension_desc(&nvbmp_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &nvbmp_extension_desc));

        CommonExtDecoderTestWithPathAndFormat::CreateDecoder();
    }
};

TEST_P(NvbmpExtDecoderTest, SingleImage)
{
    TestSingleImage(image_path, sample_format);
}

INSTANTIATE_TEST_SUITE_P(NVBMP_DECODE,
    NvbmpExtDecoderTest,
    Combine(
        Values(
            "bmp/cat-111793_640.bmp"
        ), Values (
            NVIMGCODEC_SAMPLEFORMAT_I_RGB,
            NVIMGCODEC_SAMPLEFORMAT_P_RGB
        )
    )
);

}} // namespace nvimgcodec::test
