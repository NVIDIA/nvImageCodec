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

#include <extensions/libtiff/libtiff_ext.h>
#include "common_ext_decoder_test.h"
#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <parsers/tiff.h>
#include <parsers/parser_test_utils.h>
#include <test_utils.h>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <filesystem>
#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {

class LibtiffExtDecoderTest : public ::testing::Test, public CommonExtDecoderTest
{
  public:
    LibtiffExtDecoderTest() {}

    void SetUp() override
    {
        CommonExtDecoderTest::SetUp();

        nvimgcodecExtensionDesc_t tiff_parser_extension_desc{
            NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_tiff_parser_extension_desc(&tiff_parser_extension_desc));
        extensions_.emplace_back();
        nvimgcodecExtensionCreate(instance_, &extensions_.back(), &tiff_parser_extension_desc);

        nvimgcodecExtensionDesc_t libtiff_extension_desc{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(nvimgcodecExtensionDesc_t), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, get_libtiff_extension_desc(&libtiff_extension_desc));
        extensions_.emplace_back();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance_, &extensions_.back(), &libtiff_extension_desc));
    }

    void TearDown() override
    {
        CommonExtDecoderTest::TearDown();
    }
};

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_RGB_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_I_RGB);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_BGR_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_I_BGR);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_UNCHANGED_I)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_RGB_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_P_RGB);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_BGR_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_P_BGR);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_UNCHANGED_P)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED);
}

TEST_F(LibtiffExtDecoderTest, TIFF_SingleImage_Grayscale)
{
    TestSingleImage("tiff/cat-1245673_640.tiff", NVIMGCODEC_SAMPLEFORMAT_P_Y);
}

}} // namespace nvimgcodec::test
