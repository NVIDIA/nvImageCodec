/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <memory>
#include <string>
#include <utility>

#include "../src/code_stream.h"
#include "mock_codec.h"
#include "mock_codec_registry.h"
#include "mock_image_parser.h"
#include "mock_iostream_factory.h"

namespace nvimgcodec { namespace test {

using ::testing::_;
using ::testing::ByMove;
using ::testing::Const;
using ::testing::Matcher;
using ::testing::Return;
using ::testing::ReturnRef;

TEST(CodeStreamTest, test_parse_from_file)
{
    const std::string codec_name("test_codec");
    MockCodec codec;
    EXPECT_CALL(codec, name()).WillRepeatedly(ReturnRef(codec_name));

    std::unique_ptr<MockImageParser> parser = std::make_unique<MockImageParser>();

    MockCodecRegistry codec_registry;
    EXPECT_CALL(codec_registry, getParser(_))
        .Times(1)
        .WillRepeatedly(Return(ByMove(std::move(parser))));

    std::unique_ptr<MockIoStreamFactory> iostream_factory = std::make_unique<MockIoStreamFactory>();
    EXPECT_CALL(*iostream_factory.get(), createFileIoStream(_, _, _, false)).Times(1);

    CodeStream code_stream(&codec_registry, std::move(iostream_factory));
    code_stream.parseFromFile("test_file");

    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    code_stream.getImageInfo(&image_info);
}

TEST(CodeStreamTest, test_parse_from_mem)
{
    const std::string codec_name("test_codec");
    MockCodec codec;
    EXPECT_CALL(codec, name()).WillRepeatedly(ReturnRef(codec_name));

    std::unique_ptr<MockImageParser> parser = std::make_unique<MockImageParser>();
    MockCodecRegistry codec_registry;
    EXPECT_CALL(codec_registry, getParser(_))
        .Times(1)
        .WillRepeatedly(Return(ByMove(std::move(parser))));

    std::unique_ptr<MockIoStreamFactory> iostream_factory = std::make_unique<MockIoStreamFactory>();
    EXPECT_CALL(*iostream_factory.get(), createMemIoStream(Matcher<const unsigned char*>(_), _)).Times(1);

    CodeStream code_stream(&codec_registry, std::move(iostream_factory));
    code_stream.parseFromMem(nullptr, 0);

    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), 0};
    code_stream.getImageInfo(&image_info);
}

}} // namespace nvimgcodec::test
