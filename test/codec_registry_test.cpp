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
#include "../src/codec_registry.h"
#include "mock_codec.h"
#include "mock_image_parser.h"
#include "mock_logger.h"

namespace nvimgcodec { namespace test {

using ::testing::ByMove;
using ::testing::Return;
using ::testing::ReturnPointee;
using ::testing::ReturnRef;
using ::testing::Eq;

TEST(codec_registry, get_codec_by_name)
{
    const std::string codec_name1("test_codec_1");
    std::unique_ptr<MockCodec> codec1 = std::make_unique<MockCodec>();
    MockCodec* codec1_ptr             = codec1.get();
    EXPECT_CALL(*codec1_ptr, name()).WillRepeatedly(ReturnRef(codec_name1));
    std::unique_ptr<MockCodec> codec2 = std::make_unique<MockCodec>();
    MockCodec* codec2_ptr             = codec2.get();
    const std::string codec_name2("test_codec_2");
    EXPECT_CALL(*codec2_ptr, name()).WillRepeatedly(ReturnRef(codec_name2));

    MockLogger logger;
    CodecRegistry codec_registry(&logger);

    codec_registry.registerCodec(std::move(codec1));
    codec_registry.registerCodec(std::move(codec2));

    EXPECT_EQ(codec_registry.getCodecByName(codec_name2.c_str()), codec2_ptr);
    EXPECT_EQ(codec_registry.getCodecByName(codec_name1.c_str()), codec1_ptr);
}

TEST(codec_registry, get_parser_for_given_code_stream)
{
    nvimgcodecCodeStreamDesc_t code_stream1;
    nvimgcodecCodeStreamDesc_t code_stream2;

    std::unique_ptr<MockImageParser> parser1 = std::make_unique<MockImageParser>();
    MockImageParser* parser1_ptr             = parser1.get();
    std::unique_ptr<MockImageParser> parser2 = std::make_unique<MockImageParser>();
    MockImageParser* parser2_ptr             = parser2.get();

    const std::string codec_name1("test_codec_1");
    std::unique_ptr<MockCodec> codec1 = std::make_unique<MockCodec>();
    MockCodec* codec1_ptr             = codec1.get();
    EXPECT_CALL(*codec1_ptr, name()).WillRepeatedly(ReturnRef(codec_name1));
    EXPECT_CALL(*codec1_ptr, createParser(Eq(&code_stream1))).WillOnce(Return(ByMove(std::move(parser1))));
    EXPECT_CALL(*codec1_ptr, createParser(Eq(&code_stream2)))
        .WillOnce(Return(ByMove(std::move(std::unique_ptr<MockImageParser>()))));

    std::unique_ptr<MockCodec> codec2 = std::make_unique<MockCodec>();
    MockCodec* codec2_ptr             = codec2.get();
    const std::string codec_name2("test_codec_2");
    EXPECT_CALL(*codec2_ptr, name()).WillRepeatedly(ReturnRef(codec_name2));
    EXPECT_CALL(*codec2_ptr, createParser(Eq(&code_stream2))).WillOnce(Return(ByMove(std::move(parser2))));

    MockLogger logger;
    CodecRegistry codec_registry(&logger);
    codec_registry.registerCodec(std::move(codec1));
    codec_registry.registerCodec(std::move(codec2));

    auto parser_2 = codec_registry.getParser(&code_stream2);
    auto parser_1 = codec_registry.getParser(&code_stream1);

    EXPECT_EQ(parser_2.get(), parser2_ptr);
    EXPECT_EQ(parser_1.get(), parser1_ptr);
}

}} // namespace nvimgcodec::test