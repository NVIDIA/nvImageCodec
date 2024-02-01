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

#include "../src/codec.h"
#include "../src/iimage_parser.h"
#include "../src/iimage_parser_factory.h"
#include "mock_image_parser_factory.h"
#include "mock_logger.h"

namespace nvimgcodec { namespace test {

TEST(CodecTest, parsers_are_probet_in_priority_order_when_registered_in_order_123)
{
    MockLogger logger;
    Codec codec(&logger, "test_codec");
    std::unique_ptr<MockImageParserFactory> factory1 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory2 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory3 = std::make_unique<MockImageParserFactory>();

    ::testing::Sequence s1;
    EXPECT_CALL(*factory1.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory2.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory3.get(), canParse(nullptr)).Times(1).InSequence(s1);

    codec.registerParserFactory(std::move(factory1), 1);
    codec.registerParserFactory(std::move(factory2), 2);
    codec.registerParserFactory(std::move(factory3), 3);

    nvimgcodecCodeStreamDesc_t* code_stream = nullptr;
    std::unique_ptr<IImageParser> parser  = codec.createParser(code_stream);
}

TEST(CodecTest, parsers_are_probet_in_priority_order_when_registered_in_order_231)
{
    MockLogger logger;
    Codec codec(&logger, "test_codec");
    std::unique_ptr<MockImageParserFactory> factory1 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory2 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory3 = std::make_unique<MockImageParserFactory>();

    ::testing::Sequence s1;
    EXPECT_CALL(*factory1.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory2.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory3.get(), canParse(nullptr)).Times(1).InSequence(s1);

    codec.registerParserFactory(std::move(factory2), 2);
    codec.registerParserFactory(std::move(factory3), 3);
    codec.registerParserFactory(std::move(factory1), 1);

    nvimgcodecCodeStreamDesc_t* code_stream = nullptr;
    std::unique_ptr<IImageParser> parser  = codec.createParser(code_stream);
}

TEST(CodecTest, parsers_are_probet_in_priority_order_when_registered_in_order_321)
{
    MockLogger logger;
    Codec codec(&logger, "test_codec");
    std::unique_ptr<MockImageParserFactory> factory1 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory2 = std::make_unique<MockImageParserFactory>();
    std::unique_ptr<MockImageParserFactory> factory3 = std::make_unique<MockImageParserFactory>();

    ::testing::Sequence s1;
    EXPECT_CALL(*factory1.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory2.get(), canParse(nullptr)).Times(1).InSequence(s1);
    EXPECT_CALL(*factory3.get(), canParse(nullptr)).Times(1).InSequence(s1);

    codec.registerParserFactory(std::move(factory3), 3);
    codec.registerParserFactory(std::move(factory2), 2);
    codec.registerParserFactory(std::move(factory1), 1);

    nvimgcodecCodeStreamDesc_t* code_stream = nullptr;
    std::unique_ptr<IImageParser> parser  = codec.createParser(code_stream);
}

}} // namespace nvimgcodec::test
