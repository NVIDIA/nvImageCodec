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

#pragma once

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "../src/icodec.h"
#include "../src/iimage_decoder.h"
#include "../src/iimage_decoder_factory.h"
#include "../src/iimage_encoder.h"
#include "../src/iimage_encoder_factory.h"
#include "../src/iimage_parser.h"
#include "../src/iimage_parser_factory.h"

#include <memory>

namespace nvimgcodec { namespace test {

class MockCodec : public ICodec
{
  public:
    MOCK_METHOD(const std::string&, name, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageParser>, createParser,
        (nvimgcodecCodeStreamDesc_t* code_stream), (const, override));
    MOCK_METHOD(int, getDecodersNum, (), (const, override));
    MOCK_METHOD(IImageDecoderFactory*, getDecoderFactory, (int index), (const, override));
    MOCK_METHOD(int, getEncodersNum, (), (const, override));
    MOCK_METHOD(IImageEncoderFactory*, getEncoderFactory, (int index), (const, override));
    MOCK_METHOD(void, registerEncoderFactory,
        (std::unique_ptr<IImageEncoderFactory> factory, float priority), (override));
    MOCK_METHOD(void, unregisterEncoderFactory,(const std::string encoder_id), (override));
    MOCK_METHOD(void, registerDecoderFactory,
        (std::unique_ptr<IImageDecoderFactory> factory, float priority), (override));
    MOCK_METHOD(void, unregisterDecoderFactory, (const std::string decoder_id), (override));
    MOCK_METHOD(void, registerParserFactory,
        (std::unique_ptr<IImageParserFactory> factory, float priority), (override));
    MOCK_METHOD(void, unregisterParserFactory, (const std::string parser_id) ,(override));
};

}} // namespace nvimgcodec::test