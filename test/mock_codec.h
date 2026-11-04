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
#include <string>
#include <vector>

namespace nvimgcodec { namespace test {

class MockCodec : public ICodec
{
  public:
    MOCK_METHOD(const std::string&, name, (), (const, override));
    MOCK_METHOD(bool, hasRegistrations, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageParser>, createParser,
        (nvimgcodecCodeStreamDesc_t* code_stream), (const, override));
    MOCK_METHOD((std::vector<IImageParserFactory*>), getParserFactories, (), (const, override));
    MOCK_METHOD((std::vector<IImageDecoderFactory*>), getDecoderFactories, (), (const, override));
    MOCK_METHOD((std::vector<IImageEncoderFactory*>), getEncoderFactories, (), (const, override));
    MOCK_METHOD(void, registerEncoderFactory,
        (std::unique_ptr<IImageEncoderFactory> factory, float priority), (override));
    MOCK_METHOD(bool, unregisterEncoderFactory, (const nvimgcodecEncoderDesc_t* desc), (override));
    MOCK_METHOD(void, registerDecoderFactory,
        (std::unique_ptr<IImageDecoderFactory> factory, float priority), (override));
    MOCK_METHOD(bool, unregisterDecoderFactory, (const nvimgcodecDecoderDesc_t* desc), (override));
    MOCK_METHOD(void, registerParserFactory,
        (std::unique_ptr<IImageParserFactory> factory, float priority), (override));
    MOCK_METHOD(bool, unregisterParserFactory, (const nvimgcodecParserDesc_t* desc), (override));
    MOCK_METHOD(void, registerFileExtension, (const std::string& file_extension), (override));
    MOCK_METHOD(void, unregisterFileExtension, (const std::string& file_extension), (override));
    MOCK_METHOD(const std::vector<std::string>&, getFileExtensions, (), (const, override));
    MOCK_METHOD(bool, hasFileExtension, (const std::string& file_extension), (const, override));
    MOCK_METHOD(bool, isDefaultFileExtension, (const std::string& file_extension), (const, override));
    MOCK_METHOD(void, setDefaultFileExtension, (const std::string& file_extension, bool is_default), (override));
};

}} // namespace nvimgcodec::test