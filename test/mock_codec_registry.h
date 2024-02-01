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
#include "../src/icodec_registry.h"
#include "../src/iimage_parser.h"
#include "../src/icodec.h"
#include <memory>

namespace nvimgcodec { namespace test {

class MockCodecRegistry : public ICodecRegistry
{
  public:
    MOCK_METHOD(void, registerCodec, (std::unique_ptr<ICodec> codec), (override));
    MOCK_METHOD((std::unique_ptr<IImageParser>), getParser,(
        nvimgcodecCodeStreamDesc_t* code_stream), (const, override));
    MOCK_METHOD(ICodec*, getCodecByName, (const char* name), (override));
    MOCK_METHOD(size_t, getCodecsCount, (), (const, override));
    MOCK_METHOD(ICodec*, getCodecByIndex, (size_t i), (override));
};

}} // namespace nvimgcodec::test