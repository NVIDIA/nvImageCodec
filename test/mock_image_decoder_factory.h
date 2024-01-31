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

#pragma once

#include <nvimgcodec.h>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "../src/iimage_decoder.h"
#include "../src/iimage_decoder_factory.h"

namespace nvimgcodec {
namespace test {

class MockImageDecoderFactory : public IImageDecoderFactory
{
  public:
    MOCK_METHOD(std::string, getDecoderId, (), (const, override));
    MOCK_METHOD(std::string, getCodecName, (), (const, override));
    MOCK_METHOD(nvimgcodecBackendKind_t, getBackendKind, (), (const, override));
    MOCK_METHOD(std::unique_ptr<IImageDecoder>, createDecoder,
        (const nvimgcodecExecutionParams_t*, const char*), (const, override));
};

}}