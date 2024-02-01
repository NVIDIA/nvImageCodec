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
#include <nvimgcodec.h>
#include "../src/iimage_parser.h"
#include <memory>

namespace nvimgcodec { namespace test {

class MockImageParser : public IImageParser
{
  public:
    MOCK_METHOD(std::string, getParserId,(), (const, override));
    MOCK_METHOD(std::string, getCodecName,(), (const, override));
    MOCK_METHOD(nvimgcodecStatus_t, getImageInfo,(
        nvimgcodecCodeStreamDesc_t* code_stream, nvimgcodecImageInfo_t* image_info) ,(override));
};

}} // namespace nvimgcodec::test