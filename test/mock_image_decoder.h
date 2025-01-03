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

#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "../src/idecode_state.h"
#include "../src/icode_stream.h"
#include "../src/iimage.h"

namespace nvimgcodec { namespace test {

class MockImageDecoder : public IImageDecoder
{
  public:
    MOCK_METHOD(nvimgcodecBackendKind_t, getBackendKind, (), (const, override));
    MOCK_METHOD(void, canDecode,
        (const std::vector<ICodeStream*>&, const std::vector<IImage*>&, const nvimgcodecDecodeParams_t*, std::vector<bool>*,
            std::vector<nvimgcodecProcessingStatus_t>*),
        (const, override));
    MOCK_METHOD(ProcessingResultsPromise::FutureImpl, decode,
        (IDecodeState*, const std::vector<ICodeStream*>&, const std::vector<IImage*>&, const nvimgcodecDecodeParams_t*), (override));
    MOCK_METHOD(const char*, decoderId, (), (const, override));
};


}} // namespace nvimgcodec::test