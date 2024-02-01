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

#include <gmock/gmock.h>
#include "../src/ilogger.h"

namespace nvimgcodec {

class MockLogger : public ILogger
{
  public:
    MOCK_METHOD(void, log,
        (const nvimgcodecDebugMessageSeverity_t message_severity, const nvimgcodecDebugMessageCategory_t message_type,
            const std::string& message),
        (override));
    MOCK_METHOD(void, log,
        (const nvimgcodecDebugMessageSeverity_t message_severity, const nvimgcodecDebugMessageCategory_t message_type,
            const nvimgcodecDebugMessageData_t* data),
        (override));
    MOCK_METHOD(void, registerDebugMessenger, (IDebugMessenger * messenger), (override));
    MOCK_METHOD(void, unregisterDebugMessenger, (IDebugMessenger * messenger), (override));
};

} // namespace nvimgcodec
