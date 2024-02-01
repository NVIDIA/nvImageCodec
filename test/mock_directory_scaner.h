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
#include "../src/idirectory_scaner.h"

#include <filesystem>

namespace nvimgcodec {

namespace fs = std::filesystem;

class MockDirectoryScaner : public IDirectoryScaner
{
  public:
    MOCK_METHOD(void, start, (const fs::path& directory), (override));
    MOCK_METHOD(bool, hasMore, (), (override));
    MOCK_METHOD(fs::path, next, (), (override));
    MOCK_METHOD(fs::file_status, symlinkStatus, (const fs::path& p), (override));
    MOCK_METHOD(bool, exists, (const fs::path& p), (override));
};

} // namespace nvimgcodec