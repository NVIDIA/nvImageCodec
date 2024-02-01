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
#include <nvimgcodec.h>
#include "idecode_state.h"

namespace nvimgcodec {

class DecodeStateBatch : public IDecodeState
{
  public:
    DecodeStateBatch() = default;
    ~DecodeStateBatch() override = default;
    void setPromise(const ProcessingResultsPromise& promise) override;
    const ProcessingResultsPromise& getPromise() override;
  private:
    std::unique_ptr<ProcessingResultsPromise> promise_;
};
} // namespace nvimgcodec
