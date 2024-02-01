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

#include <memory>
#include "processing_results.h"
#include "work.h"

namespace nvimgcodec {

template<typename T>
class IWorkManager
{
  public:
    virtual ~IWorkManager() = default;
    virtual std::unique_ptr<Work<T>> createNewWork(
        const ProcessingResultsPromise& results, const void* params) = 0;
    virtual void recycleWork(std::unique_ptr<Work<T>> work) = 0;
    virtual void combineWork(
        Work<T>* target, std::unique_ptr<Work<T>> source) = 0;
};

} // namespace nvimgcodec
