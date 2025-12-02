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

#include <pybind11/pybind11.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class BackendParams
{
  public:
    BackendParams(float load_hint = 1.0f, nvimgcodecLoadHintPolicy_t load_hint_policy = NVIMGCODEC_LOAD_HINT_POLICY_FIXED)
    : backend_params_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, load_hint, load_hint_policy}
    {
    }

    float getLoadHint() { return backend_params_.load_hint; }
    nvimgcodecLoadHintPolicy_t getLoadHintPolicy() { return backend_params_.load_hint_policy; }

    nvimgcodecBackendParams_t backend_params_;
};

} // namespace nvimgcodec
