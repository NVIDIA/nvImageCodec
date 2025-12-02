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

#include "backend_kind.h"
#include "backend_params.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Backend
{
  public:
    Backend(nvimgcodecBackendKind_t backend_kind = NVIMGCODEC_BACKEND_KIND_GPU_ONLY, float load_hint = 1.0f, nvimgcodecLoadHintPolicy_t load_hint_policy = NVIMGCODEC_LOAD_HINT_POLICY_FIXED);
    nvimgcodecBackendKind_t getBackendKind() { return backend_.kind; }
    float getLoadHint() { return backend_params_.getLoadHint(); }
    nvimgcodecLoadHintPolicy_t getLoadHintPolicy() { return backend_params_.getLoadHintPolicy(); }

    static void exportToPython(py::module& m);

    BackendParams backend_params_;
    nvimgcodecBackend_t backend_;

};

} // namespace nvimgcodec
