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
    BackendParams();

    float getLoadHint() { return backend_params_.load_hint; }
    void setLoadHint(float load_hint) { backend_params_.load_hint = load_hint; };

    bool getLoadHintPolicy() { return backend_params_.load_hint_policy; }
    void setLoadHintPolicy(nvimgcodecLoadHintPolicy_t load_hint_policy) { backend_params_.load_hint_policy = load_hint_policy; };


    static void exportToPython(py::module& m);

    nvimgcodecBackendParams_t backend_params_;
};

} // namespace nvimgcodec
