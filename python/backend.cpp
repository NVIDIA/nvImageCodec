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

#include "backend.h"
#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

Backend::Backend()
    : backend_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr, NVIMGCODEC_BACKEND_KIND_GPU_ONLY,
          {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}}
{
}

void Backend::exportToPython(py::module& m)
{
    py::class_<Backend>(m, "Backend")
        .def(py::init([]() { return Backend{}; }), "Default constructor")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, float load_hint, nvimgcodecLoadHintPolicy_t load_hint_policy) {
            Backend p;
            p.backend_.kind = backend_kind;
            p.backend_.params.load_hint = load_hint;
            p.backend_.params.load_hint_policy = load_hint_policy;
            return p;
        }),
            "backend_kind"_a, "load_hint"_a = 1.0f, "load_hint_policy"_a = NVIMGCODEC_LOAD_HINT_POLICY_FIXED, "Constructor with parameters")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, BackendParams backend_params) {
            Backend p;
            p.backend_.kind = backend_kind;
            p.backend_.params = backend_params.backend_params_;
            return p;
        }),
            "backend_kind"_a, "backend_params"_a, "Constructor with backend parameters")
        .def_property("backend_kind", &Backend::getBackendKind, &Backend::setBackendKind, "Backend kind (e.g. GPU_ONLY or CPU_ONLY).")
        .def_property("load_hint", &Backend::getLoadHint, &Backend::setLoadHint,
            "Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower "
            "priority backend.")
        .def_property("load_hint_policy", &Backend::getLoadHintPolicy, &Backend::setLoadHintPolicy,
            "Defines how to use the load hint")
        .def_property("backend_params", &Backend::getBackendParams, &Backend::setBackendParams, "Backend parameters.");
}
 
} // namespace nvimgcodec
