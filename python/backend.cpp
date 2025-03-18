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
    // clang-format off
    py::class_<Backend>(m, "Backend", "Class representing the backend configuration for image processing tasks.")
        .def(py::init([]() { return Backend{}; }), 
            "Default constructor initializes the backend with GPU_ONLY backend kind and default parameters.")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, float load_hint, nvimgcodecLoadHintPolicy_t load_hint_policy) {
                Backend p;
                p.backend_.kind = backend_kind;
                p.backend_.params.load_hint = load_hint;
                p.backend_.params.load_hint_policy = load_hint_policy;
                return p;
            }),
            "backend_kind"_a, "load_hint"_a = 1.0f, "load_hint_policy"_a = NVIMGCODEC_LOAD_HINT_POLICY_FIXED, 
            R"pbdoc(
            Constructor with parameters.
            
            Args:
                backend_kind: Specifies the type of backend (e.g., GPU_ONLY, CPU_ONLY).

                load_hint: Fraction of the batch samples that will be processed by this backend (default is 1.0).
                This is just a hint for performance balancing, so particular extension can ignore it and work on all images it recognizes.
                
                load_hint_policy: Policy for using the load hint, affecting how processing is distributed.
            )pbdoc")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, BackendParams backend_params) {
                Backend p;
                p.backend_.kind = backend_kind;
                p.backend_.params = backend_params.backend_params_;
                return p;
            }),
            "backend_kind"_a, "backend_params"_a, 
            R"pbdoc(
            Constructor with backend parameters.
            
            Args:
                backend_kind: Type of backend (e.g., GPU_ONLY, CPU_ONLY).
                
                backend_params: Additional parameters that define how the backend should operate.
            )pbdoc")
        .def_property("backend_kind", &Backend::getBackendKind, &Backend::setBackendKind, 
            R"pbdoc(
            The backend kind determines whether processing is done on GPU, CPU, or a hybrid of both.
            )pbdoc")
        .def_property("load_hint", &Backend::getLoadHint, &Backend::setLoadHint,
            R"pbdoc(
            Load hint is a fraction representing the portion of the workload assigned to this backend. 
            Adjusting this may optimize resource use across available backends.
            )pbdoc")
        .def_property("load_hint_policy", &Backend::getLoadHintPolicy, &Backend::setLoadHintPolicy,
            R"pbdoc(
            The load hint policy defines how the load hint is interpreted, affecting dynamic load distribution.
            )pbdoc")
        .def_property("backend_params", &Backend::getBackendParams, &Backend::setBackendParams, 
            R"pbdoc(
            Backend parameters include detailed configurations that control backend behavior and performance.
            )pbdoc");
    // clang-format on
}

 
} // namespace nvimgcodec
