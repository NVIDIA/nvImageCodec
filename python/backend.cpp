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

Backend::Backend(nvimgcodecBackendKind_t backend_kind, float load_hint, nvimgcodecLoadHintPolicy_t load_hint_policy)
    : backend_params_{load_hint, load_hint_policy}
    , backend_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr, backend_kind,
          backend_params_.backend_params_}
{
}

void Backend::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Backend>(m, "Backend", "Class representing the backend configuration for image processing tasks.")
        .def(py::init([]() { return Backend{}; }), 
            "Default constructor initializes the backend with GPU_ONLY backend kind and default parameters.")
        .def(py::init([](nvimgcodecBackendKind_t backend_kind, float load_hint, nvimgcodecLoadHintPolicy_t load_hint_policy) {
                return Backend (backend_kind, load_hint, load_hint_policy);
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
        .def_property_readonly("backend_kind", &Backend::getBackendKind,
            R"pbdoc(
            The backend kind determines whether processing is done on GPU, CPU, or a hybrid of both.
            )pbdoc")
        .def_property_readonly("load_hint", &Backend::getLoadHint,
            R"pbdoc(
            Load hint is a fraction representing the portion of the workload assigned to this backend. 
            Adjusting this may optimize resource use across available backends.
            )pbdoc")
        .def_property_readonly("load_hint_policy", &Backend::getLoadHintPolicy,
            R"pbdoc(
            The load hint policy defines how the load hint is interpreted, affecting dynamic load distribution.
            )pbdoc");
    // clang-format on
    py::implicitly_convertible<nvimgcodecBackendKind_t, Backend>();
}

 
} // namespace nvimgcodec
