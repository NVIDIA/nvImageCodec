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

#include "backend_params.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

BackendParams::BackendParams()
    : backend_params_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}
{
}

void BackendParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<BackendParams>(m, "BackendParams", "Class for configuring backend parameters like load hint and load hint policy.")
        .def(py::init([]() { return BackendParams{}; }), "Default constructor",
            R"pbdoc(
            Creates a BackendParams object with default settings.

            By default, the load hint is set to 1.0 and the load hint policy is set to a fixed value.
            )pbdoc")
        .def(py::init([](float load_hint, nvimgcodecLoadHintPolicy_t load_hint_policy) {
            BackendParams p;
            p.backend_params_.load_hint = load_hint;
            p.backend_params_.load_hint_policy = load_hint_policy;
            return p;
        }),
            "load_hint"_a = 1.0f,
            "load_hint_policy"_a = NVIMGCODEC_LOAD_HINT_POLICY_FIXED,
            "Constructor with load parameters",
            R"pbdoc(
            Creates a BackendParams object with specified load parameters.

            Args:
                load_hint: A float representing the fraction of the batch samples that will be picked by this backend.
                The remaining samples will be picked by the next lower priority backend.
                This is just a hint for performance balancing, so particular extension can ignore it and work on all images it recognizes.

                load_hint_policy: Defines how the load hint is used. Different policies can dictate whether to ignore,
                fix, or adaptively change the load hint.
                
            )pbdoc")
        .def_property("load_hint", &BackendParams::getLoadHint, &BackendParams::setLoadHint,
            R"pbdoc(
            Fraction of the batch samples that will be picked by this backend.

            The remaining samples will be picked by the next lower priority backend.
            This is just a hint for performance balancing, so particular extension can ignore it and work on all images it recognizes.
            )pbdoc")
        .def_property("load_hint_policy", &BackendParams::getLoadHintPolicy, &BackendParams::setLoadHintPolicy,
            R"pbdoc(
            Defines how to use the load hint.

            This property controls the interpretation of load hints, with options to ignore,
            use as fixed or adaptively alter the hint according to workload.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
