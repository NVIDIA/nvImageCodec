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
    : backend_params_{NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f}
{
}

void BackendParams::exportToPython(py::module& m)
{
    py::class_<BackendParams>(m, "BackendParams")
        .def(py::init([]() { return BackendParams{}; }), "Default constructor")
        .def(py::init([](bool load_hint) {
            BackendParams p;
            p.backend_params_.load_hint = load_hint;
            return p;
        }),
            "load_hint"_a = 1.0f, "Constructor with load_hint parameters")
        .def_property("load_hint", &BackendParams::getLoadHint, &BackendParams::setLoadHint,
            "Fraction of the batch samples that will be picked by this backend. The remaining samples will be picked by the next lower "
            "priority backend. This is just hint so particular codec can ignore this "
            "value");
}

} // namespace nvimgcodec
