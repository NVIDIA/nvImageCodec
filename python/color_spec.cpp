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

#include "color_spec.h"

namespace nvimgcodec {

void ColorSpec::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecColorSpec_t>(m, "ColorSpec")
        .value("UNCHANGED", NVIMGCODEC_COLORSPEC_UNCHANGED)
        .value("YCC", NVIMGCODEC_COLORSPEC_SYCC)
        .value("RGB", NVIMGCODEC_COLORSPEC_SRGB)
        .value("GRAY", NVIMGCODEC_COLORSPEC_GRAY)
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec
