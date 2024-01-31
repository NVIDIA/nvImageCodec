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

#include "decode_params.h"

#include <iostream>

#include "error_handling.h"

namespace nvimgcodec {

DecodeParams::DecodeParams()
    : decode_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr, true, false}
    , color_spec_{NVIMGCODEC_COLORSPEC_SRGB}
    , allow_any_depth_{false}
{
}

void DecodeParams::exportToPython(py::module& m)
{
    py::class_<DecodeParams>(m, "DecodeParams")
        .def(py::init([]() { return DecodeParams{}; }), "Default constructor")
        .def(py::init([](bool apply_exif_orientation, nvimgcodecColorSpec_t color_spec, bool allow_any_depth) {
            DecodeParams p;
            p.decode_params_.apply_exif_orientation = apply_exif_orientation;
            p.color_spec_ = color_spec;
            p.allow_any_depth_ = allow_any_depth;
            return p;
        }),
            "apply_exif_orientation"_a = true, "color_spec"_a = NVIMGCODEC_COLORSPEC_SRGB, "allow_any_depth"_a = false,
            "Constructor with apply_exif_orientation, color_spec parameters, and allow_any_depth")
        .def_property("apply_exif_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            "Apply EXIF orientation if available")
        .def_property("allow_any_depth", &DecodeParams::getAllowAnyDepth, &DecodeParams::setAllowAnyDepth,
            "Allow any native bitdepth. If not enabled, the dynamic range is scaled to uint8.")
        .def_property("color_spec", &DecodeParams::getColorSpec, &DecodeParams::setColorSpec,
            "Color specification");
}

} // namespace nvimgcodec
