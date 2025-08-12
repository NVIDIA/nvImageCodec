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
    : decode_params_{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr, true}
    , color_spec_{NVIMGCODEC_COLORSPEC_SRGB}
    , allow_any_depth_{false}
{
}

void DecodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<DecodeParams>(m, "DecodeParams", "Class to define parameters for image decoding operations.")
        .def(py::init([]() { return DecodeParams{}; }), 
            "Default constructor that initializes the DecodeParams object with default settings.")
        .def(py::init([](bool apply_exif_orientation, nvimgcodecColorSpec_t color_spec, bool allow_any_depth) {
            DecodeParams p;
            p.decode_params_.apply_exif_orientation = apply_exif_orientation;
            p.color_spec_ = color_spec;
            p.allow_any_depth_ = allow_any_depth;
            return p;
        }),
            "apply_exif_orientation"_a = true, "color_spec"_a = NVIMGCODEC_COLORSPEC_SRGB, 
            "allow_any_depth"_a = false,
            R"pbdoc(
            Constructor with parameters to control the decoding process.

            Args:
                apply_exif_orientation: Boolean flag to apply EXIF orientation if available. Defaults to True.

                color_spec: Desired color specification for decoding. Defaults to sRGB.
                
                allow_any_depth: Boolean flag to allow any native bit depth. If not enabled, the 
                dynamic range is scaled to uint8. Defaults to False.
            )pbdoc")
        .def_property("apply_exif_orientation", &DecodeParams::getEnableOrientation, &DecodeParams::setEnableOrientation,
            R"pbdoc(
            Boolean property to enable or disable applying EXIF orientation during decoding.

            When set to True, the image is rotated and/or flipped according to its EXIF orientation 
            metadata if present. Defaults to True.
            )pbdoc")
        .def_property("allow_any_depth", &DecodeParams::getAllowAnyDepth, &DecodeParams::setAllowAnyDepth,
            R"pbdoc(
            Boolean property to permit any native bit depth during decoding.

            When set to True, it allows decoding of images with their native bit depth. 
            If False, the pixel values are scaled to the 8-bit range (0-255). Defaults to False.
            )pbdoc")
        .def_property("color_spec", &DecodeParams::getColorSpec, &DecodeParams::setColorSpec,
            R"pbdoc(
            Property to get or set the color specification for the decoding process.

            This determines the color space or color profile to use during decoding. 
            For instance, sRGB is a common color specification. Defaults to sRGB.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
