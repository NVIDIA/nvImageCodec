/*
 * SPDX-FileCopyrightText: Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    py::enum_<nvimgcodecColorSpec_t>(m, "ColorSpec", "Enum representing color specification for image - how the color information in samples should be interpreted.")
        .value("UNKNOWN", NVIMGCODEC_COLORSPEC_UNKNOWN, "The color specification is unknown or not specified.")
        .value("UNCHANGED", NVIMGCODEC_COLORSPEC_UNCHANGED, "The color specification should be left unchanged from the source; equivalent to UNKNOWN.")
        .value("SYCC", NVIMGCODEC_COLORSPEC_SYCC, "Specifies the sYCC color space (YCbCr with sRGB primaries).")
        .value("SRGB", NVIMGCODEC_COLORSPEC_SRGB, "Specifies the standard RGB (sRGB) color space.")
        .value("GRAY", NVIMGCODEC_COLORSPEC_GRAY, "Specifies grayscale (single channel, no color).")
        .value("CMYK", NVIMGCODEC_COLORSPEC_CMYK, "Specifies the CMYK color space (Cyan, Magenta, Yellow, Black).")
        .value("YCCK", NVIMGCODEC_COLORSPEC_YCCK, "Specifies the YCCK color space (YCbCr plus Black channel).")
        .value("PALETTE", NVIMGCODEC_COLORSPEC_PALETTE, "Sample data is represented using a palette color map.")
        .value("ICC_PROFILE", NVIMGCODEC_COLORSPEC_ICC_PROFILE, "Precise color space is provided in an ICC profile.")
        .value("UNSUPPORTED", NVIMGCODEC_COLORSPEC_UNSUPPORTED, "The color specification is unsupported by the library.")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec

