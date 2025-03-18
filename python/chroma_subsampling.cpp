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

#include "chroma_subsampling.h"

namespace nvimgcodec {

void ChromaSubsampling::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecChromaSubsampling_t>(m, "ChromaSubsampling",
        R"pbdoc(
            Enum representing different types of chroma subsampling.

            Chroma subsampling is the practice of encoding images by implementing less resolution for chroma information than for luma information. 
            This is based on the fact that the human eye is more sensitive to changes in brightness than color.
        )pbdoc")
        .value("CSS_444", NVIMGCODEC_SAMPLING_444,
            R"pbdoc(
            No chroma subsampling. Each pixel has a corresponding chroma value (full color resolution).
            )pbdoc")
        .value("CSS_422", NVIMGCODEC_SAMPLING_422,
            R"pbdoc(
            Chroma is subsampled by a factor of 2 in the horizontal direction. Each line has its chroma sampled at half the horizontal resolution of luma.
            )pbdoc")
        .value("CSS_420", NVIMGCODEC_SAMPLING_420,
            R"pbdoc(
            Chroma is subsampled by a factor of 2 both horizontally and vertically. Each block of 2x2 pixels shares a single chroma sample.
            )pbdoc")
        .value("CSS_440", NVIMGCODEC_SAMPLING_440,
            R"pbdoc(
            Chroma is subsampled by a factor of 2 in the vertical direction. Each column has its chroma sampled at half the vertical resolution of luma.
            )pbdoc")
        .value("CSS_411", NVIMGCODEC_SAMPLING_411,
            R"pbdoc(
            Chroma is subsampled by a factor of 4 in the horizontal direction. Each line has its chroma sampled at quarter the horizontal resolution of luma.
            )pbdoc")
        .value("CSS_410", NVIMGCODEC_SAMPLING_410,
            R"pbdoc(
            Chroma is subsampled by a factor of 4 horizontally and a factor of 2 vertically. Each line has its chroma sampled at quarter the horizontal and half of the vertical resolution of luma.
            )pbdoc")
        .value("CSS_GRAY", NVIMGCODEC_SAMPLING_GRAY,
            R"pbdoc(
            Grayscale image. No chroma information is present.
            )pbdoc")
        .value("CSS_410V", NVIMGCODEC_SAMPLING_410V,
            R"pbdoc(
            Chroma is subsampled by a factor of 4 horizontally and a factor of 2 vertically. Each line has its chroma sampled at quarter the horizontal and half of the vertical resolution of luma.
            Comparing to 4:1:0,  this variation modifies how vertical sampling is handled. While it also has one chroma sample for every four luma samples horizontally,
            it introduces a vertical alternation in how chroma samples are placed between rows. 
            )pbdoc")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec

