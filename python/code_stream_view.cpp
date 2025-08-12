/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "code_stream_view.h"
#include <iostream>
#include <sstream>
#include "error_handling.h"

#include <pybind11/stl.h>

namespace nvimgcodec {

void CodeStreamView::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<CodeStreamView>(m, "CodeStreamView", 
            R"pbdoc(
            Class representing a view into a code stream, specifying which image and region to access.

            A code stream view consists of an image index and an optional region of interest.
            )pbdoc")
        .def(py::init([](size_t image_idx, const std::optional<Region>& region) {
            return CodeStreamView(image_idx, region);
        }), 
            "image_idx"_a = 0, "region"_a = std::nullopt,
            R"pbdoc(
            Constructor that initializes a CodeStreamView with an image index and region.

            Args:
                image_idx: Index of the image in the code stream. Defaults to 0.
                
                region: Optional region of interest within the image.
            )pbdoc")
        .def_property_readonly("image_idx", &CodeStreamView::imageIdx, 
            R"pbdoc(
            Property to get the image index in the code stream.

            Returns:
                The index of the image.
            )pbdoc")
        .def_property_readonly("region", &CodeStreamView::region,
            R"pbdoc(
            Property to get the region of interest.

            Returns:
                The Region object specifying the area of interest, or None if no region is set.
            )pbdoc")
        .def("__repr__", [](const CodeStreamView* v) {
            std::stringstream ss;
            ss << *v;
            return ss.str();
        }, 
            R"pbdoc(
            String representation of the CodeStreamView object.

            Returns:
                A string representing the CodeStreamView.
            )pbdoc");
    // clang-format on
}

std::ostream& operator<<(std::ostream& os, const CodeStreamView& v)
{
    os << "CodeStreamView(image_idx=" << v.impl_.image_idx;
    auto region = v.region();
    if (region) {
        os << ", region=" << *region;
    } else {
        os << ", region=None";
    }
    os << ")";
    return os;
}

} // namespace nvimgcodec 