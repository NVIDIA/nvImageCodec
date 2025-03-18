/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "region.h"
#include <iostream>
#include <sstream>
#include "error_handling.h"

#include <pybind11/stl.h>

namespace nvimgcodec {

template <typename Container>
Region CreateRegion(Container&& start, Container&& end) {
    Region ret;
    int ndim = ret.impl_.ndim = start.size();
    if (start.size() != end.size() && (!start.empty() && !end.empty())) {
        throw std::runtime_error("Dimension mismatch");
    } else if (ndim > NVIMGCODEC_MAX_NUM_DIM) {
        throw std::runtime_error("Too many dimensions: " + std::to_string(ndim));
    }
    for (int i = 0; i < ndim; i++) {
        ret.impl_.start[i] = start[i];
        ret.impl_.end[i] = end[i];
    }
    return ret;
}

std::vector<int> vec(py::tuple t) {
    std::vector<int> v(t.size());
    for (size_t i = 0; i < t.size(); i++) {
        v[i] = t[i].cast<int>();
    }
    return v;
}

void Region::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Region>(m, "Region", 
            R"pbdoc(
            Class representing a region of interest within an image.

            The dimensions are oriented such that the top-left corner is (0,0).
            )pbdoc")
        .def(py::init([]() { return Region{}; }), 
            "Default constructor that initializes an empty Region object.")
        .def(py::init([](int start_y, int start_x, int end_y, int end_x) {
            return CreateRegion(std::vector<int>{start_y, start_x}, std::vector<int>{end_y, end_x});
        }), 
            "start_y"_a, "start_x"_a, "end_y"_a, "end_x"_a,
            R"pbdoc(
            Constructor that initializes a Region with specified start and end coordinates.

            Args:
                start_y: Starting Y coordinate.

                start_x: Starting X coordinate.
                
                end_y: Ending Y coordinate.
                
                end_x: Ending X coordinate.
            )pbdoc")
        .def(py::init([](const std::vector<int>& start, const std::vector<int>& end) {
            return CreateRegion(start, end);
        }), 
            "start"_a, "end"_a,
            R"pbdoc(
            Constructor that initializes a Region with start and end coordinate lists.

            Args:
                start: List of starting coordinates.

                end: List of ending coordinates.

            )pbdoc")
        .def(py::init([](py::tuple start, py::tuple end) {
            return CreateRegion(vec(start), vec(end));
        }), 
            "start"_a, "end"_a,
            R"pbdoc(
            Constructor that initializes a Region with start and end coordinate tuples.

            Args:
                start: Tuple of starting coordinates.
                
                end: Tuple of ending coordinates.

            )pbdoc")
        .def_property_readonly("ndim", &Region::ndim, 
            R"pbdoc(
            Property to get the number of dimensions in the Region.

            Returns:
                The number of dimensions.
            )pbdoc")
        .def_property_readonly("start", &Region::start, 
            R"pbdoc(
            Property to get the start coordinates of the Region.

            Returns:
                A list of starting coordinates.
            )pbdoc")
        .def_property_readonly("end", &Region::end, 
            R"pbdoc(
            Property to get the end coordinates of the Region.

            Returns:
                A list of ending coordinates.
            )pbdoc")
        .def("__repr__", [](const Region* r) {
            std::stringstream ss;
            ss << *r;
            return ss.str();
        }, 
            R"pbdoc(
            String representation of the Region object.

            Returns:
                A string representing the Region.
            )pbdoc");
    // clang-format on
}


std::ostream& operator<<(std::ostream& os, const Region& r)
{
    os << "Region("
       << "start=";
    auto print_tuple = [](std::ostream& os, const int* data, size_t ndim) {
        os << "(";
        for (size_t d = 0; d < ndim; d++) {
            if (d > 0)
                os << ", ";
            os << data[d];
        }
        os << ")";
    };
    print_tuple(os, r.impl_.start, r.impl_.ndim);
    os << " end=";
    print_tuple(os, r.impl_.end, r.impl_.ndim);
    os << ")";
    return os;
}

} // namespace nvimgcodec
