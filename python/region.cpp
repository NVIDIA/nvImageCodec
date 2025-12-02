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
#include <cassert>
#include <iostream>
#include <sstream>
#include "error_handling.h"

#include <pybind11/stl.h>

namespace nvimgcodec {

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
        .def(py::init([](int start_y, int start_x, int end_y, int end_x, fillSampleType_t out_of_bounds_sample) {
            return Region(std::vector<int>{start_y, start_x}, std::vector<int>{end_y, end_x}, out_of_bounds_sample);
        }),
            "start_y"_a, "start_x"_a, "end_y"_a, "end_x"_a,
            "out_of_bounds_sample"_a = fillSampleType_t{},
            R"pbdoc(
            Constructor that initializes a Region with specified start and end coordinates.

            Args:
                start_y: Starting Y coordinate.

                start_x: Starting X coordinate.

                end_y: Ending Y coordinate.

                end_x: Ending X coordinate.

                out_of_bounds_sample: The sample that will be set for out of bounds region pixels for all components.
                If not specified, sample value of 0 will be used.
            )pbdoc")
        .def(py::init([](const std::vector<int>& start, const std::vector<int>& end, fillSampleType_t out_of_bounds_sample) {
            return Region(start, end, out_of_bounds_sample);
        }),
            "start"_a, "end"_a,
            "out_of_bounds_sample"_a = fillSampleType_t{},
            R"pbdoc(
            Constructor that initializes a Region with start and end coordinate lists.

            Args:
                start: List of starting coordinates.

                end: List of ending coordinates.

                out_of_bounds_sample: The sample that will be set for out of bounds region pixels for all components
                If not specified, sample value of 0 will be used.

            )pbdoc")
        .def(py::init([](py::tuple start, py::tuple end, fillSampleType_t out_of_bounds_sample) {
            return Region(vec(start), vec(end), out_of_bounds_sample);
        }),
            "start"_a, "end"_a,
            "out_of_bounds_sample"_a = fillSampleType_t{},
            R"pbdoc(
            Constructor that initializes a Region with start and end coordinate tuples.

            Args:
                start: Tuple of starting coordinates.

                end: Tuple of ending coordinates.

                out_of_bounds_sample: The sample that will be set for out of bounds region pixels for all components
                If not specified, sample value of 0 will be used.

            )pbdoc")
        .def(py::init([](int start_y, int start_x, int end_y, int end_x, const std::vector<fillSampleType_t>& out_of_bounds_samples) {
            return Region(std::vector<int>{start_y, start_x}, std::vector<int>{end_y, end_x}, out_of_bounds_samples);
        }),
            "start_y"_a, "start_x"_a, "end_y"_a, "end_x"_a,
            "out_of_bounds_samples"_a,
            R"pbdoc(
            Constructor that initializes a Region with specified start and end coordinates.

            Args:
                start_y: Starting Y coordinate.

                start_x: Starting X coordinate.

                end_y: Ending Y coordinate.

                end_x: Ending X coordinate.

                out_of_bounds_samples: The list of samples that will be set for out of bounds pixel.
                List length must be at most 5, for the rest of the channels fill sample of 5th channel will be used.
                If list is shorter than 5, then fill sample for missing channels will be set to 0.
            )pbdoc")
        .def(py::init([](const std::vector<int>& start, const std::vector<int>& end, const std::vector<fillSampleType_t>& out_of_bounds_samples) {
            return Region(start, end, out_of_bounds_samples);
        }),
            "start"_a, "end"_a,
            "out_of_bounds_samples"_a,
            R"pbdoc(
            Constructor that initializes a Region with start and end coordinate lists.

            Args:
                start: List of starting coordinates.

                end: List of ending coordinates.

                out_of_bounds_samples: The list of samples that will be set for out of bounds pixel.
                List length must be at most 5, for the rest of the channels fill sample of 5th channel will be used.
                If list is shorter than 5, then fill sample for missing channels will be set to 0.
            )pbdoc")
        .def(py::init([](py::tuple start, py::tuple end, const std::vector<fillSampleType_t>& out_of_bounds_samples) {
            return Region(vec(start), vec(end), out_of_bounds_samples);
        }),
            "start"_a, "end"_a,
            "out_of_bounds_samples"_a,
            R"pbdoc(
            Constructor that initializes a Region with start and end coordinate tuples.

            Args:
                start: Tuple of starting coordinates.

                end: Tuple of ending coordinates.

                out_of_bounds_samples: The list of samples that will be set for out of bounds pixel.
                List length must be at most 5, for the rest of the channels fill sample of 5th channel will be used.
                If list is shorter than 5, then fill sample for missing channels will be set to 0.
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
        .def_property_readonly("out_of_bounds_samples", &Region::out_of_bounds_samples, 
            R"pbdoc(
            Property to get the out of bounds samples.

            Returns:
                A list of out of bounds samples.
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
    auto print_tuple = [&](const int* data, size_t ndim) {
        os << "(";
        for (size_t d = 0; d < ndim; d++) {
            if (d > 0)
                os << ", ";
            os << data[d];
        }
        os << ")";
    };

    auto print_array = [&](const std::vector<Region::fillSampleType_t>& samples) {
        os << "[";
        bool first_iter = true;
        for (const auto& sample : samples) {
            if (!first_iter) {
                os << ", ";
            }
            std::visit([&](auto&& value){os << value;}, sample);
            first_iter = false;
        }
        os << "]";
    };

    os << "Region("
       << "start=";
    print_tuple(r.impl_.start, r.impl_.ndim);
    os << " end=";
    print_tuple(r.impl_.end, r.impl_.ndim);

    assert(r.impl_.out_of_bounds_policy == NVIMGCODEC_OUT_OF_BOUNDS_POLICY_CONSTANT);
    os << " out_of_bounds_policy=CONSTANT with samples=";
    print_array(r.out_of_bounds_samples());

    os << ")";
    return os;
}

} // namespace nvimgcodec
