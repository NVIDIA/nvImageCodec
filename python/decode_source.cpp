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

#include "decode_source.h"
#include "code_stream.h"
#include <iostream>
#include "error_handling.h"

namespace nvimgcodec {

DecodeSource::DecodeSource(const CodeStream* code_stream_ptr, std::optional<Region> region)
    : code_stream_{}
    , code_stream_ptr_(code_stream_ptr)
    , region_(region)
{
}

DecodeSource::DecodeSource(std::unique_ptr<CodeStream> code_stream, std::optional<Region> region)
    : code_stream_(std::move(code_stream))
    , code_stream_ptr_(code_stream_.get())
    , region_(region)
{
}

DecodeSource::~DecodeSource()
{
}

const CodeStream* DecodeSource::code_stream() const
{
    return code_stream_ptr_;
}

std::optional<Region> DecodeSource::region() const
{
    return region_;
}

void DecodeSource::exportToPython(py::module& m, nvimgcodecInstance_t instance)
{
    py::class_<DecodeSource>(m, "DecodeSource")
        .def(py::init([](const CodeStream* code_stream, std::optional<Region> region) {
            return new DecodeSource{code_stream, region};
        }),
            "code_stream"_a, "region"_a = py::none(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def(py::init([instance](py::array_t<uint8_t> arr, std::optional<Region> region) {
            return new DecodeSource{std::make_unique<CodeStream>(instance, arr), region};
        }),
            "array"_a, "region"_a = py::none(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def(py::init([instance](py::bytes bytes, std::optional<Region> region) {
            return new DecodeSource{std::make_unique<CodeStream>(instance, bytes), region};
        }),
            "bytes"_a, "region"_a = py::none(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def(py::init([instance](const std::filesystem::path& filename, std::optional<Region> region) {
            return new DecodeSource{std::make_unique<CodeStream>(instance, filename), region};
        }),
            "filename"_a, "region"_a = py::none(), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
        .def_property_readonly("code_stream", &DecodeSource::code_stream)
        .def_property_readonly("region", &DecodeSource::region)
        .def("__repr__", [](const DecodeSource* dec_src) {
            std::stringstream ss;
            ss << *dec_src;
            return ss.str();
        });

    py::implicitly_convertible<py::bytes, DecodeSource>();
    py::implicitly_convertible<py::array_t<uint8_t>, DecodeSource>();
    py::implicitly_convertible<std::string, DecodeSource>();
    py::implicitly_convertible<py::tuple, DecodeSource>();
    py::implicitly_convertible<CodeStream, DecodeSource>();
}

std::ostream& operator<<(std::ostream& os, const DecodeSource& ds)
{
  os << "DecodeSource("
      << "code_stream=" << *ds.code_stream();
  if (ds.region()) {
      os << " region=" << ds.region().value();
  }
  os << ")";
  return os;
}

} // namespace nvimgcodec
