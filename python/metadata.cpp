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

#include "metadata.h"
#include <iostream>
#include <sstream>
#include "error_handling.h"
#include "metadata_kind.h"
#include "metadata_format.h"

#include <pybind11/stl.h>

#if defined(_MSC_VER)
#include <BaseTsd.h>
typedef SSIZE_T ssize_t;
#endif
namespace nvimgcodec {

Metadata::Metadata(const nvimgcodecMetadata_t& metadata)
    : impl_{metadata}
{
    allocateBuffer();
}

Metadata::Metadata(nvimgcodecMetadataKind_t kind, nvimgcodecMetadataFormat_t format, const py::bytes& buffer)
{
    impl_.kind = kind;
    impl_.format = format;
    
    char* buffer_ptr;
    ssize_t length;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(buffer.ptr(), &buffer_ptr, &length)) {
        throw std::runtime_error("Failed to get bytes buffer");
    }
    
    if (length > 0) {
        buffer_.assign(buffer_ptr, buffer_ptr + length);
        impl_.buffer = buffer_.data();
        impl_.buffer_size = buffer_.size();
    } else {
        impl_.buffer = nullptr;
        impl_.buffer_size = 0;
    }
}

void Metadata::allocateBuffer()
{
    if (impl_.buffer_size > 0) {
        buffer_.resize(impl_.buffer_size);
        impl_.buffer = buffer_.data();
    }
}

void Metadata::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Metadata>(m, "Metadata", 
            R"pbdoc(
            Class representing metadata associated with an image.

            This class provides access to metadata information such as EXIF data, geographic information, or medical metadata.
            )pbdoc")
        .def(py::init([]() { return Metadata{}; }), 
            "Default constructor that initializes an empty Metadata object.")
        .def(py::init([](nvimgcodecMetadataKind_t kind, nvimgcodecMetadataFormat_t format, py::bytes buffer) {
            return Metadata(kind, format, buffer);
        }), 
            "kind"_a, "format"_a, "buffer"_a,
            R"pbdoc(
            Constructor that initializes a Metadata object with specified kind, format and buffer.

            Args:
                kind: The kind of metadata (e.g. EXIF, GEO, medical formats).
                format: The format of the metadata (e.g. RAW, JSON, XML).
                buffer: The metadata content as bytes.
            )pbdoc")
        .def_property_readonly("kind", &Metadata::kind,
            R"pbdoc(
            Property to get the kind of metadata.

            Returns:
                The metadata kind (e.g. EXIF, GEO, medical formats).
            )pbdoc")
        .def_property_readonly("format", &Metadata::format,
            R"pbdoc(
            Property to get the format of the metadata.

            Returns:
                The metadata format (e.g. RAW, JSON, XML).
            )pbdoc")
        .def_property_readonly("buffer", &Metadata::buffer,
            R"pbdoc(
            Property to get the metadata content.

            Returns:
                The metadata content as bytes.
            )pbdoc")
        .def_property_readonly("buffer_size", &Metadata::buffer_size,
            R"pbdoc(
            Property to get the size of the metadata buffer in bytes.

            Returns:
                The size of the metadata buffer.
            )pbdoc")
        .def("__repr__", [](const Metadata* m) {
            std::stringstream ss;
            ss << *m;
            return ss.str();
        },
            R"pbdoc(
            String representation of the Metadata object.

            Returns:
                A string representing the metadata.
            )pbdoc");
    // clang-format on
}

std::ostream& operator<<(std::ostream& os, const Metadata& m)
{
    os << "Metadata("
       << "kind=" << py::cast(m.kind())
       << " format=" << py::cast(m.format())
       << " buffer_size=" << m.buffer_size()
       << ")";
    return os;
}

} // namespace nvimgcodec 