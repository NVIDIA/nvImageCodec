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
#include "metadata_type.h"

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


namespace {
    // Generic converter that handles single vs multiple value logic
    template<typename Converter>
    py::object convertValues(uint32_t count, Converter&& converter) {
        if (count == 1) {
            return converter(0);
        } else {
            py::list result;
            for (uint32_t i = 0; i < count; ++i) {
                result.append(converter(i));
            }
            return result;
        }
    }

    // Helper for simple type conversion
    template<typename T, typename CastType = T>
    py::object convertSimpleValues(const char* buffer_ptr, uint32_t count) {
        const T* values = reinterpret_cast<const T*>(buffer_ptr);
        return convertValues(count, [values](uint32_t i) {
            return py::cast(static_cast<CastType>(values[i]));
        });
    }

    // Helper for rational type conversion (pairs of values)
    template<typename T>
    py::object convertRationalValues(const char* buffer_ptr, uint32_t count) {
        const T* values = reinterpret_cast<const T*>(buffer_ptr);
        return convertValues(count, [values](uint32_t i) {
            return py::make_tuple(values[i*2], values[i*2+1]);
        });
    }
}

py::object Metadata::getValue() const
{
    // Convert buffer content based on metadata type
    py::bytes buffer_bytes = buffer();
    if (buffer_size() == 0) {
        return py::none();
    }
    
    std::string buffer_str = buffer_bytes;
    const char* buffer_ptr = buffer_str.data();
    size_t buf_size = buffer_size();
    nvimgcodecMetadataValueType_t type = value_type();
    uint32_t count = value_count();
    
    switch (type) {
        case NVIMGCODEC_METADATA_VALUE_TYPE_BYTE:
        case NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED:{
            if (count == 1) {
                return py::cast(static_cast<uint8_t>(buffer_ptr[0]));
            } else {
                return buffer_bytes;
            }
        }
        case NVIMGCODEC_METADATA_VALUE_TYPE_ASCII:
            return py::str(std::string(buffer_ptr, buf_size - 1/*Buffer is null terminated*/));
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_SHORT:
            return convertSimpleValues<uint16_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_LONG:
            return convertSimpleValues<uint32_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL:
            return convertRationalValues<uint32_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_SBYTE:
            return convertSimpleValues<int8_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_SSHORT:
            return convertSimpleValues<int16_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_SLONG:
            return convertSimpleValues<int32_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_SRATIONAL:
            return convertRationalValues<int32_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT:
            return convertSimpleValues<float>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE:
            return convertSimpleValues<double>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_LONG8:
        case NVIMGCODEC_METADATA_VALUE_TYPE_IFD8:
            return convertSimpleValues<uint64_t>(buffer_ptr, count);
            
        case NVIMGCODEC_METADATA_VALUE_TYPE_SLONG8:
            return convertSimpleValues<int64_t>(buffer_ptr, count);
            
        default:
            // For unknown types, return the raw buffer
            return buffer_bytes;
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
        .def_property_readonly("value_type", &Metadata::value_type,
            R"pbdoc(
            Property to get the type of the metadata when format is RAW.

            Returns:
                The metadata type (e.g. BYTE, SHORT, LONG, etc.).
            )pbdoc")
        .def_property_readonly("id", &Metadata::id,
            R"pbdoc(
            Property to get the ID of the metadata.

            Returns:
                The metadata ID. For TIFF tag metadata kind, this is the tag ID. For other metadata kinds, it is reserved for future use.
            )pbdoc")
        .def_property_readonly("value_count", &Metadata::value_count,
            R"pbdoc(
            Property to get the number of values in the metadata.

            Returns:
                The number of values. For TIFF tag metadata, this represents the count of values for the tag.
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
        .def_property_readonly("value", &Metadata::getValue,
            R"pbdoc(
            Property to get the metadata value converted to a Python type.

            Returns:
                The metadata value converted to an appropriate Python type based on the metadata type:
                - BYTE/UNDEFINED: byte or bytes
                - ASCII: string
                - SHORT: int or list of ints  
                - LONG: int or list of ints
                - RATIONAL: tuple (numerator, denominator) or list of tuples
                - SBYTE: int or list of ints
                - SSHORT: int or list of ints
                - SLONG: int or list of ints
                - SRATIONAL: tuple (numerator, denominator) or list of tuples
                - FLOAT: float or list of floats
                - DOUBLE: float or list of floats
                - LONG8/IFD8: int or list of ints
                - SLONG8: int or list of ints
                - Unknown types: raw bytes
                Returns None if buffer is empty.
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
       << " value_type=" << py::cast(m.value_type())
       << " id=" << m.id()
       << " value_count=" << m.value_count()
       << " buffer_size=" << m.buffer_size()
       << ")";
    return os;
}

} // namespace nvimgcodec 