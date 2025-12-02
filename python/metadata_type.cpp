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

#include "metadata_type.h"

namespace nvimgcodec {

void MetadataType::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecMetadataValueType_t>(m, "MetadataType",
        R"pbdoc(
            Enum representing different metadata types.

            Metadata types as defined for TIFF tag types in TIFF specification.
        )pbdoc")
        .value("UNKNOWN", NVIMGCODEC_METADATA_VALUE_TYPE_UNKNOWN,
            R"pbdoc(
            Unknown metadata type.
            )pbdoc")
        .value("BYTE", NVIMGCODEC_METADATA_VALUE_TYPE_BYTE,
            R"pbdoc(
            8-bit unsigned integer.
            )pbdoc")
        .value("ASCII", NVIMGCODEC_METADATA_VALUE_TYPE_ASCII,
            R"pbdoc(
            8-bit byte containing 7-bit ASCII code; last byte must be NUL.
            )pbdoc")
        .value("SHORT", NVIMGCODEC_METADATA_VALUE_TYPE_SHORT,
            R"pbdoc(
            16-bit (2-byte) unsigned integer.
            )pbdoc")
        .value("LONG", NVIMGCODEC_METADATA_VALUE_TYPE_LONG,
            R"pbdoc(
            32-bit (4-byte) unsigned integer.
            )pbdoc")
        .value("RATIONAL", NVIMGCODEC_METADATA_VALUE_TYPE_RATIONAL,
            R"pbdoc(
            Two LONGs: numerator and denominator.
            )pbdoc")
        .value("SBYTE", NVIMGCODEC_METADATA_VALUE_TYPE_SBYTE,
            R"pbdoc(
            8-bit signed (twos-complement) integer.
            )pbdoc")
        .value("UNDEFINED", NVIMGCODEC_METADATA_VALUE_TYPE_UNDEFINED,
            R"pbdoc(
            8-bit byte, value depends on field definition.
            )pbdoc")
        .value("SSHORT", NVIMGCODEC_METADATA_VALUE_TYPE_SSHORT,
            R"pbdoc(
            16-bit (2-byte) signed (twos-complement) integer.
            )pbdoc")
        .value("SLONG", NVIMGCODEC_METADATA_VALUE_TYPE_SLONG,
            R"pbdoc(
            32-bit (4-byte) signed (twos-complement) integer.
            )pbdoc")
        .value("SRATIONAL", NVIMGCODEC_METADATA_VALUE_TYPE_SRATIONAL,
            R"pbdoc(
            Two SLONGs: numerator and denominator.
            )pbdoc")
        .value("FLOAT", NVIMGCODEC_METADATA_VALUE_TYPE_FLOAT,
            R"pbdoc(
            4-byte IEEE floating point value.
            )pbdoc")
        .value("DOUBLE", NVIMGCODEC_METADATA_VALUE_TYPE_DOUBLE,
            R"pbdoc(
            8-byte IEEE floating point value.
            )pbdoc")
        .value("LONG8", NVIMGCODEC_METADATA_VALUE_TYPE_LONG8,
            R"pbdoc(
            8-byte (64-bit) unsigned integer (BigTIFF).
            )pbdoc")
        .value("SLONG8", NVIMGCODEC_METADATA_VALUE_TYPE_SLONG8,
            R"pbdoc(
            8-byte (64-bit) signed integer (BigTIFF).
            )pbdoc")
        .value("IFD8", NVIMGCODEC_METADATA_VALUE_TYPE_IFD8,
            R"pbdoc(
            8-byte (64-bit) unsigned integer used for offsets (BigTIFF).
            )pbdoc")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec 
