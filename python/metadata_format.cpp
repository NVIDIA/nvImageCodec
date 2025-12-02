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

#include "metadata_format.h"

namespace nvimgcodec {

void MetadataFormat::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecMetadataFormat_t>(m, "MetadataFormat",
        R"pbdoc(
            Enum representing different metadata formats.

            The format specifies how the metadata is encoded (e.g. RAW, JSON, XML).
        )pbdoc")
        .value("UNKNOWN", NVIMGCODEC_METADATA_FORMAT_UNKNOWN,
            R"pbdoc(
            Unknown metadata format.
            )pbdoc")
        .value("RAW", NVIMGCODEC_METADATA_FORMAT_RAW,
            R"pbdoc(
            Raw binary metadata format.
            )pbdoc")
        .value("JSON", NVIMGCODEC_METADATA_FORMAT_JSON,
            R"pbdoc(
            JSON metadata format.
            )pbdoc")
        .value("XML", NVIMGCODEC_METADATA_FORMAT_XML,
            R"pbdoc(
            XML metadata format.
            )pbdoc")
        .value("XMP", NVIMGCODEC_METADATA_FORMAT_XMP,
            R"pbdoc(
            XMP metadata format.
            )pbdoc")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec 