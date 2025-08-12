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

#include "metadata_kind.h"

namespace nvimgcodec {

void MetadataKind::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecMetadataKind_t>(m, "MetadataKind",
        R"pbdoc(
            Enum representing different metadata kinds.

            The kind specifies the type of metadata (e.g. EXIF, GEO, medical formats).
        )pbdoc")
        .value("UNKNOWN", NVIMGCODEC_METADATA_KIND_UNKNOWN,
            R"pbdoc(
            Unknown metadata kind.
            )pbdoc")
        .value("EXIF", NVIMGCODEC_METADATA_KIND_EXIF,
            R"pbdoc(
            EXIF metadata containing camera settings and image capture information. [Reserved for future use]
            )pbdoc")
        .value("GEO", NVIMGCODEC_METADATA_KIND_GEO,
            R"pbdoc(
            Geographic metadata as in GeoTIFF.
            )pbdoc")
        .value("MED_APERIO", NVIMGCODEC_METADATA_KIND_MED_APERIO,
            R"pbdoc(
            Medical metadata in Aperio format for whole slide imaging.
            )pbdoc")
        .value("MED_PHILIPS", NVIMGCODEC_METADATA_KIND_MED_PHILIPS,
            R"pbdoc(
            Medical metadata in Philips format for whole slide imaging.
            )pbdoc")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec 