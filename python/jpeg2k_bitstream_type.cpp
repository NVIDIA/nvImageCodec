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

#include "jpeg2k_bitstream_type.h"

namespace nvimgcodec {

void Jpeg2kBitstreamType::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecJpeg2kBitstreamType_t>(m, "Jpeg2kBitstreamType",
        R"pbdoc(
            Enum to define JPEG2000 bitstream types.

            This enum identifies the bitstream type for JPEG2000, which may be either a raw J2K codestream
            or a JP2 container format that can include additional metadata.
        )pbdoc")
        .value("J2K", NVIMGCODEC_JPEG2K_STREAM_J2K, "JPEG2000 codestream format")
        .value("JP2", NVIMGCODEC_JPEG2K_STREAM_JP2, "JPEG2000 JP2 container format")
        .export_values();
    // clang-format on
}


} // namespace nvimgcodec
