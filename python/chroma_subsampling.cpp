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

#include "chroma_subsampling.h"

namespace nvimgcodec {

void ChromaSubsampling::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecChromaSubsampling_t>(m, "ChromaSubsampling")
        .value("CSS_444", NVIMGCODEC_SAMPLING_444)
        .value("CSS_422", NVIMGCODEC_SAMPLING_422)
        .value("CSS_420", NVIMGCODEC_SAMPLING_420)
        .value("CSS_440", NVIMGCODEC_SAMPLING_440)
        .value("CSS_411", NVIMGCODEC_SAMPLING_411)
        .value("CSS_410", NVIMGCODEC_SAMPLING_410)
        .value("CSS_GRAY", NVIMGCODEC_SAMPLING_GRAY)
        .value("CSS_410V", NVIMGCODEC_SAMPLING_410V)
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec
