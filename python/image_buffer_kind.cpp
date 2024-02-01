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

#include "image_buffer_kind.h"

namespace nvimgcodec {

void ImageBufferKind::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecImageBufferKind_t>(m, "ImageBufferKind", "Defines buffer kind in which image data is stored.")
        .value("STRIDED_DEVICE", NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE, "GPU-accessible with planes in pitch-linear layout.") 
        .value("STRIDED_HOST", NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST, "Host-accessible with planes in pitch-linear layout.")
        .export_values();
    // clang-format on
};

} // namespace nvimgcodec
