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

#include "backend_kind.h"

namespace nvimgcodec {

void BackendKind::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecBackendKind_t>(m, "BackendKind",
        R"pbdoc(
            Enum representing backend kinds used in nvImageCodec for decoding/encoding operations.

            This enum helps specify where (CPU, GPU, both, or GPU hardware engine) the image processing tasks are executed.
        )pbdoc")
        .value("CPU_ONLY", NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
            R"pbdoc(
                Backend kind specifying that decoding/encoding is executed only on CPU.
            )pbdoc")
        .value("GPU_ONLY", NVIMGCODEC_BACKEND_KIND_GPU_ONLY,
            R"pbdoc(
                Backend kind specifying that decoding/encoding is executed only on GPU.
            )pbdoc")
        .value("HYBRID_CPU_GPU", NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
            R"pbdoc(
                Backend kind specifying that decoding/encoding is executed on both CPU and GPU.
            )pbdoc")
        .value("HW_GPU_ONLY", NVIMGCODEC_BACKEND_KIND_HW_GPU_ONLY,
            R"pbdoc(
                Backend kind specifying that decoding/encoding is executed on GPU dedicated hardware engine.
            )pbdoc")
        .export_values();
    // clang-format on
};

} // namespace nvimgcodec
