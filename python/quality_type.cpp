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

#include "quality_type.h"

namespace nvimgcodec {

void QualityType::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecQualityType_t>(m, "QualityType", "Supported quality types (algorithms), which determines how `quality_value` is interpreted..")
        .value("DEFAULT", NVIMGCODEC_QUALITY_TYPE_DEFAULT,
            R"pbdoc(
            Each plugin decides its default quality setting. `quality_value` is ignored in this case.
            )pbdoc")
        .value("LOSSLESS", NVIMGCODEC_QUALITY_TYPE_LOSSLESS,
            R"pbdoc(
            Image encoding is reversible and keeps original image quality. `quality_value` is ignored,  except for the CUDA tiff encoder backend,
            for which `quality_value=0` means no compression, and `quality_value=1` means LZW compression..
            )pbdoc")
        .value("QUALITY", NVIMGCODEC_QUALITY_TYPE_QUALITY,
            R"pbdoc(
            `quality_value` is interpreted as JPEG-like quality in range from 1 (worst) to 100 (best).
            )pbdoc")
        .value("QUANTIZATION_STEP", NVIMGCODEC_QUALITY_TYPE_QUANTIZATION_STEP,
            R"pbdoc(
            `quality_value` is interpreted as quantization step (by how much pixel data will be divided).
            The higher the value, the worse quality image is produced.
            )pbdoc")
        .value("PSNR", NVIMGCODEC_QUALITY_TYPE_PSNR,
            R"pbdoc(
            `quality_value` is interpreted as desired Peak Signal-to-Noise Ratio (PSNR) target for the encoded image.
            The higher the value, the better quality image is produced. Value should be positive.
            )pbdoc")
        .value("SIZE_RATIO", NVIMGCODEC_QUALITY_TYPE_SIZE_RATIO,
            R"pbdoc(
            `quality_value` is interpreted as desired encoded image size ratio compared to original size, should be floating point in range (0.0, 1.0).
            E.g. value 0.1 means target size of 10% of original image.
            )pbdoc")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec
