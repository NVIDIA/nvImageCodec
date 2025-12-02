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

#include "sample_format.h"

namespace nvimgcodec {

void SampleFormat::exportToPython(py::module& m)
{
    // clang-format off
    py::enum_<nvimgcodecSampleFormat_t>(m, "SampleFormat", "Enum representing sample format for image. Sample format describes how color components are matched to channels in given order and channels are matched to planes.")
        .value("UNKNOWN", NVIMGCODEC_SAMPLEFORMAT_UNKNOWN, "Unknown sample format.")
        .value("P_UNCHANGED", NVIMGCODEC_SAMPLEFORMAT_P_UNCHANGED, "Unchanged planar format.")
        .value("I_UNCHANGED", NVIMGCODEC_SAMPLEFORMAT_I_UNCHANGED, "Unchanged interleaved format.")
        .value("P_Y", NVIMGCODEC_SAMPLEFORMAT_P_Y, "Y component only (grayscale). For 3-dimensional shape is defined as (1, H, W) and this is only difference with I_Y.")
        .value("I_Y", NVIMGCODEC_SAMPLEFORMAT_I_Y, "Interleaved Y component only (grayscale). For 3-dimensional shape is defined as (H, W, 1) and this is only difference with P_Y.")
        .value("P_YA", NVIMGCODEC_SAMPLEFORMAT_P_YA, "Planar Y component with alpha.")
        .value("I_YA", NVIMGCODEC_SAMPLEFORMAT_I_YA, "Interleaved Y component with alpha.")
        .value("P_RGB", NVIMGCODEC_SAMPLEFORMAT_P_RGB, "Planar RGB format.")
        .value("I_RGB", NVIMGCODEC_SAMPLEFORMAT_I_RGB, "Interleaved RGB format.")
        .value("P_BGR", NVIMGCODEC_SAMPLEFORMAT_P_BGR, "Planar BGR format.")
        .value("I_BGR", NVIMGCODEC_SAMPLEFORMAT_I_BGR, "Interleaved BGR format.")
        .value("P_YUV", NVIMGCODEC_SAMPLEFORMAT_P_YUV, "YUV planar format.")
        .value("P_YCC", NVIMGCODEC_SAMPLEFORMAT_P_YCC, "YCC planar format (alias for P_YUV).")
        .value("I_YUV", NVIMGCODEC_SAMPLEFORMAT_I_YUV, "Interleaved YUV format.")
        .value("I_YCC", NVIMGCODEC_SAMPLEFORMAT_I_YCC, "Interleaved YCC format (alias for I_YUV).")
        .value("P_RGBA", NVIMGCODEC_SAMPLEFORMAT_P_RGBA, "Planar RGBA format.")
        .value("I_RGBA", NVIMGCODEC_SAMPLEFORMAT_I_RGBA, "Interleaved RGBA format.")
        .value("P_YCCK", NVIMGCODEC_SAMPLEFORMAT_P_YCCK, "Planar YCCK format.")
        .value("I_YCCK", NVIMGCODEC_SAMPLEFORMAT_I_YCCK, "Interleaved YCCK format.")
        .value("P_CMYK", NVIMGCODEC_SAMPLEFORMAT_P_CMYK, "Planar CMYK format.")
        .value("I_CMYK", NVIMGCODEC_SAMPLEFORMAT_I_CMYK, "Interleaved CMYK format.")
        .value("UNSUPPORTED", NVIMGCODEC_SAMPLEFORMAT_UNSUPPORTED, "Unsupported sample format.")
        .export_values();
    // clang-format on
}

} // namespace nvimgcodec
