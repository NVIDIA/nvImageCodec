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

#include "encode_params.h"

#include <iostream>
#include <optional>

#include "error_handling.h"

namespace nvimgcodec {

EncodeParams::EncodeParams()
    : jpeg2k_encode_params_{}
    , jpeg_encode_params_{}
    , encode_params_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr, 95, 50}
    , chroma_subsampling_{NVIMGCODEC_SAMPLING_444}
    , color_spec_{NVIMGCODEC_COLORSPEC_UNCHANGED}
{
}

void EncodeParams::exportToPython(py::module& m)
{
    py::class_<EncodeParams>(m, "EncodeParams")
        .def(py::init([]() { return EncodeParams{}; }), "Default constructor")
        .def(py::init([](float quality, float target_psnr, nvimgcodecColorSpec_t color_spec, nvimgcodecChromaSubsampling_t chroma_subsampling,
                          std::optional<JpegEncodeParams> jpeg_encode_params, std::optional<Jpeg2kEncodeParams> jpeg2k_encode_params) {
            EncodeParams p;
            p.encode_params_.quality = quality;
            p.encode_params_.target_psnr = target_psnr;
            p.color_spec_ = color_spec;
            p.chroma_subsampling_ = chroma_subsampling;
            p.jpeg_encode_params_ = jpeg_encode_params.has_value() ? jpeg_encode_params.value() : JpegEncodeParams();
            p.jpeg2k_encode_params_ = jpeg2k_encode_params.has_value() ? jpeg2k_encode_params.value() : Jpeg2kEncodeParams();

            return p;
        }),
            // clang-format off
            "quality"_a = 95, 
            "target_psnr"_a = 50, 
            "color_spec"_a = NVIMGCODEC_COLORSPEC_UNCHANGED, 
            "chroma_subsampling"_a = NVIMGCODEC_SAMPLING_444,
            "jpeg_encode_params"_a = py::none(),
            "jpeg2k_encode_params"_a = py::none(),
            "Constructor with quality, target_psnr, color_spec, chroma_subsampling etc. parameters")
        // clang-format on
        .def_property("quality", &EncodeParams::getQuality, &EncodeParams::setQuality, "Quality value 0-100 (default 95)")
        .def_property("target_psnr", &EncodeParams::getTargetPsnr, &EncodeParams::setTargetPsnr, "Target psnr (default 50)")
        .def_property("color_spec", &EncodeParams::getColorSpec, &EncodeParams::setColorSpec,
            "Output color specification (default ColorSpec.UNCHANGED)")
        .def_property("chroma_subsampling", &EncodeParams::getChromaSubsampling, &EncodeParams::setChromaSubsampling,
            "Chroma subsampling (default ChromaSubsampling.CSS_444)")
        .def_property("jpeg_params", &EncodeParams::getJpegEncodeParams, &EncodeParams::setJpegEncodeParams, "Jpeg encode parameters")
        .def_property(
            "jpeg2k_params", &EncodeParams::getJpeg2kEncodeParams, &EncodeParams::setJpeg2kEncodeParams, "Jpeg2000 encode parameters");
}

} // namespace nvimgcodec
