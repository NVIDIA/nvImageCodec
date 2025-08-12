/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
    , encode_params_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr}
    , chroma_subsampling_{NVIMGCODEC_SAMPLING_444}
    , color_spec_{NVIMGCODEC_COLORSPEC_UNCHANGED}
{
}

void EncodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<EncodeParams>(m, "EncodeParams", "Class to define parameters for image encoding operations.")
        .def(py::init([]() { return EncodeParams{}; }), "Default constructor that initializes the EncodeParams object with default settings.")
        .def(py::init([](nvimgcodecQualityType_t quality_type, float quality_value, nvimgcodecColorSpec_t color_spec, nvimgcodecChromaSubsampling_t chroma_subsampling,
                          std::optional<JpegEncodeParams> jpeg_encode_params, std::optional<Jpeg2kEncodeParams> jpeg2k_encode_params) {
            EncodeParams p;
            p.encode_params_.quality_type = quality_type;
            p.encode_params_.quality_value = quality_value;
            p.color_spec_ = color_spec;
            p.chroma_subsampling_ = chroma_subsampling;
            p.jpeg_encode_params_ = jpeg_encode_params.has_value() ? jpeg_encode_params.value() : JpegEncodeParams();
            p.jpeg2k_encode_params_ = jpeg2k_encode_params.has_value() ? jpeg2k_encode_params.value() : Jpeg2kEncodeParams();

            return p;
        }),
            "quality_type"_a = NVIMGCODEC_QUALITY_TYPE_DEFAULT,
            "quality_value"_a = 0,
            "color_spec"_a = NVIMGCODEC_COLORSPEC_UNCHANGED, 
            "chroma_subsampling"_a = NVIMGCODEC_SAMPLING_444,
            "jpeg_encode_params"_a = py::none(),
            "jpeg2k_encode_params"_a = py::none(),
            R"pbdoc(
            Constructor with parameters to control the encoding process.

            Args:
                quality_type (QualityType): Quality type (algorithm) that will be used to encode image.

                quality_value (float): Specifies how good encoded image should look like. Refer to the QualityType enum for the allowed values for each quality type.

                color_spec (ColorSpec): Output color specification. Defaults to UNCHANGED.

                chroma_subsampling (ChromaSubsampling): Chroma subsampling format. Defaults to CSS_444.

                jpeg_encode_params (JpegEncodeParams): Optional JPEG specific encoding parameters.
                
                jpeg2k_encode_params (Jpeg2kEncodeParams): Optional JPEG2000 specific encoding parameters.
            )pbdoc")
        .def_property("quality_type", &EncodeParams::getQualityType, &EncodeParams::setQualityType,
            R"pbdoc(
            Quality type (algorithm) that will be used to encode image.
            )pbdoc")
        .def_property("quality_value", &EncodeParams::getQualityValue, &EncodeParams::setQualityValue,
            R"pbdoc(
            Specifies how good encoded image should look like. Refer to the QualityType enum for the allowed values for each quality type.
            )pbdoc")
        .def_property("color_spec", &EncodeParams::getColorSpec, &EncodeParams::setColorSpec,
            R"pbdoc(
            Defines the expected color specification for the output. Defaults to ColorSpec.UNCHANGED.
            )pbdoc")
        .def_property("chroma_subsampling", &EncodeParams::getChromaSubsampling, &EncodeParams::setChromaSubsampling,
            R"pbdoc(
            Specifies the chroma subsampling format for encoding. Defaults to CSS_444 so not chroma subsampling.
            )pbdoc")
        .def_property("jpeg_params", &EncodeParams::getJpegEncodeParams, &EncodeParams::setJpegEncodeParams,
            R"pbdoc(
            Optional, additional JPEG-specific encoding parameters.
            )pbdoc")
        .def_property("jpeg2k_params", &EncodeParams::getJpeg2kEncodeParams, &EncodeParams::setJpeg2kEncodeParams,
            R"pbdoc(
            Optional, additional JPEG2000-specific encoding parameters.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
