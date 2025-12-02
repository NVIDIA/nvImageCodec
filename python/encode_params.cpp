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
#include <stdexcept>

#include "error_handling.h"

namespace nvimgcodec {

void EncodeParams::setupStructChain()
{
    tile_geometry_.impl_.struct_next = nullptr;
    jpeg2k_encode_params_.impl_.struct_next = &tile_geometry_.impl_;
    jpeg_encode_params_.impl_.struct_next = &jpeg2k_encode_params_.impl_;
    impl_.struct_next = &jpeg_encode_params_.impl_;
}

EncodeParams::EncodeParams()
    : jpeg2k_encode_params_{}
    , jpeg_encode_params_{}
    , tile_geometry_{}
    , chroma_subsampling_{std::nullopt}
    , color_spec_{NVIMGCODEC_COLORSPEC_UNCHANGED}
    , impl_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr}

{
    setupStructChain();
}

EncodeParams::EncodeParams(const EncodeParams& other)
    : jpeg2k_encode_params_{other.jpeg2k_encode_params_}
    , jpeg_encode_params_{other.jpeg_encode_params_}
    , tile_geometry_{other.tile_geometry_}
    , chroma_subsampling_{other.chroma_subsampling_}
    , color_spec_{other.color_spec_}
    , impl_{NVIMGCODEC_STRUCTURE_TYPE_ENCODE_PARAMS, sizeof(nvimgcodecEncodeParams_t), nullptr}
{
    // Set up the struct_next chain (same as default constructor)
    setupStructChain();
    
    // Copy the impl_ fields except struct_next
    impl_.quality_type = other.impl_.quality_type;
    impl_.quality_value = other.impl_.quality_value;
}

EncodeParams& EncodeParams::operator=(const EncodeParams& other)
{
    if (this != &other) {
        // Copy all member objects
        jpeg2k_encode_params_ = other.jpeg2k_encode_params_;
        jpeg_encode_params_ = other.jpeg_encode_params_;
        tile_geometry_ = other.tile_geometry_;
        chroma_subsampling_ = other.chroma_subsampling_;
        color_spec_ = other.color_spec_;
        
        // Copy impl_ fields except struct_next
        impl_.quality_type = other.impl_.quality_type;
        impl_.quality_value = other.impl_.quality_value;
        
        // Re-establish the struct_next chain
        setupStructChain();
    }
    return *this;
}

void EncodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<EncodeParams>(m, "EncodeParams", "Class to define parameters for image encoding operations.")
        .def(py::init([]() { return EncodeParams{}; }), "Default constructor that initializes the EncodeParams object with default settings.")
        .def(py::init([](nvimgcodecQualityType_t quality_type, float quality_value, nvimgcodecColorSpec_t color_spec, std::optional<nvimgcodecChromaSubsampling_t> chroma_subsampling,
                          std::optional<JpegEncodeParams> jpeg_encode_params, std::optional<Jpeg2kEncodeParams> jpeg2k_encode_params,
                          std::optional<uint32_t> tile_width, std::optional<uint32_t> tile_height) {
            EncodeParams p;
            p.impl_.quality_type = quality_type;
            p.impl_.quality_value = quality_value;
            p.color_spec_ = color_spec;
            p.chroma_subsampling_ = chroma_subsampling;
            p.jpeg_encode_params_ = jpeg_encode_params.has_value() ? jpeg_encode_params.value() : JpegEncodeParams();
            p.jpeg2k_encode_params_ = jpeg2k_encode_params.has_value() ? jpeg2k_encode_params.value() : Jpeg2kEncodeParams();
            
            // Validate that both tile width and height are provided together, or neither is provided
            bool has_tile_width = tile_width.has_value();
            bool has_tile_height = tile_height.has_value();
            if (has_tile_width != has_tile_height) {
                throw std::invalid_argument("Both tile_width and tile_height must be provided together, or neither should be provided");
            }
            
            if (tile_width.has_value()) {
                p.tile_geometry_.setTileWidth(tile_width.value());
            }
            if (tile_height.has_value()) {
                p.tile_geometry_.setTileHeight(tile_height.value());
            }

            return p;
        }),
            "quality_type"_a = NVIMGCODEC_QUALITY_TYPE_DEFAULT,
            "quality_value"_a = 0,
            "color_spec"_a = NVIMGCODEC_COLORSPEC_UNCHANGED, 
            "chroma_subsampling"_a = py::none(),
            "jpeg_encode_params"_a = py::none(),
            "jpeg2k_encode_params"_a = py::none(),
            "tile_width"_a = py::none(),
            "tile_height"_a = py::none(),
            R"pbdoc(
            Constructor with parameters to control the encoding process.

            Args:
                quality_type (QualityType): Quality type (algorithm) that will be used to encode image.

                quality_value (float): Specifies how good encoded image should look like. Refer to the QualityType enum for the allowed values for each quality type.

                color_spec (ColorSpec): Output color specification. Defaults to UNCHANGED.

                chroma_subsampling (ChromaSubsampling): Chroma subsampling format. When not specified, defaults to CSS_GRAY for single-channel (grayscale) images and CSS_444 for multi-channel images.

                jpeg_encode_params (JpegEncodeParams): Optional JPEG specific encoding parameters.
                
                jpeg2k_encode_params (Jpeg2kEncodeParams): Optional JPEG2000 specific encoding parameters.
                
                tile_width (int): Optional tile width for tiled encoding. When set to a value greater than 0, enables tiled encoding.
                
                tile_height (int): Optional tile height for tiled encoding. When set to a value greater than 0, enables tiled encoding.
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
            Specifies the chroma subsampling format for encoding. When not specified, defaults to CSS_GRAY for single-channel (grayscale) images and CSS_444 for multi-channel images.
            )pbdoc")
        .def_property("jpeg_params", &EncodeParams::getJpegEncodeParams, &EncodeParams::setJpegEncodeParams,
            R"pbdoc(
            Optional, additional JPEG-specific encoding parameters.
            )pbdoc")
        .def_property("jpeg2k_params", &EncodeParams::getJpeg2kEncodeParams, &EncodeParams::setJpeg2kEncodeParams,
            R"pbdoc(
            Optional, additional JPEG2000-specific encoding parameters.
            )pbdoc")
        .def_property("tile_width", &EncodeParams::getTileWidth, &EncodeParams::setTileWidth,
            R"pbdoc(
            Width of tiles for tiled encoding.

            When set to a value greater than 0, enables tiled encoding with the specified tile width.
            )pbdoc")
        .def_property("tile_height", &EncodeParams::getTileHeight, &EncodeParams::setTileHeight,
            R"pbdoc(
            Height of tiles for tiled encoding.

            When set to a value greater than 0, enables tiled encoding with the specified tile height.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
