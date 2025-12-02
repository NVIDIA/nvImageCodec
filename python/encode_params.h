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

#pragma once

#include <string>
#include <vector>
#include <tuple>
#include <optional>

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jpeg2k_encode_params.h"
#include "jpeg_encode_params.h"
#include "tile_geometry.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class EncodeParams
{
  public:
    EncodeParams();
    
    EncodeParams(const EncodeParams& other);
    EncodeParams& operator=(const EncodeParams& other);

    nvimgcodecQualityType_t getQualityType() { return impl_.quality_type; }
    void setQualityType(nvimgcodecQualityType_t quality_type) { impl_.quality_type = quality_type; };

    float getQualityValue() { return impl_.quality_value; }
    void setQualityValue(float quality_value) { impl_.quality_value = quality_value; };

    nvimgcodecColorSpec_t getColorSpec() { return color_spec_; }
    void setColorSpec(nvimgcodecColorSpec_t color_spec) { color_spec_ = color_spec; };

    std::optional<nvimgcodecChromaSubsampling_t> getChromaSubsampling() { return chroma_subsampling_; }
    void setChromaSubsampling(std::optional<nvimgcodecChromaSubsampling_t> chroma_subsampling) { chroma_subsampling_ = chroma_subsampling; }

    Jpeg2kEncodeParams& getJpeg2kEncodeParams() { return jpeg2k_encode_params_; }
    void setJpeg2kEncodeParams(Jpeg2kEncodeParams jpeg2k_encode_params) { jpeg2k_encode_params_ = jpeg2k_encode_params; }

    JpegEncodeParams& getJpegEncodeParams() { return jpeg_encode_params_; }
    void setJpegEncodeParams(JpegEncodeParams jpeg_encode_params) { jpeg_encode_params_ = jpeg_encode_params; }

    uint32_t getTileWidth() { return tile_geometry_.getTileWidth(); }
    void setTileWidth(uint32_t tile_width) { tile_geometry_.setTileWidth(tile_width); }

    uint32_t getTileHeight() { return tile_geometry_.getTileHeight(); }
    void setTileHeight(uint32_t tile_height) { tile_geometry_.setTileHeight(tile_height); }

    nvimgcodecEncodeParams_t* handle() { return &impl_; }

    static void exportToPython(py::module& m);

    void setupStructChain();

    Jpeg2kEncodeParams jpeg2k_encode_params_;
    JpegEncodeParams jpeg_encode_params_;
    TileGeometry tile_geometry_;
    std::optional<nvimgcodecChromaSubsampling_t> chroma_subsampling_;
    nvimgcodecColorSpec_t color_spec_;
    nvimgcodecEncodeParams_t impl_;
};

} // namespace nvimgcodec
