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

#pragma once

#include <string>
#include <vector>
#include <tuple>

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "jpeg2k_encode_params.h"
#include "jpeg_encode_params.h"

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class EncodeParams
{
  public:
    EncodeParams();

    float getQuality() { return encode_params_.quality; }
    void setQuality(float quality) { encode_params_.quality = quality; };

    float getTargetPsnr() { return encode_params_.target_psnr; }
    void setTargetPsnr(float target_psnr) { encode_params_.target_psnr = target_psnr; };

    nvimgcodecColorSpec_t getColorSpec() { return color_spec_; }
    void setColorSpec(nvimgcodecColorSpec_t color_spec) { color_spec_ = color_spec; };

    nvimgcodecChromaSubsampling_t getChromaSubsampling() { return chroma_subsampling_; }
    void setChromaSubsampling(nvimgcodecChromaSubsampling_t chroma_subsampling) { chroma_subsampling_ = chroma_subsampling; }

    Jpeg2kEncodeParams& getJpeg2kEncodeParams() { return jpeg2k_encode_params_; }
    void setJpeg2kEncodeParams(Jpeg2kEncodeParams jpeg2k_encode_params) { jpeg2k_encode_params_ = jpeg2k_encode_params; }

    JpegEncodeParams& getJpegEncodeParams() { return jpeg_encode_params_; }
    void setJpegEncodeParams(JpegEncodeParams jpeg_encode_params) { jpeg_encode_params_ = jpeg_encode_params; }
    static void exportToPython(py::module& m);

    Jpeg2kEncodeParams jpeg2k_encode_params_;
    JpegEncodeParams jpeg_encode_params_;
    nvimgcodecEncodeParams_t encode_params_;
    nvimgcodecChromaSubsampling_t chroma_subsampling_;
    nvimgcodecColorSpec_t color_spec_;
};

} // namespace nvimgcodec
