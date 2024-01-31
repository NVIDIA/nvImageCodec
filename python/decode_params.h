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

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class DecodeParams
{
  public:
    DecodeParams();
    bool getEnableOrientation() {return decode_params_.apply_exif_orientation;}
    void setEnableOrientation(bool enable){decode_params_.apply_exif_orientation = enable;};
    nvimgcodecColorSpec_t getColorSpec() {return color_spec_;}
    void setColorSpec(nvimgcodecColorSpec_t color_spec){color_spec_ = color_spec;};
    bool getAllowAnyDepth() {return allow_any_depth_;}
    void setAllowAnyDepth(bool allow_any_depth){allow_any_depth_ = allow_any_depth;};

    static void exportToPython(py::module& m);

    nvimgcodecDecodeParams_t decode_params_;
    nvimgcodecColorSpec_t color_spec_;
    bool allow_any_depth_;
};

} // namespace nvimgcodec
