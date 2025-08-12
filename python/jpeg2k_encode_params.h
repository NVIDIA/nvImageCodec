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
#include <tuple>
#include <vector>

#include <nvimgcodec.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class Jpeg2kEncodeParams
{
  public:
    Jpeg2kEncodeParams();

    bool getJpeg2kHT() { return nvimgcodec_jpeg2k_encode_params_.ht; }
    void setJpeg2kHT(bool ht) { nvimgcodec_jpeg2k_encode_params_.ht = ht; }

    std::tuple<int, int> getJpeg2kCodeBlockSize()
    {
        return std::make_tuple<int, int>(static_cast<int>(nvimgcodec_jpeg2k_encode_params_.code_block_w),
            static_cast<int>(nvimgcodec_jpeg2k_encode_params_.code_block_h));
    }
    void setJpeg2kCodeBlockSize(std::tuple<int, int> size)
    {
        nvimgcodec_jpeg2k_encode_params_.code_block_w = std::get<0>(size);
        nvimgcodec_jpeg2k_encode_params_.code_block_h = std::get<1>(size);
    }
    int getJpeg2kNumResoulutions() { return nvimgcodec_jpeg2k_encode_params_.num_resolutions; }
    void setJpeg2kNumResoulutions(int num_resolutions) { nvimgcodec_jpeg2k_encode_params_.num_resolutions = num_resolutions; };

    nvimgcodecJpeg2kBitstreamType_t getJpeg2kBitstreamType() { return nvimgcodec_jpeg2k_encode_params_.stream_type; }
    void setJpeg2kBitstreamType(nvimgcodecJpeg2kBitstreamType_t bistream_type)
    {
        nvimgcodec_jpeg2k_encode_params_.stream_type = bistream_type;
    };

    nvimgcodecJpeg2kProgOrder_t getJpeg2kProgOrder() { return nvimgcodec_jpeg2k_encode_params_.prog_order; }
    void setJpeg2kProgOrder(nvimgcodecJpeg2kProgOrder_t prog_order) { nvimgcodec_jpeg2k_encode_params_.prog_order = prog_order; };

    static void exportToPython(py::module& m);

    nvimgcodecJpeg2kEncodeParams_t nvimgcodec_jpeg2k_encode_params_;
};

} // namespace nvimgcodec
