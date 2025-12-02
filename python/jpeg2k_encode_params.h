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
    
    Jpeg2kEncodeParams(const Jpeg2kEncodeParams& other);
    Jpeg2kEncodeParams& operator=(const Jpeg2kEncodeParams& other);

    bool getJpeg2kHT() { return impl_.ht; }
    void setJpeg2kHT(bool ht) { impl_.ht = ht; }

    int getJpeg2kMctMode() { return impl_.mct_mode; }
    void setJpeg2kMctMode(int mct_mode) { impl_.mct_mode = mct_mode; }

    std::tuple<int, int> getJpeg2kCodeBlockSize()
    {
        return std::make_tuple<int, int>(static_cast<int>(impl_.code_block_w),
            static_cast<int>(impl_.code_block_h));
    }
    void setJpeg2kCodeBlockSize(std::tuple<int, int> size)
    {
        impl_.code_block_w = std::get<0>(size);
        impl_.code_block_h = std::get<1>(size);
    }
    int getJpeg2kNumResoulutions() { return impl_.num_resolutions; }
    void setJpeg2kNumResoulutions(int num_resolutions) { impl_.num_resolutions = num_resolutions; };

    nvimgcodecJpeg2kBitstreamType_t getJpeg2kBitstreamType() { return impl_.stream_type; }
    void setJpeg2kBitstreamType(nvimgcodecJpeg2kBitstreamType_t bistream_type)
    {
        impl_.stream_type = bistream_type;
    };

    nvimgcodecJpeg2kProgOrder_t getJpeg2kProgOrder() { return impl_.prog_order; }
    void setJpeg2kProgOrder(nvimgcodecJpeg2kProgOrder_t prog_order) { impl_.prog_order = prog_order; };

    static void exportToPython(py::module& m);

    nvimgcodecJpeg2kEncodeParams_t impl_;
};

} // namespace nvimgcodec
