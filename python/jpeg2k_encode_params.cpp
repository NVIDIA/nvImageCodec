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

#include "error_handling.h"

namespace nvimgcodec {

Jpeg2kEncodeParams::Jpeg2kEncodeParams()
    : nvimgcodec_jpeg2k_encode_params_{NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), nullptr, NVIMGCODEC_JPEG2K_STREAM_JP2, NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL,
          5, 64, 64, true}
{
}

void Jpeg2kEncodeParams::exportToPython(py::module& m)
{
    py::class_<Jpeg2kEncodeParams>(m, "Jpeg2kEncodeParams")
        .def(py::init([]() { return Jpeg2kEncodeParams{}; }), "Default constructor")
        .def(py::init([](bool reversible, std::tuple<int, int> code_block_size, int num_resolutions,
                          nvimgcodecJpeg2kBitstreamType_t bitstream_type, nvimgcodecJpeg2kProgOrder_t prog_order) {
            Jpeg2kEncodeParams p;
            p.nvimgcodec_jpeg2k_encode_params_.irreversible = !reversible;
            p.nvimgcodec_jpeg2k_encode_params_.code_block_w = std::get<0>(code_block_size);
            p.nvimgcodec_jpeg2k_encode_params_.code_block_h = std::get<1>(code_block_size);
            p.nvimgcodec_jpeg2k_encode_params_.num_resolutions = num_resolutions;
            p.nvimgcodec_jpeg2k_encode_params_.stream_type = bitstream_type;
            p.nvimgcodec_jpeg2k_encode_params_.prog_order = prog_order;

            return p;
        }),
        // clang-format off
            "reversible"_a = false,
            "code_block_size"_a = std::make_tuple<int, int>(64, 64), 
            "num_resolutions"_a = 5,
            "bitstream_type"_a = NVIMGCODEC_JPEG2K_STREAM_JP2, 
            "prog_order"_a = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL,
            "Constructor with reversible, code_block_size, num_resolutions, bitstream_type, prog_order parameters")
        // clang-format on
        .def_property("reversible", &Jpeg2kEncodeParams::getJpeg2kReversible, &Jpeg2kEncodeParams::setJpeg2kReversible,
            "Use reversible Jpeg 2000 transform (default False)")
        .def_property("code_block_size", &Jpeg2kEncodeParams::getJpeg2kCodeBlockSize, &Jpeg2kEncodeParams::setJpeg2kCodeBlockSize,
            "Jpeg 2000 code block width and height (default 64x64)")
        .def_property("num_resolutions", &Jpeg2kEncodeParams::getJpeg2kNumResoulutions,
            &Jpeg2kEncodeParams::setJpeg2kNumResoulutions, "Jpeg 2000 number of resolutions - decomposition levels (default 5)")
        .def_property("bitstream_type", &Jpeg2kEncodeParams::getJpeg2kBitstreamType, &Jpeg2kEncodeParams::setJpeg2kBitstreamType,
            "Jpeg 2000 bitstream type (default JP2)")
        .def_property("prog_order", &Jpeg2kEncodeParams::getJpeg2kProgOrder, &Jpeg2kEncodeParams::setJpeg2kProgOrder,
            "Jpeg 2000 progression order (default RPCL)");
}

} // namespace nvimgcodec
