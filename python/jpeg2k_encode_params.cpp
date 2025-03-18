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
          6, 64, 64, true}
{
}

void Jpeg2kEncodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Jpeg2kEncodeParams>(m, "Jpeg2kEncodeParams", "Class to define parameters for JPEG2000 image encoding operations.")
        .def(py::init([]() { return Jpeg2kEncodeParams{}; }), 
            "Default constructor that initializes the Jpeg2kEncodeParams object with default settings.")
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
            "reversible"_a = false,
            "code_block_size"_a = std::make_tuple<int, int>(64, 64), 
            "num_resolutions"_a = 6,
            "bitstream_type"_a = NVIMGCODEC_JPEG2K_STREAM_JP2, 
            "prog_order"_a = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL,
            R"pbdoc(
            Constructor with parameters to control the JPEG2000 encoding process.

            Args:
                reversible: Boolean flag to use reversible JPEG2000 transform. Defaults to False (irreversible).

                code_block_size: Tuple representing the height and width of code blocks in the encoding. Defaults to (64, 64).
                
                num_resolutions: Number of resolution levels for the image. Defaults to 6.
                
                bitstream_type: Type of JPEG2000 bitstream, either raw codestream or JP2 container. Defaults to JP2.
                
                prog_order: Progression order for the JPEG2000 encoding. Defaults to RPCL (Resolution-Position-Component-Layer).
            )pbdoc")
        .def_property("reversible", &Jpeg2kEncodeParams::getJpeg2kReversible, &Jpeg2kEncodeParams::setJpeg2kReversible,
            R"pbdoc(
            Boolean property to enable or disable the reversible JPEG2000 transform.

            When set to True, uses a reversible transform ensuring lossless compression. Defaults to False (irreversible).
            )pbdoc")
        .def_property("code_block_size", &Jpeg2kEncodeParams::getJpeg2kCodeBlockSize, &Jpeg2kEncodeParams::setJpeg2kCodeBlockSize,
            R"pbdoc(
            Property to get or set the code block width and height for encoding.

            Defines the size of code blocks used in JPEG2000 encoding. Defaults to (64, 64).
            )pbdoc")
        .def_property("num_resolutions", &Jpeg2kEncodeParams::getJpeg2kNumResoulutions, &Jpeg2kEncodeParams::setJpeg2kNumResoulutions,
            R"pbdoc(
            Property to get or set the number of resolution levels.

            Determines the number of levels for the image's resolution pyramid. Each additional level represents a halving of the resolution.
            Defaults to 6.
            )pbdoc")
        .def_property("bitstream_type", &Jpeg2kEncodeParams::getJpeg2kBitstreamType, &Jpeg2kEncodeParams::setJpeg2kBitstreamType,
            R"pbdoc(
            Property to get or set the JPEG2000 bitstream type.

            Determines the type of container or codestream for the encoded image. Defaults to JP2.
            )pbdoc")
        .def_property("prog_order", &Jpeg2kEncodeParams::getJpeg2kProgOrder, &Jpeg2kEncodeParams::setJpeg2kProgOrder,
            R"pbdoc(
            Property to get or set the progression order for the JPEG2000 encoding.

            Specifies the order in which the encoded data is organized. It can affect decoding performance and streaming. Defaults to RPCL.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
