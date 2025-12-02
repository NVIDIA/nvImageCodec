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

#include "error_handling.h"

namespace nvimgcodec {

Jpeg2kEncodeParams::Jpeg2kEncodeParams()
    : impl_{NVIMGCODEC_STRUCTURE_TYPE_JPEG2K_ENCODE_PARAMS, sizeof(nvimgcodecJpeg2kEncodeParams_t), nullptr, NVIMGCODEC_JPEG2K_STREAM_JP2, NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL,
          6, 64, 64, 0, false}
{
}

Jpeg2kEncodeParams::Jpeg2kEncodeParams(const Jpeg2kEncodeParams& other)
    : impl_(other.impl_)
{
    // Set struct_next to nullptr as it should be managed separately
    impl_.struct_next = nullptr;
}

Jpeg2kEncodeParams& Jpeg2kEncodeParams::operator=(const Jpeg2kEncodeParams& other)
{
    if (this != &other) {
        // Copy all fields except struct_next
        impl_ = other.impl_;
        // Set struct_next to nullptr as it should be managed separately
        impl_.struct_next = nullptr;
    }
    return *this;
}

void Jpeg2kEncodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<Jpeg2kEncodeParams>(m, "Jpeg2kEncodeParams", "Class to define parameters for JPEG2000 image encoding operations.")
        .def(py::init([]() { return Jpeg2kEncodeParams{}; }), 
            "Default constructor that initializes the Jpeg2kEncodeParams object with default settings.")
        .def(py::init([](std::tuple<int, int> code_block_size, int num_resolutions,
                          nvimgcodecJpeg2kBitstreamType_t bitstream_type, nvimgcodecJpeg2kProgOrder_t prog_order, int mct_mode, bool ht) {
            Jpeg2kEncodeParams p;
            p.impl_.code_block_w = std::get<0>(code_block_size);
            p.impl_.code_block_h = std::get<1>(code_block_size);
            p.impl_.num_resolutions = num_resolutions;
            p.impl_.stream_type = bitstream_type;
            p.impl_.prog_order = prog_order;
            p.impl_.mct_mode = mct_mode;
            p.impl_.ht = ht;
            return p;
        }),
            "code_block_size"_a = std::make_tuple<int, int>(64, 64), 
            "num_resolutions"_a = 6,
            "bitstream_type"_a = NVIMGCODEC_JPEG2K_STREAM_JP2, 
            "prog_order"_a = NVIMGCODEC_JPEG2K_PROG_ORDER_RPCL,
            "mct_mode"_a = 0,
            "ht"_a = false,
            R"pbdoc(
            Constructor with parameters to control the JPEG2000 encoding process.

            Args:
                code_block_size: Tuple representing the height and width of code blocks in the encoding. Defaults to (64, 64).
                
                num_resolutions: Number of resolution levels for the image. Defaults to 6.
                
                bitstream_type: Type of JPEG2000 bitstream, either raw codestream or JP2 container. Defaults to JP2.
                
                prog_order: Progression order for the JPEG2000 encoding. Defaults to RPCL (Resolution-Position-Component-Layer).
                
                mct_mode: Integer flag to use multiple component transform. Defaults to 0 (do not use MCT). Valid values are 0 or 1.
                
                ht: Boolean flag to use High-Throughput JPEG2000 encoder. Defaults to False (do not use HT).
            )pbdoc")
        .def_property("mct_mode", &Jpeg2kEncodeParams::getJpeg2kMctMode, &Jpeg2kEncodeParams::setJpeg2kMctMode,
            R"pbdoc(
            Integer property to set the multiple component transform mode.

            When set to 1, uses the multiple component transform (MCT). Defaults to 0.
            
            It can be used to convert SRGB input color space to SYCC for coding efficiency. Irreversible 
            component transformation used with the 9-7 irreversible filter (lossy compression). Reversible component transformation 
            used with the 5-3 reversible filter (lossless compression).

            Multiple component transformation can be used only with SRGB input color space and with chroma subsampling 444.
        )pbdoc")
        .def_property("ht", &Jpeg2kEncodeParams::getJpeg2kHT, &Jpeg2kEncodeParams::setJpeg2kHT,
            R"pbdoc(
            Boolean property to enable or disable the High-Throughput JPEG2000 encoder.

            When set to True, uses the High-Throughput JPEG2000 encoder. Defaults to False.
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
