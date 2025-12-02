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

JpegEncodeParams::JpegEncodeParams()
    : impl_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), nullptr, false}
    , nvimgcodec_jpeg_image_info_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr, NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT}
{
}

JpegEncodeParams::JpegEncodeParams(const JpegEncodeParams& other)
    : impl_{ other.impl_}
    , nvimgcodec_jpeg_image_info_{ other.nvimgcodec_jpeg_image_info_}
{
    impl_.struct_next = nullptr;
    nvimgcodec_jpeg_image_info_.struct_next = nullptr;
}

JpegEncodeParams& JpegEncodeParams::operator=(const JpegEncodeParams& other)
{
    if (this != &other) {
        impl_ = other.impl_;
        nvimgcodec_jpeg_image_info_ = other.nvimgcodec_jpeg_image_info_;
        impl_.struct_next = nullptr;
        nvimgcodec_jpeg_image_info_.struct_next = nullptr;
    }
    return *this;
}

void JpegEncodeParams::exportToPython(py::module& m)
{
    // clang-format off
    py::class_<JpegEncodeParams>(m, "JpegEncodeParams", "Class to define parameters for JPEG image encoding operations."
        " It provides settings to configure JPEG encoding such as enabling progressive encoding and optimizing Huffman tables.")
        .def(py::init([]() { return JpegEncodeParams{}; }), 
            "Default constructor that initializes the JpegEncodeParams object with default settings.")
        .def(py::init([](bool jpeg_progressive, bool jpeg_optimized_huffman) {
            JpegEncodeParams p;
            p.impl_.optimized_huffman = jpeg_optimized_huffman;
            p.nvimgcodec_jpeg_image_info_.encoding =
                jpeg_progressive ? NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT;
            return p;
        }),
            "progressive"_a = false,
            "optimized_huffman"_a = false,
            R"pbdoc(
            Constructor with parameters to control the JPEG encoding process.

            Args:
                progressive: Boolean flag to use progressive JPEG encoding. Defaults to False.
                
                optimized_huffman: Boolean flag to use optimized Huffman tables for JPEG encoding. Defaults to False.
            )pbdoc")
        .def_property("progressive", &JpegEncodeParams::getJpegProgressive, &JpegEncodeParams::setJpegProgressive,
            R"pbdoc(
            Boolean property to enable or disable progressive JPEG encoding.

            When set to True, the encoded JPEG will be progressive, meaning it can be rendered in successive waves of detail. Defaults to False.
            )pbdoc")
        .def_property("optimized_huffman", &JpegEncodeParams::getJpegOptimizedHuffman, &JpegEncodeParams::setJpegOptimizedHuffman,
            R"pbdoc(
            Boolean property to enable or disable the use of optimized Huffman tables in JPEG encoding.

            When set to True, the JPEG encoding process will use optimized Huffman tables which produce smaller file sizes but may require more processing time. Defaults to False.
            )pbdoc");
    // clang-format on
}


} // namespace nvimgcodec
