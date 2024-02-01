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
    : nvimgcodec_jpeg_encode_params_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_ENCODE_PARAMS, sizeof(nvimgcodecJpegEncodeParams_t), nullptr, false}
    , nvimgcodec_jpeg_image_info_{NVIMGCODEC_STRUCTURE_TYPE_JPEG_IMAGE_INFO, sizeof(nvimgcodecJpegImageInfo_t), nullptr, NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT}
{
}

void JpegEncodeParams::exportToPython(py::module& m)
{
    py::class_<JpegEncodeParams>(m, "JpegEncodeParams")
        .def(py::init([]() { return JpegEncodeParams{}; }), "Default constructor")
        .def(py::init([](bool jpeg_progressive, bool jpeg_optimized_huffman) {
            JpegEncodeParams p;
            p.nvimgcodec_jpeg_encode_params_.optimized_huffman = jpeg_optimized_huffman;
            p.nvimgcodec_jpeg_image_info_.encoding =
                jpeg_progressive ? NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT;
            return p;
        }),
            // clang-format off
            "progressive"_a = false,
            "optimized_huffman"_a = false,
            "Constructor with progressive, optimized_huffman parameters")
        // clang-format on
        .def_property("progressive", &JpegEncodeParams::getJpegProgressive, &JpegEncodeParams::setJpegProgressive,
            "Use Jpeg progressive encoding (default False)")
        .def_property("optimized_huffman", &JpegEncodeParams::getJpegOptimizedHuffman, &JpegEncodeParams::setJpegOptimizedHuffman,
            "Use Jpeg encoding with optimized Huffman (default False)");
}

} // namespace nvimgcodec
