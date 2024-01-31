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

namespace nvimgcodec {

namespace py = pybind11;
using namespace py::literals;

class JpegEncodeParams
{
  public:
    JpegEncodeParams();

    bool getJpegProgressive(){
        return nvimgcodec_jpeg_image_info_.encoding == NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN;}
    void setJpegProgressive(bool progressive) {
        nvimgcodec_jpeg_image_info_.encoding =
            progressive ? NVIMGCODEC_JPEG_ENCODING_PROGRESSIVE_DCT_HUFFMAN : NVIMGCODEC_JPEG_ENCODING_BASELINE_DCT; }

    bool getJpegOptimizedHuffman(){
        return nvimgcodec_jpeg_encode_params_.optimized_huffman;}
    void setJpegOptimizedHuffman(bool optimized_huffman) {
       nvimgcodec_jpeg_encode_params_.optimized_huffman = optimized_huffman; }

    static void exportToPython(py::module& m);

    nvimgcodecJpegEncodeParams_t nvimgcodec_jpeg_encode_params_;
    nvimgcodecJpegImageInfo_t nvimgcodec_jpeg_image_info_;
};

} // namespace nvimgcodec
