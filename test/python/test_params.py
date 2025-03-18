# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations
import os
import sys
import pytest as t
import numpy as np
from nvidia import nvimgcodec
from utils import *
import cv2

def test_bankend_params():
    # BackendParams default constructor
    backend_params = nvimgcodec.BackendParams()
    assert (backend_params.load_hint == t.approx(1.0))

    # BackendParams constructor with load_hint parameters
    backend_params = nvimgcodec.BackendParams(0.0)
    assert (backend_params.load_hint == t.approx(0.0))

    backend_params.load_hint = 0.5 
    assert (backend_params.load_hint == t.approx(0.5))

    # Backend default constructor
    backend = nvimgcodec.Backend()
    assert(backend.backend_kind == nvimgcodec.GPU_ONLY)
    assert(backend.load_hint == t.approx(1.0))

    # Backend constructor with parameters
    backend = nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5)
    assert(backend.backend_kind == nvimgcodec.GPU_ONLY)
    assert(backend.load_hint == t.approx(0.5))

    backend.backend_kind = nvimgcodec.CPU_ONLY
    backend.load_hint = 0.7
    assert(backend.backend_kind == nvimgcodec.CPU_ONLY)
    assert(backend.load_hint == t.approx(0.7))

    # Backend constructor with backend parameters
    backend_params = nvimgcodec.BackendParams()
    backend = nvimgcodec.Backend(nvimgcodec.GPU_ONLY, backend_params)
    assert (backend.load_hint == t.approx(1.0))

    backend_params = backend.backend_params
    assert (backend_params.load_hint == t.approx(1.0))

    backend_params.load_hint = 0.5
    backend.backend_params = backend_params
    assert (backend.load_hint == t.approx(0.5))

def test_decode_params():
    # DecodeParams default constructor
    decode_params = nvimgcodec.DecodeParams()
    assert (decode_params.apply_exif_orientation == True)
    assert (decode_params.allow_any_depth == False)
    assert (decode_params.color_spec == nvimgcodec.RGB)

    decode_params.apply_exif_orientation = False
    assert (decode_params.apply_exif_orientation == False)

    decode_params.allow_any_depth = True
    assert (decode_params.allow_any_depth == True)

    decode_params.color_spec = nvimgcodec.YCC
    assert (decode_params.color_spec == nvimgcodec.YCC)

    decode_params.color_spec = nvimgcodec.GRAY
    assert (decode_params.color_spec == nvimgcodec.GRAY)

    decode_params.color_spec = nvimgcodec.UNCHANGED
    assert (decode_params.color_spec == nvimgcodec.UNCHANGED)

    decode_params.color_spec = nvimgcodec.RGB
    assert (decode_params.color_spec == nvimgcodec.RGB)

    # DecodeParams constructor with parameters
    decode_params = nvimgcodec.DecodeParams(False, nvimgcodec.YCC, True)
    assert (decode_params.apply_exif_orientation == False)
    assert (decode_params.allow_any_depth == True)
    assert (decode_params.color_spec == nvimgcodec.YCC)

    decode_params = nvimgcodec.DecodeParams(allow_any_depth=True, color_spec=nvimgcodec.GRAY, apply_exif_orientation=False)
    assert (decode_params.apply_exif_orientation == False)
    assert (decode_params.allow_any_depth == True)
    assert (decode_params.color_spec == nvimgcodec.GRAY)

def test_encode_params():
    # EncodeParams default constructor
    encode_params = nvimgcodec.EncodeParams()
    assert (encode_params.quality == 95)
    assert (encode_params.target_psnr == 50)
    assert (encode_params.color_spec == nvimgcodec.UNCHANGED)
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_444)

    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == False)
    assert (jpeg_params.optimized_huffman == False)

    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.reversible == False)
    assert (jpeg2k_params.code_block_size == (64, 64))
    assert (jpeg2k_params.num_resolutions == 6)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_params.prog_order == nvimgcodec.RPCL)

    encode_params.quality = 100
    assert (encode_params.quality == 100)

    encode_params.target_psnr = 45
    assert (encode_params.target_psnr == 45)

    encode_params.color_spec = nvimgcodec.YCC
    assert (encode_params.color_spec == nvimgcodec.YCC)

    encode_params.color_spec = nvimgcodec.GRAY
    assert (encode_params.color_spec == nvimgcodec.GRAY)

    encode_params.color_spec = nvimgcodec.UNCHANGED
    assert (encode_params.color_spec == nvimgcodec.UNCHANGED)

    encode_params.color_spec = nvimgcodec.RGB
    assert (encode_params.color_spec == nvimgcodec.RGB)

    encode_params.chroma_subsampling = nvimgcodec.CSS_422
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_422)

    encode_params.chroma_subsampling = nvimgcodec.CSS_444
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_444)

    encode_params.chroma_subsampling = nvimgcodec.CSS_420
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_420)

    encode_params.chroma_subsampling = nvimgcodec.CSS_440
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_440)

    encode_params.chroma_subsampling = nvimgcodec.CSS_411
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_411)

    encode_params.chroma_subsampling = nvimgcodec.CSS_410
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_410)

    encode_params.chroma_subsampling = nvimgcodec.CSS_GRAY
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_GRAY)

    encode_params.chroma_subsampling = nvimgcodec.CSS_410V
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_410V)

    encode_params.jpeg_params = nvimgcodec.JpegEncodeParams(False, False)
    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == False)
    assert (jpeg_params.optimized_huffman == False)

    encode_params.jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams(True, (786, 659), 897, nvimgcodec.J2K, nvimgcodec.PCRL)
    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.reversible == True)
    assert (jpeg2k_params.code_block_size == (786, 659))
    assert (jpeg2k_params.num_resolutions == 897)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.J2K)
    assert (jpeg2k_params.prog_order == nvimgcodec.PCRL)

    # EncodeParams constructor with parameters
    encode_params = nvimgcodec.EncodeParams(46, 78, nvimgcodec.GRAY, nvimgcodec.CSS_410)
    assert (encode_params.quality == 46)
    assert (encode_params.target_psnr == 78)
    assert (encode_params.color_spec == nvimgcodec.GRAY)
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_410)

    # jpeg_encode_params is optional, if not, it is default value.
    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == False)
    assert (jpeg_params.optimized_huffman == False)

    # jpeg2k_encode_params is optional, if not, it is default value.
    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.reversible == False)
    assert (jpeg2k_params.code_block_size == (64, 64))
    assert (jpeg2k_params.num_resolutions == 6)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_params.prog_order == nvimgcodec.RPCL)

    jpeg_params = nvimgcodec.JpegEncodeParams(True, False)
    jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams(True, (786, 659), 897, nvimgcodec.J2K, nvimgcodec.PCRL)
    encode_params = nvimgcodec.EncodeParams(46, 78, nvimgcodec.GRAY, nvimgcodec.CSS_410, jpeg_params, jpeg2k_params)

    # jpeg_encode_params is optional, if yes, it is set value in construtor.
    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == True)
    assert (jpeg_params.optimized_huffman == False)

    # jpeg2k_encode_params is optional, if yes, it is set value in construtor.
    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.reversible == True)
    assert (jpeg2k_params.code_block_size == (786, 659))
    assert (jpeg2k_params.num_resolutions == 897)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.J2K)
    assert (jpeg2k_params.prog_order == nvimgcodec.PCRL)

    encode_params = nvimgcodec.EncodeParams(chroma_subsampling=nvimgcodec.CSS_411, target_psnr=65, color_spec=nvimgcodec.YCC, quality=32)
    assert (encode_params.quality == 32)
    assert (encode_params.target_psnr == 65)
    assert (encode_params.color_spec == nvimgcodec.YCC)
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_411)

def test_jpeg_encode_params():
    # JpegEncodeParams default constructor
    jpeg_encode_params = nvimgcodec.JpegEncodeParams()
    assert (jpeg_encode_params.progressive == False)
    assert (jpeg_encode_params.optimized_huffman == False)

    jpeg_encode_params.progressive = True
    assert (jpeg_encode_params.progressive == True)

    jpeg_encode_params.optimized_huffman = True
    assert (jpeg_encode_params.optimized_huffman == True)

    # JpegEncodeParams constructor with parameters
    jpeg_encode_params = nvimgcodec.JpegEncodeParams(False, False)
    assert (jpeg_encode_params.progressive == False)
    assert (jpeg_encode_params.optimized_huffman == False)

    jpeg_encode_params = nvimgcodec.JpegEncodeParams(False, True)
    assert (jpeg_encode_params.progressive == False)
    assert (jpeg_encode_params.optimized_huffman == True)

    jpeg_encode_params = nvimgcodec.JpegEncodeParams(True, False)
    assert (jpeg_encode_params.progressive == True)
    assert (jpeg_encode_params.optimized_huffman == False)

    jpeg_encode_params = nvimgcodec.JpegEncodeParams(True, True)
    assert (jpeg_encode_params.progressive == True)
    assert (jpeg_encode_params.optimized_huffman == True)

    jpeg_encode_params = nvimgcodec.JpegEncodeParams(optimized_huffman=True, progressive=False)
    assert (jpeg_encode_params.progressive == False)
    assert (jpeg_encode_params.optimized_huffman == True)

def test_jpeg2k_encode_params():
    # Jpeg2kEncodeParams default constructor
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    assert (jpeg2k_encode_params.reversible == False)
    assert (jpeg2k_encode_params.code_block_size == (64, 64))
    assert (jpeg2k_encode_params.num_resolutions == 6)
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.RPCL)

    jpeg2k_encode_params.reversible = False
    assert (jpeg2k_encode_params.reversible == False)

    jpeg2k_encode_params.code_block_size = (32, 32)
    assert (jpeg2k_encode_params.code_block_size == (32, 32))

    jpeg2k_encode_params.num_resolutions = 8
    assert (jpeg2k_encode_params.num_resolutions == 8)

    jpeg2k_encode_params.bitstream_type = nvimgcodec.J2K
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.J2K)

    jpeg2k_encode_params.bitstream_type = nvimgcodec.JP2
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.JP2)

    jpeg2k_encode_params.prog_order = nvimgcodec.LRCP
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.LRCP)

    jpeg2k_encode_params.prog_order = nvimgcodec.RLCP
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.RLCP)

    jpeg2k_encode_params.prog_order = nvimgcodec.RPCL
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.RPCL)

    jpeg2k_encode_params.prog_order = nvimgcodec.PCRL
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.PCRL)

    jpeg2k_encode_params.prog_order = nvimgcodec.CPRL
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.CPRL)

    # Jpeg2kEncodeParams constructor with parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams(True, (128, 256), 73, nvimgcodec.J2K, nvimgcodec.PCRL)
    assert (jpeg2k_encode_params.reversible == True)
    assert (jpeg2k_encode_params.code_block_size == (128, 256))
    assert (jpeg2k_encode_params.num_resolutions == 73)
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.J2K)
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.PCRL)

    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams(prog_order=nvimgcodec.CPRL, num_resolutions=76, code_block_size=(28, 86), bitstream_type=nvimgcodec.JP2, reversible=False)
    assert (jpeg2k_encode_params.reversible == False)
    assert (jpeg2k_encode_params.code_block_size == (28, 86))
    assert (jpeg2k_encode_params.num_resolutions == 76)
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.CPRL)

def test_region_params():
    # Region default constructor
    region = nvimgcodec.Region()
    assert (region.ndim == 0)

    # Region constructor with parameters
    # dimension = 1
    region = nvimgcodec.Region(end=(10,), start=(20,))
    assert (region.ndim == 1)
    assert (region.start == (20,))
    assert (region.end == (10,))

    region = nvimgcodec.Region([10], [20])
    assert (region.ndim == 1)
    assert (region.start == (10,))
    assert (region.end == (20,))

    print(region)

    # dimension = 2
    region = nvimgcodec.Region(189, 19, 380, 89)
    assert (region.ndim == 2)
    assert (region.start == (189, 19))
    assert (region.end == (380, 89))

    region = nvimgcodec.Region(end_y=189, start_x=19, end_x=380, start_y=89)
    assert (region.ndim == 2)
    assert (region.start == (89, 19))
    assert (region.end == (189, 380))

    print(region)

    region = nvimgcodec.Region((189, 19), (380, 89))
    assert (region.ndim == 2)
    assert (region.start == (189, 19))
    assert (region.end == (380, 89))

    region = nvimgcodec.Region(start=[87, 78], end=[45, 809])
    assert (region.ndim == 2)
    assert (region.start == (87, 78))
    assert (region.end == (45, 809))

    print(region)

    # dimension = 3
    region = nvimgcodec.Region(start=(13, 76, 67), end=(87, 65, 98))
    assert (region.ndim == 3)
    assert (region.start == (13, 76, 67))
    assert (region.end == (87, 65, 98))

    region = nvimgcodec.Region([13, 76, 67], [87, 65, 98])
    assert (region.ndim == 3)
    assert (region.start == (13, 76, 67))
    assert (region.end == (87, 65, 98))

    print(region)

    # dimension = 4
    region = nvimgcodec.Region((93, 6, 7, 8), (7, 5, 9, 10))
    assert (region.ndim == 4)
    assert (region.start == (93, 6, 7, 8))
    assert (region.end == (7, 5, 9, 10))

    region = nvimgcodec.Region(start=[3, 6, 7, 8], end=[7, 5, 9, 10])
    assert (region.ndim == 4)
    assert (region.start == (3, 6, 7, 8))
    assert (region.end == (7, 5, 9, 10))

    print(region)

    # dimension = 5
    region = nvimgcodec.Region(start=(93, 6, 7, 8, 9), end=(7, 5, 9, 10, 11))
    assert (region.ndim == 5)
    assert (region.start == (93, 6, 7, 8, 9))
    assert (region.end == (7, 5, 9, 10, 11))

    region = nvimgcodec.Region([3, 6, 7, 8, 9], [7, 5, 9, 10, 11])
    assert (region.ndim == 5)
    assert (region.start == (3, 6, 7, 8, 9))
    assert (region.end == (7, 5, 9, 10, 11))

    print(region)

    # negative tests
    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region(start=(8192, 28672), end=(10, 11, 12))
    assert (str(excinfo.value) == "Dimension mismatch")

    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region(start=(1, 2, 3, 4, 5, 6), end=(1, 2, 3, 4, 5, 6))
    assert (str(excinfo.value) == "Too many dimensions: 6")
