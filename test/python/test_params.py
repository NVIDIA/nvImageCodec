# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

def test_backend_params():
    # Backend default constructor
    backend = nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)
    assert (backend.backend_kind == nvimgcodec.BackendKind.HYBRID_CPU_GPU)
    assert (backend.load_hint == t.approx(1.0))
    assert (backend.load_hint_policy == nvimgcodec.LoadHintPolicy.FIXED)


    # Backen constructor with load_hint parameters
    backend = nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY, 0.0)
    assert (backend.backend_kind == nvimgcodec.BackendKind.CPU_ONLY)
    assert (backend.load_hint == t.approx(0.0))
    assert (backend.load_hint_policy == nvimgcodec.LoadHintPolicy.FIXED)


    # Backend constructor with all parameters
    backend = nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY, 0.8, nvimgcodec.LoadHintPolicy.ADAPTIVE_MINIMIZE_IDLE_TIME)
    assert(backend.backend_kind == nvimgcodec.BackendKind.CPU_ONLY)
    assert(backend.load_hint == t.approx(0.8))
    assert(backend.load_hint_policy == nvimgcodec.LoadHintPolicy.ADAPTIVE_MINIMIZE_IDLE_TIME)


def test_decode_params():
    # DecodeParams default constructor
    decode_params = nvimgcodec.DecodeParams()
    assert (decode_params.apply_exif_orientation == True)
    assert (decode_params.allow_any_depth == False)
    assert (decode_params.color_spec == nvimgcodec.SRGB)

    decode_params.apply_exif_orientation = False
    assert (decode_params.apply_exif_orientation == False)

    decode_params.allow_any_depth = True
    assert (decode_params.allow_any_depth == True)

    decode_params.color_spec = nvimgcodec.SYCC
    assert (decode_params.color_spec == nvimgcodec.SYCC)

    decode_params.color_spec = nvimgcodec.GRAY
    assert (decode_params.color_spec == nvimgcodec.GRAY)

    decode_params.color_spec = nvimgcodec.UNCHANGED
    assert (decode_params.color_spec == nvimgcodec.UNCHANGED)

    decode_params.color_spec = nvimgcodec.SRGB
    assert (decode_params.color_spec == nvimgcodec.SRGB)

    # DecodeParams constructor with parameters
    decode_params = nvimgcodec.DecodeParams(False, nvimgcodec.SYCC, True)
    assert (decode_params.apply_exif_orientation == False)
    assert (decode_params.allow_any_depth == True)
    assert (decode_params.color_spec == nvimgcodec.SYCC)

    decode_params = nvimgcodec.DecodeParams(allow_any_depth=True, color_spec=nvimgcodec.GRAY, apply_exif_orientation=False)
    assert (decode_params.apply_exif_orientation == False)
    assert (decode_params.allow_any_depth == True)
    assert (decode_params.color_spec == nvimgcodec.GRAY)

def test_encode_params():
    # EncodeParams default constructor
    encode_params = nvimgcodec.EncodeParams()
    assert (encode_params.quality_type == nvimgcodec.QualityType.DEFAULT)
    assert (encode_params.quality_value == 0)
    assert (encode_params.color_spec == nvimgcodec.UNCHANGED)
    assert (encode_params.chroma_subsampling is None)  # Defaults to None, inferred from image at encode time

    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == False)
    assert (jpeg_params.optimized_huffman == False)

    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.code_block_size == (64, 64))
    assert (jpeg2k_params.num_resolutions == 6)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_params.prog_order == nvimgcodec.RPCL)
    assert (jpeg2k_params.ht == False)

    encode_params.quality_type = nvimgcodec.QualityType.QUALITY
    assert (encode_params.quality_type == nvimgcodec.QualityType.QUALITY)

    encode_params.quality_value = 45
    assert (encode_params.quality_value == 45)

    encode_params.color_spec = nvimgcodec.SYCC
    assert (encode_params.color_spec == nvimgcodec.SYCC)

    encode_params.color_spec = nvimgcodec.GRAY
    assert (encode_params.color_spec == nvimgcodec.GRAY)

    encode_params.color_spec = nvimgcodec.UNCHANGED
    assert (encode_params.color_spec == nvimgcodec.UNCHANGED)

    encode_params.color_spec = nvimgcodec.SRGB
    assert (encode_params.color_spec == nvimgcodec.SRGB)

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

    encode_params.jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams((786, 659), 897, nvimgcodec.J2K, nvimgcodec.PCRL, 1, True)
    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.code_block_size == (786, 659))
    assert (jpeg2k_params.num_resolutions == 897)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.J2K)
    assert (jpeg2k_params.prog_order == nvimgcodec.PCRL)
    assert (jpeg2k_params.ht == True)

    # EncodeParams constructor with parameters
    encode_params = nvimgcodec.EncodeParams(nvimgcodec.QualityType.PSNR, 22.5, nvimgcodec.GRAY, nvimgcodec.CSS_410)
    assert (encode_params.quality_type == nvimgcodec.QualityType.PSNR)
    assert (encode_params.quality_value == 22.5)
    assert (encode_params.color_spec == nvimgcodec.GRAY)
    assert (encode_params.chroma_subsampling == nvimgcodec.CSS_410)

    # jpeg_encode_params is optional, if not, it is default value.
    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == False)
    assert (jpeg_params.optimized_huffman == False)

    # jpeg2k_encode_params is optional, if not, it is default value.
    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.code_block_size == (64, 64))
    assert (jpeg2k_params.num_resolutions == 6)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_params.prog_order == nvimgcodec.RPCL)
    assert (jpeg2k_params.ht == False)

    jpeg_params = nvimgcodec.JpegEncodeParams(True, False)
    jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams((786, 659), 897, nvimgcodec.J2K, nvimgcodec.PCRL, 1, True)
    encode_params = nvimgcodec.EncodeParams(nvimgcodec.QualityType.QUALITY, 78, nvimgcodec.GRAY, nvimgcodec.CSS_410, jpeg_params, jpeg2k_params)

    # jpeg_encode_params is optional, if yes, it is set value in construtor.
    jpeg_params = encode_params.jpeg_params
    assert (jpeg_params.progressive == True)
    assert (jpeg_params.optimized_huffman == False)

    # jpeg2k_encode_params is optional, if yes, it is set value in construtor.
    jpeg2k_params = encode_params.jpeg2k_params
    assert (jpeg2k_params.code_block_size == (786, 659))
    assert (jpeg2k_params.num_resolutions == 897)
    assert (jpeg2k_params.bitstream_type == nvimgcodec.J2K)
    assert (jpeg2k_params.prog_order == nvimgcodec.PCRL)
    assert (jpeg2k_params.ht == True)

    encode_params = nvimgcodec.EncodeParams(chroma_subsampling=nvimgcodec.CSS_411, quality_value=0.25, color_spec=nvimgcodec.SYCC, quality_type=nvimgcodec.QualityType.SIZE_RATIO)
    assert (encode_params.quality_type == nvimgcodec.QualityType.SIZE_RATIO)
    assert (encode_params.quality_value == 0.25)
    assert (encode_params.color_spec == nvimgcodec.SYCC)
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
    assert (jpeg2k_encode_params.code_block_size == (64, 64))
    assert (jpeg2k_encode_params.num_resolutions == 6)
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.RPCL)
    assert (jpeg2k_encode_params.mct_mode == 0)
    assert (jpeg2k_encode_params.ht == False)

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

    jpeg2k_encode_params.mct_mode = 1
    assert (jpeg2k_encode_params.mct_mode == 1)

    jpeg2k_encode_params.ht = True
    assert (jpeg2k_encode_params.ht == True)

    # Jpeg2kEncodeParams constructor with parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams((128, 256), 73, nvimgcodec.J2K, nvimgcodec.PCRL, 1, True)
    assert (jpeg2k_encode_params.code_block_size == (128, 256))
    assert (jpeg2k_encode_params.num_resolutions == 73)
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.J2K)
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.PCRL)
    assert (jpeg2k_encode_params.mct_mode == 1)
    assert (jpeg2k_encode_params.ht == True)

    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams(prog_order=nvimgcodec.CPRL, num_resolutions=76, code_block_size=(28, 86), mct_mode=1, ht=True, bitstream_type=nvimgcodec.JP2)
    assert (jpeg2k_encode_params.code_block_size == (28, 86))
    assert (jpeg2k_encode_params.num_resolutions == 76)
    assert (jpeg2k_encode_params.mct_mode == 1)
    assert (jpeg2k_encode_params.bitstream_type == nvimgcodec.JP2)
    assert (jpeg2k_encode_params.prog_order == nvimgcodec.CPRL)
    assert (jpeg2k_encode_params.ht == True)

    # setting just one parameter should leave others with default values
    jpeg2k_encode_params_default = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_encode_params_default.num_resolutions = 15
    jpeg2k_encode_params_custom = nvimgcodec.Jpeg2kEncodeParams(num_resolutions=15)

    assert (jpeg2k_encode_params_default.code_block_size == jpeg2k_encode_params_custom.code_block_size)
    assert (jpeg2k_encode_params_default.num_resolutions == jpeg2k_encode_params_custom.num_resolutions)
    assert (jpeg2k_encode_params_default.bitstream_type == jpeg2k_encode_params_custom.bitstream_type)
    assert (jpeg2k_encode_params_default.prog_order == jpeg2k_encode_params_custom.prog_order)
    assert (jpeg2k_encode_params_default.ht == jpeg2k_encode_params_custom.ht)

def test_region_params():
    # Region default constructor
    region = nvimgcodec.Region()
    assert (region.ndim == 0)
    assert region.out_of_bounds_samples == [0] * 5

    # Region constructor with parameters
    # dimension = 1
    region = nvimgcodec.Region(end=(110,), start=(20,))
    assert (region.ndim == 1)
    assert (region.start == (20,))
    assert (region.end == (110,))

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

    region = nvimgcodec.Region(start=[87, 78], end=[145, 809])
    assert (region.ndim == 2)
    assert (region.start == (87, 78))
    assert (region.end == (145, 809))

    print(region)

    # dimension = 3
    region = nvimgcodec.Region(start=(13, 76, 67), end=(87, 165, 98))
    assert (region.ndim == 3)
    assert (region.start == (13, 76, 67))
    assert (region.end == (87, 165, 98))

    region = nvimgcodec.Region([13, 76, 67], [87, 165, 98])
    assert (region.ndim == 3)
    assert (region.start == (13, 76, 67))
    assert (region.end == (87, 165, 98))

    print(region)

    # dimension = 4
    region = nvimgcodec.Region((93, 6, 7, 8), (107, 15, 9, 10))
    assert (region.ndim == 4)
    assert (region.start == (93, 6, 7, 8))
    assert (region.end == (107, 15, 9, 10))

    region = nvimgcodec.Region(start=[3, 6, 7, 8], end=[7, 15, 9, 10])
    assert (region.ndim == 4)
    assert (region.start == (3, 6, 7, 8))
    assert (region.end == (7, 15, 9, 10))

    print(region)

    # dimension = 5
    region = nvimgcodec.Region(start=(93, 6, 7, 8, 9), end=(107, 15, 9, 10, 11))
    assert (region.ndim == 5)
    assert (region.start == (93, 6, 7, 8, 9))
    assert (region.end == (107, 15, 9, 10, 11))

    region = nvimgcodec.Region([3, 6, 7, 8, 9], [7, 15, 9, 10, 11])
    assert (region.ndim == 5)
    assert (region.start == (3, 6, 7, 8, 9))
    assert (region.end == (7, 15, 9, 10, 11))

    print(region)

    # verify that can create region with fill value from uint, int and float
    region = nvimgcodec.Region([10], [20], 255)
    assert all(val == 255 for val in region.out_of_bounds_samples)

    region = nvimgcodec.Region([10], [20], -10)
    assert all(val == -10 for val in region.out_of_bounds_samples)

    region = nvimgcodec.Region([10], [20], 0.5)
    assert all(val == 0.5 for val in region.out_of_bounds_samples)

    # check that can be created from 5 values
    vals = [1, 3, 10, 1213, 32423]
    region = nvimgcodec.Region([10], [20], vals)
    assert region.out_of_bounds_samples == vals

    # check that different types are supported
    vals = [1, -40, 2.5, 2, -5.5]
    region = nvimgcodec.Region([10], [20], vals)
    assert region.out_of_bounds_samples == vals

    # check that skipped values are shown as 0
    vals = [0, 125, 255]
    region = nvimgcodec.Region([10], [20], vals)
    assert region.out_of_bounds_samples[:len(vals)] == vals
    assert all(val == 0 for val in region.out_of_bounds_samples[len(vals):])

    # negative tests
    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region(start=(8192, 28672), end=(10, 11, 12))
    assert str(excinfo.value) == "Dimension mismatch"

    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region(start=(1, 2, 3, 4, 5, 6), end=(11, 12, 13, 14, 15, 16))
    assert str(excinfo.value) == "Too many dimensions: 6, at most 5 are allowed."

    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region(start=(100, 200), end=(10, 300))
    assert str(excinfo.value) == "Invalid dimension on index 0; start = 100, end = 10"

    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region(start=(100, 400), end=(120, 300))
    assert str(excinfo.value) == "Invalid dimension on index 1; start = 400, end = 300"

    with t.raises(Exception) as excinfo:
        region = nvimgcodec.Region([0, 100], [200, 300], [10, 10, 10, 10, 10, 10])
    assert str(excinfo.value) == "Too many fill values: 6, at most 5 are allowed."
