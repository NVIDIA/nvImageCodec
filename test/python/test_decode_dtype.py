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
import numpy as np
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path, is_nvjpeg2k_supported
import nvjpeg_test_speedup

debug_output = False

def impl_decode_single_jpeg2k_dtype_with_precision(img_path, shape, dtype, precision):
    input_img_path = os.path.join(img_dir_path, img_path)
    decoder = nvimgcodec.Decoder()

    type_precision = (dtype(0).nbytes * 8)

    # First decode to the original bitdepth
    img_any_depth = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(allow_any_depth=True))
    assert img_any_depth.shape == shape
    assert img_any_depth.dtype == dtype
    img_precision = img_any_depth.precision or type_precision
    assert img_precision == precision
    data_any_depth = np.array(img_any_depth.cpu())
    # Scale it down, to compare later
    data_any_depth_converted_u8 = (data_any_depth * (255 / ((2**precision)-1))).astype(np.uint8)
    if debug_output:
        nvimgcodec.Encoder().write("a.bmp", data_any_depth_converted_u8)

    # Now decode without extra parameters, meaning we will decode to HWC RGB u8 always (scaling the 12 bit dynamic range to 8 bit dynamic range)
    img_u8 = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(allow_any_depth=False))
    assert img_u8.shape == shape
    assert img_u8.dtype == np.uint8
    assert img_u8.precision == 0
    data_u8 = np.array(img_u8.cpu())
    if debug_output:
        nvimgcodec.Encoder().write("b.bmp", data_u8)

    atol = 1 if precision != 0 and precision != type_precision else 0
    np.testing.assert_allclose(data_u8, data_any_depth_converted_u8, atol=atol)

def test_decode_single_jpeg2k_16bit():
    impl_decode_single_jpeg2k_dtype_with_precision("jpeg2k/cat-1046544_640-16bit.jp2", (475, 640, 3), np.uint16, 16)

def test_decode_single_jpeg2k_12bit():
    impl_decode_single_jpeg2k_dtype_with_precision("jpeg2k/cat-1245673_640-12bit.jp2", (423, 640, 3), np.uint16, 12)

@t.mark.skipif(not is_nvjpeg2k_supported(), reason="nvjpeg2k decoder not yet supported on aarch64")
def test_decode_single_jpeg2k_5bit():
    impl_decode_single_jpeg2k_dtype_with_precision("jpeg2k/cat-1245673_640-5bit.jp2", (423, 640, 3), np.uint8, 5)