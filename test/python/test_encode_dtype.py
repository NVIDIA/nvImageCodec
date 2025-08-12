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
import numpy as np
import pytest as t
from nvidia import nvimgcodec
from utils import *
import nvjpeg_test_speedup

def impl_encode_single_jpeg2k_dtype_with_precision(img_path, shape, dtype, precision):
    input_img_path = os.path.join(img_dir_path, img_path)
    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder()

    type_precision = (dtype(0).nbytes * 8)

    # First decode to the original bitdepth
    reference = decoder.read(
        input_img_path, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))
    assert reference.shape == shape
    assert reference.dtype == dtype
    img_precision = reference.precision or type_precision
    assert img_precision == precision

    # Encoded unchanged
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded = encoder.encode(reference, "jpeg2k", params=encode_params)
    
    # Decode and verify
    tested = decoder.decode(encoded, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))
    assert tested.shape == shape
    assert tested.dtype == dtype
    img_precision = tested.precision or type_precision
    assert img_precision == precision
    
    atol = 1 if precision != 0 and precision != type_precision else 0
    np.testing.assert_allclose(reference.cpu(), tested.cpu(), atol=atol)


@t.mark.skipif(not is_nvjpeg2k_supported(), reason="nvjpeg2k encoder not yet supported on aarch64")
def test_encode_single_jpeg2k_16bit():
    impl_encode_single_jpeg2k_dtype_with_precision(
        "jpeg2k/cat-111793_640-16bit-gray.jp2", (426, 640, 1), np.uint16, 16)



