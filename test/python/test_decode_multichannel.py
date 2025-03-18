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
from utils import is_nvjpeg2k_supported, img_dir_path

params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)
backends_gpu_only=[nvimgcodec.Backend(nvimgcodec.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)]
backends_cpu_only=[nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]

@t.mark.parametrize("path,shape,dtype,backends",
                    [
                        ("tiff/multichannel/cat-1245673_640_multichannel.tif", (423, 640, 6), np.uint8, backends_cpu_only), # nvTIFF support up to 4 channels
                        ("tiff/with_alpha_16bit/4ch16bpp.tiff", (497, 497, 4), np.uint16, backends_gpu_only),
                        ("tiff/with_alpha_16bit/4ch16bpp.tiff", (497, 497, 4), np.uint16, backends_cpu_only),
                        ("png/with_alpha_16bit/4ch16bpp.png", (497, 497, 4), np.uint16, None),
                        ("png/with_alpha/cat-111793_640-alpha.png", (426, 640, 4), np.uint8, None),
                        ("jpeg2k/with_alpha/cat-111793_640-alpha.jp2", (426, 640, 4), np.uint8, backends_cpu_only),
                        ("jpeg2k/with_alpha/cat-111793_640-alpha.jp2", (426, 640, 4), np.uint8, backends_gpu_only),
                        ("jpeg2k/with_alpha_16bit/4ch16bpp.jp2", (497, 497, 4), np.uint16, backends_cpu_only),
                        ("jpeg2k/with_alpha_16bit/4ch16bpp.jp2", (497, 497, 4), np.uint16, backends_gpu_only),
                        ("jpeg2k/tiled-cat-1046544_640_gray.jp2", (475, 640, 1), np.uint8, backends_cpu_only),
                        ("jpeg2k/tiled-cat-1046544_640_gray.jp2", (475, 640, 1), np.uint8, backends_gpu_only),
                    ],
                    )
def test_decode_single_multichannel(path, shape, dtype, backends):
    # TODO(janton): remove this when nvjpeg2k is released for arm
    if 'jpeg2k' in path and not is_nvjpeg2k_supported():
        t.skip("nvjpeg2k decoder not supported in this platform")

    input_img_path = os.path.join(img_dir_path, path)

    decoder = nvimgcodec.Decoder(backends=backends)

    # By default the decoder works on RGB and uint8, ignoring extra channels
    img_RGB = decoder.read(input_img_path)
    expected_shape_rgb = shape[:2] + (3,)
    assert img_RGB.shape == expected_shape_rgb
    assert img_RGB.dtype == np.uint8

    # Using UNCHANGED
    params_unchanged=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    img_unchanged = decoder.read(input_img_path, params=params_unchanged)
    assert img_unchanged.shape == shape
    assert img_unchanged.dtype == dtype

    # Gray
    params_gray=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY)
    img_gray = decoder.read(input_img_path, params=params_gray)
    expected_shape_gray = shape[:2] + (1,)
    assert img_gray.shape == expected_shape_gray
    assert img_gray.dtype == np.uint8
