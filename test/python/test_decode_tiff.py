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
from utils import is_nvcomp_supported

img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
backends_list=[
    [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
    [nvimgcodec.Backend(nvimgcodec.GPU_ONLY)],
    None, # use default backend
]

@t.mark.parametrize("backends", backends_list)
@t.mark.parametrize("full_precision", [True, False])
def test_decode_tiff_palette(backends, full_precision):
    if not is_nvcomp_supported():
        if (backends is not None and backends[0].backend_kind == nvimgcodec.GPU_ONLY):
            t.skip("nvCOMP is not supported on this platform")

    path_regular = os.path.join(img_dir_path, "tiff/cat-300572_640.tiff")
    path_palette = os.path.join(img_dir_path, "tiff/cat-300572_640_palette.tiff")

    decoder = nvimgcodec.Decoder(backends=backends)
    decode_params=nvimgcodec.DecodeParams(allow_any_depth=full_precision)
    img_regular = np.array(decoder.read(path_regular).cpu())
    img_palette = np.array(decoder.read(path_palette, params=decode_params).cpu())

    if full_precision:
        assert img_palette.dtype.itemsize == 2
        precision = 16
    else:
        assert img_palette.dtype.itemsize == 1
        precision = 8

    delta = np.abs(img_regular / 256 - img_palette / 2 ** precision)
    assert np.quantile(delta, 0.9) < 0.05, "Original and palette TIFF differ significantly"

@t.mark.parametrize(
    "other_image_path, other_image_precision",
    [
        ("tiff/cat-300572_640_uint16.tiff", 16),
        ("tiff/cat-300572_640_uint32.tiff", 32),
        ("tiff/cat-300572_640_fp32.tiff", 32),
    ]
)
def test_decode_tiff_cross_precision_validation(other_image_path, other_image_precision):
# only nvTIFF can decode 32 bit images
    if not is_nvcomp_supported() and other_image_precision == 32:
        t.skip("nvCOMP is not supported on this platform")

    path_regular = os.path.join(img_dir_path, "tiff/cat-300572_640.tiff")
    path_other = os.path.join(img_dir_path, other_image_path)

    decode_params=nvimgcodec.DecodeParams(allow_any_depth=True)
    decoder = nvimgcodec.Decoder()
    img_regular = np.array(decoder.read(path_regular).cpu())
    
    other_image = decoder.read(path_other, params=decode_params).cpu()
    assert other_image.precision == other_image_precision
    img_other = np.asarray(other_image.cpu())

    if "fp32" in other_image_path:
        delta = np.abs(img_regular / 256 - img_other)
    else:
        delta = np.abs(img_regular / 256 - img_other / 2 ** other_image_precision)
    assert np.max(delta) < 1.1 / 256, "Images differ significantly"

@t.mark.parametrize("backends", backends_list)
def test_decode_tiff_uint16_reference(backends):
    path_u16 = os.path.join(img_dir_path, "tiff/uint16.tiff")
    path_u16_npy = os.path.join(img_dir_path, "tiff/uint16.npy")
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    dec = nvimgcodec.Decoder(backends=backends)
    img_decoded = dec.read(path_u16, params)
    decoded = np.array(img_decoded.cpu())[: , :, 0]  # nvImageCodec gives an extra dimension
    reference = np.load(path_u16_npy)
    np.testing.assert_array_equal(decoded, reference)

# This tests used a crafted TIFF to provoke an OOM error,
# to check that we don't crash and gracefully raise an error
# See:
# https://gitlab.com/libtiff/libtiff/-/issues/621
# https://bugzilla.redhat.com/show_bug.cgi?id=2251326
# https://access.redhat.com/security/cve/CVE-2023-52355

# First the original test from https://gitlab.com/libtiff/libtiff/-/issues/621
def test_decode_tiff_too_many_planes():
    assert None == nvimgcodec.Decoder().read(
        os.path.join(img_dir_path, "tiff/error/too_many_planes.tiff"))

# Now reduce the number of planes to the maximum allowed (32) so that nvimagecodec
# doesn't throw an error early
def test_decode_tiff_oom():
    assert None == nvimgcodec.Decoder().read(
        os.path.join(img_dir_path, "tiff/error/oom.tiff"))
