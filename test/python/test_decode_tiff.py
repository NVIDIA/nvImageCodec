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

img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
backends_cpu_only=[nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]

def test_decode_tiff_palette():
    path_regular = os.path.join(img_dir_path, "tiff/cat-300572_640.tiff")
    path_palette = os.path.join(img_dir_path, "tiff/cat-300572_640_palette.tiff")

    decoder = nvimgcodec.Decoder(backends=backends_cpu_only)
    img_regular = np.array(decoder.read(path_regular).cpu())
    img_palette = np.array(decoder.read(path_palette).cpu())

    delta = np.abs(img_regular.astype('float') - img_palette.astype('float')) / 256
    assert np.quantile(delta, 0.9) < 0.05, "Original and palette TIFF differ significantly"

def test_decode_tiff_uint16():
    path_u16 = os.path.join(img_dir_path, "tiff/uint16.tiff")
    path_u16_npy = os.path.join(img_dir_path, "tiff/uint16.npy")
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    dec = nvimgcodec.Decoder()
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
