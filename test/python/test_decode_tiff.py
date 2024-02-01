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

