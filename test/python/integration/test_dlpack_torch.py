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
import numpy as np
import pytest as t
from nvidia import nvimgcodec
try:
    import torch
    has_torch = torch.cuda.is_available()
except:
    print("Torch is not available, will skip related tests")
    has_torch = False

@t.mark.skipif(not has_torch, reason="Torch with CUDA is not available")
@t.mark.parametrize("shape,dtype", [
    ((640, 480, 3), np.int8),
    ((640, 480, 3), np.uint8),
    ((640, 480, 3), np.int16),
])
def test_dlpack_import_from_torch(shape, dtype):
    rng = np.random.default_rng()
    host_array = rng.integers(0, 128, shape, dtype)
    dev_array = torch.as_tensor(host_array, device="cuda")

    # Since nvimgcodec.as_image can understand both dlpack and cuda_array_interface,
    # and we don't know a priori which interfaces it'll use (torch provides both),
    # let's create one object with only the dlpack interface.
    class DLPackObject:
        pass

    o = DLPackObject()
    o.__dlpack__ = dev_array.__dlpack__
    o.__dlpack_device__ = dev_array.__dlpack_device__

    img = nvimgcodec.as_image(o)
    assert img.shape == shape
    assert img.dtype == dtype
    assert img.ndim == len(shape)
    
    assert (host_array == torch.from_dlpack(img).cpu().numpy()).all()

@t.mark.skipif(not has_torch, reason="Torch with CUDA is not available")
@t.mark.parametrize("shape,dtype",
                    [
                        ((640, 480, 3), np.int8),
                        ((640, 480, 3), np.uint8),
                        ((640, 480, 3), np.int16),
                    ],
)
def test_dlpack_export_to_torch(shape, dtype):
    rng = np.random.default_rng()
    host_array = rng.integers(0, 128, shape, dtype)
    dev_array = torch.as_tensor(host_array, device="cuda")

    img = nvimgcodec.as_image(dev_array)

    cap = img.to_dlpack()
    
    assert (host_array == torch.from_dlpack(cap).cpu().numpy()).all()
    
