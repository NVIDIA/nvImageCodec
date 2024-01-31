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
import cupy as cp
import pytest as t
from nvidia import nvimgcodec

@t.mark.parametrize("src_cuda_stream", [cp.cuda.Stream.null, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("dst_cuda_stream", [cp.cuda.Stream.null, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("shape,dtype",
                    [
                        ((640, 480, 3), np.int8),
                        ((640, 480, 3), np.uint8),
                        ((640, 480, 3), np.int16),
                    ],
                    )
def test_dlpack_import_from_cupy(shape, dtype, src_cuda_stream, dst_cuda_stream):
    with src_cuda_stream:
        rng = np.random.default_rng()
        host_array = rng.integers(0, 128, shape, dtype)
        cp_img = cp.asarray(host_array)
    
        # Since nvimgcodec.as_image can understand both dlpack and cuda_array_interface,
        # and we don't know a priori which interfaces it'll use (cupy provides both),
        # let's create one object with only the dlpack interface.
        class DLPackObject:
            pass

        o = DLPackObject()
        o.__dlpack__ = cp_img.__dlpack__
        o.__dlpack_device__ = cp_img.__dlpack_device__

        nv_img = nvimgcodec.as_image(o, dst_cuda_stream.ptr)
        converted = np.array(nv_img.cpu())
        assert (host_array == converted).all()
 

@t.mark.parametrize("shape,dtype",
                    [
                        ((640, 480, 3), np.int8),
                        ((640, 480, 3), np.uint8),
                        ((640, 480, 3), np.int16),
                    ],
                    )
def test_dlpack_export_to_cupy(shape, dtype):
    rng = np.random.default_rng()
    host_array = rng.integers(0, 128, shape, dtype)
    dev_array = cp.asarray(host_array)

    img = nvimgcodec.as_image(dev_array)
    
    cap = img.to_dlpack()
    cp_img = cp.from_dlpack(cap)
    
    assert (host_array == cp.asnumpy(cp_img)).all()
