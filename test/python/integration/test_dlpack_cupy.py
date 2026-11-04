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
import sys
import os
import pytest as t

import numpy as np
try:
    import cupy as cp
    _has_cupy = True
except:
    print("CuPy is not available, will skip related tests")
    _has_cupy = False

pytestmark = t.mark.skipif(not _has_cupy, reason="CuPy is not available")

# Use string IDs to avoid creating CUDA streams at collection time (which can segfault
# if CUDA is not fully initialized during pytest collection).
cuda_stream_ids = ["null", "non_blocking", "blocking"] if _has_cupy else []

def _make_stream(stream_id):
    if stream_id == "null":
        return cp.cuda.Stream.null
    elif stream_id == "non_blocking":
        return cp.cuda.Stream(non_blocking=True)
    else:
        return cp.cuda.Stream(non_blocking=False)


from nvidia import nvimgcodec
from utils import as_cuda_stream_arg

@t.mark.parametrize("src_stream_id", cuda_stream_ids)
@t.mark.parametrize("dst_stream_id", cuda_stream_ids)
@t.mark.parametrize("shape,dtype",
                    [
                        ((640, 480, 3), np.int8),
                        ((640, 480, 3), np.uint8),
                        ((640, 480, 3), np.int16),
                    ],
                    )
def test_dlpack_import_from_cupy(shape, dtype, src_stream_id, dst_stream_id):
    src_cuda_stream = _make_stream(src_stream_id)
    dst_cuda_stream = _make_stream(dst_stream_id)
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

        nv_img = nvimgcodec.as_image(o, as_cuda_stream_arg(dst_cuda_stream))
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

def test_dlpack_reuse():
    """Test that from_dlpack can be called multiple times on the same object"""
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../resources/jpeg/cat-1245673_640_444.jpg"))
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)])
    img = decoder.read(img_path)
    cp_img1 = cp.from_dlpack(img)
    cp_img2 = cp.from_dlpack(img)
    np.testing.assert_allclose(cp.asnumpy(cp_img1), cp.asnumpy(cp_img2))

def test_dlpack_capsule_single_use():
    """Test that a DLPack capsule can only be consumed once"""
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../resources/jpeg/cat-1245673_640_444.jpg"))
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)])
    img = decoder.read(img_path)
    cap = img.to_dlpack()
    # First consumption should work
    cp.from_dlpack(cap)
    # Second consumption of same capsule should fail
    with t.raises(Exception):
        cp.from_dlpack(cap)

def test_dlpack_lifetime():
    """Test that the exported tensor remains valid after the original image is deleted"""
    img_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../resources/jpeg/cat-1245673_640_444.jpg"))
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)])
    img = decoder.read(img_path)
    ref_cpu = np.array(img.cpu())
    shape = img.shape

    # Export the image to a CuPy array
    buffer = cp.from_dlpack(img)
    # Delete the original image
    del img
    # Verify that the exported tensor is still valid
    assert buffer.shape == shape
    np.testing.assert_array_equal(ref_cpu, cp.asnumpy(buffer))

def test_dlpack_import_and_reexport():
    """Test that we can import a DLPack tensor from CuPy and re-export it (like PyTorch/CuPy do)"""
    # Create a CuPy array
    original = cp.array([[[1, 2, 3], [4, 5, 6]], [[7, 8, 9], [10, 11, 12]]], dtype=cp.uint8)
    ref_cpu = cp.asnumpy(original)

    # Import into nvimgcodec via DLPack
    img = nvimgcodec.from_dlpack(original)
    np.testing.assert_array_equal(ref_cpu, np.array(img.cpu()))

    # Re-export back to DLPack (this should work like PyTorch/CuPy)
    buffer = cp.from_dlpack(img)
    np.testing.assert_array_equal(ref_cpu, cp.asnumpy(buffer))

    # Delete the nvimgcodec Image - both original and buffer should still be valid
    del img
    np.testing.assert_array_equal(cp.asnumpy(original), cp.asnumpy(buffer))


def test_dlpack_import_rejects_non_row_only_strides():
    """DLPack tensors with non-row-only strides must be rejected at import.

    The cross-device path in Image.cuda() performs a flat cudaMemcpyPeerAsync
    over height * row_stride bytes; a tensor with a non-trivial inner stride
    would silently copy non-image memory. The __cuda_array_interface__ path
    already enforces this via is_padding_correct; this is the DLPack analog.
    """
    # Allocate a contiguous HWC buffer with 6 channels, then take every other
    # channel via slicing. This yields a tensor with strides[2] = 2 elements
    # (instead of 1) -- exactly the "non-row-only stride" case the CAI guard
    # already rejects.
    base = cp.zeros((48, 64, 6), dtype=cp.uint8)
    strided = base[..., ::2]
    assert strided.strides[2] != strided.itemsize  # sanity: non-contiguous in channels

    capsule = strided.toDlpack()
    with t.raises(RuntimeError):
        nvimgcodec.as_image(capsule)
