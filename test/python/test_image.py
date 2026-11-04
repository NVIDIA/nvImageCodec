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
import gc
import os
import numpy as np
try:
    import cupy as cp
    img = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8) # Force to load necessary libriaries
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None so it can be referenced in parametrize decorators

try:
    from cuda.bindings import runtime as cudart
    CUDART_AVAILABLE = True
except Exception:
    CUDART_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    print("PyTorch is not available, will skip related tests")
    TORCH_AVAILABLE = False
    torch = None  # Define torch as None so it can be referenced in parametrize decorators

import pytest as t
from nvidia import nvimgcodec
import sys
from utils import *


def test_image_cpu_exports_to_host():
    decoder = nvimgcodec.Decoder()
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)

    ref_img = get_opencv_reference(input_img_path)
    test_image = decoder.read(input_img_path)
    
    host_image = test_image.cpu()

    compare_host_images([host_image], [ref_img])

def test_image_cpu_when_image_is_in_host_mem_returns_the_same_object():
    decoder = nvimgcodec.Decoder()
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)
    test_image = decoder.read(input_img_path)
    host_image = test_image.cpu()
    # Python 3.14+ changed refcount behavior, expecting 1 less than previous versions
    expected_initial = 1 if sys.version_info >= (3, 14) else 2
    assert (sys.getrefcount(host_image) == expected_initial)

    host_image_2 = host_image.cpu()

    assert (sys.getrefcount(host_image) == expected_initial + 1)
    assert (sys.getrefcount(host_image_2) == expected_initial + 1)

def test_image_cuda_exports_to_device():
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)
    ref_img = get_opencv_reference(input_img_path)
    host_img = nvimgcodec.as_image(ref_img)

    device_img = host_img.cuda()

    compare_device_with_host_images([device_img], [ref_img])
    

def test_image_cuda_when_image_is_in_device_mem_returns_the_same_object():
    decoder = nvimgcodec.Decoder()
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)
    device_img = decoder.read(input_img_path)
    # Python 3.14+ changed refcount behavior, expecting 1 less than previous versions
    expected_initial = 1 if sys.version_info >= (3, 14) else 2
    assert (sys.getrefcount(device_img) == expected_initial)

    device_img_2 = device_img.cuda()

    assert (sys.getrefcount(device_img) == expected_initial + 1)
    assert (sys.getrefcount(device_img_2) == expected_initial + 1)


@t.mark.parametrize(
    "input_img_file, shape",
    [
        ("bmp/cat-111793_640.bmp", (426, 640, 3)),
        ("jpeg/padlock-406986_640_410.jpg", (426, 640, 3)),
        ("jpeg2k/tiled-cat-1046544_640.jp2", (475, 640, 3))
    ]
)
def test_array_interface_export(input_img_file, shape):
    input_img_path = os.path.join(img_dir_path, input_img_file)

    ref_img = get_opencv_reference(input_img_path)
    decoder = nvimgcodec.Decoder()
    device_img = decoder.read(input_img_path)
    
    host_img = device_img.cpu()
    array_interface = host_img.__array_interface__

    assert(array_interface['strides'] == None)
    assert(array_interface['shape'] == ref_img.__array_interface__['shape'])
    assert(array_interface['typestr'] == ref_img.__array_interface__['typestr'])
    assert(host_img.ndim == 3)
    assert(host_img.dtype == ref_img.dtype)
    assert(host_img.shape == ref_img.shape)
    assert(host_img.strides == ref_img.strides)
    
    compare_host_images([host_img], [ref_img])


@t.mark.parametrize(
    "input_img_file",
    [
        "bmp/cat-111793_640.bmp",
        "jpeg/padlock-406986_640_410.jpg", 
        "jpeg2k/tiled-cat-1046544_640.jp2",
    ]
)
def test_array_interface_import(input_img_file):
    input_img_path = os.path.join(img_dir_path, input_img_file)

    ref_img = get_opencv_reference(input_img_path)
    host_img = nvimgcodec.as_image(ref_img)
    
    assert (host_img.__array_interface__['strides'] == ref_img.__array_interface__['strides'])
    assert (host_img.__array_interface__['shape'] == ref_img.__array_interface__['shape'])
    assert (host_img.__array_interface__['typestr'] == ref_img.__array_interface__['typestr'])
      
    compare_host_images([host_img], [ref_img])

def test_image_buffer_kind():
    input_img_path = os.path.join(
        img_dir_path, "jpeg/padlock-406986_640_410.jpg")

    ref_img = get_opencv_reference(input_img_path)
    host_img = nvimgcodec.as_image(ref_img)
    assert (host_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST)
    
    device_img = host_img.cuda()
    assert (device_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE)


@t.mark.parametrize(
    "shape, sample_format",
    [
        # Contiguous interleaved (HWC): single-plane whole-buffer fast path.
        ((16, 24, 3), nvimgcodec.SampleFormat.I_RGB),
        # Contiguous planar (CHW): multi-plane whole-buffer fast path - the case
        # the single cudaMemcpy must cover instead of one copy per plane.
        ((3, 16, 24), nvimgcodec.SampleFormat.P_RGB),
    ],
)
def test_image_cpu_cuda_roundtrip_contiguous(shape, sample_format):
    """A fully-contiguous buffer round-trips host->device->host unchanged via
    the whole-buffer single-copy fast path (GetBufferSize == GetImageSize)."""
    arr = np.random.randint(0, 255, shape, np.uint8)
    host_img = nvimgcodec.as_image(arr, sample_format=sample_format)

    device_img = host_img.cuda()  # H2D
    assert device_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    back = np.asarray(device_img.cpu())  # D2H
    np.testing.assert_array_equal(back, arr)


def test_image_cpu_cuda_roundtrip_row_padded():
    """A row-padded source has padding (GetBufferSize != GetImageSize), so the
    transfer must take the per-plane cudaMemcpy2DAsync fallback, not the
    whole-buffer fast path, and still round-trip its logical content."""
    full = np.random.randint(0, 255, (16, 24, 3), np.uint8)
    padded = full[:, :12]  # view with row padding: shape (16, 12, 3)
    assert padded.strides == (24 * 3, 3, 1)

    host_img = nvimgcodec.as_image(padded)
    device_img = host_img.cuda()         # H2D, padded source -> fallback path
    back = np.asarray(device_img.cpu())  # D2H, packed device source -> fast path
    np.testing.assert_array_equal(back, padded)


def test_image_device_id_and_cuda_stream_on_host():
    """Host image should report ``device_id`` and ``cuda_stream`` as ``None``
    rather than leaking internal sentinel values."""
    ref_img = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref_img)
    assert host_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert host_img.device_id is None
    assert host_img.cuda_stream is None


def test_image_device_id_and_cuda_stream_after_cpu():
    """A device image moved to CPU via ``.cpu()`` should also report
    ``device_id`` and ``cuda_stream`` as ``None``."""
    decoder = nvimgcodec.Decoder()
    input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    device_img = decoder.read(input_img_path)
    assert device_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    assert isinstance(device_img.device_id, int)

    host_img = device_img.cpu()
    assert host_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert host_img.device_id is None
    assert host_img.cuda_stream is None


def test_image_array_interface_import_two_dimensions_assume_one_channel():
    ref_img = np.zeros((2, 3))
    host_img = nvimgcodec.as_image(ref_img)
    assert(host_img.shape[0] == 2)
    assert(host_img.shape[1] == 3)
    assert(host_img.shape[2] == 1)
    assert(host_img.sample_format == nvimgcodec.SampleFormat.I_Y)
    
def test_image_array_interface_import_three_dimensions():
    ref_img = np.zeros((1, 2, 3))
    host_img = nvimgcodec.as_image(ref_img)
    assert(host_img.shape[0] == 1)
    assert(host_img.shape[1] == 2)
    assert(host_img.shape[2] == 3)
    assert(host_img.sample_format == nvimgcodec.SampleFormat.I_RGB)

def test_image_array_interface_import_less_than_two_dimensions_throws():
    ref_img = np.zeros((5))
    with t.raises(Exception) as excinfo:
        host_img = nvimgcodec.as_image(ref_img)
    assert (str(excinfo.value) == "Unexpected number of dimensions. At least 2 dimensions are expected.")
    
def test_image_array_interface_import_more_than_three_dimensions_throws():
    ref_img = np.zeros((1, 2, 3 ,4))
    with t.raises(Exception) as excinfo:
        host_img = nvimgcodec.as_image(ref_img)
    assert (str(excinfo.value) == "Unexpected number of dimensions. At most 3 dimensions are expected.")

def test_image_array_interface_import_not_accepted_number_of_channels_to_be_zero_throws():
    ref_img = np.zeros((1, 2, 0))
    with t.raises(Exception) as excinfo:
        host_img = nvimgcodec.as_image(ref_img)
    assert (str(excinfo.value) == "Unexpected number of channels. At least 1 channel is expected.")


def test_image_array_interface_import_non_c_style_contiguous_array_throws():
    img_rgba = np.random.randint(0, 255, (10, 10, 4), np.uint8)  # Create dummy image

    # Drop the last channel (alpha)
    # numpy is just creating  view on the same memory by changing strides and shape values,  
    # so ___array_interface__  is like {'data': (1043196352, False), 'strides': (40, 4, 1), 'descr': [('', '|u1')], 'typestr': '|u1', 'shape': (10, 10, 3), 'version': 3} 
    # strides are not like in "packed" array anymore so it is not C-style contiguous array 
    # For C-style contiguous array strides should be either None (https://numpy.org/doc/2.1/reference/arrays.interface.html) or in this case(30, 3, 1)
    img_rgb = img_rgba[..., :3] 

    with t.raises(Exception) as excinfo:
        host_img = nvimgcodec.as_image(img_rgb)
    assert (str(excinfo.value) == "Unexpected array style. Padding is only allowed for rows. Other dimensions should have contiguous strides.")

def test_image_array_interface_import_image_with_padding_works():
    img_rgb = np.random.randint(0, 255, (10, 10, 3), np.uint8)  # Create dummy image

    # Drop some of the columns, which can be interpreted as using padding for rows
    img_rgb = img_rgb[:, :5]
    assert img_rgb.shape == (10, 5, 3)
    assert img_rgb.strides == (30, 3, 1)

    try:
        host_img = nvimgcodec.as_image(img_rgb)
        assert host_img.shape == (10, 5, 3)
        assert host_img.strides == (30, 3, 1)
    except Exception as e:
        assert False, f"An exception ({type(e).__name__}) was raised: {e} where it should not"

def test_image_array_interface_import_when_strides_none_does_not_throw():
    img_rgba = np.random.randint(0, 255, (10, 10, 4), np.uint8) # Create dummy image
    img_rgb = img_rgba[..., :3] # creating just view with non-contiguous array
    img_rgb = np.array(img_rgb) # it makes copy and packs array
    
    assert (img_rgb.__array_interface__["strides"] == None)
    try:
        host_img = nvimgcodec.as_image(img_rgb)
    except Exception as e:
        assert False, f"An exception ({type(e).__name__}) was raised: {e} where it should not"

def test_image_size_and_capacity_with_external_host_buffer():
    """Test that size and capacity properties work for external host buffer Images."""
    
    # Create Image with external host buffer (from numpy array)
    ref_img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    external_img = nvimgcodec.as_image(ref_img)
    
    assert external_img.size == ref_img.nbytes
    assert external_img.capacity == ref_img.nbytes
    assert external_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_image_size_and_capacity_with_external_device_buffer():
    """Test that size and capacity properties work for external device buffer Images."""
    
    # Create Image with external device buffer (from cupy array)
    ref_img = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8)
    external_img = nvimgcodec.as_image(ref_img)
    
    assert external_img.size == ref_img.nbytes
    assert external_img.capacity == ref_img.nbytes
    assert external_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE

def test_image_size_and_capacity_with_internal_buffer():
    """Test that size and capacity properties work for internal buffer Images."""
    decoder = nvimgcodec.Decoder()

    # Create Image with internal buffer (via decoder)
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)
    code_stream = nvimgcodec.CodeStream(input_img_path)
    internal_img = decoder.decode(code_stream)
    
    # Both should have valid size and capacity properties
    assert internal_img.size == internal_img.shape[0] * internal_img.shape[1] * internal_img.shape[2] * internal_img.dtype.itemsize
    assert internal_img.capacity >= internal_img.size

def image_conversion_impl(image):
    if not CUPY_AVAILABLE:
        t.skip("cupy is not available")
    device_image = image.cuda()
    host_image = image.cpu()

    assert host_image.shape == image.shape
    assert device_image.shape == image.shape

    assert host_image.size == image.size
    assert device_image.size == image.size

    assert host_image.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert device_image.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE

    if image.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE:
        numpy_reference = cp.asnumpy(image)
    else:
        numpy_reference = np.asarray(image)

    np.testing.assert_allclose(host_image, numpy_reference)
    np.testing.assert_allclose(cp.asnumpy(device_image), numpy_reference)

BUFFER_CREATE_LIST = [np.random.randint]
if CUPY_AVAILABLE:
    BUFFER_CREATE_LIST.append(cp.random.randint)

@t.mark.parametrize("buffer_create", BUFFER_CREATE_LIST)
def test_image_conversion_from_external_source(buffer_create):
    buffer = buffer_create(0, 255, (250, 331, 2), dtype=np.uint8)
    image = nvimgcodec.as_image(buffer)
    image_conversion_impl(image)

@t.mark.parametrize("buffer_create", BUFFER_CREATE_LIST)
def test_image_conversion_from_external_source_slice(buffer_create):
    buffer = buffer_create(0, 255, (250, 331, 2), dtype=np.uint8)
    slice = buffer[3:-5,4:-7]
    image = nvimgcodec.as_image(slice)
    image_conversion_impl(image)

@t.mark.parametrize("backends", [
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)], # tests image created in host memory
    [nvimgcodec.BackendKind.GPU_ONLY, nvimgcodec.BackendKind.HYBRID_CPU_GPU] # test image created in device memory
])
def test_image_conversion_from_internal_source(backends):
    image_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_420.jpg")
    decoder = nvimgcodec.Decoder(backends=backends)
    image = decoder.decode(image_path)
    image_conversion_impl(image)

@t.mark.parametrize(
    "array_module, expected_buffer_kind",
    [
        (np, nvimgcodec.ImageBufferKind.STRIDED_HOST),
        t.param(cp, nvimgcodec.ImageBufferKind.STRIDED_DEVICE, 
                marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
@t.mark.parametrize(
    "shape, expected_sample_format, expected_color_spec",
    [
        ((10, 10, 1), nvimgcodec.SampleFormat.I_Y, nvimgcodec.ColorSpec.GRAY),
        ((10, 10, 2), nvimgcodec.SampleFormat.I_YA, nvimgcodec.ColorSpec.GRAY),
        ((10, 10, 3), nvimgcodec.SampleFormat.I_RGB, nvimgcodec.ColorSpec.SRGB),
        ((10, 10, 4), nvimgcodec.SampleFormat.I_RGBA, nvimgcodec.ColorSpec.SRGB),
        ((10, 10, 5), nvimgcodec.SampleFormat.UNKNOWN, nvimgcodec.ColorSpec.UNKNOWN),
    ]
)
def test_as_image_default_sample_format_and_color_spec(array_module, expected_buffer_kind, shape, 
                                                        expected_sample_format, expected_color_spec):
    """Test that as_image infers correct default sample_format and color_spec based on number of channels."""
    ref_img = array_module.random.randint(0, 255, shape, dtype=array_module.uint8)
    img = nvimgcodec.as_image(ref_img)
    
    assert img.sample_format == expected_sample_format, \
        f"Expected sample_format {expected_sample_format} for shape {shape}, got {img.sample_format}"
    assert img.color_spec == expected_color_spec, \
        f"Expected color_spec {expected_color_spec} for shape {shape}, got {img.color_spec}"
    assert img.buffer_kind == expected_buffer_kind


@t.mark.parametrize(
    "array_module",
    [
        np,
        t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
def test_as_image_override_sample_format(array_module):
    """Test that explicitly setting sample_format overrides the default."""
    ref_img = array_module.random.randint(0, 255, (10, 10, 3), dtype=array_module.uint8)
    
    # Default should be I_RGB
    img_default = nvimgcodec.as_image(ref_img)
    assert img_default.sample_format == nvimgcodec.SampleFormat.I_RGB
    
    # Override to I_BGR
    img_bgr = nvimgcodec.as_image(ref_img, sample_format=nvimgcodec.SampleFormat.I_BGR)
    assert img_bgr.sample_format == nvimgcodec.SampleFormat.I_BGR
    
    # Override to I_YUV
    img_yuv = nvimgcodec.as_image(ref_img, sample_format=nvimgcodec.SampleFormat.I_YUV)
    assert img_yuv.sample_format == nvimgcodec.SampleFormat.I_YUV


@t.mark.parametrize(
    "array_module",
    [
        np,
        t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
def test_as_image_override_color_spec(array_module):
    """Test that explicitly setting color_spec overrides the default."""
    ref_img = array_module.random.randint(0, 255, (10, 10, 3), dtype=array_module.uint8)
    
    # Default should be SRGB
    img_default = nvimgcodec.as_image(ref_img)
    assert img_default.color_spec == nvimgcodec.ColorSpec.SRGB
    
    # Override to SYCC
    img_sycc = nvimgcodec.as_image(ref_img, color_spec=nvimgcodec.ColorSpec.SYCC)
    assert img_sycc.color_spec == nvimgcodec.ColorSpec.SYCC
    
    # Override to UNKNOWN
    img_unknown = nvimgcodec.as_image(ref_img, color_spec=nvimgcodec.ColorSpec.UNKNOWN)
    assert img_unknown.color_spec == nvimgcodec.ColorSpec.UNKNOWN

@t.mark.parametrize(
    "array_module",
    [
        np,
        t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
def test_as_images_default_sample_format_and_color_spec(array_module):
    """Test that as_images infers correct default sample_format and color_spec for all images."""
    # Create images with different channel counts
    img_1ch = array_module.random.randint(0, 255, (10, 10, 1), dtype=array_module.uint8)
    img_3ch = array_module.random.randint(0, 255, (10, 10, 3), dtype=array_module.uint8)
    img_4ch = array_module.random.randint(0, 255, (10, 10, 4), dtype=array_module.uint8)
    
    images = nvimgcodec.as_images([img_1ch, img_3ch, img_4ch])
    
    # Check 1-channel image
    assert images[0].sample_format == nvimgcodec.SampleFormat.I_Y
    assert images[0].color_spec == nvimgcodec.ColorSpec.GRAY
    
    # Check 3-channel image
    assert images[1].sample_format == nvimgcodec.SampleFormat.I_RGB
    assert images[1].color_spec == nvimgcodec.ColorSpec.SRGB
    
    # Check 4-channel image
    assert images[2].sample_format == nvimgcodec.SampleFormat.I_RGBA
    assert images[2].color_spec == nvimgcodec.ColorSpec.SRGB


@t.mark.parametrize(
    "array_module",
    [
        np,
        t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
def test_as_images_override_sample_format_and_color_spec(array_module):
    """Test that as_images applies the same sample_format and color_spec override to all images."""
    img1 = array_module.random.randint(0, 255, (10, 10, 3), dtype=array_module.uint8)
    img2 = array_module.random.randint(0, 255, (15, 15, 3), dtype=array_module.uint8)
    img3 = array_module.random.randint(0, 255, (20, 20, 3), dtype=array_module.uint8)
    
    # Override both parameters for all images
    images = nvimgcodec.as_images([img1, img2, img3],
                                  sample_format=nvimgcodec.SampleFormat.I_BGR,
                                  color_spec=nvimgcodec.ColorSpec.SYCC)
    
    # All images should have the overridden values
    for img in images:
        assert img.sample_format == nvimgcodec.SampleFormat.I_BGR
        assert img.color_spec == nvimgcodec.ColorSpec.SYCC


@t.mark.parametrize(
    "num_channels, invalid_sample_format, min_required_channels",
    [
        (3, nvimgcodec.SampleFormat.I_RGBA, 4),  # I_RGBA needs at least 4 channels, but has 3
        (2, nvimgcodec.SampleFormat.I_RGB, 3),   # I_RGB needs at least 3 channels, but has 2
        (1, nvimgcodec.SampleFormat.I_RGBA, 4),  # I_RGBA needs at least 4 channels, but has 1
        (2, nvimgcodec.SampleFormat.I_CMYK, 4),  # I_CMYK needs at least 4 channels, but has 2
    ]
)
def test_as_image_invalid_sample_format_for_channels_throws(num_channels, invalid_sample_format, min_required_channels):
    """Interleaved sample_format with too few channels in an HWC array raises
    up-front with the same shape-aware error message as the planar branch."""
    shape = (10, 10, num_channels) if num_channels > 1 else (10, 10)
    ref_img = np.random.randint(0, 255, shape, dtype=np.uint8)

    with t.raises(ValueError, match="Interleaved sample_format") as excinfo:
        nvimgcodec.as_image(ref_img, sample_format=invalid_sample_format)

    assert f"(H, W, {min_required_channels})" in str(excinfo.value)
    assert f"got shape ({', '.join(str(s) for s in shape)})" in str(excinfo.value)


def test_as_image_sample_format_unknown_throws():
    """``sample_format=UNKNOWN`` is not a meaningful as_image argument - the
    caller must either pick a concrete format, use ``*_UNCHANGED`` as a
    layout hint, or omit the argument and accept the channel-count default."""
    arr = np.zeros((10, 10, 3), np.uint8)
    with t.raises(ValueError, match="sample_format=UNKNOWN"):
        nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.UNKNOWN)


def test_as_image_planar_sample_format_with_too_few_channels_throws():
    """A concrete planar sample_format paired with a shape that provides FEWER
    channels than the format requires is rejected up-front; we never silently
    fall back to HWC labelling."""
    # P_YA needs (>=2, H, W) - a 2-D array provides effectively 1 channel.
    ref_2d = np.random.randint(0, 255, (10, 10), dtype=np.uint8)
    with t.raises(ValueError, match="Planar sample_format P_YA"):
        nvimgcodec.as_image(ref_2d, sample_format=nvimgcodec.SampleFormat.P_YA)

    # P_RGB needs (>=3, H, W) - this CHW array only has 2 planes.
    ref_2ch_chw = np.random.randint(0, 255, (2, 10, 10), dtype=np.uint8)
    with t.raises(ValueError, match="Planar sample_format P_RGB"):
        nvimgcodec.as_image(ref_2ch_chw, sample_format=nvimgcodec.SampleFormat.P_RGB)


@t.mark.parametrize(
    "array_module",
    [
        np,
        t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
@t.mark.parametrize("sample_format, match", [
    (nvimgcodec.SampleFormat.I_RGB, "Interleaved sample_format I_RGB"),
    (nvimgcodec.SampleFormat.P_RGB, "Planar sample_format P_RGB"),
])
def test_as_image_2d_cannot_be_rgb(array_module, sample_format, match):
    """A 2-D (H, W) array is single-channel, so an RGB sample_format is rejected
    for both interleaved and planar layouts."""
    arr = array_module.random.randint(0, 255, (10, 10), dtype=array_module.uint8)
    with t.raises(ValueError, match=match):
        nvimgcodec.as_image(arr, sample_format=sample_format)

    # P_RGBA needs (>=4, H, W) - this CHW array only has 3 planes.
    ref_3ch_chw = np.random.randint(0, 255, (3, 10, 10), dtype=np.uint8)
    with t.raises(ValueError, match="Planar sample_format P_RGBA"):
        nvimgcodec.as_image(ref_3ch_chw, sample_format=nvimgcodec.SampleFormat.P_RGBA)


def test_as_image_planar_sample_format_extra_planes_are_kept():
    """A 3-D CHW array with MORE planes than the planar format requires is
    accepted - the extra planes are kept on the Image and the sample_format
    label points at the leading planes (parallel to the HWC behaviour where
    extra channels on the last axis are tolerated)."""
    # (4, H, W) + P_RGB: extra plane is preserved.
    ref = np.random.randint(0, 255, (4, 10, 10), dtype=np.uint8)
    img = nvimgcodec.as_image(ref, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGB
    assert img.shape == (4, 10, 10)

    # (5, H, W) + P_Y: 4 extra planes preserved.
    ref5 = np.random.randint(0, 255, (5, 10, 10), dtype=np.uint8)
    img5 = nvimgcodec.as_image(ref5, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img5.sample_format == nvimgcodec.SampleFormat.P_Y
    assert img5.shape == (5, 10, 10)


def test_as_image_more_channels_than_required_for_interleaved_is_allowed():
    """For interleaved sample_formats, an HWC array with strictly more channels
    than the format names is accepted (e.g. an RGBA buffer labelled as I_RGB)."""
    # 4-channel HWC array labelled I_RGB (needs >=3 channels).
    ref_img_4ch = np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
    img_rgb = nvimgcodec.as_image(ref_img_4ch, sample_format=nvimgcodec.SampleFormat.I_RGB)
    assert img_rgb.sample_format == nvimgcodec.SampleFormat.I_RGB

    # 5-channel HWC array labelled I_RGBA (needs >=4).
    ref_img_5ch = np.random.randint(0, 255, (10, 10, 5), dtype=np.uint8)
    img_rgba_5ch = nvimgcodec.as_image(ref_img_5ch, sample_format=nvimgcodec.SampleFormat.I_RGBA)
    assert img_rgba_5ch.sample_format == nvimgcodec.SampleFormat.I_RGBA


def test_as_image_flexible_sample_formats_accept_any_channels():
    """``I_UNCHANGED`` is HWC label-only - it accepts any number of channels
    on the HWC last axis. ``P_UNCHANGED`` is a layout hint that drives CHW
    interpretation but doesn't validate the channel count against any arity.
    (``UNKNOWN`` is rejected outright by ``as_image``; see
    ``test_as_image_sample_format_unknown_throws``.)"""
    # I_UNCHANGED + 3-channel HWC -> shape preserved as HWC.
    ref_img_3ch = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    img_i_unchanged = nvimgcodec.as_image(ref_img_3ch, sample_format=nvimgcodec.SampleFormat.I_UNCHANGED)
    assert img_i_unchanged.sample_format == nvimgcodec.SampleFormat.I_UNCHANGED
    assert img_i_unchanged.shape == (10, 10, 3)

    # I_UNCHANGED + 5-channel HWC -> still HWC, label kept.
    ref_img_5ch = np.random.randint(0, 255, (10, 10, 5), dtype=np.uint8)
    img_5ch_i_unchanged = nvimgcodec.as_image(
        ref_img_5ch, sample_format=nvimgcodec.SampleFormat.I_UNCHANGED)
    assert img_5ch_i_unchanged.sample_format == nvimgcodec.SampleFormat.I_UNCHANGED
    assert img_5ch_i_unchanged.shape == (10, 10, 5)

    # P_UNCHANGED is layout-aware: 3-D input is read as CHW, the channel
    # count is whatever the leading axis says (no arity validation).
    ref_img_3hw = np.random.randint(0, 255, (3, 10, 12), dtype=np.uint8)
    img_p_unchanged = nvimgcodec.as_image(ref_img_3hw, sample_format=nvimgcodec.SampleFormat.P_UNCHANGED)
    assert img_p_unchanged.sample_format == nvimgcodec.SampleFormat.P_UNCHANGED
    assert img_p_unchanged.shape == (3, 10, 12)

    # P_UNCHANGED accepts any plane count.
    ref_img_5hw = np.random.randint(0, 255, (5, 10, 12), dtype=np.uint8)
    img_5p_unchanged = nvimgcodec.as_image(ref_img_5hw, sample_format=nvimgcodec.SampleFormat.P_UNCHANGED)
    assert img_5p_unchanged.sample_format == nvimgcodec.SampleFormat.P_UNCHANGED
    assert img_5p_unchanged.shape == (5, 10, 12)

    # P_UNCHANGED on a 2-D array -> single-plane CHW.
    ref_img_2d = np.random.randint(0, 255, (10, 12), dtype=np.uint8)
    img_2d_p_unchanged = nvimgcodec.as_image(ref_img_2d, sample_format=nvimgcodec.SampleFormat.P_UNCHANGED)
    assert img_2d_p_unchanged.sample_format == nvimgcodec.SampleFormat.P_UNCHANGED
    assert img_2d_p_unchanged.shape == (1, 10, 12)


@t.mark.parametrize(
    "shape, valid_sample_format",
    [
        # Interleaved (HWC) cases - shape matches the format's channel count.
        ((10, 10),    nvimgcodec.SampleFormat.I_Y),
        ((10, 10, 2), nvimgcodec.SampleFormat.I_YA),
        ((10, 10, 3), nvimgcodec.SampleFormat.I_RGB),
        ((10, 10, 3), nvimgcodec.SampleFormat.I_BGR),
        ((10, 10, 3), nvimgcodec.SampleFormat.I_YUV),
        ((10, 10, 4), nvimgcodec.SampleFormat.I_RGBA),
        ((10, 10, 4), nvimgcodec.SampleFormat.I_CMYK),
        ((10, 10, 4), nvimgcodec.SampleFormat.I_YCCK),
        # Planar (CHW) cases - shape[0] must equal the format's channel arity,
        # except for the single-channel P_Y which also accepts a 2-D shape.
        ((10, 10),    nvimgcodec.SampleFormat.P_Y),
        ((1, 10, 10), nvimgcodec.SampleFormat.P_Y),
        ((2, 10, 10), nvimgcodec.SampleFormat.P_YA),
        ((3, 10, 10), nvimgcodec.SampleFormat.P_RGB),
        ((4, 10, 10), nvimgcodec.SampleFormat.P_RGBA),
    ]
)
def test_as_image_valid_sample_format_for_channels_succeeds(shape, valid_sample_format):
    """Test that providing a compatible sample_format for a matching shape
    works for both layouts."""
    ref_img = np.random.randint(0, 255, shape, dtype=np.uint8)

    # Should not raise an exception.
    img = nvimgcodec.as_image(ref_img, sample_format=valid_sample_format)
    assert img.sample_format == valid_sample_format


def test_numpy_asarray_keeps_image_alive():
    """Test numpy arrays keep Image alive"""
    decoder = nvimgcodec.Decoder()
    input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    
    image = decoder.read(input_img_path).cpu()
    arr = np.asarray(image)
    
    assert arr.shape == (426, 640, 3) and arr.dtype == np.uint8
    assert 0 <= arr.min() <= arr.max() <= 255

    image2 = decoder.read(input_img_path).cpu()
    arr2 = np.asarray(image2)

    del image2
    gc.collect()

    np.testing.assert_array_equal(arr, arr2)


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_numpy_array_on_gpu_image_raises_helpful_error():
    """np.array(img) on a GPU image should raise RuntimeError with a .cpu() hint, not segfault."""
    gpu_arr = cp.random.randint(0, 255, (64, 64, 3), dtype=cp.uint8)
    img = nvimgcodec.as_image(gpu_arr)
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE

    with t.raises(RuntimeError, match=r"\.cpu\(\)"):
        np.array(img)


def test_numpy_array_on_cpu_image_works():
    """np.array(img) on a CPU image should work without calling .cpu() first."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = nvimgcodec.as_image(ref)
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST

    result = np.array(img)
    np.testing.assert_array_equal(result, ref)


def test_numpy_array_dtype_conversion():
    """np.array(img, dtype=...) should perform dtype conversion."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = nvimgcodec.as_image(ref)

    result_f32 = np.array(img, dtype=np.float32)
    assert result_f32.dtype == np.float32
    np.testing.assert_array_equal(result_f32, ref.astype(np.float32))

    result_u8 = np.array(img, dtype=np.uint8)
    assert result_u8.dtype == np.uint8
    np.testing.assert_array_equal(result_u8, ref)


def test_numpy_array_copy_true_returns_copy():
    """np.array(img, copy=True) should always return a new copy not sharing memory."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = nvimgcodec.as_image(ref)

    view = np.asarray(img)
    result = np.array(img, copy=True)
    np.testing.assert_array_equal(result, ref)
    assert not np.shares_memory(result, view)


def test_numpy_array_copy_false_returns_view():
    """np.array(img, copy=False) should return a view sharing memory with the image buffer."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    img = nvimgcodec.as_image(ref)

    view = np.asarray(img)
    result = np.array(img, copy=False)
    np.testing.assert_array_equal(result, ref)
    assert np.shares_memory(result, view)


# ---------------------------------------------------------------------------
# Image.cuda() same-device short-circuit
# ---------------------------------------------------------------------------

def test_image_cuda_same_device_returns_self():
    """img.cuda(device_id=N) when already on device N should return the same object.

    Covers both the explicit-device-id form and the default (no-args) form.
    """
    decoder = nvimgcodec.Decoder(device_id=0)
    img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    img_dev0 = decoder.read(img_path)
    assert img_dev0 is not None

    same = img_dev0.cuda(device_id=0)
    assert same is img_dev0

    same_default = img_dev0.cuda()
    assert same_default is img_dev0


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_image_cuda_same_device_ignores_cuda_stream():
    """When image is already on the target device, cuda_stream is ignored and self is returned."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    img_dev0 = decoder.read(nvimgcodec.CodeStream(img_path))

    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        same = img_dev0.cuda(device_id=0, cuda_stream=stream)
        assert same is img_dev0
    finally:
        cudart.cudaStreamDestroy(stream)


# ---------------------------------------------------------------------------
# Image.cuda_stream and Image.device_id properties (single-GPU cases)
# ---------------------------------------------------------------------------

def test_image_cuda_stream_property_default():
    """Device image created with default stream should have cuda_stream reported as an int."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img = decoder.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")))
    assert img.cuda_stream is not None
    assert isinstance(img.cuda_stream, int)


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_image_cuda_stream_property_explicit():
    """Device image created with an explicit stream should report that stream."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    stream_int = int(stream)
    try:
        device_img = host_img.cuda(device_id=0, cuda_stream=stream_int)
        # Compare as int: cudaStream_t equality with int is implementation-dependent;
        # casting to int keeps the assertion independent of cuda-bindings behavior.
        assert int(device_img.cuda_stream) == stream_int
    finally:
        cudart.cudaStreamDestroy(stream)


def test_image_device_id_property_default():
    """Decoded image on GPU 0 should report device_id == 0 as an int."""
    decoder = nvimgcodec.Decoder(device_id=0)
    img = decoder.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")))
    assert img.device_id is not None
    assert isinstance(img.device_id, int)
    assert img.device_id == 0


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_decoder_read_stream_lifetime_contract():
    """A decoded Image must not retain a caller-owned CUDA stream in its
    buffer deleter after Decoder.read returns.

    Keep the Decoder alive while destroying the stream first, then release the
    Image. This is the lifetime order that previously made stream-ordered
    output allocation call cudaFreeAsync with a destroyed stream.
    """
    decoder = nvimgcodec.Decoder(device_id=0)
    image_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess

    image = None
    try:
        image = decoder.read(nvimgcodec.CodeStream(image_path), cuda_stream=int(stream))
        assert image is not None
    finally:
        try:
            cudart.cudaStreamSynchronize(stream)
        finally:
            cudart.cudaStreamDestroy(stream)

    del image

    # The Decoder is intentionally still alive and remains usable.
    assert decoder.read(nvimgcodec.CodeStream(image_path)) is not None


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_decoder_reuse_image_stream_lifetime_contract():
    """The reusable-Image decode path must also detach its buffer deleter
    from the caller-owned CUDA stream.
    """
    decoder = nvimgcodec.Decoder(device_id=0)
    code_stream = nvimgcodec.CodeStream(
        os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg"))
    image = decoder.decode(code_stream)
    assert image is not None

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess

    reused_image = None
    try:
        reused_image = decoder.decode(code_stream, image=image, cuda_stream=int(stream))
        assert reused_image is image
    finally:
        try:
            cudart.cudaStreamSynchronize(stream)
        finally:
            cudart.cudaStreamDestroy(stream)

    del reused_image
    del image

    # The Decoder is intentionally still alive and remains usable.
    assert decoder.decode(code_stream) is not None


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_image_cuda_h2d_stream_lifetime_contract():
    """Regression: destroying the user's stream before the resulting Image is
    released must not terminate the process (host-to-device path).

    The library contract is scoped to the Image's buffer deallocation:
    after Image.cuda returns, Image destruction will not touch the stream.
    Other Image methods that enqueue work on the stream (e.g. Image.cpu) must
    still be called while the stream is alive -- they read from
    image_info.cuda_stream. This test exercises the H2D
    (STRIDED_HOST -> STRIDED_DEVICE) async branch with ``synchronize=False``,
    which is the exact hazard path the library-side fix addresses.

    The D2D (cross-device) counterpart lives in test_multigpu.py.
    """
    ref = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess

    device_img = host_img.cuda(device_id=0, cuda_stream=stream, synchronize=False)
    # Read back while `stream` is still alive (Image.cpu uses it internally).
    cudart.cudaStreamSynchronize(stream)
    np.testing.assert_array_equal(np.asarray(device_img.cpu()), ref)

    # Now destroy `stream` and let the Image drop. The buffer deleter was
    # detached from `stream` by Image.cuda, so destruction must not crash.
    cudart.cudaStreamDestroy(stream)
    del device_img


class _FakeCudaStream:
    """Minimal CUDA Stream Protocol object for tests (no cuda.core dependency)."""

    def __init__(self, ptr):
        self._ptr = int(ptr)

    def __cuda_stream__(self):
        return (0, self._ptr)


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_image_cuda_accepts_cuda_stream_protocol():
    """Image.cuda accepts __cuda_stream__ protocol objects as well as ints."""
    ref = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    stream_int = int(stream)
    try:
        device_img = host_img.cuda(device_id=0, cuda_stream=_FakeCudaStream(stream_int))
        assert int(device_img.cuda_stream) == stream_int
        np.testing.assert_array_equal(np.asarray(device_img.cpu()), ref)

        # Bare int still works.
        device_img2 = host_img.cuda(device_id=0, cuda_stream=stream_int)
        assert int(device_img2.cuda_stream) == stream_int
    finally:
        cudart.cudaStreamDestroy(stream)


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_image_cuda_stream_protocol_keeps_owner_alive():
    """Passing a protocol stream object pins it on the returned Image."""
    import weakref

    ref = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        proto = _FakeCudaStream(int(stream))
        wr = weakref.ref(proto)
        device_img = host_img.cuda(device_id=0, cuda_stream=proto)
        del proto
        gc.collect()
        assert wr() is not None, "Image must keep the protocol stream object alive"

        del device_img
        gc.collect()
        assert wr() is None, "Protocol stream object should be released with the Image"
    finally:
        cudart.cudaStreamDestroy(stream)


@t.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_as_image_explicit_cuda_stream_wins_over_cai():
    """as_image(cupy_array, cuda_stream=protocol) keeps the argument, not CAI.

    CuPy CAI ``stream`` is the current stream at access time (often 1 outside
    ``with stream:``). An explicit ``cuda_stream=`` must still pin the protocol
    object and report that handle on ``Image.cuda_stream``.
    """
    import weakref

    cp_img = cp.zeros((16, 16, 3), dtype=cp.uint8)
    cai_stream = cp_img.__cuda_array_interface__.get("stream")

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    stream_int = int(stream)
    try:
        proto = _FakeCudaStream(stream_int)
        wr = weakref.ref(proto)
        img = nvimgcodec.as_image(cp_img, cuda_stream=proto)
        assert int(img.cuda_stream) == stream_int
        assert int(img.__cuda_array_interface__["stream"]) == stream_int
        if cai_stream is not None:
            assert int(cai_stream) != stream_int

        del proto
        gc.collect()
        assert wr() is not None, "Image must keep the protocol stream object alive"

        del img
        gc.collect()
        assert wr() is None, "Protocol stream object should be released with the Image"
    finally:
        cudart.cudaStreamDestroy(stream)


@t.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_as_image_unspecified_cuda_stream_inherits_cai():
    """Omitting cuda_stream or passing None inherits CAI's stream."""
    cp_img = cp.zeros((16, 16, 3), dtype=cp.uint8)
    cai_stream = cp_img.__cuda_array_interface__.get("stream")
    if cai_stream is None:
        t.skip("CAI did not advertise stream")

    omitted = nvimgcodec.as_image(cp_img)
    none_arg = nvimgcodec.as_image(cp_img, cuda_stream=None)
    assert int(omitted.cuda_stream) == int(cai_stream)
    assert int(none_arg.cuda_stream) == int(cai_stream)


@t.mark.skipif(not CUPY_AVAILABLE, reason="CuPy not available")
def test_as_image_explicit_zero_wins_over_cai():
    """cuda_stream=0 is specified and is not replaced by CAI's stream."""
    cp_img = cp.zeros((16, 16, 3), dtype=cp.uint8)
    cai_stream = cp_img.__cuda_array_interface__.get("stream")
    img = nvimgcodec.as_image(cp_img, cuda_stream=0)
    assert int(img.cuda_stream) == 0
    # CAI forbids stream 0; the export maps the legacy default to 1.
    assert int(img.__cuda_array_interface__["stream"]) == 1
    if cai_stream is not None:
        assert int(cai_stream) != 0


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_decoder_accepts_cuda_stream_protocol():
    decoder = nvimgcodec.Decoder(device_id=0)
    image_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        image = decoder.read(nvimgcodec.CodeStream(image_path), cuda_stream=_FakeCudaStream(int(stream)))
        assert image is not None
        assert int(image.cuda_stream) == int(stream)
    finally:
        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_encoder_accepts_cuda_stream_protocol():
    decoder = nvimgcodec.Decoder(device_id=0)
    encoder = nvimgcodec.Encoder(device_id=0)
    image_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    image = decoder.read(image_path)
    assert image is not None

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        encoded = encoder.encode(image, "jpeg", cuda_stream=_FakeCudaStream(int(stream)))
        assert encoded is not None
    finally:
        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)


@t.mark.skipif(not CUDART_AVAILABLE, reason="cuda.bindings.runtime not available")
def test_to_dlpack_accepts_cuda_stream_protocol():
    decoder = nvimgcodec.Decoder(device_id=0)
    image_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    image = decoder.read(image_path)
    assert image is not None

    cudart.cudaSetDevice(0)
    err, stream = cudart.cudaStreamCreate()
    assert err == cudart.cudaError_t.cudaSuccess
    try:
        proto = _FakeCudaStream(int(stream))
        cap = image.to_dlpack(cuda_stream=proto)
        assert cap is not None
        cap2 = image.__dlpack__(stream=proto)
        assert cap2 is not None
    finally:
        cudart.cudaStreamSynchronize(stream)
        cudart.cudaStreamDestroy(stream)


def test_cuda_stream_protocol_rejects_bad_inputs():
    ref = np.zeros((8, 8, 3), dtype=np.uint8)
    host_img = nvimgcodec.as_image(ref)

    class BadVersion:
        def __cuda_stream__(self):
            return (1, 0)

    class BadTuple:
        def __cuda_stream__(self):
            return (0,)

    class NotAStream:
        pass

    class NotCallable:
        __cuda_stream__ = 0

    with t.raises(ValueError, match="Unsupported __cuda_stream__ protocol version"):
        host_img.cuda(cuda_stream=BadVersion())
    with t.raises(TypeError, match=r"must return a 2-tuple"):
        host_img.cuda(cuda_stream=BadTuple())
    with t.raises(TypeError, match=r"must be an int \(cudaStream_t handle\)"):
        host_img.cuda(cuda_stream=NotAStream())
    host_img.cuda(cuda_stream=None)
    with t.raises(TypeError, match="must be a callable method"):
        host_img.cuda(cuda_stream=NotCallable())
    with t.raises(ValueError, match="Unsupported __cuda_stream__ protocol version"):
        host_img.to_dlpack(cuda_stream=BadVersion())


# ----------------------------------------------------------------------
# Tests for the `precision` parameter on as_image / as_images.
# ----------------------------------------------------------------------

ARRAY_MODULES = [
    np,
    t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
]


@t.mark.parametrize(
    "array_module, dtype_name, expected_precision",
    [
        (np, "uint8", 8),
        (np, "uint16", 16),
        (np, "int16", 16),
        (np, "float32", 32),
        t.param(cp, "uint8", 8, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
        t.param(cp, "uint16", 16, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
    ]
)
def test_as_image_default_precision_resolves_to_dtype_bitdepth(array_module, dtype_name, expected_precision):
    """When precision is not provided, image.precision resolves to the dtype bitdepth."""
    dtype = getattr(array_module, dtype_name)
    ref_img = array_module.zeros((10, 10, 3), dtype=dtype)
    img = nvimgcodec.as_image(ref_img)
    assert img.precision == expected_precision


@t.mark.parametrize("array_module", ARRAY_MODULES)
def test_as_image_explicit_none_precision_resolves_to_dtype_bitdepth(array_module):
    """Passing precision=None is equivalent to omitting it."""
    ref_img = array_module.zeros((10, 10, 3), dtype=array_module.uint16)
    img = nvimgcodec.as_image(ref_img, precision=None)
    assert img.precision == 16


@t.mark.parametrize("array_module", ARRAY_MODULES)
@t.mark.parametrize(
    "dtype_name, precision",
    [
        ("uint16", 12),  # 12-bit data carried in a uint16 buffer (primary use case)
        ("uint16", 10),
        ("uint16", 1),
        ("uint8", 1),
        ("uint8", 7),
        ("int16", 12),
        ("int8", 4),
    ],
)
def test_as_image_precision_within_bitdepth_is_accepted(array_module, dtype_name, precision):
    """Any precision in [1, bitdepth(dtype)] is accepted and round-trips through .precision."""
    dtype = getattr(array_module, dtype_name)
    ref_img = array_module.zeros((10, 10, 3), dtype=dtype)
    img = nvimgcodec.as_image(ref_img, precision=precision)
    assert img.precision == precision


@t.mark.parametrize("array_module", ARRAY_MODULES)
@t.mark.parametrize(
    "dtype_name, bitdepth",
    [("uint8", 8), ("int8", 8), ("uint16", 16), ("int16", 16), ("uint32", 32), ("int32", 32)],
)
def test_as_image_precision_equal_to_bitdepth_is_accepted(array_module, dtype_name, bitdepth):
    """Boundary case: precision exactly equal to the dtype bitdepth is valid."""
    dtype = getattr(array_module, dtype_name)
    ref_img = array_module.zeros((10, 10, 3), dtype=dtype)
    img = nvimgcodec.as_image(ref_img, precision=bitdepth)
    assert img.precision == bitdepth


@t.mark.parametrize("array_module", ARRAY_MODULES)
@t.mark.parametrize(
    "dtype_name, bad_precision",
    [
        ("uint8", 9),    # > 8
        ("uint8", 16),
        ("int8", 9),
        ("uint16", 17),  # > 16
        ("int16", 32),
        ("uint32", 33),
    ],
)
def test_as_image_precision_exceeding_bitdepth_raises(array_module, dtype_name, bad_precision):
    """Precision greater than the dtype bitdepth must be rejected."""
    dtype = getattr(array_module, dtype_name)
    ref_img = array_module.zeros((10, 10, 3), dtype=dtype)
    with t.raises(Exception) as excinfo:
        nvimgcodec.as_image(ref_img, precision=bad_precision)
    msg = str(excinfo.value)
    assert "Precision" in msg
    assert "exceeds the bitdepth" in msg


@t.mark.parametrize("array_module", ARRAY_MODULES)
def test_as_image_negative_precision_raises(array_module):
    """Negative precision must be rejected."""
    ref_img = array_module.zeros((10, 10, 3), dtype=array_module.uint16)
    with t.raises(Exception) as excinfo:
        nvimgcodec.as_image(ref_img, precision=-1)
    assert "Precision must be a non-negative integer" in str(excinfo.value)


@t.mark.parametrize("array_module", ARRAY_MODULES)
@t.mark.parametrize("dtype_name", ["float16", "float32", "float64"])
def test_as_image_sub_bitdepth_precision_for_float_raises(array_module, dtype_name):
    """Floating-point dtypes reject a precision below the dtype's bitdepth."""
    dtype = getattr(array_module, dtype_name)
    ref_img = array_module.zeros((10, 10, 3), dtype=dtype)
    with t.raises(Exception) as excinfo:
        nvimgcodec.as_image(ref_img, precision=12)
    msg = str(excinfo.value)
    assert "floating-point" in msg


@t.mark.parametrize("array_module", ARRAY_MODULES)
def test_as_image_precision_zero_for_float_is_accepted(array_module):
    """precision=0 (full bitdepth) is always valid, including for floats.
    The Python property resolves 0 to the dtype's actual bitdepth (32 for float32)."""
    ref_img = array_module.zeros((10, 10, 3), dtype=array_module.float32)
    img = nvimgcodec.as_image(ref_img, precision=0)
    assert img.precision == 32


@t.mark.parametrize("array_module", ARRAY_MODULES)
@t.mark.parametrize(
    "dtype_name, bitdepth",
    [("float16", 16), ("float32", 32), ("float64", 64)],
)
def test_as_image_precision_equal_to_bitdepth_for_float_is_accepted(
        array_module, dtype_name, bitdepth):
    """precision == dtype bitdepth is the numeric equivalent of the 0 sentinel and must
    be accepted for float dtypes too, matching the integer-dtype behavior."""
    dtype = getattr(array_module, dtype_name)
    ref_img = array_module.zeros((10, 10, 3), dtype=dtype)
    img = nvimgcodec.as_image(ref_img, precision=bitdepth)
    assert img.precision == bitdepth


@t.mark.parametrize("array_module", ARRAY_MODULES)
def test_as_images_applies_precision_uniformly(array_module):
    """as_images applies the same precision to every produced Image."""
    img1 = array_module.zeros((10, 10, 3), dtype=array_module.uint16)
    img2 = array_module.zeros((15, 15, 3), dtype=array_module.uint16)
    img3 = array_module.zeros((20, 20, 3), dtype=array_module.uint16)
    images = nvimgcodec.as_images([img1, img2, img3], precision=12)
    for img in images:
        assert img.precision == 12


@t.mark.parametrize("array_module", ARRAY_MODULES)
def test_as_images_precision_validated_against_each_dtype(array_module):
    """Validation runs per-image: a mismatch in any element raises."""
    img_u16 = array_module.zeros((10, 10, 3), dtype=array_module.uint16)
    img_u8 = array_module.zeros((10, 10, 3), dtype=array_module.uint8)
    with t.raises(Exception) as excinfo:
        nvimgcodec.as_images([img_u16, img_u8], precision=12)
    assert "exceeds the bitdepth" in str(excinfo.value)


# ---------------------------------------------------------------------------
# as_image with explicit sample_format - layout interpretation
#
# Covers the "planar (CHW) zero-copy wrap" opt-in: when the caller passes a
# concrete planar sample_format together with a shape that matches that
# format's channel arity, the resulting Image is built with planar layout.
# Mismatched shapes fall through to the legacy interleaved (HWC) labelling
# path.
# ---------------------------------------------------------------------------

def _make_sample_format_host_array_hwc(num_channels):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(64, 96, num_channels), dtype=np.uint8)


def _make_sample_format_host_array_chw(num_channels):
    rng = np.random.default_rng(0)
    return rng.integers(0, 256, size=(num_channels, 48, 80), dtype=np.uint8)


def test_as_image_sample_format_interleaved_default():
    arr = _make_sample_format_host_array_hwc(3)
    img = nvimgcodec.as_image(arr)
    assert img.sample_format == nvimgcodec.SampleFormat.I_RGB
    assert img.shape == (64, 96, 3)


def test_as_image_sample_format_explicit_interleaved_rgb():
    arr = _make_sample_format_host_array_hwc(3)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.I_RGB)
    assert img.sample_format == nvimgcodec.SampleFormat.I_RGB
    assert img.shape == (64, 96, 3)


def test_as_image_sample_format_explicit_interleaved_rgba():
    arr = _make_sample_format_host_array_hwc(4)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.I_RGBA)
    assert img.sample_format == nvimgcodec.SampleFormat.I_RGBA
    assert img.shape == (64, 96, 4)


def test_as_image_sample_format_planar_rgb_from_chw():
    arr = _make_sample_format_host_array_chw(3)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGB
    assert img.shape == (3, 48, 80)
    assert img.color_spec == nvimgcodec.ColorSpec.SRGB


def test_as_image_sample_format_planar_rgba_from_chw():
    arr = _make_sample_format_host_array_chw(4)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_RGBA)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGBA
    assert img.shape == (4, 48, 80)


def test_as_image_sample_format_planar_y_from_2d():
    # A 2-D (H, W) grayscale array + sample_format=P_Y -> planar Image with
    # shape (1, H, W). No (H, W, 1) wrapper required from the caller.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(48, 80), dtype=np.uint8)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.P_Y
    assert img.color_spec == nvimgcodec.ColorSpec.GRAY
    assert img.shape == (1, 48, 80)
    # Pixel content survives the wrap (P_Y at arity 1 has the same byte
    # layout as the input).
    np.testing.assert_array_equal(np.asarray(img.cpu())[0], arr)


def test_as_image_sample_format_planar_y_from_chw_1hw():
    # The (1, H, W) form must also work, going through the existing 3-D
    # planar branch.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(1, 48, 80), dtype=np.uint8)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.P_Y
    assert img.shape == (1, 48, 80)
    np.testing.assert_array_equal(np.asarray(img.cpu()), arr)


def test_as_image_sample_format_2d_with_planar_rgb_raises():
    # A 2-D array can only enter the planar path for single-channel formats
    # (P_Y). P_RGB has arity 3, so passing it with a 2-D shape is rejected
    # up-front rather than silently labelled.
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(48, 80), dtype=np.uint8)
    with t.raises(ValueError, match="Planar sample_format P_RGB"):
        nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_RGB)


def test_as_image_sample_format_interleaved_too_few_channels_raises():
    # (H, W, 3) array with I_RGBA request: I_RGBA needs at least 4 channels.
    arr = _make_sample_format_host_array_hwc(3)
    with t.raises((ValueError, RuntimeError)):
        nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.I_RGBA)


# ---------------------------------------------------------------------------
# as_image with row-padded buffers
#
# Padding is allowed only on the row axis (the image data must be a contiguous
# region of memory in every other dimension). For PLANAR inputs the plane axis
# is treated like any inner axis - plane gaps are NOT allowed because the C
# image-info data model has no per-plane offset field; consumers compute the
# offset of plane c as `c * height * row_stride` from the buffer base, so any
# larger plane stride would be silently misread.
# ---------------------------------------------------------------------------

_PADDING_ARRAY_MODULES = [
    np,
    t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
]


def _make_padded_hwc(array_module, h, w, padded_w, c, dtype=None):
    """Return a (h, padded_w, c) backing buffer and a (h, w, c) view into it.

    The returned view exposes explicit strides via __array_interface__ /
    __cuda_array_interface__ (numpy/cupy preserve strides for non-trivial
    slices), so this exercises the strided-import path in nvimgcodec.
    """
    dtype = array_module.uint8 if dtype is None else dtype
    full = array_module.zeros((h, padded_w, c), dtype=dtype)
    full[:, :w, :] = array_module.random.randint(0, 256, (h, w, c)).astype(dtype)
    return full, full[:, :w, :]


def _make_padded_chw(array_module, c, h, w, padded_w, dtype=None):
    dtype = array_module.uint8 if dtype is None else dtype
    full = array_module.zeros((c, h, padded_w), dtype=dtype)
    full[:, :, :w] = array_module.random.randint(0, 256, (c, h, w)).astype(dtype)
    return full, full[:, :, :w]


def _make_padded_2d(array_module, h, w, padded_w, dtype=None):
    dtype = array_module.uint8 if dtype is None else dtype
    full = array_module.zeros((h, padded_w), dtype=dtype)
    full[:, :w] = array_module.random.randint(0, 256, (h, w)).astype(dtype)
    return full, full[:, :w]


def _to_host(arr):
    """numpy passthrough, cupy -> numpy."""
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_hwc_rows(array_module):
    """Row-padded interleaved HWC view: row stride > W*C."""
    _, view = _make_padded_hwc(array_module, h=20, w=16, padded_w=24, c=3)

    img = nvimgcodec.as_image(view)
    assert img.shape == (20, 16, 3)
    assert img.sample_format == nvimgcodec.SampleFormat.I_RGB
    # row stride is the padded row size, inner strides are natural.
    assert img.strides == (72, 3, 1)
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_hwc_rgba(array_module):
    """Row-padded HWC RGBA - exercises the 4-channel default branch."""
    _, view = _make_padded_hwc(array_module, h=20, w=16, padded_w=20, c=4)

    img = nvimgcodec.as_image(view)
    assert img.shape == (20, 16, 4)
    assert img.sample_format == nvimgcodec.SampleFormat.I_RGBA
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_chw_planar(array_module):
    """Row-padded planar CHW with sample_format=P_RGB."""
    _, view = _make_padded_chw(array_module, c=3, h=20, w=16, padded_w=24)

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.shape == (3, 20, 16)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGB
    # row stride 24, plane stride 24*20=480.
    assert img.strides == (480, 24, 1)
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_chw_rgba_planar(array_module):
    """Row-padded planar CHW with sample_format=P_RGBA (4-plane)."""
    _, view = _make_padded_chw(array_module, c=4, h=20, w=16, padded_w=24)

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGBA)
    assert img.shape == (4, 20, 16)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGBA
    assert img.strides == (480, 24, 1)
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_chw_p_unchanged(array_module):
    """Row-padded planar with sample_format=P_UNCHANGED (no arity constraint)."""
    _, view = _make_padded_chw(array_module, c=2, h=20, w=16, padded_w=24)

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_UNCHANGED)
    assert img.shape == (2, 20, 16)
    assert img.sample_format == nvimgcodec.SampleFormat.P_UNCHANGED
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_bottom_padded_hwc(array_module):
    """Bottom-only padding: a prefix slice arr[:H] of a taller buffer.

    Numpy / cupy expose this as `strides=None` (the prefix is C-contiguous),
    but we still get the correct shape and pixel content. Verifies the
    `strides=None` short-circuit path of is_padding_correct stays correct.
    """
    full = array_module.random.randint(0, 256, (24, 16, 3)).astype(array_module.uint8)
    view = full[:20, :, :]
    img = nvimgcodec.as_image(view)
    assert img.shape == (20, 16, 3)
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


# ---- 2-D grayscale (H, W) ----

@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_2d_grayscale_default(array_module):
    """Contiguous 2-D (H, W) array, no sample_format override: defaults to I_Y."""
    arr = array_module.random.randint(0, 256, (20, 16)).astype(array_module.uint8)
    img = nvimgcodec.as_image(arr)
    assert img.sample_format == nvimgcodec.SampleFormat.I_Y
    assert img.color_spec == nvimgcodec.ColorSpec.GRAY
    assert img.shape == (20, 16, 1)
    np.testing.assert_array_equal(_to_host(img.cpu())[..., 0], _to_host(arr))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_2d_grayscale_default(array_module):
    """Row-padded 2-D (H, W) array: default sample_format still infers I_Y."""
    _, view = _make_padded_2d(array_module, h=20, w=16, padded_w=24)

    img = nvimgcodec.as_image(view)
    assert img.sample_format == nvimgcodec.SampleFormat.I_Y
    assert img.shape == (20, 16, 1)
    assert img.strides == (24, 1, 1)
    np.testing.assert_array_equal(_to_host(img.cpu())[..., 0], _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_2d_grayscale_p_y(array_module):
    """Row-padded 2-D (H, W) array with sample_format=P_Y: planar single-plane."""
    _, view = _make_padded_2d(array_module, h=20, w=16, padded_w=24)

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.P_Y
    assert img.color_spec == nvimgcodec.ColorSpec.GRAY
    assert img.shape == (1, 20, 16)
    # Row stride preserved, plane stride = row_stride * height = 480.
    assert img.strides == (480, 24, 1)
    np.testing.assert_array_equal(_to_host(img.cpu())[0], _to_host(view))


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_2d_grayscale_i_y(array_module):
    """Row-padded 2-D (H, W) array with sample_format=I_Y: interleaved single-channel."""
    _, view = _make_padded_2d(array_module, h=20, w=16, padded_w=24)

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.I_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.I_Y
    assert img.shape == (20, 16, 1)
    np.testing.assert_array_equal(_to_host(img.cpu())[..., 0], _to_host(view))


# ---- Plane-gap inputs are rejected ----
#
# The C nvimgcodecImagePlaneInfo_t struct has no per-plane offset / stride
# field; every consumer that crosses the C boundary (encoders, decoders,
# DLPack export) computes plane c's start as `c * height * row_stride` from
# the buffer base. We therefore reject inputs whose plane stride exceeds the
# natural product up-front rather than silently misreading plane 1+ once the
# Image is handed to the codec pipeline.

@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_plane_gap_via_step_slice_raises(array_module):
    """arr[::2] of (4, H, W) -> shape (2, H, W) with plane stride 2*H*W."""
    full = array_module.random.randint(0, 256, (4, 20, 16)).astype(array_module.uint8)
    view = full[::2]  # plane stride 2*H*W

    with t.raises(RuntimeError, match="Padding is only allowed for rows"):
        nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_UNCHANGED)


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_plane_gap_with_concrete_planar_format_raises(array_module):
    """arr[::2] of (4, H, W) with sample_format=P_YA - same rejection path."""
    full = array_module.random.randint(0, 256, (4, 20, 16)).astype(array_module.uint8)
    view = full[::2]

    with t.raises(RuntimeError, match="Padding is only allowed for rows"):
        nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_YA)


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_plane_gap_combined_with_row_padding_raises(array_module):
    """A 3-D CHW input that combines a step-slice along the plane axis with
    row padding is still rejected - plane gaps cannot be expressed in the C
    image-info contract regardless of row layout."""
    H, W, padded_W = 20, 16, 24
    full = array_module.zeros((6, H, padded_W), dtype=array_module.uint8)
    view = full[::2, :, :W]  # shape (3, H, W); plane stride 2*H*padded_W

    with t.raises(RuntimeError, match="Padding is only allowed for rows"):
        nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)


@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_chw_contiguous_prefix_slice_works(array_module):
    """arr[:3] of (4, H, W) -> shape (3, H, W) with no plane gap; accepted."""
    full = array_module.random.randint(0, 256, (4, 20, 16)).astype(array_module.uint8)
    view = full[:3]

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.shape == (3, 20, 16)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGB
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(view))


# ---- Round-trip pixel content for padded planar inputs ----

@t.mark.parametrize("array_module", _PADDING_ARRAY_MODULES)
def test_as_image_padded_chw_pixel_content_round_trip(array_module):
    """Pixel content of a padded planar CHW view survives the wrap.

    Catches the case where the wrapper would read from the wrong axis (e.g.,
    confusing row stride with plane stride). Done across both numpy and cupy.
    """
    full, view = _make_padded_chw(array_module, c=3, h=12, w=10, padded_w=16)
    # Fill the padding region with a sentinel that must NOT appear in the
    # exported buffer if the row stride is honoured correctly.
    full[:, :, 10:] = 255

    img = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    host = _to_host(img.cpu())
    assert host.shape == (3, 12, 10)
    # The first 10 columns of each plane should match the view exactly; the
    # sentinel 255 column is outside the visible W.
    np.testing.assert_array_equal(host, _to_host(view))


# ---------------------------------------------------------------------------
# as_image DLPack interop
#
# Verifies that a DLPack consumer (np.from_dlpack / cp.from_dlpack) and the
# Image's own shape/strides/size agree. The two protocols share the same
# underlying buffer so a layout mismatch is a real interop bug, not just
# cosmetic.
# ---------------------------------------------------------------------------

_DLPACK_ARRAY_MODULES = [
    np,
    t.param(cp, marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
]


def _dlpack_consumer(array_module):
    """Pick the array module that should consume an Image's __dlpack__.
    numpy host buffers go through numpy.from_dlpack; cupy device buffers go
    through cupy.from_dlpack."""
    return np.from_dlpack if array_module is np else cp.from_dlpack


def _dlpack_view(array_module, img):
    """Materialise the consumer-side view of img's DLPack output, then bring
    it back to host so we can compare shape / strides / pixel content with
    plain numpy."""
    view = _dlpack_consumer(array_module)(img)
    if array_module is np:
        return view, view  # numpy view doubles as the host view
    return view, cp.asnumpy(view)


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_hwc_default(array_module):
    """Default 3-D HWC input -> I_RGB Image -> DLPack view of shape (H, W, C)."""
    arr = array_module.random.randint(0, 256, (20, 16, 3)).astype(array_module.uint8)
    img = nvimgcodec.as_image(arr)
    assert img.shape == (20, 16, 3)

    view, host = _dlpack_view(array_module, img)
    assert view.shape == img.shape
    assert view.strides == img.strides
    assert view.nbytes == img.size
    np.testing.assert_array_equal(host, _to_host(arr))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_hwc_2d_grayscale(array_module):
    """A 2-D (H, W) array becomes a single-channel I_Y Image of shape
    (H, W, 1); DLPack must agree."""
    arr = array_module.random.randint(0, 256, (20, 16)).astype(array_module.uint8)
    img = nvimgcodec.as_image(arr)
    assert img.sample_format == nvimgcodec.SampleFormat.I_Y
    assert img.shape == (20, 16, 1)

    view, host = _dlpack_view(array_module, img)
    assert view.shape == img.shape
    assert view.strides == img.strides
    assert view.nbytes == img.size
    np.testing.assert_array_equal(host.squeeze(-1), _to_host(arr))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_hwc_padded(array_module):
    """Row-padded HWC source -> DLPack consumer sees the same shape and the
    same byte strides as Image.strides."""
    full = array_module.zeros((20, 24, 3), dtype=array_module.uint8)
    full[:, :16, :] = array_module.random.randint(0, 256, (20, 16, 3)).astype(array_module.uint8)
    view = full[:, :16, :]
    img = nvimgcodec.as_image(view)
    assert img.shape == (20, 16, 3)
    assert img.strides == (24 * 3, 3, 1)

    dv, host = _dlpack_view(array_module, img)
    assert dv.shape == img.shape
    assert dv.strides == img.strides
    np.testing.assert_array_equal(host, _to_host(view))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_planar_rgb(array_module):
    """A (3, H, W) array with sample_format=P_RGB becomes a planar Image
    of shape (3, H, W); the DLPack consumer must see the same layout
    (the prior bug exported it as (H, W, 1) when num_planes==1, but P_RGB
    has num_planes=3 so the bug didn't fire here - kept for matrix coverage)."""
    arr = array_module.random.randint(0, 256, (3, 20, 16)).astype(array_module.uint8)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.shape == (3, 20, 16)

    view, host = _dlpack_view(array_module, img)
    assert view.shape == img.shape
    assert view.strides == img.strides
    assert view.nbytes == img.size
    np.testing.assert_array_equal(host, _to_host(arr))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_planar_y_from_2d(array_module):
    """Regression test for the prior dlpack_utils.cpp bug: a 2-D (H, W) array
    wrapped with sample_format=P_Y produces a planar single-plane Image of
    shape (1, H, W). The DLPack consumer must report the same shape, NOT
    the (H, W, 1) HWC layout that the old `is_interleaved || num_planes==1`
    rule collapsed it to."""
    arr = array_module.random.randint(0, 256, (20, 16)).astype(array_module.uint8)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.P_Y
    assert img.shape == (1, 20, 16)

    view, host = _dlpack_view(array_module, img)
    assert view.shape == (1, 20, 16)
    assert view.strides == img.strides
    assert view.nbytes == img.size
    np.testing.assert_array_equal(host[0], _to_host(arr))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_planar_y_from_chw_1hw(array_module):
    """The explicit (1, H, W) form of a single-plane P_Y Image goes through
    the same DLPack export path."""
    arr = array_module.random.randint(0, 256, (1, 20, 16)).astype(array_module.uint8)
    img = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.shape == (1, 20, 16)

    view, host = _dlpack_view(array_module, img)
    assert view.shape == (1, 20, 16)
    assert view.strides == img.strides
    np.testing.assert_array_equal(host, _to_host(arr))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_export_padded_planar(array_module):
    """Row-padded CHW source -> DLPack view reflects the padded row stride
    and the natural plane stride."""
    full = array_module.zeros((3, 20, 24), dtype=array_module.uint8)
    full[:, :, :16] = array_module.random.randint(0, 256, (3, 20, 16)).astype(array_module.uint8)
    src = full[:, :, :16]
    img = nvimgcodec.as_image(src, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.shape == (3, 20, 16)
    assert img.strides == (20 * 24, 24, 1)

    dv, host = _dlpack_view(array_module, img)
    assert dv.shape == img.shape
    assert dv.strides == img.strides
    np.testing.assert_array_equal(host, _to_host(src))


# ---- DLPack import (nvimgcodec.from_dlpack) ----
#
# Without a sample_format override the import path interprets the incoming
# tensor as interleaved HWC. Verify that shape, strides, size and pixel content
# survive a from_dlpack round-trip. (A layout-selecting sample_format on the
# DLPack path is covered in the next section.)

@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_import_hwc(array_module):
    """nvimgcodec.from_dlpack(tensor) materialises an Image whose shape /
    strides / size match the source tensor's interleaved layout."""
    arr = array_module.random.randint(0, 256, (20, 16, 3)).astype(array_module.uint8)
    img = nvimgcodec.from_dlpack(arr)
    assert img.shape == (20, 16, 3)
    # Contiguous interleaved -> natural row stride.
    assert img.strides == (16 * 3, 3, 1)
    assert img.size == 20 * 16 * 3
    np.testing.assert_array_equal(_to_host(img.cpu()), _to_host(arr))


@t.mark.parametrize("array_module", _DLPACK_ARRAY_MODULES)
def test_as_image_dlpack_import_then_export_roundtrip(array_module):
    """Round-trip: source tensor -> from_dlpack -> Image -> __dlpack__ -> consumer.
    Shape, strides and size are preserved end-to-end."""
    arr = array_module.random.randint(0, 256, (20, 16, 3)).astype(array_module.uint8)
    img = nvimgcodec.from_dlpack(arr)

    view, host = _dlpack_view(array_module, img)
    assert view.shape == img.shape
    assert view.strides == img.strides
    assert view.nbytes == img.size
    np.testing.assert_array_equal(host, _to_host(arr))


# ---- DLPack import: sample_format / color_spec parity with as_image ----
#
# These mirror the array-interface as_image tests above (default labels by
# channel count, sample_format / color_spec overrides, planar layouts, arity
# and UNKNOWN validation) but drive the *DLPack* path. The array-interface tests
# parametrize over numpy (host) and cupy (device); the DLPack import path
# requires CUDA-accessible memory, so these use cupy and instead parametrize
# over the two pure-DLPack input forms: a raw capsule (arr.__dlpack__()) and an
# adapter exposing only __dlpack__ / __dlpack_device__. Passing those directly
# avoids the __cuda_array_interface__ branch masking the DLPack path.

class _DLPackOnly:
    """Wraps an array exposing ONLY the DLPack protocol, so as_image cannot fall
    back to __cuda_array_interface__ / __array_interface__."""

    def __init__(self, arr):
        self._arr = arr

    def __dlpack__(self, stream=None):
        return self._arr.__dlpack__(stream=stream) if stream is not None else self._arr.__dlpack__()

    def __dlpack_device__(self):
        return self._arr.__dlpack_device__()


def _pure_dlpack_source(arr, kind):
    return arr.__dlpack__() if kind == "capsule" else _DLPackOnly(arr)


_DLPACK_SOURCE_KINDS = ["capsule", "adapter"]


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize(
    "shape, expected_sample_format, expected_color_spec",
    [
        ((10, 10, 1), nvimgcodec.SampleFormat.I_Y, nvimgcodec.ColorSpec.GRAY),
        ((10, 10, 2), nvimgcodec.SampleFormat.I_YA, nvimgcodec.ColorSpec.GRAY),
        ((10, 10, 3), nvimgcodec.SampleFormat.I_RGB, nvimgcodec.ColorSpec.SRGB),
        ((10, 10, 4), nvimgcodec.SampleFormat.I_RGBA, nvimgcodec.ColorSpec.SRGB),
        ((10, 10, 5), nvimgcodec.SampleFormat.UNKNOWN, nvimgcodec.ColorSpec.UNKNOWN),
    ],
)
def test_as_image_dlpack_default_sample_format_and_color_spec(
        kind, shape, expected_sample_format, expected_color_spec):
    """DLPack path infers the same channel-count default sample_format / color_spec
    as the array-interface path, and reports a device buffer."""
    arr = cp.random.randint(0, 255, shape, dtype=cp.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(arr, kind))
    assert img.sample_format == expected_sample_format
    assert img.color_spec == expected_color_spec
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    assert img.shape == shape


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("sample_format", [
    nvimgcodec.SampleFormat.I_RGB,
    nvimgcodec.SampleFormat.I_BGR,
    nvimgcodec.SampleFormat.I_YUV,
])
def test_as_image_dlpack_override_interleaved_sample_format(kind, sample_format):
    """An interleaved sample_format override is honoured on the DLPack path."""
    hwc = cp.random.randint(0, 256, (20, 24, 3), dtype=cp.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(hwc, kind), sample_format=sample_format)
    assert img.sample_format == sample_format
    assert img.shape == (20, 24, 3)
    np.testing.assert_array_equal(np.asarray(img.cpu()), cp.asnumpy(hwc))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("color_spec", [
    nvimgcodec.ColorSpec.SRGB,
    nvimgcodec.ColorSpec.SYCC,
    nvimgcodec.ColorSpec.UNKNOWN,
])
def test_as_image_dlpack_override_color_spec(kind, color_spec):
    """A color_spec override is honoured on the DLPack path."""
    hwc = cp.random.randint(0, 256, (20, 24, 3), dtype=cp.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(hwc, kind), color_spec=color_spec)
    assert img.color_spec == color_spec


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("sample_format, shape", [
    (nvimgcodec.SampleFormat.P_RGB, (3, 20, 24)),
    (nvimgcodec.SampleFormat.P_BGR, (3, 20, 24)),
    (nvimgcodec.SampleFormat.P_Y, (1, 20, 24)),
    (nvimgcodec.SampleFormat.P_UNCHANGED, (3, 20, 24)),
])
def test_as_image_dlpack_planar_layout(kind, sample_format, shape):
    """A planar sample_format makes the DLPack path interpret the tensor as CHW;
    shape and pixel content are preserved."""
    chw = cp.random.randint(0, 256, shape, dtype=cp.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(chw, kind), sample_format=sample_format)
    assert img.sample_format == sample_format
    assert img.shape == shape
    np.testing.assert_array_equal(np.asarray(img.cpu()), cp.asnumpy(chw))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("sample_format, expected_sample_format, expected_shape", [
    (None, nvimgcodec.SampleFormat.I_Y, (20, 24, 1)),          # default -> grayscale interleaved
    (nvimgcodec.SampleFormat.I_Y, nvimgcodec.SampleFormat.I_Y, (20, 24, 1)),
    (nvimgcodec.SampleFormat.P_Y, nvimgcodec.SampleFormat.P_Y, (1, 20, 24)),
])
def test_as_image_dlpack_2d_grayscale(kind, sample_format, expected_sample_format, expected_shape):
    """A 2-D (H, W) DLPack tensor is accepted as a single-channel image (matching
    the array-interface as_image path): default and I_Y give interleaved
    (H, W, 1), P_Y gives planar (1, H, W); color_spec defaults to GRAY."""
    hw = cp.random.randint(0, 256, (20, 24), dtype=cp.uint8)
    kwargs = {} if sample_format is None else {"sample_format": sample_format}
    img = nvimgcodec.as_image(_pure_dlpack_source(hw, kind), **kwargs)
    assert img.sample_format == expected_sample_format
    assert img.color_spec == nvimgcodec.ColorSpec.GRAY
    assert img.shape == expected_shape
    np.testing.assert_array_equal(np.asarray(img.cpu()).reshape(20, 24), cp.asnumpy(hw))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_as_image_dlpack_2d_grayscale_row_padded():
    """A row-padded 2-D (H, W) DLPack tensor (a sliced view) is accepted; only the
    row dimension may carry padding."""
    full = cp.random.randint(0, 256, (20, 32), dtype=cp.uint8)
    view = full[:, :24]  # row-padded view: shape (20, 24), row stride 32
    img = nvimgcodec.as_image(view.__dlpack__())
    assert img.shape == (20, 24, 1)
    np.testing.assert_array_equal(np.asarray(img.cpu()).reshape(20, 24), cp.asnumpy(view))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_from_dlpack_2d_grayscale():
    """nvimgcodec.from_dlpack accepts a 2-D (H, W) grayscale tensor."""
    hw = cp.random.randint(0, 256, (20, 24), dtype=cp.uint8)
    img = nvimgcodec.from_dlpack(hw)
    assert img.sample_format == nvimgcodec.SampleFormat.I_Y
    assert img.shape == (20, 24, 1)
    np.testing.assert_array_equal(np.asarray(img.cpu()).reshape(20, 24), cp.asnumpy(hw))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("sample_format, match", [
    (nvimgcodec.SampleFormat.I_RGB, "Interleaved sample_format I_RGB"),
    (nvimgcodec.SampleFormat.P_RGB, "Planar sample_format P_RGB"),
])
def test_as_image_dlpack_2d_cannot_be_rgb(kind, sample_format, match):
    """A 2-D (H, W) DLPack tensor is single-channel, so an RGB sample_format is
    rejected on the DLPack path (interleaved and planar alike)."""
    hw = cp.random.randint(0, 256, (20, 24), dtype=cp.uint8)
    with t.raises(ValueError, match=match):
        nvimgcodec.as_image(_pure_dlpack_source(hw, kind), sample_format=sample_format)


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_from_dlpack_planar_layout_honored():
    """nvimgcodec.from_dlpack(tensor, sample_format=P_RGB) reads (C, H, W) as planar."""
    chw = cp.random.randint(0, 256, (3, 20, 24), dtype=cp.uint8)
    img = nvimgcodec.from_dlpack(chw, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGB
    assert img.shape == (3, 20, 24)
    np.testing.assert_array_equal(np.asarray(img.cpu()), cp.asnumpy(chw))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_from_dlpack_precision_override():
    """nvimgcodec.from_dlpack(tensor, precision=N) records sub-container precision
    (e.g. 12-bit data held in a uint16 buffer), like as_image."""
    arr = cp.random.randint(0, 1 << 12, (16, 24, 3), dtype=cp.uint16)
    img = nvimgcodec.from_dlpack(arr.__dlpack__(), precision=12)
    assert img.precision == 12
    assert img.dtype == np.uint16
    np.testing.assert_array_equal(np.asarray(img.cpu()), cp.asnumpy(arr))


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_as_image_dlpack_planar_arity_mismatch_raises():
    """A planar sample_format whose arity exceeds the tensor's plane count is
    rejected on the DLPack path (1-plane tensor cannot be P_RGB)."""
    chw = cp.random.randint(0, 256, (1, 20, 24), dtype=cp.uint8)
    with t.raises(ValueError, match=r"Planar sample_format .* CHW-shaped input"):
        nvimgcodec.as_image(chw.__dlpack__(), sample_format=nvimgcodec.SampleFormat.P_RGB)


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_as_image_dlpack_interleaved_arity_mismatch_raises():
    """An interleaved sample_format whose arity exceeds the tensor's channel count
    is rejected on the DLPack path (3-channel tensor cannot be I_RGBA)."""
    hwc = cp.random.randint(0, 256, (20, 24, 3), dtype=cp.uint8)
    with t.raises(ValueError, match=r"Interleaved sample_format .* HWC-shaped input"):
        nvimgcodec.as_image(hwc.__dlpack__(), sample_format=nvimgcodec.SampleFormat.I_RGBA)


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_as_image_dlpack_unknown_sample_format_raises():
    """sample_format=UNKNOWN is rejected on the DLPack path, matching as_image's
    array-interface behaviour."""
    hwc = cp.random.randint(0, 256, (20, 24, 3), dtype=cp.uint8)
    with t.raises(ValueError, match="UNKNOWN"):
        nvimgcodec.as_image(hwc.__dlpack__(), sample_format=nvimgcodec.SampleFormat.UNKNOWN)


# ---------------------------------------------------------------------------
# DLPack import: CPU (host) source
#
# A DLPack exporter whose only device kind is kDLCPU (e.g. a PyTorch CPU
# tensor, which exposes neither __array_interface__ nor
# __cuda_array_interface__) used to be unconditionally rejected with
# "Unsupported device in DLTensor. Only CUDA-accessible memory buffers can be
# wrapped", even though the same buffer works fine through the
# __array_interface__ path. These mirror the CUDA-only parity tests above but
# drive plain host memory through the pure-DLPack path (capsule / adapter),
# the same way the CUDA-only tests avoid the __cuda_array_interface__ branch.
# A numpy array's own __dlpack__ already reports device kDLCPU (verified: it
# rejects a non-None `stream`, exactly like PyTorch does), so numpy is enough
# to exercise the fix without adding a hard test dependency on torch. A
# dedicated PyTorch-tensor test below covers the literal real-world scenario.
# ---------------------------------------------------------------------------

@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize(
    "shape, expected_sample_format, expected_color_spec",
    [
        ((10, 10, 1), nvimgcodec.SampleFormat.I_Y, nvimgcodec.ColorSpec.GRAY),
        ((10, 10, 2), nvimgcodec.SampleFormat.I_YA, nvimgcodec.ColorSpec.GRAY),
        ((10, 10, 3), nvimgcodec.SampleFormat.I_RGB, nvimgcodec.ColorSpec.SRGB),
        ((10, 10, 4), nvimgcodec.SampleFormat.I_RGBA, nvimgcodec.ColorSpec.SRGB),
        ((10, 10, 5), nvimgcodec.SampleFormat.UNKNOWN, nvimgcodec.ColorSpec.UNKNOWN),
    ],
)
def test_as_image_dlpack_cpu_default_sample_format_and_color_spec(
        kind, shape, expected_sample_format, expected_color_spec):
    """DLPack CPU path infers the same channel-count default sample_format /
    color_spec as the CUDA DLPack and array-interface paths, and reports a
    host buffer with no device_id, matching the __array_interface__ path."""
    arr = np.random.randint(0, 255, shape, dtype=np.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(arr, kind))
    assert img.sample_format == expected_sample_format
    assert img.color_spec == expected_color_spec
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert img.device_id is None
    assert img.shape == shape
    np.testing.assert_array_equal(np.asarray(img.cpu()), arr)


@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("sample_format", [
    nvimgcodec.SampleFormat.I_RGB,
    nvimgcodec.SampleFormat.I_BGR,
    nvimgcodec.SampleFormat.I_YUV,
])
def test_as_image_dlpack_cpu_override_interleaved_sample_format(kind, sample_format):
    """An interleaved sample_format override is honoured on the DLPack CPU path."""
    hwc = np.random.randint(0, 256, (20, 24, 3), dtype=np.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(hwc, kind), sample_format=sample_format)
    assert img.sample_format == sample_format
    assert img.shape == (20, 24, 3)
    np.testing.assert_array_equal(np.asarray(img.cpu()), hwc)


@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
@t.mark.parametrize("sample_format, shape", [
    (nvimgcodec.SampleFormat.P_RGB, (3, 20, 24)),
    (nvimgcodec.SampleFormat.P_Y, (1, 20, 24)),
])
def test_as_image_dlpack_cpu_planar_layout(kind, sample_format, shape):
    """A planar sample_format makes the DLPack CPU path interpret the tensor as
    CHW; shape and pixel content are preserved."""
    chw = np.random.randint(0, 256, shape, dtype=np.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(chw, kind), sample_format=sample_format)
    assert img.sample_format == sample_format
    assert img.shape == shape
    np.testing.assert_array_equal(np.asarray(img.cpu()), chw)


@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
def test_as_image_dlpack_cpu_2d_grayscale(kind):
    """A 2-D (H, W) DLPack CPU tensor is accepted as a single-channel image,
    matching the CUDA DLPack and array-interface paths."""
    hw = np.random.randint(0, 256, (20, 24), dtype=np.uint8)
    img = nvimgcodec.as_image(_pure_dlpack_source(hw, kind))
    assert img.sample_format == nvimgcodec.SampleFormat.I_Y
    assert img.color_spec == nvimgcodec.ColorSpec.GRAY
    assert img.shape == (20, 24, 1)
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    np.testing.assert_array_equal(np.asarray(img.cpu()).reshape(20, 24), hw)


def test_from_dlpack_cpu_capsule():
    """nvimgcodec.from_dlpack accepts a raw CPU DLPack capsule directly."""
    hwc = np.random.randint(0, 256, (20, 24, 3), dtype=np.uint8)
    img = nvimgcodec.from_dlpack(hwc.__dlpack__())
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert img.shape == (20, 24, 3)
    np.testing.assert_array_equal(np.asarray(img.cpu()), hwc)


def test_as_image_dlpack_unsupported_non_cpu_non_cuda_device_raises():
    """A DLPack device that is neither kDLCPU nor CUDA-accessible is still
    rejected, with a message covering both accepted kinds."""
    class _FakeUnsupportedDevice:
        def __dlpack_device__(self):
            return (99, 0)  # not a real DLDeviceType

        def __dlpack__(self, stream=None):
            raise AssertionError("__dlpack__ must not be called once the device check rejects it")

    with t.raises(RuntimeError, match="Only CPU or CUDA-accessible memory buffers can be wrapped"):
        nvimgcodec.as_image(_FakeUnsupportedDevice())


@t.mark.parametrize("kind", _DLPACK_SOURCE_KINDS)
def test_as_image_dlpack_cpu_negative_row_stride_raises(kind):
    """A reversed/flipped CPU view (negative row stride) is rejected rather
    than silently wrapping into a huge size_t row_stride. Same underlying
    check (are_strides_row_padded_only in type_utils.h) also guards the
    __array_interface__ path; see the next test."""
    hwc = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    flipped = hwc[::-1]
    assert flipped.strides[0] < 0
    with t.raises(RuntimeError, match="Reversed \\(negative-stride\\) views are not supported"):
        nvimgcodec.as_image(_pure_dlpack_source(flipped, kind))


def test_as_image_array_interface_negative_row_stride_raises():
    """The __array_interface__ path rejects the same reversed/flipped view."""
    hwc = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    flipped = hwc[::-1]
    assert flipped.__array_interface__["strides"][0] < 0
    with t.raises(RuntimeError, match="Reversed \\(negative-stride\\) views are not supported"):
        nvimgcodec.as_image(flipped)


@t.mark.skipif(not TORCH_AVAILABLE, reason="pytorch is not available")
def test_as_image_dlpack_cpu_torch_tensor():
    """A PyTorch CPU tensor (which exposes only __dlpack__ / __dlpack_device__,
    neither __array_interface__ nor __cuda_array_interface__) is wrapped
    instead of raising "Unsupported device in DLTensor"."""
    t_cpu = torch.randint(0, 256, (20, 24, 3), dtype=torch.uint8)
    assert not hasattr(t_cpu, "__array_interface__")
    assert not hasattr(t_cpu, "__cuda_array_interface__")

    img = nvimgcodec.as_image(t_cpu)
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert img.device_id is None
    assert img.shape == (20, 24, 3)
    np.testing.assert_array_equal(np.asarray(img.cpu()), t_cpu.numpy())

    # cuda_stream is meaningless for a CPU exporter; passing one must not raise
    # (PyTorch's own __dlpack__ rejects a non-None stream for CPU tensors, so
    # this also guards against ever passing one through for a CPU source).
    img2 = nvimgcodec.as_image(t_cpu, cuda_stream=1)
    assert img2.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST


@t.mark.skipif(not TORCH_AVAILABLE or not torch.cuda.is_available(),
                reason="pytorch with CUDA support is not available")
def test_as_image_dlpack_cuda_host_torch_tensor():
    """A pinned-host PyTorch tensor reports DLPack device kDLCUDAHost (3) via
    __dlpack_device__(), not kDLCPU - is_cuda_accessible() already treats
    kDLCUDAHost as CUDA-accessible (that classification predates this MR).
    PyTorch's own __dlpack__() rejects a non-None stream for a pinned tensor
    the same way it does for a plain CPU tensor, so wrapping one must not
    raise. Verified: the DLTensor actually embedded in the capsule reports
    kDLCPU regardless (torch::Device for a pinned tensor is still "cpu"), so
    buffer_kind/device_id land the same as the plain-CPU case; only the
    __dlpack_device__()-reported kDLCUDAHost matters, and only for deciding
    whether to pass a stream."""
    t_pinned = torch.randint(0, 256, (20, 24, 3), dtype=torch.uint8).pin_memory()
    assert t_pinned.is_pinned()
    assert not hasattr(t_pinned, "__array_interface__")
    assert not hasattr(t_pinned, "__cuda_array_interface__")
    assert t_pinned.__dlpack_device__()[0] == 3  # kDLCUDAHost

    img = nvimgcodec.as_image(t_pinned)
    assert img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert img.device_id is None
    assert img.shape == (20, 24, 3)
    np.testing.assert_array_equal(np.asarray(img.cpu()), t_pinned.numpy())

    # An explicit cuda_stream must not raise either.
    img2 = nvimgcodec.as_image(t_pinned, cuda_stream=1)
    assert img2.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST


# ---------------------------------------------------------------------------
# Image.num_channels
# ---------------------------------------------------------------------------

@t.mark.parametrize(
    "shape, sample_format, expected_num_channels",
    [
        ((10, 10, 1), None, 1),
        ((10, 10, 2), None, 2),
        ((10, 10, 3), None, 3),
        ((10, 10, 4), None, 4),
        ((10, 10), None, 1),  # 2-D (H, W) -> single channel
        ((3, 10, 10), nvimgcodec.SampleFormat.P_RGB, 3),   # planar: 3 planes x 1 channel each
        ((4, 10, 10), nvimgcodec.SampleFormat.P_RGBA, 4),  # planar: 4 planes x 1 channel each
        ((1, 10, 10), nvimgcodec.SampleFormat.P_Y, 1),     # planar: 1 plane x 1 channel
    ]
)
def test_image_num_channels_host(shape, sample_format, expected_num_channels):
    """Image.num_channels returns total channels across all planes for host images."""
    arr = np.random.randint(0, 255, shape, dtype=np.uint8)
    kwargs = {} if sample_format is None else {"sample_format": sample_format}
    img = nvimgcodec.as_image(arr, **kwargs)
    assert img.num_channels == expected_num_channels


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize(
    "shape, sample_format, expected_num_channels",
    [
        ((10, 10, 1), None, 1),
        ((10, 10, 3), None, 3),
        ((10, 10, 4), None, 4),
        ((3, 10, 10), nvimgcodec.SampleFormat.P_RGB, 3),
    ]
)
def test_image_num_channels_device(shape, sample_format, expected_num_channels):
    """Image.num_channels returns total channels across all planes for device images."""
    arr = cp.random.randint(0, 255, shape, dtype=cp.uint8)
    kwargs = {} if sample_format is None else {"sample_format": sample_format}
    img = nvimgcodec.as_image(arr, **kwargs)
    assert img.num_channels == expected_num_channels


def test_image_num_channels_from_decoded_file():
    """Image.num_channels matches CodeStream.num_channels for a decoded image."""
    input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_410.jpg")
    decoder = nvimgcodec.Decoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY,
        backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
    )
    code_stream = nvimgcodec.CodeStream(input_img_path)
    img = decoder.decode(code_stream)
    assert img.num_channels == code_stream.num_channels
