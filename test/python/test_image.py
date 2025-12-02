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
try:
    import cupy as cp
    img = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8) # Force to load necessary libriaries
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None so it can be referenced in parametrize decorators

import pytest as t
from nvidia import nvimgcodec
import sys
from utils import *
import nvjpeg_test_speedup

def test_image_cpu_exports_to_host():
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)

    ref_img = get_opencv_reference(input_img_path)
    test_image = decoder.read(input_img_path)
    
    host_image = test_image.cpu()

    compare_host_images([host_image], [ref_img])

def test_image_cpu_when_image_is_in_host_mem_returns_the_same_object():
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
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
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
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
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
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
        (3, nvimgcodec.SampleFormat.I_RGBA, 4),  # RGBA needs at least 4 channels, but has 3
        (2, nvimgcodec.SampleFormat.I_RGB, 3),   # I_RGB needs at least 3 channels, but has 2
        (1, nvimgcodec.SampleFormat.I_RGBA, 4),  # I_RGBA needs at least 4 channels, but has 1
        (2, nvimgcodec.SampleFormat.I_CMYK, 4),  # I_CMYK needs at least 4 channels, but has 2
        (1, nvimgcodec.SampleFormat.P_YA, 2),    # P_YA needs at least 2 channels, but has 1
    ]
)
def test_as_image_invalid_sample_format_for_channels_throws(num_channels, invalid_sample_format, min_required_channels):
    """Test that providing sample_format with insufficient channels raises an error."""
    shape = (10, 10, num_channels) if num_channels > 1 else (10, 10)
    ref_img = np.random.randint(0, 255, shape, dtype=np.uint8)
    
    with t.raises(Exception) as excinfo:
        img = nvimgcodec.as_image(ref_img, sample_format=invalid_sample_format)
    
    assert "Invalid sample_format for the number of channels" in str(excinfo.value)
    assert f"requires at least {min_required_channels} channel(s)" in str(excinfo.value)
    assert f"has only {num_channels} channel(s)" in str(excinfo.value)


def test_as_image_more_channels_than_required_is_allowed():
    """Test that having more channels than required by sample_format is allowed."""
    # 4-channel RGBA image with I_RGB format (only needs 3 channels) - should work
    ref_img_4ch = np.random.randint(0, 255, (10, 10, 4), dtype=np.uint8)
    img_rgb = nvimgcodec.as_image(ref_img_4ch, sample_format=nvimgcodec.SampleFormat.I_RGB)
    assert img_rgb.sample_format == nvimgcodec.SampleFormat.I_RGB
    
    # 4-channel image with P_Y format (only needs 1 channel) - should work
    img_y = nvimgcodec.as_image(ref_img_4ch, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img_y.sample_format == nvimgcodec.SampleFormat.P_Y
    
    # 3-channel image with P_Y format (only needs 1 channel) - should work
    ref_img_3ch = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    img_y_3ch = nvimgcodec.as_image(ref_img_3ch, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img_y_3ch.sample_format == nvimgcodec.SampleFormat.P_Y
    
    # 5-channel image with I_RGBA format (only needs 4 channels) - should work
    ref_img_5ch = np.random.randint(0, 255, (10, 10, 5), dtype=np.uint8)
    img_rgba_5ch = nvimgcodec.as_image(ref_img_5ch, sample_format=nvimgcodec.SampleFormat.I_RGBA)
    assert img_rgba_5ch.sample_format == nvimgcodec.SampleFormat.I_RGBA


def test_as_image_flexible_sample_formats_accept_any_channels():
    """Test that UNCHANGED and UNKNOWN sample formats accept any number of channels."""
    # Test with 3 channels using UNCHANGED formats
    ref_img_3ch = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
    
    img_p_unchanged = nvimgcodec.as_image(ref_img_3ch, sample_format=nvimgcodec.SampleFormat.P_UNCHANGED)
    assert img_p_unchanged.sample_format == nvimgcodec.SampleFormat.P_UNCHANGED
    
    img_i_unchanged = nvimgcodec.as_image(ref_img_3ch, sample_format=nvimgcodec.SampleFormat.I_UNCHANGED)
    assert img_i_unchanged.sample_format == nvimgcodec.SampleFormat.I_UNCHANGED
    
    img_unknown = nvimgcodec.as_image(ref_img_3ch, sample_format=nvimgcodec.SampleFormat.UNKNOWN)
    assert img_unknown.sample_format == nvimgcodec.SampleFormat.UNKNOWN
    
    # Test with 5 channels (unusual) using UNKNOWN format
    ref_img_5ch = np.random.randint(0, 255, (10, 10, 5), dtype=np.uint8)
    img_5ch_unknown = nvimgcodec.as_image(ref_img_5ch, sample_format=nvimgcodec.SampleFormat.UNKNOWN)
    assert img_5ch_unknown.sample_format == nvimgcodec.SampleFormat.UNKNOWN


@t.mark.parametrize(
    "num_channels, valid_sample_format",
    [
        (1, nvimgcodec.SampleFormat.P_Y),
        (1, nvimgcodec.SampleFormat.I_Y),
        (2, nvimgcodec.SampleFormat.P_YA),
        (2, nvimgcodec.SampleFormat.I_YA),
        (3, nvimgcodec.SampleFormat.I_RGB),
        (3, nvimgcodec.SampleFormat.I_BGR),
        (3, nvimgcodec.SampleFormat.P_RGB),
        (3, nvimgcodec.SampleFormat.I_YUV),
        (4, nvimgcodec.SampleFormat.I_RGBA),
        (4, nvimgcodec.SampleFormat.P_RGBA),
        (4, nvimgcodec.SampleFormat.I_CMYK),
        (4, nvimgcodec.SampleFormat.I_YCCK),
    ]
)
def test_as_image_valid_sample_format_for_channels_succeeds(num_channels, valid_sample_format):
    """Test that providing compatible sample_format for the number of channels works correctly."""
    shape = (10, 10, num_channels) if num_channels > 1 else (10, 10)
    ref_img = np.random.randint(0, 255, shape, dtype=np.uint8)
    
    # Should not raise an exception
    img = nvimgcodec.as_image(ref_img, sample_format=valid_sample_format)
    assert img.sample_format == valid_sample_format

