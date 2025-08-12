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
    assert (sys.getrefcount(host_image) == 2)

    host_image_2 = host_image.cpu()

    assert (sys.getrefcount(host_image) == 3)
    assert (sys.getrefcount(host_image_2) == 3)

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
    assert (sys.getrefcount(device_img) == 2)

    device_img_2 = device_img.cuda()

    assert (sys.getrefcount(device_img) == 3)
    assert (sys.getrefcount(device_img_2) == 3)


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
    
def test_image_array_interface_import_three_dimensions():
    ref_img = np.zeros((1, 2, 3))
    host_img = nvimgcodec.as_image(ref_img)
    assert(host_img.shape[0] == 1)
    assert(host_img.shape[1] == 2)
    assert(host_img.shape[2] == 3)

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

@t.mark.parametrize(
    "num_channels",
    [
        0,
        2,
        4, 
        5,
    ]
)
def test_image_array_interface_import_not_accepted_number_of_channels_throws(num_channels):
    ref_img = np.zeros((1, 2, num_channels))
    with t.raises(Exception) as excinfo:
        host_img = nvimgcodec.as_image(ref_img)
    assert (str(excinfo.value) == "Unexpected number of channels. Only 3 channels for RGB or 1 channel for gray scale are accepted.")

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
