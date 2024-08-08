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
import cv2
import cupy as cp
import pytest as t
from nvidia import nvimgcodec
import sys
from utils import *

def test_image_cpu_exports_to_host():
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
    input_img_file = "jpeg/padlock-406986_640_410.jpg"
    input_img_path = os.path.join(img_dir_path, input_img_file)

    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
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
    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR)
    test_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    host_img = nvimgcodec.as_image(test_img)

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

    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

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

    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)  
    
    host_img = nvimgcodec.as_image(ref_img)
    
    assert (host_img.__array_interface__['strides'] == ref_img.__array_interface__['strides'])
    assert (host_img.__array_interface__['shape'] == ref_img.__array_interface__['shape'])
    assert (host_img.__array_interface__['typestr'] == ref_img.__array_interface__['typestr'])
      
    compare_host_images([host_img], [ref_img])

def test_image_buffer_kind():
    input_img_path = os.path.join(
        img_dir_path, "jpeg/padlock-406986_640_410.jpg")

    ref_img = cv2.imread(
        input_img_path, cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)  
    
    host_img = nvimgcodec.as_image(ref_img)
    assert (host_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST)
    
    device_img = host_img.cuda()
    assert (device_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE)
   
