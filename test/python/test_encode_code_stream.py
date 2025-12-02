# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys
import pytest as t
import numpy as np
    
from nvidia import nvimgcodec
from utils import *
import nvjpeg_test_speedup

@t.mark.parametrize("codec", ["jpeg2k", "jpeg", "tiff"])
@t.mark.parametrize("sample_type", [np.uint8, np.uint16])
def test_encode_to_code_stream(codec, sample_type):
    """Test encoding to CodeStream directly."""
    if sample_type == np.uint16 and codec == "jpeg":
        t.skip("JPEG does not support 16-bit images")
        return
    
    encoder = nvimgcodec.Encoder()
    
    # Create a simple test image
    arr = np.zeros((32, 32, 3), dtype=sample_type) + 128

    nv_img = nvimgcodec.as_image(arr)
    
    # Encode to CodeStream
    encoded_code_stream = encoder.encode(nv_img, codec=codec)
    assert isinstance(encoded_code_stream, nvimgcodec.CodeStream)
    assert encoded_code_stream.width == 32
    assert encoded_code_stream.height == 32
    #assert encoded_code_stream.channels == 3 #TODO: fix code stream bug for channels
    assert encoded_code_stream.codec_name == codec
    assert encoded_code_stream.size > 0
    assert encoded_code_stream.dtype == sample_type


def test_encode_batch_to_code_stream():
    """Test batch encoding to CodeStream directly."""
    encoder = nvimgcodec.Encoder()
    
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]
    
    # Encode to CodeStream batch
    encoded_code_stream_list = encoder.encode(nv_imgs, codec="jpeg2k")
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 32
    assert encoded_code_stream_list[0].height == 32
    assert encoded_code_stream_list[1].width == 64
    assert encoded_code_stream_list[1].height == 64
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "jpeg2k"
    assert encoded_code_stream_list[1].codec_name == "jpeg2k"

@t.mark.parametrize("pin_memory", [True, False])
@t.mark.parametrize("preallocation_size", [1024, 2048, 4096])
def test_encode_with_code_stream_reuse(pin_memory, preallocation_size):
    """Test reusing a CodeStream for multiple encodings."""
    encoder = nvimgcodec.Encoder()
    
    # Create CodeStream with preallocation
    code_stream = nvimgcodec.CodeStream(preallocation_size, pin_memory=pin_memory)
    
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    nv_img1 = nvimgcodec.as_image(arr1)

    code_stream = encoder.encode(nv_img1, codec="jpeg2k", code_stream_s=code_stream)
    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 32
    assert code_stream.height == 32
    assert code_stream.size > 0
    assert code_stream.codec_name == "jpeg2k"
    assert code_stream.size <= preallocation_size
    assert code_stream.capacity == preallocation_size

def test_encode_with_preallocated_code_stream_capacity_grows_for_large_image():
    """Test that CodeStream capacity grows if input image is larger than preallocated size."""
    encoder = nvimgcodec.Encoder()
    preallocation_size = 1024
    code_stream = nvimgcodec.CodeStream(preallocation_size, pin_memory=True)

    # Create a large image that will require more than preallocated size
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    nv_img = nvimgcodec.as_image(arr)

    # Encode the large image
    code_stream = encoder.encode(nv_img, codec="jpeg2k", code_stream_s=code_stream)

    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 512
    assert code_stream.height == 512
    assert code_stream.size > 0
    assert code_stream.size <= code_stream.capacity
    # The capacity should have grown to accommodate the encoded image
    assert code_stream.capacity > preallocation_size

@t.mark.parametrize("pin_memory", [True, False])
@t.mark.parametrize("preallocation_size", [1024, 2048, 4096])
def test_encode_with_code_stream_reuse_batch(pin_memory, preallocation_size):
    """Test reusing a CodeStream for multiple encodings in batch."""
    encoder = nvimgcodec.Encoder()

    # Create CodeStream with preallocation
    code_streams = [nvimgcodec.CodeStream(preallocation_size, pin_memory=pin_memory) for _ in range(2)]

    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]

    # Encode batch to CodeStream
    encoded_code_stream_list = encoder.encode(nv_imgs, codec="jpeg2k", code_stream_s=code_streams)
    assert isinstance(encoded_code_stream_list, list)
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 32
    assert encoded_code_stream_list[0].height == 32
    assert encoded_code_stream_list[1].width == 64
    assert encoded_code_stream_list[1].height == 64
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "jpeg2k"
    assert encoded_code_stream_list[1].codec_name == "jpeg2k"
    assert encoded_code_stream_list[0].size <= preallocation_size
    assert encoded_code_stream_list[1].size <= preallocation_size
    assert encoded_code_stream_list[0].capacity == preallocation_size
    assert encoded_code_stream_list[1].capacity == preallocation_size

def test_encode_with_code_stream_reuse_batch_insufficient_code_streams():
    """Test that encoding a batch with fewer CodeStreams than images raises an exception."""
    encoder = nvimgcodec.Encoder()

    # Only one CodeStream for two images
    code_streams = [nvimgcodec.CodeStream(1024, pin_memory=True)]

    # Create two test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]

    # Should raise an exception due to insufficient CodeStreams
    with t.raises(Exception) as e:
        encoder.encode(nv_imgs, codec="jpeg2k", code_stream_s=code_streams)
        
    assert (isinstance(e.value, ValueError) )



def test_encode_with_previous_encode_result_code_stream_reuse():
    """Test reusing a CodeStream for multiple encodings with previous encode result."""
    encoder = nvimgcodec.Encoder()
        
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64

    nv_img1 = nvimgcodec.as_image(arr1)
    nv_img2 = nvimgcodec.as_image(arr2)
    
    # Encode first image
    code_stream = encoder.encode(nv_img1, codec="jpeg2k")
    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 32
    assert code_stream.height == 32
    assert code_stream.size > 0
    assert code_stream.codec_name == "jpeg2k"
    
    # Encode second image to the same code stream previously created
    code_stream = encoder.encode(nv_img2, codec="jpeg", code_stream_s=code_stream)
    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 64
    assert code_stream.height == 64
    assert code_stream.size > 0
    assert code_stream.codec_name == "jpeg"
    assert code_stream.size == code_stream.size

def test_encode_with_previous_encode_result_code_stream_reuse_batch():
    """Test reusing a CodeStream for multiple encodings in batch with previous encode result."""
    encoder = nvimgcodec.Encoder()
    
    # Create CodeStream with preallocation
    code_stream = nvimgcodec.CodeStream(1024, pin_memory=True)
    
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs_batch1 = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]
    
    arr3 = np.zeros((128, 128, 3), dtype=np.uint8) + 128
    arr4 = np.zeros((256, 256, 3), dtype=np.uint8) + 64
    nv_imgs_batch2 = [nvimgcodec.as_image(arr3), nvimgcodec.as_image(arr4)]
    
    # Encode to CodeStream batch
    encoded_code_stream_list = encoder.encode(nv_imgs_batch1, codec="jpeg2k")
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 32
    assert encoded_code_stream_list[0].height == 32
    assert encoded_code_stream_list[1].width == 64
    assert encoded_code_stream_list[1].height == 64
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "jpeg2k"
    assert encoded_code_stream_list[1].codec_name == "jpeg2k"
    
    # Encode to CodeStream batch
    encoded_code_stream_list = encoder.encode(nv_imgs_batch2, codec="tiff", code_stream_s = encoded_code_stream_list)
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 128
    assert encoded_code_stream_list[1].width == 256
    assert encoded_code_stream_list[0].height == 128
    assert encoded_code_stream_list[1].height == 256
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "tiff"
    assert encoded_code_stream_list[1].codec_name == "tiff"
    
def test_encode_returns_code_stream_with_buffer_protocol():
    """Test that CodeStream implements buffer protocol correctly."""
    encoder = nvimgcodec.Encoder()
    
    # Create test image
    arr = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    nv_img = nvimgcodec.as_image(arr)
    
    # Encode to CodeStream
    encoded_code_stream = encoder.encode(nv_img, codec="jpeg2k")
    
    # Test buffer protocol
    buffer = memoryview(encoded_code_stream)
    assert len(buffer) > 0










