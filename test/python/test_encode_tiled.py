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
import pytest as t

import numpy as np
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None so it can be referenced in parametrize decorators
    
from nvidia import nvimgcodec
from utils import *

def verify_tiled_encoding(encoded_data, expected_tile_width, expected_tile_height, expected_num_tiles_x, expected_num_tiles_y, ref_img):
    """Verify that the encoded data contains the expected tile information and pixel values match."""
    decoder = nvimgcodec.Decoder()
    
    # Create CodeStream from encoded data
    code_stream = nvimgcodec.CodeStream(encoded_data)
    
    # Verify tile information
    # Check tile dimensions
    assert code_stream.tile_width == expected_tile_width
    assert code_stream.tile_height == expected_tile_height
    
    # Check number of tiles
    assert code_stream.num_tiles_x == expected_num_tiles_x
    assert code_stream.num_tiles_y == expected_num_tiles_y
        
    # Decode the image to verify it can be properly decoded
    decoded_img = decoder.decode(code_stream)
    assert decoded_img is not None
    assert decoded_img.shape[0] > 0 and decoded_img.shape[1] > 0
    
    # Convert decoded image to numpy array and compare with reference
    decoded_np = np.asarray(decoded_img.cpu())
    compare_image(decoded_np, ref_img)
    

def calculate_expected_tiles(image_width, image_height, tile_width, tile_height):
    """Calculate expected number of tiles based on image dimensions and tile size."""
    if tile_width <= 0 or tile_height <= 0:
        # When no tiling is specified, the entire image is treated as one tile
        return image_width, image_height, 1, 1
    
    num_tiles_x = (image_width + tile_width - 1) // tile_width
    num_tiles_y = (image_height + tile_height - 1) // tile_height
    
    return tile_width, tile_height, num_tiles_x, num_tiles_y

@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("backends", [[nvimgcodec.BackendKind.GPU_ONLY]])
@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("ht", [True, False])
@t.mark.parametrize("tile_width", [0, 64, 128, 256])
@t.mark.parametrize("tile_height", [0, 64, 128, 256])
@t.mark.parametrize(
    "input_img_file",
    [
        "bmp/cat-111793_640.bmp",
        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg2k/cat-111793_640.jp2",
    ]
)
def test_encode_tiled_jpeg2k(tmp_path, input_img_file, tile_width, tile_height, max_num_cpu_threads, backends, ht):
    """Test tiled encoding with JPEG2000 codec."""
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=max_num_cpu_threads, backends=backends)

    input_img_path = os.path.join(img_dir_path, input_img_file)
    ref_img = get_opencv_reference(input_img_path)
    cp_ref_img = cp.asarray(ref_img)

    nv_ref_img = nvimgcodec.as_image(cp_ref_img)
    
    # Create JPEG2000 encode parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_encode_params.ht = ht  
    
    # Create encode parameters with tiling
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=jpeg2k_encode_params,
        tile_width=tile_width,
        tile_height=tile_height
    )
    
    # Check if tile size is valid for the image
    image_width, image_height = ref_img.shape[1], ref_img.shape[0]
    if tile_width > 0 and tile_height > 0 and (tile_width > image_width or tile_height > image_height):
        # Skip this test case as tile size is larger than image
        t.skip(f"Tile size {tile_width}x{tile_height} is larger than image size {image_width}x{image_height}")
    
    test_encoded_img = encoder.encode(
        nv_ref_img, codec="jpeg2k", params=encode_params)

    # Verify that encoding was successful
    assert test_encoded_img is not None
    assert test_encoded_img.size > 0
        

    # Calculate expected tile information
    expected_tile_width, expected_tile_height, expected_num_tiles_x, expected_num_tiles_y = calculate_expected_tiles(
        ref_img.shape[1], ref_img.shape[0], tile_width, tile_height)
    
    # Verify tiling information in the encoded data
    verify_tiled_encoding(test_encoded_img, expected_tile_width, expected_tile_height, expected_num_tiles_x, expected_num_tiles_y, ref_img)

@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize("backends", [[nvimgcodec.BackendKind.GPU_ONLY]])
@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("ht", [True, False])
@t.mark.parametrize(
    "input_images_batch",
    [
        ("bmp/cat-111793_640.bmp",
         "jpeg/padlock-406986_640_410.jpg",
         "jpeg/padlock-406986_640_gray.jpg",
         "jpeg2k/cat-111793_640.jp2",)
    ]
)

def test_encode_tiled_jpeg2k_batch(tmp_path, input_images_batch, max_num_cpu_threads, backends, ht):
    """Test batch tiled encoding with JPEG2000 codec."""
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=max_num_cpu_threads, backends=backends)

    input_img_paths = [os.path.join(img_dir_path, input_img_file) for input_img_file in input_images_batch]
    ref_imgs = [get_opencv_reference(input_img_path) for input_img_path in input_img_paths]
    cp_ref_imgs = [cp.asarray(ref_img) for ref_img in ref_imgs]

    nv_ref_imgs = [nvimgcodec.as_image(cp_ref_img) for cp_ref_img in cp_ref_imgs]
    
    # Create JPEG2000 encode parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    jpeg2k_encode_params.ht = ht  # Enable/disable HTJ2K based on parameter
    
    # Create encode parameters with tiling
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=jpeg2k_encode_params,
        tile_width=256,
        tile_height=256
    )
    
    test_encoded_imgs = encoder.encode(
        nv_ref_imgs, codec="jpeg2k", params=encode_params)

    # Verify that encoding was successful
    assert test_encoded_imgs is not None
    assert len(test_encoded_imgs) == len(input_images_batch)
    for encoded_img in test_encoded_imgs:
        assert encoded_img is not None
        assert encoded_img.size > 0

    # Verify tiling information for each encoded image
    for i, (encoded_img, ref_img) in enumerate(zip(test_encoded_imgs, ref_imgs)):
        expected_tile_width, expected_tile_height, expected_num_tiles_x, expected_num_tiles_y = calculate_expected_tiles(
            ref_img.shape[1], ref_img.shape[0], 256, 256)
        
        verify_tiled_encoding(encoded_img, expected_tile_width, expected_tile_height, expected_num_tiles_x, expected_num_tiles_y, ref_img)

def test_encode_tiled_jpeg2k_property_setters():
    """Test setting tile properties after EncodeParams creation."""
    # Create JPEG2000 encode parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    
    # Create encode parameters without tiling
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=jpeg2k_encode_params
    )
    
    # Verify default values
    assert encode_params.tile_width == 0
    assert encode_params.tile_height == 0
    
    # Set tile properties
    encode_params.tile_width = 128
    encode_params.tile_height = 256
    
    # Verify properties are set correctly
    assert encode_params.tile_width == 128
    assert encode_params.tile_height == 256

def test_encode_tiled_jpeg2k_constructor_parameters():
    """Test setting tile parameters in EncodeParams constructor."""
    # Create JPEG2000 encode parameters
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    
    # Create encode parameters with tiling in constructor
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=jpeg2k_encode_params,
        tile_width=512,
        tile_height=512
    )
    
    # Verify tile parameters are set correctly
    assert encode_params.tile_width == 512
    assert encode_params.tile_height == 512


def test_encode_tiled_jpeg2k_exceed_tile_limit():
    """Test encoding JPEG2000 with number of tiles exceeding the JPEG2000 limit (65535)."""
    # Image dimensions and tile size chosen to exceed 65535 tiles
    image_width = 4096
    image_height = 4096
    tile_width = 8
    tile_height = 8

    # Calculate expected number of tiles
    num_tiles_x = (image_width + tile_width - 1) // tile_width
    num_tiles_y = (image_height + tile_height - 1) // tile_height
    total_tiles = num_tiles_x * num_tiles_y
    assert total_tiles > 65535, "Test setup error: total tiles does not exceed JPEG2000 limit"

    # Create a dummy image
    arr = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)

    # Set up encode parameters with excessive tiling
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=jpeg2k_encode_params,
        tile_width=tile_width,
        tile_height=tile_height
    )

    encoder = nvimgcodec.Encoder()
    encoded_data = encoder.encode(arr, codec="jpeg2k", params=encode_params)

    # The encoder should return None
    assert encoded_data is None, "Encoder did not return None as expected when exceeding tile limit"



@t.mark.parametrize("codec", ["bmp", "jpeg", "tiff", "pnm", "png", "webp"])
def test_encode_tiled_not_supported_codecs_return_none(codec):
    """
    Test that encoding with tiling for codecs that do not support tiling returns None.
    """

    # Create a dummy image
    image_width = 512
    image_height = 512
    arr = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)

    # Set up encode parameters with tiling
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        tile_width=128,
        tile_height=128
    )

    encoder = nvimgcodec.Encoder()
    encoded_data = encoder.encode(arr, codec=codec, params=encode_params)

    assert encoded_data is None, f"Encoder for codec '{codec}' did not return None when tiling is not supported"


def test_encode_tiled_jpeg2k_cpu_backend_returns_none():
    """
    Test that encoding with tiling and JPEG2000 codec on CPU-only backend returns None,
    since it should fallback to OpenCV encoder which does not support tiling.
    """
    # Create a dummy image
    image_width = 512
    image_height = 512
    arr = np.random.randint(0, 256, (image_height, image_width, 3), dtype=np.uint8)

    # Set up encode parameters with tiling
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams()
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=jpeg2k_encode_params,
        tile_width=128,
        tile_height=128
    )

    # Force CPU-only backend 
    encoder = nvimgcodec.Encoder(backends=[nvimgcodec.BackendKind.CPU_ONLY])
    
    encoded_data = encoder.encode(arr, codec="jpeg2k", params=encode_params)

    assert encoded_data is None, (
        "Encoder did not return None as expected when encoding tiled JPEG2000 with CPU-only backend"
    )
