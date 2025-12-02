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
import sys
import pytest as t

import numpy as np
try:
    import cupy as cp
    cuda_streams = [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)]
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    cuda_streams = []
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None so it can be referenced in parametrize decorators
from nvidia import nvimgcodec
from utils import *
import nvjpeg_test_speedup

@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("cuda_stream", cuda_streams)
@t.mark.parametrize("encode_to_data", [True, False])
@t.mark.parametrize(
    "input_img_file",
    [
        "bmp/cat-111793_640.bmp",

        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_411.jpg",
        "jpeg/padlock-406986_640_420.jpg",
        "jpeg/padlock-406986_640_422.jpg",
        "jpeg/padlock-406986_640_440.jpg",
        "jpeg/padlock-406986_640_444.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg/ycck_colorspace.jpg",
        "jpeg/cmyk.jpg",
        "jpeg/cmyk-dali.jpg",
        "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

        "jpeg/exif/padlock-406986_640_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
        "jpeg/exif/padlock-406986_640_no_orientation.jpg",
        "jpeg/exif/padlock-406986_640_rotate_180.jpg",
        "jpeg/exif/padlock-406986_640_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",

        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-111793_640.jp2",
        "jpeg2k/tiled-cat-1046544_640.jp2",
        "jpeg2k/tiled-cat-111793_640.jp2",
        "jpeg2k/cat-111793_640-16bit.jp2",
        "jpeg2k/cat-1245673_640-12bit.jp2",
    ]
)
@t.mark.parametrize("file_ext", [".png", ".jp2"])
def test_encode_single_image(tmp_path, input_img_file, encode_to_data, cuda_stream, max_num_cpu_threads, file_ext):
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=max_num_cpu_threads)

    input_img_path = os.path.join(img_dir_path, input_img_file)
    ref_img = get_opencv_reference(input_img_path)
    cp_ref_img = cp.asarray(ref_img)

    nv_ref_img = nvimgcodec.as_image(cp_ref_img)
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    
    if encode_to_data:
        if cuda_stream:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec=file_ext, params = encode_params, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec=file_ext, params = encode_params)
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + file_ext)
        if cuda_stream:
            encoder.write(output_img_path, nv_ref_img,
                          params=encode_params, cuda_stream=cuda_stream.ptr)
        else:
            encoder.write(output_img_path, nv_ref_img,
                          params=encode_params)
        with open(output_img_path, 'rb') as in_file:
            test_encoded_img = in_file.read()

    test_img = get_opencv_reference(np.asarray(bytearray(test_encoded_img)))
    compare_image(np.asarray(test_img), np.asarray(ref_img))


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("cuda_stream", cuda_streams)
@t.mark.parametrize("encode_to_data", [True, False])
@t.mark.parametrize(
    "input_images_batch",
    [
        ("bmp/cat-111793_640.bmp",

         "jpeg/padlock-406986_640_410.jpg",
         "jpeg/padlock-406986_640_411.jpg",
         "jpeg/padlock-406986_640_420.jpg",
         "jpeg/padlock-406986_640_422.jpg",
         "jpeg/padlock-406986_640_440.jpg",
         "jpeg/padlock-406986_640_444.jpg",
         "jpeg/padlock-406986_640_gray.jpg",
         "jpeg/ycck_colorspace.jpg",
         "jpeg/cmyk.jpg",
         "jpeg/cmyk-dali.jpg",
         "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

         "jpeg/exif/padlock-406986_640_horizontal.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
         "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
         "jpeg/exif/padlock-406986_640_no_orientation.jpg",
         "jpeg/exif/padlock-406986_640_rotate_180.jpg",
         "jpeg/exif/padlock-406986_640_rotate_270.jpg",
         "jpeg/exif/padlock-406986_640_rotate_90.jpg",

         "jpeg2k/cat-1046544_640.jp2",
         "jpeg2k/cat-1046544_640.jp2",
         "jpeg2k/cat-111793_640.jp2",
         "jpeg2k/tiled-cat-1046544_640.jp2",
         "jpeg2k/tiled-cat-111793_640.jp2",
         "jpeg2k/cat-111793_640-16bit.jp2",
         "jpeg2k/cat-1245673_640-12bit.jp2",)
    ]
)

def test_encode_batch_image(tmp_path, input_images_batch, encode_to_data, cuda_stream, max_num_cpu_threads):
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=max_num_cpu_threads)
    input_images = [os.path.join(img_dir_path, img) for img in input_images_batch]
    ref_images = [get_opencv_reference(img) for img in input_images]
    cp_ref_images = [cp.asarray(ref_img) for ref_img in ref_images]
    nv_ref_images = [nvimgcodec.as_image(cp_ref_img) for cp_ref_img in cp_ref_images]

    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)

    if encode_to_data:
        if cuda_stream:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=encode_params, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=encode_params)
    else:
        output_img_paths = [os.path.join(tmp_path, os.path.splitext(
            os.path.basename(img))[0] + ".jp2") for img in input_images]
        if cuda_stream:
            encoder.write(output_img_paths, nv_ref_images,
                          params=encode_params, cuda_stream=cuda_stream.ptr)
        else:
            encoder.write(output_img_paths, nv_ref_images,
                          params=encode_params)
        test_encoded_images = []
        for out_img_path in output_img_paths:
            with open(out_img_path, 'rb') as in_file:
                test_encoded_img = in_file.read()
                test_encoded_images.append(test_encoded_img)

    test_decoded_images = [get_opencv_reference(np.asarray(bytearray(img))) for img in test_encoded_images]
    compare_host_images(test_decoded_images, ref_images)

def test_encode_jpeg2k_small_image():
    encoder = nvimgcodec.Encoder()
    arr = np.zeros((5,5,3), dtype=np.uint8) + 128
    encoded_image = encoder.encode(arr, codec="jpeg2k")
    decoder = nvimgcodec.Decoder()
    arr2 = decoder.decode(encoded_image).cpu()
    np.testing.assert_array_almost_equal(arr, arr2)

def test_encode_jpeg2k_2d():
    encoder = nvimgcodec.Encoder()
    arr = np.zeros((32,32), dtype=np.uint8) + 128
    encoded_image = encoder.encode(arr, codec="jpeg2k")
    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    arr2 = np.array(decoder.decode(encoded_image, params=params).cpu()).squeeze()
    np.testing.assert_array_almost_equal(arr, arr2)


def test_encode_jpeg2k_uint16():
    arr = np.zeros((256,256,3), dtype=np.uint16) + np.uint16(0.9 * np.iinfo(np.uint16).max)
    arr[100:120, 200:210, 0] = np.uint16(0.1 * np.iinfo(np.uint16).max)
    arr[100:120, 200:210, 1] = np.uint16(0.6 * np.iinfo(np.uint16).max)
    arr[100:120, 200:210, 2] = np.uint16(0.4 * np.iinfo(np.uint16).max)

    encoder = nvimgcodec.Encoder()
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded_image = encoder.encode(arr, codec="jpeg2k", params=encode_params)

    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB, allow_any_depth=True)
    arr2 = np.array(decoder.decode(encoded_image, params=params).cpu())

    np.testing.assert_array_equal(arr, arr2)


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
@t.mark.parametrize(
    "input_images_batch",
    [("bmp/cat-111793_640.bmp",

      "jpeg/padlock-406986_640_410.jpg",
      "jpeg/padlock-406986_640_411.jpg",
      "jpeg/padlock-406986_640_420.jpg",
      "jpeg/padlock-406986_640_422.jpg",
      "jpeg/padlock-406986_640_440.jpg",
      "jpeg/padlock-406986_640_444.jpg",
      "jpeg/padlock-406986_640_gray.jpg",
      "jpeg/cmyk-dali.jpg",
      "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

      "jpeg/exif/padlock-406986_640_horizontal.jpg",
      "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
      "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
      "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
      "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
      "jpeg/exif/padlock-406986_640_no_orientation.jpg",
      "jpeg/exif/padlock-406986_640_rotate_180.jpg",
      "jpeg/exif/padlock-406986_640_rotate_270.jpg",
      "jpeg/exif/padlock-406986_640_rotate_90.jpg",

      "jpeg2k/cat-1046544_640.jp2",
      "jpeg2k/cat-1046544_640.jp2",
      "jpeg2k/cat-111793_640.jp2",
      "jpeg2k/tiled-cat-1046544_640.jp2",
      "jpeg2k/tiled-cat-111793_640.jp2",
      "jpeg2k/cat-111793_640-16bit.jp2",
      "jpeg2k/cat-1245673_640-12bit.jp2")
     ]
)

def test_encode_with_as_images_from_cuda_array_interface(input_images_batch):
    input_images = [os.path.join(img_dir_path, img) for img in input_images_batch]
    ref_images = [get_opencv_reference(img) for img in input_images]
    cp_ref_images = [cp.asarray(ref_img) for ref_img in ref_images]
    nv_ref_images = nvimgcodec.as_images(cp_ref_images)
    encoder = nvimgcodec.Encoder()
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    test_encoded_images = encoder.encode(nv_ref_images, codec="jpeg2k", params=encode_params)
    test_decoded_images = [get_opencv_reference(np.asarray(bytearray(img)))
                           for img in test_encoded_images]

    compare_host_images(test_decoded_images, ref_images)

@t.mark.parametrize(
    "input_image",
    [
        "jpeg/padlock-406986_640_420.jpg",
    ]
)
def test_encode_images_with_hardware_backend(input_image):
    # Read image and decode
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, input_image)
    decoder = nvimgcodec.Decoder()
    original_img = decoder.read(fname).cpu()
    
    # Encode using hardware engine
    hw_backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY)]
    try:
        hw_encoder = nvimgcodec.Encoder(backends=hw_backends)
    except:
        t.skip(f"nvJPEG hardware encoder is not supported on this platform or failed for {input_image}")
    # It's important to pass chroma subsampling 420, otherwise the default chroma subsampling (444) is not supported by hardware
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=100, chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420)
    encoded_img = hw_encoder.encode(original_img, codec="jpeg", params=encode_params)
        
    # Decode then compare with reference
    decoded_img = decoder.read(np.asarray(bytearray(encoded_img))).cpu()
    decoded_np = np.asarray(decoded_img)
    ref_img = get_opencv_reference(fname)
    ref_np = np.asarray(ref_img)
    compare_host_images(decoded_np, ref_np, 50)

def test_encode_jpeg_gray():
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, 'bmp/cat-111793_640_grayscale.bmp')
    backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY), 
                nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
                nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]
    decoder = nvimgcodec.Decoder(backends=backends)
    params1 = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY, allow_any_depth=True)
    arr = np.array(decoder.read(fname, params=params1).cpu())
    assert arr.shape[-1] == 1
    encoder = nvimgcodec.Encoder()
    arr2 = encoder.encode(arr, codec="jpeg")
    params3 = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)
    arr3 = np.array(decoder.decode(arr2, params=params3).cpu())
    assert arr3.shape == arr.shape, f"{arr3.shape} != {arr.shape}"
    ref = get_opencv_reference(np.asarray(bytearray(arr2)), nvimgcodec.ColorSpec.GRAY)
    np.testing.assert_allclose(ref, arr3, atol=1)

@t.mark.parametrize("codec,file_ext", [("jpeg2k", ".jp2"), ("tiff", ".tiff")])
def test_encode_grayscale_with_default_encode_params(codec, file_ext):
    """
    Test that grayscale images can be encoded with default EncodeParams (chroma_subsampling=None).
    This verifies that chroma_subsampling defaults to GRAY for single-channel images
    and to CSS_444 for multi-channel images when not explicitly specified.
    """
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, 'bmp/cat-111793_640_grayscale.bmp')
    
    backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)]
    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder(backends=backends)
    
    # Decode as grayscale (1 channel)
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY, allow_any_depth=True)
    gray_img = decoder.read(fname, params=params).cpu()
    arr = np.array(gray_img)
    assert arr.shape[-1] == 1, f"Expected 1 channel, got {arr.shape[-1]}"
    
    # Encode with default EncodeParams (chroma_subsampling defaults to None)
    # This should automatically use GRAY for single-channel images
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded = encoder.encode(gray_img, codec=codec, params=encode_params)
    
    assert encoded is not None, f"Failed to encode grayscale image as {codec}"
    assert encoded.size > 0, f"Encoded {codec} data is empty"
    
    # Decode back and verify
    decoded_img = decoder.decode(encoded, params=params).cpu()
    decoded_arr = np.array(decoded_img)
    
    # For lossless, the images should be identical
    np.testing.assert_array_equal(decoded_arr, arr)

def test_encode_explicit_chroma_subsampling_override():
    """
    Test that explicit chroma_subsampling parameter is respected and overrides the default.
    """
    fname = os.path.join(img_dir_path, 'bmp/cat-111793_640.bmp')
    
    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder()
    
    # Decode as RGB (3 channels)
    rgb_img = decoder.read(fname).cpu()
    assert rgb_img.shape[-1] == 3, f"Expected 3 channels, got {rgb_img.shape[-1]}"

    # Encode with explicit chroma_subsampling
    enc_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.QUALITY,
        quality_value = 5, 
        chroma_subsampling = nvimgcodec.ChromaSubsampling.CSS_GRAY)
    encoded = encoder.encode(rgb_img, codec="jpeg", params=enc_params)
    
    assert encoded is not None, f"Failed to encode image as jpeg with CSS_GRAY"
    assert encoded.size > 0, f"Encoded jpeg data is empty"

@t.mark.parametrize("encode_to_data", [True, False])
def test_encode_single_image_with_unsupported_codec_returns_none(tmp_path, encode_to_data):
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, "bmp/cat-111793_640.bmp")
    decoder = nvimgcodec.Decoder()
    img = decoder.read(fname).cpu()
    
    backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)] # we do not have GPU webp encoder and we use it here for testing unsupported codec
    encoder = nvimgcodec.Encoder(backends=backends)
    
    if encode_to_data:
        encoded_img = encoder.encode(img, codec="webp")
        assert(encoded_img == None)
    else:
        encoded_file = encoder.write(os.path.join(tmp_path,  "bad.jpeg"), img)
        assert(encoded_file == None)
        
@t.mark.parametrize("encode_to_data", [True, False])
def test_encode_batch_with_unsupported_images_returns_none_on_corresponding_positions(tmp_path, encode_to_data):
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, "bmp/cat-111793_640.bmp")
    decoder = nvimgcodec.Decoder()
    np_img = decoder.read(fname).cpu()

    unsupported_img  = np.random.rand(10, 10, 4)

    images = [np_img, unsupported_img, np_img]
    
    encoder = nvimgcodec.Encoder()
    
    if encode_to_data:
        encoded_imgs = encoder.encode(images, codec="jpeg")
        assert(len(encoded_imgs) == len(images))
        assert(encoded_imgs[0] != None)
        assert(encoded_imgs[1] == None)
        assert(encoded_imgs[2] != None)
    else:
        output_img_paths = [
            os.path.join(tmp_path,  "ok0.jpeg"),
            os.path.join(tmp_path,  "bad1.jpeg"), 
            os.path.join(tmp_path,  "ok2.jpeg")]
        
        encoded_files = encoder.write(output_img_paths, images)
        assert(encoded_files[0] == output_img_paths[0])
        assert(encoded_files[1] == None)
        assert(encoded_files[2] == output_img_paths[2])

def test_encode_batch_with_size_mismatch_throws(tmp_path):
    img  = np.random.rand(10, 10, 3)
    images = [img, img, img]
    output_img_paths = [
        os.path.join(tmp_path,  "ok0.jpeg"),
        os.path.join(tmp_path,  "ok1.jpeg")]
    encoder = nvimgcodec.Encoder()
    
    with t.raises(Exception) as excinfo:
        encoder.write(output_img_paths, images)
    assert (str(excinfo.value) == "Size mismatch - filenames list has 2 items, but images list has 3 items.")
 
def test_encode_batch_with_unspecified_codec_throws():
    img  = np.random.rand(10, 10, 3)
    images = [img, img, img]
    encoder = nvimgcodec.Encoder()

    with t.raises(Exception) as excinfo:
        encoder.encode(images, codec="")
    assert (str(excinfo.value) == "Unspecified codec.")

def test_encode_single_image_with_unsupported_codec_throws():
    img  = np.random.rand(10, 10, 3)
    encoder = nvimgcodec.Encoder()
    
    with t.raises(Exception) as excinfo:
        encoder.encode(img, codec=".jxr")
    assert (str(excinfo.value) == "Unsupported codec.")

def test_encode_unsupported_image_returns_none():
    def gen_img(shape):
        return np.random.randint(0, 255, shape, np.uint8)

    img  = gen_img((10, 10, 5)) # only 1, 3 or 4 channels are supported
    encoder = nvimgcodec.Encoder()

    res = encoder.encode(img, codec=".jpeg")
    assert res is None

    img2  = gen_img((10, 13, 5)) # only 1, 3 or 4 channels are supported
    img3  = gen_img((15, 10, 5)) # only 1, 3 or 4 channels are supported
    res_list = encoder.encode([img, img2, img3], codec=".jpeg")
    for res in res_list:
        assert res is None

    valid_img = gen_img((10, 10, 3))
    valid_img2 = gen_img((20, 20, 4))
    res_list = encoder.encode([img, valid_img, img2, valid_img2], codec=".png") # use png for lossless
    assert res_list[0] is None
    assert res_list[1] is not None
    assert res_list[2] is None
    assert res_list[3] is not None

    decoder = nvimgcodec.Decoder()
    dec_1, dec_2 = decoder.decode([res_list[1], res_list[3]], params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED))
    np.testing.assert_array_equal(dec_1.cpu(), valid_img)
    np.testing.assert_array_equal(dec_2.cpu(), valid_img2)

def test_encode_none():
    encoder = nvimgcodec.Encoder()

    assert encoder.encode(None, codec=".jpeg") is None

    res = encoder.encode([], codec=".jpeg")
    assert len(res) == 0

    res = encoder.encode([None], codec=".jpeg")
    assert len(res) == 1
    assert res[0] is None

    res = encoder.encode([None, None], codec=".jpeg")
    assert len(res) == 2
    assert res[0] is None
    assert res[1] is None

def test_encode_unsupported_codec():
    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder()

    nv_img_jpg = decoder.read(os.path.join(img_dir_path, "bmp/cat-111793_640.bmp"))
    assert nv_img_jpg is not None

    assert encoder.encode(nv_img_jpg, "wrong_codec") is None

@t.mark.parametrize("cuda_stream", cuda_streams)
@t.mark.parametrize("encode_to_data", [True, False])
@t.mark.parametrize(
    "input_img_file",
    [
        "bmp/cat-111793_640.bmp",
        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg2k/cat-111793_640.jp2",
        "jpeg2k/cat-111793_640-16bit.jp2",
    ]
)
def test_encode_nvtiff(tmp_path, input_img_file, encode_to_data, cuda_stream):
    backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)]
    encoder = nvimgcodec.Encoder(backends=backends, max_num_cpu_threads=1)

    input_img_path = os.path.join(img_dir_path, input_img_file)
    ref_img = get_opencv_reference(input_img_path)
    cp_ref_img = cp.asarray(ref_img)

    nv_ref_img = nvimgcodec.as_image(cp_ref_img)
    
    if encode_to_data:
        if cuda_stream:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="tiff", params=None, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="tiff", params=None)
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + ".tiff")
        if cuda_stream:
            encoder.write(output_img_path, nv_ref_img, params=None, cuda_stream=cuda_stream.ptr)
        else:
            encoder.write(output_img_path, nv_ref_img, params=None)
        with open(output_img_path, 'rb') as in_file:
            test_encoded_img = in_file.read()

    test_img = get_opencv_reference(np.asarray(bytearray(test_encoded_img)))
    np.testing.assert_array_equal(test_img, ref_img)

@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("cuda_stream", cuda_streams)
@t.mark.parametrize("encode_to_data", [True, False])
@t.mark.parametrize(
    "input_images_batch",
    [
        ("bmp/cat-111793_640.bmp",

         "jpeg/padlock-406986_640_410.jpg",
         "jpeg/padlock-406986_640_411.jpg",
         "jpeg/padlock-406986_640_420.jpg",
         "jpeg/padlock-406986_640_422.jpg",
         "jpeg/padlock-406986_640_440.jpg",
         "jpeg/padlock-406986_640_444.jpg",
         "jpeg/padlock-406986_640_gray.jpg",
         "jpeg/ycck_colorspace.jpg",
         "jpeg/cmyk.jpg",
         "jpeg/cmyk-dali.jpg",
         "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",

         "jpeg/exif/padlock-406986_640_horizontal.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
         "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
         "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
         "jpeg/exif/padlock-406986_640_no_orientation.jpg",
         "jpeg/exif/padlock-406986_640_rotate_180.jpg",
         "jpeg/exif/padlock-406986_640_rotate_270.jpg",
         "jpeg/exif/padlock-406986_640_rotate_90.jpg",

         "jpeg2k/cat-1046544_640.jp2",
         "jpeg2k/cat-111793_640.jp2",
         "jpeg2k/tiled-cat-1046544_640.jp2",
         "jpeg2k/tiled-cat-111793_640.jp2",
         "jpeg2k/cat-111793_640-16bit.jp2",
         "jpeg2k/cat-1245673_640-12bit.jp2",)
    ]
)
def test_encode_nvtiff_batch(tmp_path, input_images_batch, encode_to_data, cuda_stream, max_num_cpu_threads):
    backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)]
    encoder = nvimgcodec.Encoder(backends=backends, max_num_cpu_threads=max_num_cpu_threads)
    
    input_images = [os.path.join(img_dir_path, img) for img in input_images_batch]
    ref_images = [get_opencv_reference(img) for img in input_images]
    cp_ref_images = [cp.asarray(ref_img) for ref_img in ref_images]
    nv_ref_images = [nvimgcodec.as_image(cp_ref_img) for cp_ref_img in cp_ref_images]

    if encode_to_data:
        if cuda_stream:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="tiff", params=None, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="tiff", params=None)
    else:
        output_img_paths = []
        for i, img in enumerate(input_images):
            base, _ = os.path.splitext(os.path.basename(img))
            out_name = f"{base}_{i}.tiff"
            output_img_paths.append(os.path.join(tmp_path, out_name))

        if cuda_stream:
            encoder.write(output_img_paths, nv_ref_images, params=None, cuda_stream=cuda_stream.ptr)
        else:
            encoder.write(output_img_paths, nv_ref_images, params=None)
        test_encoded_images = []
        for out_img_path in output_img_paths:
            with open(out_img_path, 'rb') as in_file:
                test_encoded_img = in_file.read()
                test_encoded_images.append(test_encoded_img)

    test_decoded_images = [get_opencv_reference(np.asarray(bytearray(img))) for img in test_encoded_images]
    for i, (test_img, ref_img) in enumerate(zip(test_decoded_images, ref_images)):
        np.testing.assert_array_equal(test_img, ref_img) 

@t.mark.parametrize("mct_mode", [0, 1])
def test_encode_image_previously_decoded_with_unchanged_color_spec(mct_mode):
    """
    Test of encoding of image which was decoded with color_spec=UNCHANGED.
    
    This test replicates the scenario where:
    1. An image is decoded with ColorSpec.UNCHANGED 
    2. Then that image is encoded with a UNCHANGED color_spec
    """
    encoder = nvimgcodec.Encoder()
    decoder = nvimgcodec.Decoder()
    
    # Decode with UNCHANGED color_spec 
    input_img_path = os.path.join(img_dir_path, "tiff/cat-1245673_640.tiff")
    decode_params = nvimgcodec.DecodeParams(color_spec = nvimgcodec.ColorSpec.UNCHANGED)
    decoded_image = decoder.read(input_img_path, params = decode_params)
    assert decoded_image is not None
    assert decoded_image.color_spec == nvimgcodec.ColorSpec.SRGB
        
    encode_params = nvimgcodec.EncodeParams(
        color_spec = nvimgcodec.ColorSpec.UNCHANGED,
        quality_type = nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams(
            bitstream_type = nvimgcodec.Jpeg2kBitstreamType.JP2,
            mct_mode = mct_mode
        )
    )
    
    # Verify that we can encode the image
    final_encoded = encoder.encode(decoded_image, "jpeg2k", params = encode_params)
    assert final_encoded is not None
    assert final_encoded.color_spec == nvimgcodec.ColorSpec.SRGB
    
    # create new code stream from the encoded data to verify that color spec is preserved after p,arsing
    parsed_code_stream = nvimgcodec.CodeStream(bytes(final_encoded))
    assert parsed_code_stream.color_spec == nvimgcodec.ColorSpec.SRGB
    
    # Verify we can decode the final result
    final_decoded = decoder.decode(parsed_code_stream, params = decode_params)
    assert final_decoded is not None
    assert final_decoded.color_spec == nvimgcodec.ColorSpec.SRGB  


@t.mark.parametrize("input_color_spec, output_color_spec", [
    # Invalid cases: mct_mode=True with non-SRGB input or output
    (nvimgcodec.ColorSpec.SRGB, nvimgcodec.ColorSpec.SYCC),
    (nvimgcodec.ColorSpec.SRGB, nvimgcodec.ColorSpec.GRAY),
    
    (nvimgcodec.ColorSpec.GRAY, nvimgcodec.ColorSpec.GRAY),
    (nvimgcodec.ColorSpec.GRAY, nvimgcodec.ColorSpec.SRGB),
    (nvimgcodec.ColorSpec.GRAY, nvimgcodec.ColorSpec.UNCHANGED),

    
    #(nvimgcodec.ColorSpec.SYCC, nvimgcodec.ColorSpec.SYCC),
    #(nvimgcodec.ColorSpec.SYCC, nvimgcodec.ColorSpec.UNCHANGED),
    #(nvimgcodec.ColorSpec.SYCC, nvimgcodec.ColorSpec.SRGB),
])
def test_encode_jpeg2k_with_mct_mode_and_invalid_color_specs(input_color_spec, output_color_spec):
    """
    Test that MCT mode fails with invalid color spec combinations.
    """
    encoder = nvimgcodec.Encoder()
    decoder = nvimgcodec.Decoder()
    if input_color_spec == nvimgcodec.ColorSpec.GRAY:
        input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_gray.jpg")
    else:
        input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_444.jpg")
    decode_params = nvimgcodec.DecodeParams(color_spec=input_color_spec)
    image = decoder.read(input_img_path, params=decode_params)
    assert image is not None

    # Create encode parameters with mct_mode=1
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        color_spec=output_color_spec,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_444,
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(
            bitstream_type=nvimgcodec.Jpeg2kBitstreamType.JP2,
            mct_mode=1
        )
    )

    cs = encoder.encode(image, "jpeg2k", params=encode_params)
    assert cs is None


@t.mark.parametrize("input_color_spec, output_color_spec", [
    # Valid cases: SRGB input with SRGB or UNCHANGED output
    (nvimgcodec.ColorSpec.SRGB, nvimgcodec.ColorSpec.SRGB),
    (nvimgcodec.ColorSpec.SRGB, nvimgcodec.ColorSpec.UNCHANGED),
])
@t.mark.parametrize("mct_mode, quality_type", [
    #(0, nvimgcodec.QualityType.QUALITY), # For quality type QUALITY, mct_mode must be 1
    (1, nvimgcodec.QualityType.QUALITY),
    (0, nvimgcodec.QualityType.LOSSLESS),
    (1, nvimgcodec.QualityType.LOSSLESS)])
@t.mark.parametrize("bitstream_type", [
    nvimgcodec.Jpeg2kBitstreamType.J2K,
    nvimgcodec.Jpeg2kBitstreamType.JP2
])
def test_encode_jpeg2k_with_mct_mode_for_valid_cases(input_color_spec, output_color_spec, mct_mode, quality_type, bitstream_type):
    """
    Test that MCT mode works correctly with valid color spec combinations for both lossless and lossy compression,
    and for both J2K and JP2 bitstream types.
    """

    encoder = nvimgcodec.Encoder()
    decoder = nvimgcodec.Decoder()

    # Use an SRGB image
    input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_444.jpg")
    decode_params = nvimgcodec.DecodeParams(color_spec=input_color_spec)
    image = decoder.read(input_img_path, params=decode_params)
    assert image is not None
    assert image.color_spec == input_color_spec

    # Create encode parameters
    jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams(
        bitstream_type=bitstream_type,
        mct_mode=1
    )

    encode_params = nvimgcodec.EncodeParams(
        quality_type=quality_type,
        quality_value=90.0 if quality_type == nvimgcodec.QualityType.QUALITY else 0.0,
        color_spec=output_color_spec,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_444,
        jpeg2k_encode_params=jpeg2k_params
    )

    # Encode
    encoded = encoder.encode(image, "jpeg2k", params=encode_params)
    assert encoded is not None

    # Decode and verify
    decoded = decoder.decode(encoded)
    assert decoded is not None
    # Output color_spec should be as requested (SRGB or UNCHANGED, which preserves SRGB)
    assert decoded.color_spec == image.color_spec or decoded.color_spec == output_color_spec
    assert decoded.shape == image.shape or decoded.shape[:-1] == image.shape[:-1]

    # For lossless with MCT, we expect the images to be identical
    if quality_type == nvimgcodec.QualityType.LOSSLESS:
        image_np = np.array(image.cpu())
        decoded_np = np.array(decoded.cpu())
        assert np.array_equal(image_np, decoded_np), "Lossless encoding did not produce identical output"


@t.mark.parametrize("chroma_subsampling", [
    nvimgcodec.ChromaSubsampling.CSS_422,
    nvimgcodec.ChromaSubsampling.CSS_420,
])
def test_encode_jpeg2k_with_mct_mode_and_non_444_chroma_subsampling_should_fail(chroma_subsampling):
    """
    Test that enabling MCT mode with non-444 chroma subsampling fails as expected.

    MCT (Multiple Component Transform) is only supported for 4:4:4 chroma subsampling.
    Attempting to use MCT with 4:2:2 or 4:2:0 should fail.
    """
    encoder = nvimgcodec.Encoder()
    decoder = nvimgcodec.Decoder()

    # Use an SRGB image
    input_img_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_444.jpg")
    decode_params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SRGB)
    image = decoder.read(input_img_path, params=decode_params)
    assert image is not None
    assert image.color_spec == nvimgcodec.ColorSpec.SRGB

    # Create encode parameters with MCT enabled (mct_mode=1)
    jpeg2k_params = nvimgcodec.Jpeg2kEncodeParams(
        bitstream_type=nvimgcodec.Jpeg2kBitstreamType.JP2,
        mct_mode=1
    )

    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        color_spec=nvimgcodec.ColorSpec.SRGB,
        chroma_subsampling=chroma_subsampling,
        jpeg2k_encode_params=jpeg2k_params
    )

    image = encoder.encode(image, "jpeg2k", params=encode_params)
    assert image is None

def test_encode_png_with_4_channels():
    encoder = nvimgcodec.Encoder()
    arr = np.random.randint(0, 255, (1024, 1024, 4), dtype=np.uint8)
    enc_code_stream = encoder.encode(arr, "png")
    assert enc_code_stream is not None
    assert enc_code_stream.color_spec == nvimgcodec.ColorSpec.SRGB
    assert enc_code_stream.sample_format == nvimgcodec.SampleFormat.I_RGBA
    assert enc_code_stream.width == 1024
    assert enc_code_stream.height == 1024
    assert enc_code_stream.num_channels == 4
    assert enc_code_stream.dtype == np.uint8
    assert enc_code_stream.codec_name == "png"
    assert enc_code_stream.size > 0