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
pytestmark = t.mark.skipif(sys.version_info >= (3, 13), reason="Requires Python version lower than 3.13")

import numpy as np
try:
    import cupy as cp
    cuda_streams = [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)]
except:
    print("CuPy is not available, will skip related tests")
    cuda_streams = []
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
@t.mark.skipif(not is_nvjpeg2k_supported(), reason="nvjpeg2k encoder not yet supported on aarch64")
def test_encode_single_image(tmp_path, input_img_file, encode_to_data, cuda_stream, max_num_cpu_threads):
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=max_num_cpu_threads)

    input_img_path = os.path.join(img_dir_path, input_img_file)
    ref_img = get_opencv_reference(input_img_path)
    cp_ref_img = cp.asarray(ref_img)

    nv_ref_img = nvimgcodec.as_image(cp_ref_img)
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    
    if encode_to_data:
        if cuda_stream:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="jpeg2k", params = encode_params, cuda_stream=cuda_stream.ptr)
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="jpeg2k", params = encode_params)
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + ".jp2")
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
@t.mark.skipif(not is_nvjpeg2k_supported(), reason="nvjpeg2k encoder not yet supported on aarch64")
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
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.RGB, allow_any_depth=True)
    arr2 = np.array(decoder.decode(encoded_image, params=params).cpu())

    np.testing.assert_array_equal(arr, arr2)


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
@t.mark.skipif(not is_nvjpeg2k_supported(), reason="nvjpeg2k encoder not yet supported on aarch64")
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

def test_encode_jpeg_gray():
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, 'bmp/cat-111793_640_grayscale.bmp')
    backends = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY), 
                nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU),
                nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]
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

@t.mark.parametrize("encode_to_data", [True, False])
def test_encode_single_image_with_unsupported_codec_returns_none(tmp_path, encode_to_data):
    img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))
    fname = os.path.join(img_dir_path, "bmp/cat-111793_640.bmp")
    decoder = nvimgcodec.Decoder()
    img = decoder.read(fname).cpu()
    
    backends = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY)] # we do not have GPU webp encoder and we use it here for testing unsupported codec
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

    img  = gen_img((10, 10, 4)) # only 1 or 3 channels are supported
    encoder = nvimgcodec.Encoder()

    res = encoder.encode(img, codec=".jpeg")
    assert res is None

    img2  = gen_img((10, 13, 4)) # only 1 or 3 channels are supported
    img3  = gen_img((15, 10, 4)) # only 1 or 3 channels are supported
    res_list = encoder.encode([img, img2, img3], codec=".jpeg")
    for res in res_list:
        assert res is None

    valid_img = gen_img((10, 10, 3))
    valid_img2 = gen_img((20, 20, 3))
    res_list = encoder.encode([img, valid_img, img2, valid_img2], codec=".bmp") # use bmp for lossless
    assert res_list[0] is None
    assert res_list[1] is not None
    assert res_list[2] is None
    assert res_list[3] is not None

    decoder = nvimgcodec.Decoder()
    dec_1, dec_2 = decoder.decode([res_list[1], res_list[3]])
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
    backends = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY)]
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
         "jpeg2k/cat-1046544_640.jp2",
         "jpeg2k/cat-111793_640.jp2",
         "jpeg2k/tiled-cat-1046544_640.jp2",
         "jpeg2k/tiled-cat-111793_640.jp2",
         "jpeg2k/cat-111793_640-16bit.jp2",
         "jpeg2k/cat-1245673_640-12bit.jp2",)
    ]
)
@t.mark.skipif(platform.machine() == "aarch64", reason="Test can hang on Orin (IMGCODECS-1491)")
def test_encode_nvtiff_batch(tmp_path, input_images_batch, encode_to_data, cuda_stream, max_num_cpu_threads):
    backends = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY)]
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
        output_img_paths = [os.path.join(tmp_path, os.path.splitext(
            os.path.basename(img))[0] + ".tiff") for img in input_images]
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
