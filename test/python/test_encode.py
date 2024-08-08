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
from utils import *

@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
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
    ref_img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)
    ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)

    cp_ref_img = cp.asarray(ref_img)

    nv_ref_img = nvimgcodec.as_image(cp_ref_img)
    encode_params = nvimgcodec.EncodeParams(jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(reversible=True))
    
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

    test_img = cv2.imdecode(
        np.asarray(bytearray(test_encoded_img)), cv2.IMREAD_COLOR)

    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    compare_image(np.asarray(test_img), np.asarray(ref_img))


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
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

    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [cv2.cvtColor(cv2.imread(img, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in input_images]
    cp_ref_images = [cp.asarray(ref_img) for ref_img in ref_images]
    nv_ref_images = [nvimgcodec.as_image(cp_ref_img) for cp_ref_img in cp_ref_images]

    encode_params = nvimgcodec.EncodeParams(jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(reversible=True))

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

    test_decoded_images = [cv2.cvtColor(cv2.imdecode(
        np.asarray(bytearray(img)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in test_encoded_images]

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
    jpeg2k_encode_params = nvimgcodec.Jpeg2kEncodeParams(reversible=True)
    encode_params = nvimgcodec.EncodeParams(jpeg2k_encode_params=jpeg2k_encode_params)
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
    ref_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in input_images]
    cp_ref_images = [cp.asarray(ref_img) for ref_img in ref_images]
    nv_ref_images = nvimgcodec.as_images(cp_ref_images)
    encoder = nvimgcodec.Encoder()
    encode_params = nvimgcodec.EncodeParams(jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(reversible=True))
    test_encoded_images = encoder.encode(nv_ref_images, codec="jpeg2k", params=encode_params)
    test_decoded_images = [cv2.cvtColor(cv2.imdecode(
        np.asarray(bytearray(img)), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB) for img in test_encoded_images]

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
    ref = np.expand_dims(np.array(cv2.imdecode(np.asarray(bytearray(arr2)), cv2.IMREAD_GRAYSCALE)), -1)
    np.testing.assert_allclose(ref, arr3, atol=1)
