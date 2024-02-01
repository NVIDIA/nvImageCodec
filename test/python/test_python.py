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

def load_single_image(file_path: str, load_mode: str = None):
    """
    Loads a single image to de decoded.
    :param file_path: Path to file with the image to be loaded.
    :param load_mode: In what format the image shall be loaded:
                        "numpy"  - loading using `np.fromfile`,
                        "python" - loading using Python's `open`,
                        "path"   - loading skipped, image path will be returned.
    :return: Encoded image.
    """
    if load_mode == "numpy":
        return np.fromfile(file_path, dtype=np.uint8)
    elif load_mode == "python":
        with open(file_path, 'rb') as in_file:
            return in_file.read()
    elif load_mode == "path":
        return file_path
    else:
        raise RuntimeError(f"Unknown load mode: {load_mode}")


def load_batch(file_paths: list[str], load_mode: str = None):
    return [load_single_image(f, load_mode) for f in file_paths]


def decode_single_image_test(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads):
    if backends:
        decoder = nvimgcodec.Decoder(max_num_cpu_threads=max_num_cpu_threads,
            backends=backends, options=get_default_decoder_options())
    else:
        decoder = nvimgcodec.Decoder(options=get_default_decoder_options())

    input_img_path = os.path.join(img_dir_path, input_img_file)

    decoder_input = load_single_image(input_img_path, input_format)

    if input_format == "path":
        test_img = decoder.read(decoder_input)
    else:
        test_img = decoder.decode(decoder_input)

    ref_img = cv2.imread(input_img_path, cv2.IMREAD_COLOR)

    compare_device_with_host_images([test_img], [ref_img])


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_img_file",
    ["bmp/cat-111793_640.bmp",

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
    "jpeg2k/cat-1245673_640-12bit.jp2",
     ]
)
def test_decode_single_image_common(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads):
    decode_single_image_test(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads)


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_img_file",
    ["jpeg/ycck_colorspace.jpg",
     "jpeg/cmyk.jpg",
    ]
)
@t.mark.skipif(nvimgcodec.__cuda_version__ < 12010,  reason="requires CUDA >= 12.1")
def test_decode_single_image_cuda12_only(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads):
    decode_single_image_test(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads)


@t.mark.parametrize(
    "input_img_file",
    ["jpeg2k/tiled-cat-1046544_640_gray.jp2",])
def test_decode_single_image_unchanged(input_img_file):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
    test_img = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))
    ref_img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)

    test_img = np.asarray(test_img.cpu())

    test_img = test_img.reshape(test_img.__array_interface__['shape'][0:2])

    compare_host_images([test_img], [ref_img])

@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("cuda_stream", [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
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
def test_decode_batch(tmp_path, input_images_batch, input_format, backends, cuda_stream, max_num_cpu_threads):
    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [cv2.imread(img, cv2.IMREAD_COLOR) for img in input_images]
    if backends:
        decoder = nvimgcodec.Decoder(
            max_num_cpu_threads=max_num_cpu_threads, backends=backends, options=get_default_decoder_options())
    else:
        decoder = nvimgcodec.Decoder(
            max_num_cpu_threads=max_num_cpu_threads, options=get_default_decoder_options())

    encoded_images = load_batch(input_images, input_format)

    if input_format == "path":
        test_images = decoder.read(encoded_images, cuda_stream=0 if cuda_stream is None else cuda_stream.ptr)
    else:
        test_images = decoder.decode(encoded_images, cuda_stream=0 if cuda_stream is None else cuda_stream.ptr)
    compare_device_with_host_images(test_images, ref_images)

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
    assert (array_interface['shape'] == shape)
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
    
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
    dec_device_img = decoder.read(input_img_path)
    assert (dec_device_img.buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE)


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
def test_as_images_with_cuda_array_interface(input_images_batch):
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


@t.mark.parametrize(
    "input_img_file, precision",
    [("jpeg2k/tiled-cat-1046544_640_gray.jp2", 0),
     ("jpeg2k/cat-111793_640-16bit.jp2", 0),
     ("jpeg2k/cat-1245673_640-12bit.jp2", 12)])
def test_image_precision(input_img_file, precision):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
    test_img = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))

    assert test_img.precision == precision
