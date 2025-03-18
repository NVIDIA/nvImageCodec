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

def expected_buffer_kind(backends):
    if backends is None or len(backends) == 0:
        return nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    for backend in backends:
        if backend.backend_kind != nvimgcodec.CPU_ONLY:
            return nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    return nvimgcodec.ImageBufferKind.STRIDED_HOST


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
    
    assert (test_img.buffer_kind == expected_buffer_kind(backends))

    ref_img = get_opencv_reference(input_img_path)
    compare_device_with_host_images([test_img], [ref_img])


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
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
    [
        "jpeg/ycck_colorspace.jpg",
        "jpeg/cmyk.jpg",
    ]
)
@t.mark.skipif(nvimgcodec.__cuda_version__ < 12010,  reason="requires CUDA >= 12.1")
def test_decode_single_image_cuda12_only(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads):
    decode_single_image_test(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads)

@t.mark.parametrize(
    "input_img_file",
    [
        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_411.jpg",
        "jpeg/padlock-406986_640_420.jpg",
        "jpeg/padlock-406986_640_422.jpg",
        "jpeg/padlock-406986_640_440.jpg",
        "jpeg/padlock-406986_640_444.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",
        "jpeg/exif/padlock-406986_640_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",

        "jpeg2k/tiled-cat-1046544_640_gray.jp2",
        "jpeg2k/with_alpha/cat-111793_640-alpha.jp2",
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/tiled-cat-1046544_640.jp2",
        "jpeg2k/cat-111793_640-16bit.jp2",
        "jpeg2k/cat-1245673_640-12bit.jp2"
    ]
)
@t.mark.parametrize(
    "color_spec",
    [nvimgcodec.ColorSpec.RGB,
     nvimgcodec.ColorSpec.UNCHANGED,
     nvimgcodec.ColorSpec.GRAY])
def test_decode_color_spec(input_img_file, color_spec):
    debug = False
    input_img_path = os.path.join(img_dir_path, input_img_file)
    backends = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY),
                nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU),
                nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options(), backends=backends)
    params = nvimgcodec.DecodeParams(
        color_spec=color_spec, allow_any_depth=False, apply_exif_orientation=True)
    test_img = decoder.read(input_img_path, params=params)
    test_img = np.asarray(test_img.cpu())
    ref_img = get_opencv_reference(input_img_path, color_spec)
    if debug:
        import cv2
        cv2.imwrite("ref.bmp", ref_img)
        cv2.imwrite("test.bmp", test_img)
    assert test_img.shape == ref_img.shape, f"{test_img.shape} != {ref_img.shape}"
    compare_host_images([test_img], [ref_img])
    if color_spec == nvimgcodec.ColorSpec.GRAY:
        assert test_img.shape[-1] == 1
    elif color_spec == nvimgcodec.ColorSpec.RGB:
        assert test_img.shape[-1] == 3
    else:
        assert color_spec == nvimgcodec.ColorSpec.UNCHANGED
        if 'gray' in input_img_file:
            expected_nchannels = 1
        elif 'alpha' in input_img_file:
            expected_nchannels = 4
        else:
            expected_nchannels = 3
        assert expected_nchannels == test_img.shape[-1]

@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("cuda_stream", cuda_streams)
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_images_batch",
    [(
        "bmp/cat-111793_640.bmp",

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
        "jpeg2k/cat-1245673_640-12bit.jp2"
    )]
)
def test_decode_batch(tmp_path, input_images_batch, input_format, backends, cuda_stream, max_num_cpu_threads):
    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [get_opencv_reference(img) for img in input_images]
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

@t.mark.parametrize(
    "input_img_file, precision",
    [
        ("jpeg2k/tiled-cat-1046544_640_gray.jp2", 8),
        ("jpeg2k/cat-111793_640-16bit.jp2", 16),
        ("jpeg2k/cat-1245673_640-12bit.jp2", 12),
        ("tiff/cat-300572_640_uint16.tiff", 16),
        ("tiff/uint16.tiff", 16),
    ]
)
def test_decode_image_check_precision(input_img_file, precision):
    input_img_path = os.path.join(img_dir_path, input_img_file)
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options())
    test_img = decoder.read(input_img_path, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True, apply_exif_orientation=False))
    assert test_img.precision == precision
    test_img = np.asarray(test_img.cpu())
    ref_img = get_opencv_reference(input_img_path, nvimgcodec.ColorSpec.UNCHANGED, True)
    compare_host_images([test_img], [ref_img])


@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
def test_decode_buffer_type(backends):
    path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_440.jpg")
    decoder = nvimgcodec.Decoder(backends=backends)
    image = decoder.read(path)
    assert image.buffer_kind == expected_buffer_kind(backends)
    assert image.cpu().buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert image.cuda().buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.GPU_ONLY, load_hint=0.5), nvimgcodec.Backend(
                                     nvimgcodec.HYBRID_CPU_GPU), nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                                 [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("cuda_stream", cuda_streams)
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_images_batch",
    [(
        "jpegxr/cat-111793_640.jxr",
        "bmp/cat-111793_640.bmp",
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg/padlock-406986_640_410.jpg",
        "jpegxr/Weimaraner.jxr",
        "jpeg2k/cat-1046544_640.jp2",
     )]
)
def test_decode_batch_with_unsupported_formats(tmp_path, input_images_batch, input_format, backends, cuda_stream, max_num_cpu_threads):
    input_images = [os.path.join(img_dir_path, img)
                    for img in input_images_batch]
    ref_images = [get_opencv_reference(img) for img in input_images]
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
        
    assert(test_images[0] == None)
    assert(test_images[4] == None) 
    del test_images[0]
    del test_images[3]   
    del ref_images[0]
    del ref_images[3]
    compare_device_with_host_images(test_images, ref_images)
