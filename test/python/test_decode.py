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
    img = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8) # Force to load necessary libriaries
    cuda_streams = [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)]
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    cuda_streams = []
    CUPY_AVAILABLE = False

from nvidia import nvimgcodec

from utils import *
import nvjpeg_test_speedup

def expected_buffer_kind(backends):
    if backends is None or len(backends) == 0:
        return nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    for backend in backends:
        # backend can be a Backend instance (with .backend_kind) or a BackendKind enum/int
        if hasattr(backend, "backend_kind"):
            kind = backend.backend_kind
        else:
            kind = backend
        if kind != nvimgcodec.BackendKind.CPU_ONLY:
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
    assert test_img.sample_format == nvimgcodec.SampleFormat.I_RGB
    assert test_img.shape[2] == 3

    ref_img = get_opencv_reference(input_img_path)
    compare_device_with_host_images([test_img], [ref_img])


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY, load_hint=0.5), nvimgcodec.BackendKind.HYBRID_CPU_GPU, nvimgcodec.BackendKind.CPU_ONLY],
                                 [nvimgcodec.BackendKind.CPU_ONLY]])
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
        "jpeg2k/cat-1245673_640-12bit.jp2"
    ]
)
def test_decode_single_image_common(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads):
    decode_single_image_test(tmp_path, input_img_file, input_format, backends, max_num_cpu_threads)


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY, load_hint=0.5), nvimgcodec.BackendKind.HYBRID_CPU_GPU, nvimgcodec.BackendKind.CPU_ONLY],
                                 [nvimgcodec.BackendKind.CPU_ONLY]])
@t.mark.parametrize("input_format", ["numpy", "python", "path"])
@t.mark.parametrize(
    "input_img_file",
    [
        "jpeg/ycck_colorspace.jpg",
        "jpeg/cmyk.jpg",
    ]
)
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
    [nvimgcodec.ColorSpec.SRGB,
     nvimgcodec.ColorSpec.UNCHANGED,
     nvimgcodec.ColorSpec.GRAY])
def test_decode_color_spec(input_img_file, color_spec):
    debug = False
    input_img_path = os.path.join(img_dir_path, input_img_file)
    backends = [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY),
                nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
                nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]
    decoder = nvimgcodec.Decoder(options=get_default_decoder_options(), backends=backends)
    params = nvimgcodec.DecodeParams(
        color_spec=color_spec, allow_any_depth=False, apply_exif_orientation=True)
    test_img = decoder.read(input_img_path, params=params)

    ref_img = get_opencv_reference(input_img_path, color_spec)
    if debug:
        import cv2
        cv2.imwrite("ref.bmp", ref_img)
        cv2.imwrite("test.bmp", np.asarray(test_img.cpu()))
    assert test_img.shape == ref_img.shape, f"{test_img.shape} != {ref_img.shape}"
    compare_host_images([np.asarray(test_img.cpu())], [ref_img])
    if color_spec == nvimgcodec.ColorSpec.GRAY:
        assert test_img.shape[-1] == 1
        assert test_img.color_spec == nvimgcodec.ColorSpec.GRAY
    elif color_spec == nvimgcodec.ColorSpec.SRGB:
        assert test_img.shape[-1] == 3
        assert test_img.color_spec == nvimgcodec.ColorSpec.SRGB
    else:
        assert color_spec == nvimgcodec.ColorSpec.UNCHANGED
        code_stream = nvimgcodec.CodeStream(input_img_path)
        if color_spec ==  nvimgcodec.ColorSpec.UNCHANGED and code_stream.color_spec != nvimgcodec.ColorSpec.SRGB and code_stream.color_spec != nvimgcodec.ColorSpec.GRAY:
            assert test_img.color_spec == nvimgcodec.ColorSpec.SRGB
        else:
           t.skip("Currently we don't support decoding with UNCHANGED color space for inputs with  color space other than sRGB or GRAY")
           
        if 'gray' in input_img_file:
            expected_nchannels = 1
            assert test_img.color_spec == nvimgcodec.ColorSpec.GRAY
        elif 'alpha' in input_img_file:
            expected_nchannels = 4
            assert test_img.color_spec == nvimgcodec.ColorSpec.SRGB
        else:
            expected_nchannels = 3

        assert expected_nchannels == test_img.shape[-1]

@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY, load_hint=0.5), nvimgcodec.BackendKind.HYBRID_CPU_GPU, nvimgcodec.BackendKind.CPU_ONLY],
                                 [nvimgcodec.BackendKind.CPU_ONLY]])
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
                                 [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY, load_hint=0.5), nvimgcodec.BackendKind.HYBRID_CPU_GPU, nvimgcodec.BackendKind.CPU_ONLY],
                                 [nvimgcodec.BackendKind.CPU_ONLY]])
def test_decode_buffer_type(backends):
    path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_440.jpg")
    decoder = nvimgcodec.Decoder(backends=backends)
    image = decoder.read(path)
    assert image.buffer_kind == expected_buffer_kind(backends)
    assert image.cpu().buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_HOST
    assert image.cuda().buffer_kind == nvimgcodec.ImageBufferKind.STRIDED_DEVICE


@t.mark.parametrize("max_num_cpu_threads", [0, 1, 5])
@t.mark.parametrize("backends", [None,
                                 [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY, load_hint=0.5), nvimgcodec.BackendKind.HYBRID_CPU_GPU, nvimgcodec.BackendKind.CPU_ONLY],
                                 [nvimgcodec.BackendKind.CPU_ONLY]])
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


def verify_image_roi(img_roi, ref_img, region, fill_value):
    np_roi = np.asarray(img_roi.cpu())
    np_ref = np.asarray(ref_img.cpu())

    # check oob roi value and crop image so that reference and roi image dimensions are the same
    start_y, start_x = region.start
    end_y, end_x = region.end
    if start_x < 0:
        assert (np_roi[:, : -start_x] == fill_value).all()
        np_roi = np_roi[:, -start_x:]
    else:
        np_ref = np_ref[:, start_x:]
        end_x -= start_x

    if start_y < 0:
        assert (np_roi[: -start_y] == fill_value).all()
        np_roi = np_roi[-start_y:]
    else:
        np_ref = np_ref[start_y:]
        end_y -= start_y

    height = np_ref.shape[0]
    width = np_ref.shape[1]

    if end_x > width:
        assert (np_roi[:, width:] == fill_value).all()
        np_roi = np_roi[:, :width]
    else:
        np_ref = np_ref[:, :end_x]

    if end_y > height:
        assert (np_roi[height:] == fill_value).all()
        np_roi = np_roi[:height]
    else:
        np_ref = np_ref[:end_y]

    assert np_ref.shape == np_roi.shape
    assert (np_ref == np_roi).all()


def roi_test_impl(decoder, image_path, start_offset_y, start_offset_x, end_offset_y, end_offset_x, fill_value=120, params=None):
    input_image = os.path.join(img_dir_path, image_path)

    cs = nvimgcodec.CodeStream(input_image)
    roi = nvimgcodec.Region(start_offset_y, start_offset_x, end_offset_y + cs.height, end_offset_x + cs.width, fill_value)

    img_roi = decoder.decode(cs.get_sub_code_stream(region=roi), params=params)
    assert img_roi is not None

    ref_img = decoder.decode(cs, params=params)
    assert ref_img is not None

    verify_image_roi(img_roi, ref_img, roi, fill_value)


is_aarch64 = platform.machine() == "aarch64"

@t.mark.parametrize("input_image", [
    "jpeg/padlock-406986_640_410.jpg",
    "jpeg2k/cat-1046544_640.jp2",       # to test direct decode without conversion
    "jpeg2k/cat-1046544_640-16bit.jp2", # to test decode with convert kernel (as we decode to 8bit)
    "jpeg2k/tiled-cat-1046544_640.jp2", # to test tiled decode
    "tiff/cat-300572_640_no_compression.tiff",
    "tiff/cat-300572_640_palette.tiff",
    t.param(
        "tiff/cat-300572_640_striped.tiff",
        marks=t.mark.skipif(is_aarch64, reason="We don't have yet nvcomp needed for Deflate decode on Tegra")
    ),
    t.param(
        "tiff/cat-300572_640_tiled.tiff",
        marks=t.mark.skipif(is_aarch64, reason="We don't have yet nvcomp needed for Deflate decode on Tegra")
    ),
])
@t.mark.parametrize("start_offset_y", [-50, 55])
@t.mark.parametrize("start_offset_x", [-60, 65])
@t.mark.parametrize("end_offset_y", [-40, 45])
@t.mark.parametrize("end_offset_x", [-30, 35])
def test_decode_roi_cuda(input_image, start_offset_y, start_offset_x, end_offset_y, end_offset_x):
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)])
    roi_test_impl(decoder, input_image, start_offset_y, start_offset_x, end_offset_y, end_offset_x)

@t.mark.parametrize("input_image", [
    "jpeg/padlock-406986_640_410.jpg",
    "jpeg2k/cat-1046544_640.jp2",       # to test direct decode without conversion
    "jpeg2k/cat-1046544_640-16bit.jp2", # to test decode with convert kernel (as we decode to 8bit)
    "jpeg2k/tiled-cat-1046544_640.jp2", # to test tiled decode
    "tiff/cat-300572_640_no_compression.tiff",
    "tiff/cat-300572_640_palette.tiff",
    t.param(
        "tiff/cat-300572_640_striped.tiff",
        marks=t.mark.skipif(is_aarch64, reason="We don't have yet nvcomp needed for Deflate decode on Tegra")
    ),
    t.param(
        "tiff/cat-300572_640_tiled.tiff",
        marks=t.mark.skipif(is_aarch64, reason="We don't have yet nvcomp needed for Deflate decode on Tegra")
    ),
])
@t.mark.parametrize("fill_value", [0, 125, 255])
def test_decode_roi_cuda_fill_value(input_image, fill_value):
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)])
    roi_test_impl(decoder, input_image, -50, -40, 60, 55, fill_value)


@t.mark.skipif(sys.platform.startswith("win32") and int(os.getenv(("CUDA_VERSION_MAJOR"), 12)) < 12,  reason="temporary while waiting for fix")
def test_decode_roi_HW():
    try:
        decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY)])
    except:
        t.skip("nvJPEG HW decoder not available")
    image_path = os.path.join(img_dir_path, "jpeg/padlock-406986_640_420.jpg")
    code_stream = nvimgcodec.CodeStream(image_path)
    ref_img = decoder.decode(code_stream)
    assert ref_img is not None

    # we want to test that HW correctly sets different fill value for each batch element
    batch_elements_fill_values = [0, 125, 255]
    batch = []
    regions = []
    for start_offset_y in [-50, 55]:
        for start_offset_x in [-60, 65]:
            for end_offset_y in [-40, 45]:
                for end_offset_x in [-30, 35]:
                    fill_value = batch_elements_fill_values[len(batch) % len(batch_elements_fill_values)]
                    roi = nvimgcodec.Region(
                        start_offset_y, start_offset_x, end_offset_y + code_stream.height, end_offset_x + code_stream.width, fill_value
                    )
                    regions.append(roi)
                    batch.append(code_stream.get_sub_code_stream(region=roi))

    result = decoder.decode(batch)
    for i, (img, roi) in enumerate(zip(result, regions)):
        assert img is not None
        verify_image_roi(img, ref_img, roi, batch_elements_fill_values[i % len(batch_elements_fill_values)])


@t.mark.parametrize("start_offset_y", [-50, 55])
@t.mark.parametrize("start_offset_x", [-60, 65])
@t.mark.parametrize("end_offset_y", [-40, 45])
@t.mark.parametrize("end_offset_x", [-30, 35])
@t.mark.parametrize("fill_value", [255, 32000, 65535])
@t.mark.parametrize("input_image", 
                    ["jpeg2k/cat-1046544_640-16bit.jp2", 
                    t.param(
                        "tiff/cat-300572_640_uint16.tiff",
                        marks=t.mark.skipif(is_aarch64, reason="We don't have yet nvcomp needed for Deflate decode on Tegra")
                    ),
                     "tiff/cat-300572_640_palette.tiff"])
def test_decode_roi_16bit(start_offset_y, start_offset_x, end_offset_y, end_offset_x, fill_value, input_image):
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU)])
    params = nvimgcodec.DecodeParams(allow_any_depth=True)
    roi_test_impl(decoder, input_image, start_offset_y, start_offset_x, end_offset_y, end_offset_x, fill_value, params)


@t.mark.parametrize(
    "input_image", [
        "bmp/cat-111793_640.bmp",
        "jpeg/padlock-406986_640_410.jpg",
        "jpeg2k/cat-1046544_640.jp2",
        "png/cat-1245673_640.png",
        "pnm/cat-111793_640.ppm",
        "tiff/cat-1245673_640.tiff",
        "webp/lossless/cat-3113513_640.webp"
    ]
)
def test_decode_roi_cpu(input_image):
    decoder = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)])
    roi_test_impl(decoder, input_image, 30, 50, -20, -40)


def convert_ycc_to_rgb(numpy_image):
    xform = np.array([[1, 0, 1.402], [1, -0.34413629, -.71413629], [1, 1.772, 0]])
    rgb = numpy_image.astype(np.float32)
    rgb[:,:,[1,2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)

@t.mark.parametrize("backends", [
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
    [nvimgcodec.BackendKind.GPU_ONLY, nvimgcodec.BackendKind.HYBRID_CPU_GPU],
    [nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY)],
    None, # use default backend)
])
def test_decode_ycc(backends):
    image_batch = ["jpeg/cat-1245673_640_444.jpg", "jpeg/padlock-406986_640_444.jpg"]
    input_img_paths = [os.path.join(img_dir_path, img_file) for img_file in image_batch]

    try:
        decoder = nvimgcodec.Decoder(backends=backends)
    except:
        t.skip(f"Decoder backends are not supported for this platform: " + ", ".join(map(lambda b: str(b.backend_kind), backends)))

    rbg_images = decoder.read(input_img_paths)
    ycc_images = decoder.read(input_img_paths, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SYCC))

    assert len(rbg_images) == len(ycc_images)
    for rbg_image, ycc_image in zip(rbg_images, ycc_images):
        assert rbg_image is not None
        assert ycc_image is not None

        rbg_image = np.asarray(rbg_image.cpu())
        ycc_image = np.asarray(ycc_image.cpu())

        converted = convert_ycc_to_rgb(ycc_image)
        np.testing.assert_allclose(rbg_image, converted, atol=1)

@t.mark.parametrize("image_path", [
    # unsupported subsampling
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_410.jpg",
    "jpeg/padlock-406986_640_411.jpg",
    "jpeg/padlock-406986_640_gray.jpg",

    # unsupported colorspace
    "jpeg/cmyk-dali.jpg",
    "jpeg/ycck_colorspace.jpg",

    # unsupported codecs
    "bmp/cat-111793_640.bmp",
    "jpeg2k/cat-111793_640.jp2",
    "png/cat-1245673_640.png",
    "pnm/cat-111793_640.ppm",
    "tiff/cat-300572_640.tiff",
    "webp/lossy/cat-3113513_640.webp"
])
def test_decode_ycc_unsupported(image_path):
    image_path = os.path.join(img_dir_path, image_path)

    decoder = nvimgcodec.Decoder()

    rbg_image = decoder.read(image_path)
    ycc_image = decoder.read(image_path, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SYCC))

    assert rbg_image is not None
    assert ycc_image is None


DEFLATE_TIFF_IMAGES = [
    "tiff/cat-300572_640.tiff",
    "tiff/cat-300572_640_fp32.tiff",
    "tiff/cat-300572_640_uint16.tiff",
]

def decode_to_external_buffer_imp(image_path, dtype, decode_params, backends, buffer_create):
    if not CUPY_AVAILABLE:
        t.skip("CuPy is not available")
    
    if image_path in DEFLATE_TIFF_IMAGES and not is_nvcomp_supported():
        t.skip("nvCOMP is not supported on this platform")

    image_path = os.path.join(img_dir_path, image_path)

    try:
        decoder = nvimgcodec.Decoder(backends=backends)
    except:
        t.skip(f"Decoder backends are not supported for this platform: " + ", ".join(map(lambda b: str(b.backend_kind), backends)))

    cs = nvimgcodec.CodeStream(image_path)

    num_channels = 3
    if decode_params is not None and decode_params.color_spec == nvimgcodec.ColorSpec.GRAY:
        num_channels = 1

    array = buffer_create((cs.height, cs.width, num_channels), dtype=dtype)
    image = nvimgcodec.as_image(array)

    decoded_external = decoder.decode(cs, params=decode_params, image=image)
    reference = decoder.decode(cs, params=decode_params)

    np.testing.assert_allclose(reference.cpu(), decoded_external.cpu())
    np.testing.assert_allclose(decoded_external.cpu(), cp.asnumpy(array))

EXTERNAL_BUFFER_TEST_IMAGE_LIST = [
    "jpeg/cat-1245673_640_444.jpg",
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/cmyk-dali.jpg",
    "jpeg2k/cat-111793_640.jp2",
    "jpeg2k/tiled-cat-111793_640.jp2",
    "jpeg2k/tiled-cat-1046544_640_gray.jp2",
    "jpeg2k/cat-1245673_640-12bit.jp2",
    "jpeg2k/cat-1046544_640-16bit.jp2",
    "jpeg2k/cat-111793_640-16bit-gray.jp2",
    "jpeg2k/with_alpha/cat-111793_640-alpha.jp2",
    "jpeg2k/with_alpha_16bit/4ch16bpp.jp2",
    "tiff/cat-300572_640.tiff",
    "tiff/cat-300572_640_uint16.tiff",
    "tiff/cat-300572_640_palette.tiff",
]

EXTERNAL_BUFFER_TEST_IMAGE_LIST_uint16 = [
    "jpeg2k/cat-1046544_640-16bit.jp2",
    "jpeg2k/cat-1245673_640-12bit.jp2",
    "jpeg2k/cat-111793_640-16bit-gray.jp2",
    "jpeg2k/with_alpha_16bit/4ch16bpp.jp2",
    "tiff/cat-300572_640_uint16.tiff",
    "tiff/cat-300572_640_palette.tiff",
]

EXTERNAL_BUFFER_TEST_IMAGE_LIST_CPU_ONLY_CODECS = [
    "bmp/cat-111793_640.bmp",
    "bmp/cat-111793_640_grayscale.bmp",
    "bmp/cat-111793_640_palette_1bit.bmp",
    "bmp/cat-111793_640_palette_8bit.bmp",
    "png/cat-1245673_640.png",
    "png/with_alpha/cat-111793_640-alpha.png",
    "png/with_alpha_16bit/4ch16bpp.png",
    "pnm/cat-111793_640.ppm",
    "pnm/cat-1245673_640.pgm",
    "pnm/cat-2184682_640.pbm",
    "webp/lossy/cat-3113513_640.webp",
    "webp/lossless/cat-3113513_640.webp",
    "webp/lossless_alpha/camel-1987672_640.webp"
]

BACKEND_LIST = [
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
    [nvimgcodec.BackendKind.GPU_ONLY, nvimgcodec.BackendKind.HYBRID_CPU_GPU],
    None, # use default backend)
]

EMPTY_BUFFER_CREATE = [np.empty]
ONES_BUFFER_CREATE = [np.ones]
if CUPY_AVAILABLE:
    EMPTY_BUFFER_CREATE.append(cp.empty)
    ONES_BUFFER_CREATE.append(cp.ones)

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST + [
    "tiff/cat-300572_640_fp32.tiff",
])
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer(image_path, backends, buffer_create):
    decode_to_external_buffer_imp(image_path, np.uint8, None, backends, buffer_create)

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_CPU_ONLY_CODECS)
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_cpu_only(image_path, buffer_create):
    decode_to_external_buffer_imp(image_path, np.uint8, None, [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)], buffer_create)

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST)
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_grayscale(image_path, backends, buffer_create):
    decode_to_external_buffer_imp(
        image_path,
        np.uint8,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY),
        backends,
        buffer_create
    )

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_CPU_ONLY_CODECS)
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_cpu_only_grayscale(image_path, buffer_create):
    decode_to_external_buffer_imp(
        image_path,
        np.uint8,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY),
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
        buffer_create
    )

@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
@t.mark.parametrize("image_path", [
    "jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg",
    "jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg"
])
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_jpeg_lossless(image_path, buffer_create):
    decode_to_external_buffer_imp(
        image_path,
        np.uint16,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY, allow_any_depth=True),
        None,
        buffer_create
    )

@t.mark.parametrize("image_path", [
    "jpeg/cat-1245673_640_444.jpg",
])
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_YCC(image_path, backends, buffer_create):
    decode_to_external_buffer_imp(
        image_path,
        np.uint8,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SYCC),
        backends,
        buffer_create
    )

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_uint16)
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_uint16(image_path, backends, buffer_create):
    decode_to_external_buffer_imp(image_path, np.uint16, nvimgcodec.DecodeParams(allow_any_depth=True), backends, buffer_create)

@t.mark.parametrize("image_path", ["png/with_alpha_16bit/4ch16bpp.png"])
@t.mark.parametrize("buffer_create", EMPTY_BUFFER_CREATE)
def test_decode_to_external_buffer_cpu_only_uint16(image_path, buffer_create):
    decode_to_external_buffer_imp(
        image_path,
        np.uint16,
        nvimgcodec.DecodeParams(allow_any_depth=True),
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
        buffer_create
    )

# Folowing tests are using slice of numpy/cupy image as image buffer
# This means that buffer stride is not equal to row size (there is padding at the end of each row)
# If above tests of external buffer works, and tests below fails, then probably stride is the problem
def decode_to_external_buffer_slice_impl(image_path, dtype, decode_params, backends, buffer_create):
    if not CUPY_AVAILABLE:
        t.skip("CuPy is not available")
    
    if image_path in DEFLATE_TIFF_IMAGES and not is_nvcomp_supported():
        t.skip("nvCOMP is not supported on this platform")

    image_path = os.path.join(img_dir_path, image_path)

    try:
        decoder = nvimgcodec.Decoder(backends=backends)
    except:
        t.skip(f"Decoder backends are not supported for this platform: " + ", ".join(map(lambda b: str(b.backend_kind), backends)))

    cs = nvimgcodec.CodeStream(image_path)

    # create image larger than what is required by codestream and then slice it to correct format and check if decoding is right
    start_x_offset = 150
    start_y_offset = 100
    end_x_offset = 200
    end_y_offset = 250

    num_channels = 3
    if decode_params is not None and decode_params.color_spec == nvimgcodec.ColorSpec.GRAY:
        num_channels = 1

    FILL_VALUE = 37

    array = buffer_create(
        (start_y_offset + end_y_offset + cs.height, start_x_offset + end_x_offset + cs.width, num_channels),
        dtype=dtype
    ) * FILL_VALUE
    slice = array[start_y_offset:-end_y_offset, start_x_offset:-end_x_offset]
    image = nvimgcodec.as_image(slice)

    decoded_external = decoder.decode(cs, params=decode_params, image=image)
    reference = decoder.decode(cs, params=decode_params)

    np.testing.assert_allclose(reference.cpu(), decoded_external.cpu())
    np.testing.assert_allclose(decoded_external.cpu(), cp.asnumpy(slice))

    # check if outside of slice is left unchanged
    assert (array[:start_y_offset] == FILL_VALUE).all()
    assert (array[-end_y_offset:] == FILL_VALUE).all()
    assert (array[:, :start_x_offset] == FILL_VALUE).all()
    assert (array[:, -end_x_offset:] == FILL_VALUE).all()

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST + [
    "tiff/cat-300572_640_fp32.tiff",
])
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice(image_path, backends, buffer_create):
    decode_to_external_buffer_slice_impl(image_path, np.uint8, None, backends, buffer_create)

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_CPU_ONLY_CODECS)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_cpu_only(image_path, buffer_create):
    decode_to_external_buffer_slice_impl(image_path, np.uint8, None, [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)], buffer_create)

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST)
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_grayscale(image_path, backends, buffer_create):
    decode_to_external_buffer_slice_impl(
        image_path,
        np.uint8,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY),
        backends,
        buffer_create
    )

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_CPU_ONLY_CODECS)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_cpu_only_grayscale(image_path, buffer_create):
    decode_to_external_buffer_slice_impl(
        image_path,
        np.uint8,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY),
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
        buffer_create
    )

@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
@t.mark.parametrize("image_path", [
    "jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg",
    "jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg"
])
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_jpeg_lossless(image_path, buffer_create):
    decode_to_external_buffer_slice_impl(
        image_path,
        np.uint16,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY, allow_any_depth=True),
        None,
        buffer_create
    )

@t.mark.parametrize("image_path", [
    "jpeg/cat-1245673_640_444.jpg",
])
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_YCC(image_path, backends, buffer_create):
    decode_to_external_buffer_slice_impl(
        image_path,
        np.uint8,
        nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SYCC),
        backends,
        buffer_create
    )

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_uint16)
@t.mark.parametrize("backends", BACKEND_LIST)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_uint16(image_path, backends, buffer_create):
    decode_to_external_buffer_slice_impl(image_path, np.uint16, nvimgcodec.DecodeParams(allow_any_depth=True), backends, buffer_create)

@t.mark.parametrize("image_path", ["png/with_alpha_16bit/4ch16bpp.png"])
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_cpu_only_uint16(image_path, buffer_create):
    decode_to_external_buffer_slice_impl(
        image_path,
        np.uint16,
        nvimgcodec.DecodeParams(allow_any_depth=True),
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
        buffer_create
    )

def decode_to_external_buffer_slice_with_roi_impl(image_path, dtype, backends, buffer_create, region_offset):
    if not CUPY_AVAILABLE:
        t.skip("CuPy is not available")
    
    if image_path in DEFLATE_TIFF_IMAGES and not is_nvcomp_supported():
        t.skip("nvCOMP is not supported on this platform")

    image_path = os.path.join(img_dir_path, image_path)

    try:
        decoder = nvimgcodec.Decoder(backends=backends)
    except:
        t.skip(f"Decoder backends are not supported for this platform: " + ", ".join(map(lambda b: str(b.backend_kind), backends)))

    cs = nvimgcodec.CodeStream(image_path)
    region = nvimgcodec.Region(
        start_y=region_offset[0], start_x=region_offset[1],
        end_y=cs.height - region_offset[2], end_x=cs.width - region_offset[3]
    )
    cs = cs.get_sub_code_stream(region=region)
    region_height = cs.height - region_offset[2] - region_offset[0]
    region_width = cs.width - region_offset[3] - region_offset[1]

    # create image larger than what is required by codestream and then slice it to correct format and check if decoding is right
    start_x_offset = 150
    start_y_offset = 100
    end_x_offset = 200
    end_y_offset = 250

    num_channels = 3

    FILL_VALUE = 37

    array = buffer_create(
        (start_y_offset + end_y_offset + region_height, start_x_offset + end_x_offset + region_width, num_channels),
        dtype=dtype
    ) * FILL_VALUE
    slice = array[start_y_offset:-end_y_offset, start_x_offset:-end_x_offset]
    image = nvimgcodec.as_image(slice)

    params = nvimgcodec.DecodeParams(allow_any_depth=(dtype != np.uint8))

    decoded_external = decoder.decode(cs, params=params, image=image)
    reference = decoder.decode(cs, params=params,)

    assert decoded_external.height == region_height
    assert decoded_external.width == region_width

    np.testing.assert_allclose(reference.cpu(), decoded_external.cpu())
    np.testing.assert_allclose(decoded_external.cpu(), cp.asnumpy(slice))

    # check if outside of slice is left unchanged
    assert (array[:start_y_offset] == FILL_VALUE).all()
    assert (array[-end_y_offset:] == FILL_VALUE).all()
    assert (array[:, :start_x_offset] == FILL_VALUE).all()
    assert (array[:, -end_x_offset:] == FILL_VALUE).all()

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST + [
    "tiff/cat-300572_640_fp32.tiff",
])
@t.mark.parametrize("backends", [
    [nvimgcodec.BackendKind.GPU_ONLY, nvimgcodec.BackendKind.HYBRID_CPU_GPU],
    None, # use default backend)
])
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
@t.mark.parametrize("region_offset", [(31, 21, 37, 50), (-11, -13, -15, -17)])
def test_decode_to_external_buffer_slice_ROI(image_path, backends, buffer_create, region_offset):
    decode_to_external_buffer_slice_with_roi_impl(image_path, np.uint8, backends, buffer_create, region_offset)

@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST_uint16)
@t.mark.parametrize("backends", [
    [nvimgcodec.BackendKind.GPU_ONLY, nvimgcodec.BackendKind.HYBRID_CPU_GPU],
    None, # use default backend)
])
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
@t.mark.parametrize("region_offset", [(31, 21, 37, 50), (-11, -13, -15, -17)])
def test_decode_to_external_buffer_slice_ROI_uint16(image_path, backends, buffer_create, region_offset):
    decode_to_external_buffer_slice_with_roi_impl(image_path, np.uint16, backends, buffer_create, region_offset)

# cpu codecs supports only in bounds ROI decoding
@t.mark.parametrize("image_path", EXTERNAL_BUFFER_TEST_IMAGE_LIST + EXTERNAL_BUFFER_TEST_IMAGE_LIST_CPU_ONLY_CODECS)
@t.mark.parametrize("buffer_create", ONES_BUFFER_CREATE)
def test_decode_to_external_buffer_slice_ROI_cpu_only(image_path, buffer_create):
    decode_to_external_buffer_slice_with_roi_impl(
        image_path,
        np.uint8,
        [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
        buffer_create,
        (31, 21, 37, 50)
    )
