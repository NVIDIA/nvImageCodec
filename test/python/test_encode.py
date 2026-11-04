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
    img = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8) # Force to load necessary libraries
    cuda_streams = [None, cp.cuda.Stream(non_blocking=True), cp.cuda.Stream(non_blocking=False)]
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    cuda_streams = []
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None so it can be referenced in parametrize decorators

from nvidia import nvimgcodec
from utils import *


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
                nv_ref_img, codec=file_ext, params = encode_params, cuda_stream=as_cuda_stream_arg(cuda_stream))
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec=file_ext, params = encode_params)
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + file_ext)
        if cuda_stream:
            encoder.write(output_img_path, nv_ref_img,
                          params=encode_params, cuda_stream=as_cuda_stream_arg(cuda_stream))
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
                nv_ref_images, codec="jpeg2k", params=encode_params, cuda_stream=as_cuda_stream_arg(cuda_stream))
        else:
            test_encoded_images = encoder.encode(
                nv_ref_images, codec="jpeg2k", params=encode_params)
    else:
        output_img_paths = [os.path.join(tmp_path, os.path.splitext(
            os.path.basename(img))[0] + ".jp2") for img in input_images]
        if cuda_stream:
            encoder.write(output_img_paths, nv_ref_images,
                          params=encode_params, cuda_stream=as_cuda_stream_arg(cuda_stream))
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

def test_encode_png_different_quality_values():
    ref = np.zeros((256,256,3), dtype=np.uint8) + np.uint8(0.9 * np.iinfo(np.uint8).max)
    ref[100:120, 200:210, 0] = np.uint8(0.1 * np.iinfo(np.uint8).max)
    ref[100:120, 200:210, 1] = np.uint8(0.6 * np.iinfo(np.uint8).max)
    ref[100:120, 200:210, 2] = np.uint8(0.4 * np.iinfo(np.uint8).max)

    encoder = nvimgcodec.Encoder()
    encode_params_0 = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS, quality_value=0)
    encode_params_1 = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS, quality_value=1)
    encode_params_9 = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS, quality_value=9)

    encoded_image_0 = encoder.encode(ref, codec="png", params=encode_params_0)
    encoded_image_1 = encoder.encode(ref, codec="png", params=encode_params_1)
    encoded_image_9 = encoder.encode(ref, codec="png", params=encode_params_9)

    # Make sure that the encoded image is smaller when the quality value is higher (higher compression option is used)
    assert encoded_image_0.size > encoded_image_1.size
    assert encoded_image_1.size > encoded_image_9.size

    decoder = nvimgcodec.Decoder()
    dec_0 = np.array(decoder.decode(encoded_image_0).cpu())
    dec_1 = np.array(decoder.decode(encoded_image_1).cpu())
    dec_9 = np.array(decoder.decode(encoded_image_9).cpu())

    np.testing.assert_array_equal(dec_0, ref)
    np.testing.assert_array_equal(dec_1, ref)
    np.testing.assert_array_equal(dec_9, ref)


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

    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=100)
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
    when not explicitly specified (the multi-channel default is codec-dependent and
    covered by test_encode_default_chroma_subsampling_is_codec_dependent).
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


@t.mark.parametrize("codec,quality_type,expected_subsampling", [
    # JPEG defaults multi-channel images to 4:2:0 (broadly compatible, only mode the
    # nvJPEG hardware encoder accepts). Other codecs keep full-resolution 4:4:4.
    # jpeg2k is encoded lossless to avoid the unrelated "QUALITY needs MCT for SRGB" path.
    ("jpeg", nvimgcodec.QualityType.QUALITY, nvimgcodec.ChromaSubsampling.CSS_420),
    ("jpeg2k", nvimgcodec.QualityType.LOSSLESS, nvimgcodec.ChromaSubsampling.CSS_444),
])
def test_encode_default_chroma_subsampling_is_codec_dependent(codec, quality_type, expected_subsampling):
    """
    When chroma_subsampling is not specified, the default for a 3-channel image is
    codec-dependent: JPEG uses CSS_420 while other codecs keep full-resolution CSS_444.
    """
    fname = os.path.join(img_dir_path, 'bmp/cat-111793_640.bmp')

    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder()

    # Decode as RGB (3 channels), no explicit chroma_subsampling on encode.
    rgb_img = decoder.read(fname).cpu()
    assert rgb_img.shape[-1] == 3, f"Expected 3 channels, got {rgb_img.shape[-1]}"

    encode_params = nvimgcodec.EncodeParams(quality_type=quality_type, quality_value=75)
    assert encode_params.chroma_subsampling is None

    encoded = encoder.encode(rgb_img, codec=codec, params=encode_params)
    assert encoded is not None, f"Failed to encode image as {codec}"
    assert encoded.size > 0, f"Encoded {codec} data is empty"

    # Parse the produced bitstream back and confirm the subsampling that was actually written.
    parsed = nvimgcodec.CodeStream(bytes(bytearray(encoded)))
    assert parsed.chroma_subsampling == expected_subsampling, \
        f"{codec}: expected {expected_subsampling}, got {parsed.chroma_subsampling}"


def test_write_batch_mixed_codecs_chooses_subsampling_per_file(tmp_path):
    """Writing a batch to files with different extensions resolves the codec - and
    therefore the chroma-subsampling default - independently per file. The same
    3-channel image saved as .jpg gets 4:2:0 while saved as .jp2 keeps 4:4:4, all
    in a single write() call with no explicit codec or chroma_subsampling."""
    fname = os.path.join(img_dir_path, 'bmp/cat-111793_640.bmp')

    decoder = nvimgcodec.Decoder()
    encoder = nvimgcodec.Encoder()

    rgb_img = decoder.read(fname).cpu()
    assert rgb_img.shape[-1] == 3, f"Expected 3 channels, got {rgb_img.shape[-1]}"

    out_jpg = os.path.join(tmp_path, "out.jpg")
    out_jp2 = os.path.join(tmp_path, "out.jp2")

    # No explicit codec (inferred per file extension) and no explicit chroma_subsampling.
    written = encoder.write([out_jpg, out_jp2], [rgb_img, rgb_img])
    assert written == [out_jpg, out_jp2], f"unexpected write result: {written}"

    expected = {
        out_jpg: ("jpeg", nvimgcodec.ChromaSubsampling.CSS_420),
        out_jp2: ("jpeg2k", nvimgcodec.ChromaSubsampling.CSS_444),
    }
    for path, (codec, css) in expected.items():
        with open(path, "rb") as f:
            parsed = nvimgcodec.CodeStream(f.read())
        assert parsed.codec_name == codec, f"{path}: expected codec {codec}, got {parsed.codec_name}"
        assert parsed.chroma_subsampling == css, \
            f"{path}: expected {css}, got {parsed.chroma_subsampling}"

    # Input parsing is content-based and must not infer the codec from the filename.
    misleading_path = os.path.join(tmp_path, "jpeg_bytes_with_png_suffix.png")
    with open(out_jpg, "rb") as source, open(misleading_path, "wb") as destination:
        destination.write(source.read())
    assert nvimgcodec.CodeStream(misleading_path).codec_name == "jpeg"


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
                nv_ref_img, codec="tiff", params=None, cuda_stream=as_cuda_stream_arg(cuda_stream))
        else:
            test_encoded_img = encoder.encode(
                nv_ref_img, codec="tiff", params=None)
    else:
        base = os.path.basename(input_img_file)
        pre, ext = os.path.splitext(base)
        output_img_path = os.path.join(tmp_path, pre + ".tiff")
        if cuda_stream:
            encoder.write(output_img_path, nv_ref_img, params=None, cuda_stream=as_cuda_stream_arg(cuda_stream))
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
                nv_ref_images, codec="tiff", params=None, cuda_stream=as_cuda_stream_arg(cuda_stream))
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
            encoder.write(output_img_paths, nv_ref_images, params=None, cuda_stream=as_cuda_stream_arg(cuda_stream))
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
    rng = np.random.default_rng(2137)
    arr = rng.integers(0, 255, (1024, 1024, 4), dtype=np.uint8)
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

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(enc_code_stream, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED))
    assert decoded is not None
    decoded_numpy = np.asarray(decoded.cpu())

    np.testing.assert_array_equal(arr, decoded_numpy)

@t.mark.parametrize("lossy", [True, False])
def test_encode_jpeg2k_with_4_channels(lossy):
    encoder = nvimgcodec.Encoder()
    rng = np.random.default_rng(2137)
    arr = rng.integers(0, 255, (1024, 1024, 4), dtype=np.uint8)
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(mct_mode=1)
    )
    if lossy:
        encode_params.quality_type = nvimgcodec.QualityType.QUALITY
        encode_params.quality_value = 95

    enc_code_stream = encoder.encode(arr, "jpeg2k", params=encode_params)
    assert enc_code_stream is not None
    assert enc_code_stream.color_spec == nvimgcodec.ColorSpec.SRGB
    assert enc_code_stream.sample_format == nvimgcodec.SampleFormat.I_RGBA
    assert enc_code_stream.width == 1024
    assert enc_code_stream.height == 1024
    assert enc_code_stream.num_channels == 4
    assert enc_code_stream.dtype == np.uint8
    assert enc_code_stream.codec_name == "jpeg2k"
    assert enc_code_stream.size > 0

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(enc_code_stream, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED))
    assert decoded is not None
    decoded_numpy = np.asarray(decoded.cpu())

    if lossy:
        np.testing.assert_allclose(arr, decoded_numpy, atol=13)
    else:
        np.testing.assert_array_equal(arr, decoded_numpy)

@t.mark.parametrize("lossy", [True, False])
def test_encode_jpeg2k_ycc(lossy):
    encoder = nvimgcodec.Encoder()
    rng = np.random.default_rng(2137)
    arr = rng.integers(120, 150, (1024, 1024, 3), dtype=np.uint8)
    image = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.I_YCC, color_spec=nvimgcodec.ColorSpec.SYCC)
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        color_spec=nvimgcodec.ColorSpec.SYCC
    )
    if lossy:
        encode_params.quality_type = nvimgcodec.QualityType.QUALITY
        encode_params.quality_value = 95

    enc_code_stream = encoder.encode(image, "jpeg2k", params=encode_params)
    assert enc_code_stream is not None
    assert enc_code_stream.color_spec == nvimgcodec.ColorSpec.SYCC
    assert enc_code_stream.sample_format == nvimgcodec.SampleFormat.I_YCC
    assert enc_code_stream.width == 1024
    assert enc_code_stream.height == 1024
    assert enc_code_stream.num_channels == 3
    assert enc_code_stream.dtype == np.uint8
    assert enc_code_stream.codec_name == "jpeg2k"
    assert enc_code_stream.size > 0

    decoder = nvimgcodec.Decoder()
    # UNCHANGED decodes to rgb by default. Before this was working because nvjpeg2k didn't properly decode to RGB
    # but instead it skipped conversion YCC -> RGB, so it returned original code stream (even though we explicitly
    # requested decode to rgb via nvjpeg2kDecodeParamsSetRGBOutput)
    # The reason we decode to RGB with unchanged is because we don't properly handle decode to YCC with all codecs
    # we will need to revisit this in the future release

    # decoded = decoder.decode(enc_code_stream, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED))
    # assert decoded is not None
    # decoded_numpy = np.asarray(decoded.cpu())

    # if lossy:
    #     np.testing.assert_allclose(arr, decoded_numpy, atol=10)
    # else:
    #     np.testing.assert_array_equal(arr, decoded_numpy)

    #verify YCC output works as well
    decoded = decoder.decode(enc_code_stream, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.SYCC))
    assert decoded is not None
    decoded_numpy = np.asarray(decoded.cpu())

    if lossy:
        np.testing.assert_allclose(arr, decoded_numpy, atol=10)
    else:
        np.testing.assert_array_equal(arr, decoded_numpy)

@t.mark.parametrize(
    "dtype, precision, num_channels, color_spec, sample_format",
    [
        # Sub-8 precision carried in a uint8 buffer
        (np.uint8, 4, 1, nvimgcodec.ColorSpec.GRAY, nvimgcodec.SampleFormat.I_Y),
        (np.uint8, 5, 1, nvimgcodec.ColorSpec.GRAY, nvimgcodec.SampleFormat.I_Y),
        (np.uint8, 7, 1, nvimgcodec.ColorSpec.GRAY, nvimgcodec.SampleFormat.I_Y),
        (np.uint8, 5, 3, nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.I_RGB),
        (np.uint8, 6, 3, nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.I_RGB),
        # Supra-8 precision carried in a uint16 buffer
        (np.uint16, 10, 1, nvimgcodec.ColorSpec.GRAY, nvimgcodec.SampleFormat.I_Y),
        (np.uint16, 12, 1, nvimgcodec.ColorSpec.GRAY, nvimgcodec.SampleFormat.I_Y),
        (np.uint16, 14, 3, nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.I_RGB),
        (np.uint16, 12, 3, nvimgcodec.ColorSpec.SRGB, nvimgcodec.SampleFormat.I_RGB),
    ]
)
def test_encode_jpeg2k_custom_precision(dtype, precision, num_channels, color_spec, sample_format):
    """Encoding a buffer with plane_info.precision != dtype-bitdepth must store that
    precision in the JPEG 2000 bitstream (SIZ marker) and round-trip on decode. Covers
    both sub-8 precision (e.g. 5-bit in uint8) and supra-8 precision (e.g. 12-bit in uint16).
    Also serializes the encoded result to a Python bytes object and rebuilds a fresh
    CodeStream from those bytes to confirm the precision survives that round-trip too."""
    max_value = (1 << precision) - 1
    rng = np.random.default_rng(1729)
    shape = (64, 96, num_channels)
    arr = rng.integers(0, max_value + 1, shape, dtype=dtype)

    image = nvimgcodec.as_image(arr, sample_format=sample_format, color_spec=color_spec, precision=precision)
    assert image.precision == precision

    encoder = nvimgcodec.Encoder()
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded_image = encoder.encode(image, codec="jpeg2k", params=encode_params)
    assert encoded_image is not None
    # The encoder-returned CodeStream object itself reports the precision it just wrote.
    assert encoded_image.precision == precision

    # Serialize to raw bytes and re-parse as a fresh CodeStream. This exercises the path
    # an external consumer would take (e.g. read a file off disk), confirming the SIZ
    # marker carries the right precision through pure bitstream parsing.
    encoded_bytes = bytes(encoded_image)
    assert len(encoded_bytes) > 0
    parsed = nvimgcodec.CodeStream(encoded_bytes)
    assert parsed.precision == precision

    decoder = nvimgcodec.Decoder()
    decode_params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    # Decode via the freshly-parsed CodeStream, not the encoder's returned object, so
    # this branch verifies the bytes-only path round-trips pixels too.
    decoded = decoder.decode(parsed, params=decode_params)
    assert decoded.precision == precision

    decoded_np = np.array(decoded.cpu()).squeeze()
    np.testing.assert_array_equal(arr.squeeze(), decoded_np)


def test_encode_jpeg2k_precision_exceeding_dtype_is_rejected_by_as_image():
    """as_image guards the precision-vs-dtype invariant, so the invalid case
    is rejected at construction time before reaching the encoder."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    with t.raises(Exception) as excinfo:
        nvimgcodec.as_image(arr, precision=12)
    assert "exceeds the bitdepth" in str(excinfo.value)


def test_encode_jpeg2k_precision_8_in_uint8_round_trips():
    """Lower-bound case: precision == dtype bitdepth must still produce a
    bitstream whose parsed precision matches and whose pixels round-trip.
    Serializes the encoded image to bytes and rebuilds a fresh CodeStream from
    those bytes before checking precision and pixel preservation."""
    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
    image = nvimgcodec.as_image(arr, precision=8)
    encoder = nvimgcodec.Encoder()
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded = encoder.encode(image, codec="jpeg2k", params=encode_params)
    assert encoded is not None
    assert encoded.precision == 8

    encoded_bytes = bytes(encoded)
    assert len(encoded_bytes) > 0
    parsed = nvimgcodec.CodeStream(encoded_bytes)
    assert parsed.precision == 8

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(parsed, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))
    np.testing.assert_array_equal(arr, np.array(decoded.cpu()))


@t.mark.parametrize("lossy", [True, False])
def test_encode_jpeg2k_ycc_mct_not_supported(lossy):
    encoder = nvimgcodec.Encoder()
    rng = np.random.default_rng(2137)
    arr = rng.integers(120, 150, (1024, 1024, 3), dtype=np.uint8)
    image = nvimgcodec.as_image(arr, sample_format=nvimgcodec.SampleFormat.I_YCC, color_spec=nvimgcodec.ColorSpec.SYCC)
    encode_params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(mct_mode=1)
    )
    if lossy:
        encode_params.quality_type = nvimgcodec.QualityType.QUALITY
        encode_params.quality_value = 95

    enc_code_stream = encoder.encode(image, "jpeg2k", params=encode_params)
    assert enc_code_stream is None


# ---------------------------------------------------------------------------
# Encoding from planar (CHW) inputs
#
# Exercises the path where as_image wraps a CHW numpy/cupy array as a planar
# Image (P_RGB / P_Y / P_RGBA), the encoder reads from the planar buffer,
# and a round-trip through the decoder recovers identical pixel content (for
# lossless codecs).
# ---------------------------------------------------------------------------

_LOSSLESS_RGB_CODECS = ["png", "bmp", "tiff", "pnm"]
_LOSSLESS_GRAY_CODECS = ["png", "bmp", "tiff", "pnm"]
_LOSSLESS_RGBA_CODECS = ["png", "tiff"]


@t.mark.parametrize("codec", _LOSSLESS_RGB_CODECS)
def test_encode_planar_rgb_roundtrip_lossless(codec):
    """Encode a (3, H, W) P_RGB Image with a lossless codec and verify the
    round-tripped pixels match the input exactly."""
    rng = np.random.default_rng(0)
    arr_chw = rng.integers(0, 256, (3, 120, 160), dtype=np.uint8)
    img = nvimgcodec.as_image(arr_chw, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGB

    encoder = nvimgcodec.Encoder()
    encoded = encoder.encode(img, codec)
    assert encoded is not None, f"encoder returned None for codec={codec}"

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(encoded)
    assert decoded is not None, f"decoder returned None for codec={codec}"
    assert decoded.shape == (120, 160, 3)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), np.transpose(arr_chw, (1, 2, 0)))


def test_encode_planar_rgb_jpeg2k_lossless_roundtrip():
    """JPEG2000 lossless round-trip for a (3, H, W) P_RGB Image."""
    rng = np.random.default_rng(0)
    arr_chw = rng.integers(0, 256, (3, 120, 160), dtype=np.uint8)
    img = nvimgcodec.as_image(arr_chw, sample_format=nvimgcodec.SampleFormat.P_RGB)

    encoder = nvimgcodec.Encoder()
    params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    encoded = encoder.encode(img, "jpeg2k", params=params)
    assert encoded is not None

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(encoded)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), np.transpose(arr_chw, (1, 2, 0)))


@t.mark.parametrize("codec", _LOSSLESS_GRAY_CODECS)
def test_encode_planar_y_roundtrip_lossless(codec):
    """Encode a (1, H, W) P_Y Image with a lossless codec and verify the
    round-tripped pixels match. Decoded shape is (H, W, 1) after the default
    SRGB / GRAY conversion the decoder applies."""
    rng = np.random.default_rng(0)
    arr_1hw = rng.integers(0, 256, (1, 120, 160), dtype=np.uint8)
    img = nvimgcodec.as_image(arr_1hw, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.P_Y

    encoder = nvimgcodec.Encoder()
    encoded = encoder.encode(img, codec)
    assert encoded is not None, f"encoder returned None for codec={codec}"

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY))
    assert decoded is not None, f"decoder returned None for codec={codec}"
    assert decoded.shape == (120, 160, 1)
    np.testing.assert_array_equal(
        np.asarray(decoded.cpu()).squeeze(-1), arr_1hw[0])


def test_encode_planar_y_from_2d_roundtrip_lossless():
    """A 2-D (H, W) grayscale array wrapped as P_Y also encodes correctly."""
    rng = np.random.default_rng(0)
    arr_2d = rng.integers(0, 256, (120, 160), dtype=np.uint8)
    img = nvimgcodec.as_image(arr_2d, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert img.sample_format == nvimgcodec.SampleFormat.P_Y
    assert img.shape == (1, 120, 160)

    encoder = nvimgcodec.Encoder()
    encoded = encoder.encode(img, "png")
    assert encoded is not None

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY))
    np.testing.assert_array_equal(np.asarray(decoded.cpu()).squeeze(-1), arr_2d)


@t.mark.parametrize("codec", _LOSSLESS_RGBA_CODECS)
@t.mark.parametrize(
    "decode_sample_format, expected_layout",
    [
        (nvimgcodec.SampleFormat.I_RGBA, "interleaved"),
        (nvimgcodec.SampleFormat.P_RGBA, "planar"),
    ],
)
def test_encode_planar_rgba_roundtrip_lossless(codec, decode_sample_format, expected_layout):
    """Encode a (4, H, W) P_RGBA Image with a lossless codec preserving the
    alpha channel through the round-trip, and decode back to either layout."""
    rng = np.random.default_rng(0)
    arr_chw = rng.integers(0, 256, (4, 120, 160), dtype=np.uint8)
    img = nvimgcodec.as_image(arr_chw, sample_format=nvimgcodec.SampleFormat.P_RGBA)
    assert img.sample_format == nvimgcodec.SampleFormat.P_RGBA

    encoder = nvimgcodec.Encoder()
    encoded = encoder.encode(img, codec)
    assert encoded is not None, f"encoder returned None for codec={codec}"

    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, sample_format=decode_sample_format)
    decoded = decoder.decode(encoded, params=params)
    assert decoded is not None, f"decoder returned None for codec={codec}"
    assert decoded.sample_format == decode_sample_format
    if expected_layout == "interleaved":
        assert decoded.shape == (120, 160, 4)
        np.testing.assert_array_equal(
            np.asarray(decoded.cpu()), np.transpose(arr_chw, (1, 2, 0)))
    else:
        assert decoded.shape == (4, 120, 160)
        np.testing.assert_array_equal(np.asarray(decoded.cpu()), arr_chw)


@t.mark.parametrize("codec", _LOSSLESS_RGB_CODECS)
def test_encode_planar_vs_interleaved_same_pixels(codec):
    """Encoding the same data wrapped as planar (P_RGB on CHW) and as
    interleaved (I_RGB on HWC) must decode to the same pixels."""
    rng = np.random.default_rng(0)
    arr_chw = rng.integers(0, 256, (3, 120, 160), dtype=np.uint8)
    arr_hwc = np.ascontiguousarray(np.transpose(arr_chw, (1, 2, 0)))

    encoder = nvimgcodec.Encoder()
    enc_planar = encoder.encode(
        nvimgcodec.as_image(arr_chw, sample_format=nvimgcodec.SampleFormat.P_RGB), codec)
    enc_interleaved = encoder.encode(
        nvimgcodec.as_image(arr_hwc, sample_format=nvimgcodec.SampleFormat.I_RGB), codec)
    assert enc_planar is not None and enc_interleaved is not None

    decoder = nvimgcodec.Decoder()
    dec_planar = np.asarray(decoder.decode(enc_planar).cpu())
    dec_interleaved = np.asarray(decoder.decode(enc_interleaved).cpu())
    np.testing.assert_array_equal(dec_planar, dec_interleaved)


# ---------------------------------------------------------------------------
# Lossy planar encoding (quality=95)
#
# A pseudo-realistic source (a real decoded image) is used; uniform random
# data is a worst case for lossy codecs because there's no spatial coherence
# to exploit and the diffs blow up.
# ---------------------------------------------------------------------------

_LOSSY_DIFF_THRESHOLDS = {
    # codec -> (max_diff, mean_diff) — empirically what we get at quality=95
    # on the cat reference image. Loose enough to absorb small encoder-side
    # version drift but tight enough to catch real bit-rot.
    # "jpeg" pins 4:4:4; "jpeg_420" is the same codec at 4:2:0, which subsamples
    # chroma and so needs a looser max-diff bound.
    "jpeg":      (20, 3.0),
    "jpeg_420":  (30, 4.0),
    "webp":      (20, 2.5),
    "jpeg2k":    (15, 2.0),
    "jpeg2k_ht": (15, 2.0),
}


def _lossy_encode_params(codec_label):
    """Build EncodeParams with quality=95 and any codec-specific knobs needed
    for the lossy variant. JPEG2K needs mct_mode=1 to allow Q-factor mode;
    htj2k is selected via Jpeg2kEncodeParams(ht=True). JPEG is tested at both
    chroma subsamplings: "jpeg" pins 4:4:4 and "jpeg_420" pins 4:2:0 (the
    multi-channel default), with looser thresholds for the latter."""
    kwargs = dict(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=95)
    if codec_label == "jpeg":
        kwargs["chroma_subsampling"] = nvimgcodec.ChromaSubsampling.CSS_444
    elif codec_label == "jpeg_420":
        kwargs["chroma_subsampling"] = nvimgcodec.ChromaSubsampling.CSS_420
    elif codec_label == "jpeg2k":
        kwargs["jpeg2k_encode_params"] = nvimgcodec.Jpeg2kEncodeParams(mct_mode=1)
    elif codec_label == "jpeg2k_ht":
        kwargs["jpeg2k_encode_params"] = nvimgcodec.Jpeg2kEncodeParams(mct_mode=1, ht=True)
    return nvimgcodec.EncodeParams(**kwargs)


def _codec_name(codec_label):
    # "jpeg2k_ht" / "jpeg_420" are labels used to distinguish encode variants in
    # test ids - on the wire they go through the jpeg2k / jpeg codec respectively.
    return {"jpeg2k_ht": "jpeg2k", "jpeg_420": "jpeg"}.get(codec_label, codec_label)


def _lossy_reference_image():
    """Load a real reference image so lossy codecs behave sensibly."""
    path = os.path.join(img_dir_path, "png/cat-1245673_640.png")
    return np.asarray(nvimgcodec.Decoder().decode(path).cpu())


@t.mark.parametrize("codec_label", ["jpeg", "jpeg_420", "webp", "jpeg2k", "jpeg2k_ht"])
def test_encode_planar_lossy_q95_roundtrip(codec_label):
    """Encode a (3, H, W) P_RGB image with a lossy codec at quality=95 and
    verify the decoded image stays close to the original."""
    src_hwc = _lossy_reference_image()
    src_chw = np.ascontiguousarray(np.transpose(src_hwc, (2, 0, 1)))
    img = nvimgcodec.as_image(src_chw, sample_format=nvimgcodec.SampleFormat.P_RGB)

    encoder = nvimgcodec.Encoder()
    encoded = encoder.encode(img, _codec_name(codec_label),
                             params=_lossy_encode_params(codec_label))
    assert encoded is not None, f"encoder returned None for {codec_label}"

    decoded = np.asarray(nvimgcodec.Decoder().decode(encoded).cpu())
    diff = np.abs(decoded.astype(int) - src_hwc.astype(int))
    max_thr, mean_thr = _LOSSY_DIFF_THRESHOLDS[codec_label]
    assert diff.max() <= max_thr, \
        f"{codec_label} max diff {diff.max()} exceeds threshold {max_thr}"
    assert diff.mean() <= mean_thr, \
        f"{codec_label} mean diff {diff.mean():.2f} exceeds threshold {mean_thr}"


@t.mark.parametrize("codec_label", ["jpeg", "jpeg_420", "webp", "jpeg2k", "jpeg2k_ht"])
def test_encode_planar_lossy_q95_matches_interleaved(codec_label):
    """Encoding the same data wrapped as planar (P_RGB on CHW) and as
    interleaved (I_RGB on HWC) must decode to byte-identical pixels for a
    given lossy codec at quality=95."""
    src_hwc = _lossy_reference_image()
    src_chw = np.ascontiguousarray(np.transpose(src_hwc, (2, 0, 1)))

    encoder = nvimgcodec.Encoder()
    encoded_planar = encoder.encode(
        nvimgcodec.as_image(src_chw, sample_format=nvimgcodec.SampleFormat.P_RGB),
        _codec_name(codec_label), params=_lossy_encode_params(codec_label))
    encoded_interleaved = encoder.encode(
        nvimgcodec.as_image(src_hwc, sample_format=nvimgcodec.SampleFormat.I_RGB),
        _codec_name(codec_label), params=_lossy_encode_params(codec_label))
    assert encoded_planar is not None and encoded_interleaved is not None

    decoder = nvimgcodec.Decoder()
    dec_planar = np.asarray(decoder.decode(encoded_planar).cpu())
    dec_interleaved = np.asarray(decoder.decode(encoded_interleaved).cpu())
    np.testing.assert_array_equal(dec_planar, dec_interleaved)


# ---------------------------------------------------------------------------
# Encoding from row-padded buffers
#
# Mirrors the padded-decode matrix in test_decode_sample_format.py: the source
# is an externally-managed buffer with right-side row padding wrapped via
# as_image; the encoder must read pixels using the per-plane row_stride (not
# the packed natural row size) and must not touch the padded zone. The
# round-trip is lossless, so the decoded output equals the visible slice
# byte-for-byte.
#
# Parametrised over:
#   - backend      : default vs CPU_ONLY
#   - array_module : numpy (host) vs cupy (device)
#   - layout       : interleaved (HWC) vs planar (CHW), where applicable
#   - codec        : a lossless RGB / GRAY / RGBA codec set
# ---------------------------------------------------------------------------

PADDED_ENCODE_BACKENDS = [
    ("default", None),
    ("cpu",     [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)]),
]

PADDED_ENCODE_ARRAY_MODULES = [
    ("numpy", np),
    t.param("cupy", cp,
            marks=t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")),
]

PADDED_ENCODE_LOSSLESS_RGB_CODECS  = ["png", "bmp", "tiff", "pnm"]
PADDED_ENCODE_LOSSLESS_GRAY_CODECS = ["png", "bmp", "tiff", "pnm"]
PADDED_ENCODE_LOSSLESS_RGBA_CODECS = ["png", "tiff"]

PADDED_ENCODE_SENTINEL = 0x80


def _padded_encoder(backends):
    if backends is None:
        return nvimgcodec.Encoder()
    return nvimgcodec.Encoder(
        device_id=nvimgcodec.NVIMGCODEC_DEVICE_CPU_ONLY, backends=backends)


def _padded_encode_as_host(arr):
    if CUPY_AVAILABLE and isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _padded_encode_random(am, shape, seed):
    """Reproducible uint8 data on the given array module."""
    if am is np:
        rng = np.random.default_rng(seed)
        return rng.integers(0, 256, shape, dtype=np.uint8)
    # cupy: seed once via cupy.random, fall back to numpy + asarray for
    # determinism across cupy versions.
    arr = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    return cp.asarray(arr)


def _padded_sentinel(am, shape):
    """A sentinel-only reference array on the same module as the test buffer.
    Used to assert post-encode that the padded zone of the source buffer is
    still byte-for-byte equal to the sentinel pre-fill."""
    return np.full(shape, PADDED_ENCODE_SENTINEL, dtype=np.uint8)


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_RGB_CODECS)
def test_encode_padded_hwc_rgb_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a row-padded interleaved (H, W, 3) source. Verify the encoder
    honours the row stride, leaves the padded zone untouched, and a decode
    of the result recovers the visible pixels exactly."""
    H, W, pad = 120, 160, 32
    backing = am.full((H, W + pad, 3), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (H, W, 3), seed=1)
    backing[:, :W, :] = pixels
    view = backing[:, :W, :]
    src = nvimgcodec.as_image(view)
    assert src.shape == (H, W, 3)
    assert src.strides[0] == (W + pad) * 3

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    # Encoder must not modify the source buffer.
    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, W:, :]),
        _padded_sentinel(am, (H, pad, 3)))

    # Lossless round-trip recovers the visible pixels.
    decoded = nvimgcodec.Decoder().decode(encoded)
    assert decoded is not None
    assert decoded.shape == (H, W, 3)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), _padded_encode_as_host(pixels))


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_RGB_CODECS)
def test_encode_padded_chw_rgb_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a row-padded planar (3, H, W) source - the planar branch must
    walk each plane at the padded row stride and skip the per-row tail."""
    H, W, pad = 120, 160, 32
    backing = am.full((3, H, W + pad), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (3, H, W), seed=2)
    backing[:, :, :W] = pixels
    view = backing[:, :, :W]
    src = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert src.shape == (3, H, W)
    assert src.strides[0] == H * (W + pad)
    assert src.strides[1] == (W + pad)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, :, W:]),
        _padded_sentinel(am, (3, H, pad)))

    decoded = nvimgcodec.Decoder().decode(encoded)
    assert decoded is not None
    # Most lossless codecs decode to HWC by default; transpose CHW reference
    # for the comparison.
    assert decoded.shape == (H, W, 3)
    np.testing.assert_array_equal(
        np.asarray(decoded.cpu()),
        np.transpose(_padded_encode_as_host(pixels), (1, 2, 0)))


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_GRAY_CODECS)
def test_encode_padded_hwc_gray_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a row-padded interleaved grayscale (H, W, 1) source."""
    H, W, pad = 120, 160, 16
    backing = am.full((H, W + pad, 1), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (H, W, 1), seed=3)
    backing[:, :W, :] = pixels
    view = backing[:, :W, :]
    src = nvimgcodec.as_image(view)
    assert src.sample_format == nvimgcodec.SampleFormat.I_Y
    assert src.shape == (H, W, 1)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, W:, :]),
        _padded_sentinel(am, (H, pad, 1)))

    decoded = nvimgcodec.Decoder().decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY))
    assert decoded is not None
    assert decoded.shape == (H, W, 1)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), _padded_encode_as_host(pixels))


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_GRAY_CODECS)
def test_encode_padded_chw_gray_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a row-padded planar grayscale (1, H, W) source via P_Y."""
    H, W, pad = 120, 160, 16
    backing = am.full((1, H, W + pad), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (1, H, W), seed=4)
    backing[:, :, :W] = pixels
    view = backing[:, :, :W]
    src = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_Y)
    assert src.sample_format == nvimgcodec.SampleFormat.P_Y
    assert src.shape == (1, H, W)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, :, W:]),
        _padded_sentinel(am, (1, H, pad)))

    decoded = nvimgcodec.Decoder().decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY))
    assert decoded is not None
    assert decoded.shape == (H, W, 1)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()).squeeze(-1), _padded_encode_as_host(pixels)[0])


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_GRAY_CODECS)
def test_encode_padded_2d_gray_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a 2-D (H, W+pad) grayscale source - the default I_Y inference
    plus the row-padding handling on the 2-D shape path."""
    H, W, pad = 120, 160, 16
    backing = am.full((H, W + pad), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (H, W), seed=5)
    backing[:, :W] = pixels
    view = backing[:, :W]
    src = nvimgcodec.as_image(view)
    assert src.sample_format == nvimgcodec.SampleFormat.I_Y
    assert src.shape == (H, W, 1)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, W:]),
        _padded_sentinel(am, (H, pad)))

    decoded = nvimgcodec.Decoder().decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.GRAY))
    assert decoded is not None
    assert decoded.shape == (H, W, 1)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()).squeeze(-1), _padded_encode_as_host(pixels))


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_RGBA_CODECS)
def test_encode_padded_hwc_rgba_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a row-padded interleaved RGBA (H, W, 4) source - covers the
    4-channel path where row_stride = (W + pad) * 4."""
    H, W, pad = 120, 160, 32
    backing = am.full((H, W + pad, 4), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (H, W, 4), seed=6)
    backing[:, :W, :] = pixels
    view = backing[:, :W, :]
    src = nvimgcodec.as_image(view)
    assert src.sample_format == nvimgcodec.SampleFormat.I_RGBA
    assert src.shape == (H, W, 4)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, W:, :]),
        _padded_sentinel(am, (H, pad, 4)))

    decoded = nvimgcodec.Decoder().decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED))
    assert decoded is not None
    assert decoded.shape == (H, W, 4)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), _padded_encode_as_host(pixels))


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec", PADDED_ENCODE_LOSSLESS_RGBA_CODECS)
def test_encode_padded_chw_rgba_lossless_roundtrip(codec, backend_name, backends, am_name, am):
    """Encode a row-padded planar RGBA (4, H, W) source - 4-plane planar
    walked at the padded row stride per plane."""
    H, W, pad = 120, 160, 32
    backing = am.full((4, H, W + pad), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    pixels = _padded_encode_random(am, (4, H, W), seed=7)
    backing[:, :, :W] = pixels
    view = backing[:, :, :W]
    src = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGBA)
    assert src.sample_format == nvimgcodec.SampleFormat.P_RGBA
    assert src.shape == (4, H, W)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, codec)
    assert encoded is not None, f"encoder returned None for codec={codec} backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view), _padded_encode_as_host(pixels))
    np.testing.assert_array_equal(
        _padded_encode_as_host(backing[:, :, W:]),
        _padded_sentinel(am, (4, H, pad)))

    decoded = nvimgcodec.Decoder().decode(
        encoded, params=nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED))
    assert decoded is not None
    assert decoded.shape == (H, W, 4)
    np.testing.assert_array_equal(
        np.asarray(decoded.cpu()),
        np.transpose(_padded_encode_as_host(pixels), (1, 2, 0)))


# ---------------------------------------------------------------------------
# Padded-input LOSSY encode tests
#
# Same backend / array_module / layout matrix as the lossless padded tests
# above, but with lossy codecs (jpeg, webp, jpeg2k baseline, jpeg2k HT) at
# quality=95. The source is a real reference image so the codecs behave
# sensibly. Each test:
#
#   1. Pre-fills the backing buffer with sentinel and copies the reference
#      pixels into the visible slice.
#   2. Wraps the slice via as_image with the appropriate layout.
#   3. Encodes with the lossy codec / backend under test.
#   4. Asserts the encoder did not touch the input slice or the padded zone.
#   5. Decodes the encoded output and asserts the lossy diff stays within
#      the published thresholds vs the reference visible pixels.
#   6. Also encodes the same pixels from a packed buffer and asserts the
#      decoded output is byte-identical between the padded and packed runs
#      (proves the padded row stride is honoured by the encoder).
# ---------------------------------------------------------------------------


_PADDED_LOSSY_CODECS = ["jpeg", "jpeg_420", "webp", "jpeg2k", "jpeg2k_ht"]


def _skip_unsupported_padded_lossy(codec_label, backend_name):
    """No CPU JPEG2K encoder is shipped in this build, so jpeg2k variants
    are unavailable on the cpu-only backend. Skip those parametrisations
    up-front instead of letting `encoder.encode()` return None at runtime."""
    if backend_name == "cpu" and codec_label in ("jpeg2k", "jpeg2k_ht"):
        t.skip(f"{codec_label} encoder not available for backend=cpu")


def _padded_lossy_pixels(am):
    """The lossy reference image, on the requested array module."""
    src_hwc = _lossy_reference_image()
    if am is np:
        return src_hwc, src_hwc
    return cp.asarray(src_hwc), src_hwc


def _padded_lossy_assert_within_thresholds(codec_label, decoded_hwc, ref_hwc):
    """Apply the codec's published quality=95 thresholds."""
    max_thr, mean_thr = _LOSSY_DIFF_THRESHOLDS[codec_label]
    diff = np.abs(decoded_hwc.astype(int) - ref_hwc.astype(int))
    assert diff.max() <= max_thr, \
        f"{codec_label} max diff {diff.max()} exceeds threshold {max_thr}"
    assert diff.mean() <= mean_thr, \
        f"{codec_label} mean diff {diff.mean():.2f} exceeds threshold {mean_thr}"


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec_label", _PADDED_LOSSY_CODECS)
def test_encode_padded_hwc_lossy_q95_roundtrip(codec_label, backend_name, backends, am_name, am):
    """Lossy encode of a row-padded interleaved (H, W, 3) source at q=95.
    The decoded image must stay within the codec's published thresholds vs
    the reference visible pixels, and the encoder must not touch the
    padded zone."""
    _skip_unsupported_padded_lossy(codec_label, backend_name)
    pixels_on_am, ref_hwc = _padded_lossy_pixels(am)
    H, W = ref_hwc.shape[:2]
    pad = 32

    backing = am.full((H, W + pad, 3), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    backing[:, :W, :] = pixels_on_am
    view = backing[:, :W, :]
    src = nvimgcodec.as_image(view)
    assert src.shape == (H, W, 3)
    assert src.strides[0] == (W + pad) * 3

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, _codec_name(codec_label),
                             params=_lossy_encode_params(codec_label))
    assert encoded is not None, f"{codec_label} encoder returned None for backend={backend_name}"

    # Input slice + padded zone untouched.
    np.testing.assert_array_equal(_padded_encode_as_host(view),
                                  _padded_encode_as_host(pixels_on_am))
    np.testing.assert_array_equal(_padded_encode_as_host(backing[:, W:, :]),
                                  _padded_sentinel(am, (H, pad, 3)))

    # Lossy threshold on round-trip.
    decoded = np.asarray(nvimgcodec.Decoder().decode(encoded).cpu())
    _padded_lossy_assert_within_thresholds(codec_label, decoded, ref_hwc)


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec_label", _PADDED_LOSSY_CODECS)
def test_encode_padded_chw_lossy_q95_roundtrip(codec_label, backend_name, backends, am_name, am):
    """Lossy encode of a row-padded planar (3, H, W) source at q=95."""
    _skip_unsupported_padded_lossy(codec_label, backend_name)
    pixels_on_am, ref_hwc = _padded_lossy_pixels(am)
    H, W = ref_hwc.shape[:2]
    pad = 32

    backing = am.full((3, H, W + pad), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    if am is np:
        chw_pixels = np.ascontiguousarray(np.transpose(pixels_on_am, (2, 0, 1)))
    else:
        chw_pixels = cp.ascontiguousarray(cp.transpose(pixels_on_am, (2, 0, 1)))
    backing[:, :, :W] = chw_pixels
    view = backing[:, :, :W]
    src = nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB)
    assert src.shape == (3, H, W)
    assert src.strides[0] == H * (W + pad)
    assert src.strides[1] == (W + pad)

    encoder = _padded_encoder(backends)
    encoded = encoder.encode(src, _codec_name(codec_label),
                             params=_lossy_encode_params(codec_label))
    assert encoded is not None, f"{codec_label} encoder returned None for backend={backend_name}"

    np.testing.assert_array_equal(_padded_encode_as_host(view),
                                  _padded_encode_as_host(chw_pixels))
    np.testing.assert_array_equal(_padded_encode_as_host(backing[:, :, W:]),
                                  _padded_sentinel(am, (3, H, pad)))

    decoded = np.asarray(nvimgcodec.Decoder().decode(encoded).cpu())
    _padded_lossy_assert_within_thresholds(codec_label, decoded, ref_hwc)


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec_label", _PADDED_LOSSY_CODECS)
def test_encode_padded_hwc_lossy_matches_packed(codec_label, backend_name, backends, am_name, am):
    """Encoding the same RGB pixels from a row-padded HWC buffer and from a
    packed HWC buffer must decode to byte-identical pixels under any lossy
    codec - if the encoder mis-strides the padded layout the two outputs
    diverge."""
    _skip_unsupported_padded_lossy(codec_label, backend_name)
    pixels_on_am, ref_hwc = _padded_lossy_pixels(am)
    H, W = ref_hwc.shape[:2]
    pad = 32

    backing = am.full((H, W + pad, 3), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    backing[:, :W, :] = pixels_on_am
    view = backing[:, :W, :]
    packed = am.ascontiguousarray(pixels_on_am)

    encoder = _padded_encoder(backends)
    params = _lossy_encode_params(codec_label)
    enc_padded = encoder.encode(nvimgcodec.as_image(view), _codec_name(codec_label), params=params)
    enc_packed = encoder.encode(nvimgcodec.as_image(packed), _codec_name(codec_label), params=params)
    assert enc_padded is not None and enc_packed is not None, \
        f"{codec_label} encoder returned None for backend={backend_name}"

    decoder = nvimgcodec.Decoder()
    dec_padded = np.asarray(decoder.decode(enc_padded).cpu())
    dec_packed = np.asarray(decoder.decode(enc_packed).cpu())
    np.testing.assert_array_equal(dec_padded, dec_packed)


@t.mark.parametrize(
    "input_img_file",
    [
        "jpeg/exif/padlock-406986_640_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
        "jpeg/exif/padlock-406986_640_no_orientation.jpg",
        "jpeg/exif/padlock-406986_640_rotate_180.jpg",
        "jpeg/exif/padlock-406986_640_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",
    ]
)
def test_encode_roundtrip_exif_oriented_image(input_img_file):
    """
    An image decoded with apply_exif_orientation=True (the default) has already had its
    orientation baked into the pixels, so the decoder must clear the decoded Image's
    orientation metadata; otherwise re-encoding it would be rejected by the orientation
    check added for non-identity orientations (encoder.encode returns None on rejection).
    """
    input_img_path = os.path.join(img_dir_path, input_img_file)
    decoder = nvimgcodec.Decoder()
    with open(input_img_path, 'rb') as in_file:
        decoded_img = decoder.decode(in_file.read())

    # A 90/270 degree rotation transposes the code stream's dimensions for display.
    code_stream = nvimgcodec.CodeStream(input_img_path)
    if "rotate_90" in input_img_file or "rotate_270" in input_img_file:
        assert decoded_img.shape[:2] == (code_stream.width, code_stream.height)
    else:
        assert decoded_img.shape[:2] == (code_stream.height, code_stream.width)

    # Lossless, so the encoded pixels are exactly the oriented decode.
    encoder = nvimgcodec.Encoder()
    encoded_img = encoder.encode(decoded_img, codec=".png")
    assert encoded_img is not None

    # Decode the re-encoded image with OpenCV and compare against OpenCV's own EXIF-oriented
    # decode of the source, so both shape and pixels are checked against an independent reference.
    test_img = get_opencv_reference(np.asarray(bytearray(encoded_img)))
    ref_img = get_opencv_reference(input_img_path)
    compare_image(test_img, ref_img)


@t.mark.parametrize(
    "input_img_file,has_orientation",
    [
        ("jpeg/exif/padlock-406986_640_horizontal.jpg", False),
        ("jpeg/exif/padlock-406986_640_no_orientation.jpg", False),
        ("jpeg/exif/padlock-406986_640_mirror_horizontal.jpg", True),
        ("jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg", True),
        ("jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg", True),
        ("jpeg/exif/padlock-406986_640_mirror_vertical.jpg", True),
        ("jpeg/exif/padlock-406986_640_rotate_180.jpg", True),
        ("jpeg/exif/padlock-406986_640_rotate_270.jpg", True),
        ("jpeg/exif/padlock-406986_640_rotate_90.jpg", True),
    ]
)
def test_encode_rejects_image_decoded_without_exif_orientation(input_img_file, has_orientation):
    """
    With apply_exif_orientation=False the pixels stay in code stream orientation and the decoded
    Image keeps the EXIF orientation as a transform still to be applied. Encoders do not apply
    orientation, so encoding such an image is rejected (None) rather than silently dropping it.
    """
    input_img_path = os.path.join(img_dir_path, input_img_file)
    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(apply_exif_orientation=False)
    decoded_img = decoder.read(input_img_path, params=params)
    assert decoded_img is not None

    encoded_img = nvimgcodec.Encoder().encode(decoded_img, codec=".png")
    if has_orientation:
        assert encoded_img is None
    else:
        assert encoded_img is not None


@t.mark.parametrize("am_name,am", PADDED_ENCODE_ARRAY_MODULES)
@t.mark.parametrize("backend_name,backends", PADDED_ENCODE_BACKENDS,
                    ids=[b[0] for b in PADDED_ENCODE_BACKENDS])
@t.mark.parametrize("codec_label", _PADDED_LOSSY_CODECS)
def test_encode_padded_chw_lossy_matches_packed(codec_label, backend_name, backends, am_name, am):
    """Mirror of the HWC matches-packed test for planar layouts."""
    _skip_unsupported_padded_lossy(codec_label, backend_name)
    pixels_on_am, ref_hwc = _padded_lossy_pixels(am)
    H, W = ref_hwc.shape[:2]
    pad = 32

    if am is np:
        chw_pixels = np.ascontiguousarray(np.transpose(pixels_on_am, (2, 0, 1)))
    else:
        chw_pixels = cp.ascontiguousarray(cp.transpose(pixels_on_am, (2, 0, 1)))
    backing = am.full((3, H, W + pad), PADDED_ENCODE_SENTINEL, dtype=am.uint8)
    backing[:, :, :W] = chw_pixels
    view = backing[:, :, :W]
    packed = am.ascontiguousarray(chw_pixels)

    encoder = _padded_encoder(backends)
    params = _lossy_encode_params(codec_label)
    enc_padded = encoder.encode(
        nvimgcodec.as_image(view, sample_format=nvimgcodec.SampleFormat.P_RGB),
        _codec_name(codec_label), params=params)
    enc_packed = encoder.encode(
        nvimgcodec.as_image(packed, sample_format=nvimgcodec.SampleFormat.P_RGB),
        _codec_name(codec_label), params=params)
    assert enc_padded is not None and enc_packed is not None, \
        f"{codec_label} encoder returned None for backend={backend_name}"

    decoder = nvimgcodec.Decoder()
    dec_padded = np.asarray(decoder.decode(enc_padded).cpu())
    dec_packed = np.asarray(decoder.decode(enc_packed).cpu())
    np.testing.assert_array_equal(dec_padded, dec_packed)
