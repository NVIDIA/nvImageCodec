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
import numpy as np
from nvidia import nvimgcodec
import pytest as t
import os
from utils import is_nvjpeg_lossless_supported, img_dir_path

def get_ref(path):
    return os.path.splitext(path)[0] + '.npy'

@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
@t.mark.parametrize("image_path,dtype,precision",
                    [
                        ("jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg", np.uint16, 16),
                        ("jpeg/lossless/cat-3449999_640_grayscale_8bit.jpg", np.uint8, 8),
                        ("jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg", np.uint16, 12),
                        ("jpeg/lossless/cat-3449999_640_grayscale_16bit.jpg", np.uint16, 16),
                    ],
                    )
def test_decode_jpeg_lossless(image_path, dtype, precision):
    input_img_path = os.path.join(img_dir_path, image_path)
    ref_path = get_ref(input_img_path)
    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    img = decoder.read(input_img_path, params=params)
    img_cpu = np.array(img.cpu())
    assert(img.dtype == dtype)
    assert(img.precision == precision)
    ref = np.load(ref_path)
    np.testing.assert_allclose(img_cpu, ref)

@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
@t.mark.parametrize("image_path,dtype,precision",
                    [
                        ("jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg", np.uint16, 16),
                        ("jpeg/lossless/cat-3449999_640_grayscale_8bit.jpg", np.uint8, 8),
                        ("jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg", np.uint16, 12),
                        ("jpeg/lossless/cat-3449999_640_grayscale_16bit.jpg", np.uint16, 16),
                    ],
                    )
def test_decode_jpeg_lossless_batch(image_path, dtype, precision):
    input_img_path = os.path.join(img_dir_path, image_path)
    input_file_paths = [input_img_path] * 10
    ref_path = get_ref(input_img_path)
    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    imgs = decoder.read(input_file_paths, params=params)
    for img in imgs:
        img_cpu = np.array(img.cpu())
        assert(img.dtype == dtype)
        assert(img.precision == precision)
        ref = np.load(ref_path)
        np.testing.assert_allclose(img_cpu, ref)

@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
def test_decode_jpeg_lossless_batch_mixed_bit_depth():
    img_16bit = os.path.join(img_dir_path, "jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg")
    img_8bit = os.path.join(img_dir_path, "jpeg/lossless/cat-3449999_640_grayscale_8bit.jpg")
    img_12bit = os.path.join(img_dir_path, "jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg")

    ref_paths = [get_ref(img_16bit), get_ref(img_8bit), get_ref(img_12bit)]
    ref_dtypes = [np.uint16, np.uint8, np.uint16]
    ref_precisions = [16, 8, 12]
    input_file_paths = [img_16bit, img_8bit, img_12bit]
    
    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    images = decoder.read(input_file_paths, params=params)
    
    for ref_path, ref_dtype, ref_precision, img in zip(ref_paths, ref_dtypes, ref_precisions, images):
        ref = np.load(ref_path)
        img_cpu = np.array(img.cpu())
        assert(img.dtype == ref_dtype)
        assert(img.precision == ref_precision)
        np.testing.assert_allclose(img_cpu, ref)


@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
def test_decode_jpeg_lossless_default_decode_error():
    '''
    Conversion to uint8 RGB is not supported for lossless JPEGs, decoding should not work
    '''
    input_img_path = os.path.join(img_dir_path, "jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg")
    decoder = nvimgcodec.Decoder()
    img = decoder.read(input_img_path)
    assert img is None

@t.mark.skipif(not is_nvjpeg_lossless_supported(), reason="requires at least CUDA compute capability 6.0 (Linux) or 7.0 (Otherwise)")
def test_decode_jpeg_lossless_roi_unsupported():
    input_img_path = os.path.join(img_dir_path, "jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg")
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    roi = nvimgcodec.Region(int(100), int(100), int(200), int(200))
    cs = nvimgcodec.CodeStream(input_img_path).get_sub_code_stream(region=roi)
    decoder = nvimgcodec.Decoder()
    img = decoder.read(cs, params=params)
    assert img is None

