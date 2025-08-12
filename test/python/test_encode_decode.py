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

import numpy as np
import pytest as t
from nvidia import nvimgcodec
import nvjpeg_test_speedup

backends_cpu_only = [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]
backends_gpu_only = [nvimgcodec.Backend(nvimgcodec.GPU_ONLY), nvimgcodec.Backend(nvimgcodec.HYBRID_CPU_GPU)]
default_image_shape = (480, 640, 3)

def encode_decode(extension, backends, dtype, shape, max_mean_diff=None, encode_params=None, decode_params=None):
    encoder = nvimgcodec.Encoder(backends=backends)
    decoder = nvimgcodec.Decoder(backends=backends)

    image = np.random.randint(np.iinfo(dtype).min, np.iinfo(dtype).max, shape, dtype)
    encoded = encoder.encode(image, extension, encode_params)
    assert encoded is not None

    decoded_gpu = decoder.decode(encoded, decode_params)
    assert decoded_gpu is not None

    decoded = np.asarray(decoded_gpu.cpu())
    assert decoded.dtype == dtype

    mean_diff = np.abs(image.astype(np.int32) - decoded.astype(np.int32)).mean()

    if max_mean_diff is None:
        assert mean_diff == 0.0
    else:
        assert mean_diff > 0 and mean_diff / (np.iinfo(dtype).max - np.iinfo(dtype).min) < max_mean_diff

def encode_decode_lossless(extension, backends, dtype, shape):
    assert extension != "jpeg", "currently only lossy jpeg encoding is supported"

    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    encode_decode(extension, backends, dtype, shape, max_mean_diff=None,
                    encode_params=encode_params, decode_params=decode_params)

def encode_decode_lossy(extension, backends, dtype, shape):
    assert extension == "jpeg" or extension == "jpeg2k" or extension == "webp"

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    if extension == "webp":
        max_mean_diff = 0.2 # webp uses subsampling
    elif extension == "jpeg2k":
        max_mean_diff = 0.025
    else:
        max_mean_diff = 0.06

    encode_decode(extension, backends, dtype, shape, max_mean_diff=max_mean_diff,
                    decode_params=decode_params)

@t.mark.parametrize("extension", ["png", "bmp", "jpeg2k", "pnm", "tiff", "webp"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_uint8_lossless(extension, backends):
    encode_decode_lossless(extension, backends, np.uint8, default_image_shape)

@t.mark.parametrize("extension", ["png", "jpeg2k", "pnm", "tiff"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_uint16_lossless(extension, backends):
    encode_decode_lossless(extension, backends, np.uint16, default_image_shape)

@t.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_only_gpu_lossless(dtype):
    encode_decode_lossless("jpeg2k", backends_gpu_only, dtype, default_image_shape)

@t.mark.parametrize("extension,dtype", [
    ("jpeg", np.uint8),
    ("jpeg2k", np.uint8),
    ("jpeg2k", np.uint16),
    ("jpeg2k", np.int16),
    ("webp", np.uint8),
])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_lossy(extension, dtype, backends):
    if dtype == np.int16 and backends == backends_cpu_only:
        t.skip("CPU plugins don't support int16")

    encode_decode_lossy(extension, backends, dtype, default_image_shape)

@t.mark.parametrize("extension,dtype", [
    ("jpeg", np.uint8),
    ("jpeg2k", np.uint8),
    ("jpeg2k", np.uint16),
    ("jpeg2k", np.int16),
])
def test_only_gpu_lossy(extension, dtype):
    encode_decode_lossy(extension, backends_gpu_only, dtype, default_image_shape)

@t.mark.parametrize("reversible", [True, False])
@t.mark.parametrize("backends", [backends_gpu_only, None]) # TODO: add cpu HT jpeg2k
@t.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16])
def test_ht_jpeg2k(reversible, backends, dtype):
    encode_params=nvimgcodec.EncodeParams(
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(ht=True),
    )
    if reversible:
        encode_params.quality_type = nvimgcodec.QualityType.LOSSLESS

    max_mean_diff = None if reversible else 0.02

    if dtype != np.uint8:
        decode_params = nvimgcodec.DecodeParams(allow_any_depth=True)
    else:
        decode_params = None

    encode_decode("jpeg2k", backends, dtype, default_image_shape, max_mean_diff=max_mean_diff,
                    encode_params=encode_params, decode_params=decode_params)

def encode_decode_with_padding(extension, backends):
    img_rgb = np.random.randint(0, 255, (200, 100, 3), np.uint8)  # Create dummy image

    # Drop some of the columns, which can be interpreted as using padding for rows
    img_rgb = img_rgb[:, 10:80]
    assert img_rgb.shape == (200, 70, 3)
    assert img_rgb.strides == (300, 3, 1)

    encoder = nvimgcodec.Encoder(backends=backends)
    decoder = nvimgcodec.Decoder()

    if extension == "jpeg":
        params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.QUALITY, quality_value=95)
    else:
        params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)

    encoded = encoder.encode(img_rgb, extension, params=params)
    assert encoded is not None

    decoded = np.array(decoder.decode(encoded).cpu())

    mean_diff = np.abs(img_rgb.astype(np.int32) - decoded.astype(np.int32)).mean()
    if extension == "jpeg":
        assert mean_diff < 4
    else:
        assert mean_diff == 0

@t.mark.parametrize("extension", ["jpeg", "jpeg2k"])
@t.mark.parametrize("backends", [backends_gpu_only])
def test_encode_decode_with_padding_gpu(extension, backends):
    encode_decode_with_padding

@t.mark.parametrize("extension", ["jpeg", "png", "bmp", "jpeg2k", "pnm", "tiff", "webp"])
@t.mark.parametrize("backends", [backends_cpu_only, None])
def test_encode_decode_with_padding(extension, backends):
    encode_decode_with_padding
