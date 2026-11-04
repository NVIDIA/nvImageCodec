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

import gc
import os
import sys
import pytest as t
import numpy as np
    
from nvidia import nvimgcodec
from resource_utils import snapshot_process_resources, wait_for_process_resources_to_settle
from utils import *


@t.mark.parametrize(
    ("codec", "sample_type"),
    [("jpeg", np.uint8)] +
    [("jpeg2k", t) for t in (np.uint8, np.uint16)] +
    [("tiff", t) for t in (np.uint8, np.uint16)]
)
def test_encode_to_code_stream(codec, sample_type):
    """Test encoding to CodeStream directly."""
    encoder = nvimgcodec.Encoder()
    
    # Create a simple test image
    arr = np.zeros((32, 32, 3), dtype=sample_type) + 128

    nv_img = nvimgcodec.as_image(arr)
    
    # Encode to CodeStream
    encoded_code_stream = encoder.encode(nv_img, codec=codec)
    assert isinstance(encoded_code_stream, nvimgcodec.CodeStream)
    assert encoded_code_stream.width == 32
    assert encoded_code_stream.height == 32
    #assert encoded_code_stream.channels == 3 #TODO: fix code stream bug for channels
    assert encoded_code_stream.codec_name == codec
    assert encoded_code_stream.size > 0
    assert encoded_code_stream.dtype == sample_type


def test_encode_batch_to_code_stream():
    """Test batch encoding to CodeStream directly."""
    encoder = nvimgcodec.Encoder()
    
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]
    
    # Encode to CodeStream batch
    encoded_code_stream_list = encoder.encode(nv_imgs, codec="jpeg2k")
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 32
    assert encoded_code_stream_list[0].height == 32
    assert encoded_code_stream_list[1].width == 64
    assert encoded_code_stream_list[1].height == 64
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "jpeg2k"
    assert encoded_code_stream_list[1].codec_name == "jpeg2k"

@t.mark.parametrize("pin_memory", [True, False])
@t.mark.parametrize("preallocation_size", [1024, 2048, 4096])
def test_encode_with_code_stream_reuse(pin_memory, preallocation_size):
    """Test reusing a CodeStream for multiple encodings."""
    encoder = nvimgcodec.Encoder()
    
    # Create CodeStream with preallocation
    code_stream = nvimgcodec.CodeStream(preallocation_size, pin_memory=pin_memory)
    
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    nv_img1 = nvimgcodec.as_image(arr1)

    code_stream = encoder.encode(nv_img1, codec="jpeg2k", code_stream_s=code_stream)
    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 32
    assert code_stream.height == 32
    assert code_stream.size > 0
    assert code_stream.codec_name == "jpeg2k"
    assert code_stream.size <= preallocation_size
    assert code_stream.capacity == preallocation_size

def test_encode_with_preallocated_code_stream_capacity_grows_for_large_image():
    """Test that CodeStream capacity grows if input image is larger than preallocated size."""
    encoder = nvimgcodec.Encoder()
    preallocation_size = 1024
    code_stream = nvimgcodec.CodeStream(preallocation_size, pin_memory=True)

    # Create a large image that will require more than preallocated size
    arr = np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)
    nv_img = nvimgcodec.as_image(arr)

    # Encode the large image
    code_stream = encoder.encode(nv_img, codec="jpeg2k", code_stream_s=code_stream)

    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 512
    assert code_stream.height == 512
    assert code_stream.size > 0
    assert code_stream.size <= code_stream.capacity
    # The capacity should have grown to accommodate the encoded image
    assert code_stream.capacity > preallocation_size

@t.mark.parametrize("pin_memory", [True, False])
@t.mark.parametrize("preallocation_size", [1024, 2048, 4096])
def test_encode_with_code_stream_reuse_batch(pin_memory, preallocation_size):
    """Test reusing a CodeStream for multiple encodings in batch."""
    encoder = nvimgcodec.Encoder()

    # Create CodeStream with preallocation
    code_streams = [nvimgcodec.CodeStream(preallocation_size, pin_memory=pin_memory) for _ in range(2)]

    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]

    # Encode batch to CodeStream
    encoded_code_stream_list = encoder.encode(nv_imgs, codec="jpeg2k", code_stream_s=code_streams)
    assert isinstance(encoded_code_stream_list, list)
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 32
    assert encoded_code_stream_list[0].height == 32
    assert encoded_code_stream_list[1].width == 64
    assert encoded_code_stream_list[1].height == 64
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "jpeg2k"
    assert encoded_code_stream_list[1].codec_name == "jpeg2k"
    assert encoded_code_stream_list[0].size <= preallocation_size
    assert encoded_code_stream_list[1].size <= preallocation_size
    assert encoded_code_stream_list[0].capacity == preallocation_size
    assert encoded_code_stream_list[1].capacity == preallocation_size


def test_encode_with_code_stream_long_reuse_resources():
    """Repeated batch reuse must not accumulate native handles or host memory."""
    batch_size = 4
    iterations = int(os.environ.get("NVIMGCODEC_REUSE_STRESS_ITERATIONS", "1000"))
    rng = np.random.default_rng(12345)
    arrays = [rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(batch_size)]
    images = [nvimgcodec.as_image(array).cuda() for array in arrays]
    capacity = arrays[0].nbytes * 2
    code_streams = [nvimgcodec.CodeStream(capacity, pin_memory=True) for _ in range(batch_size)]
    original_streams = tuple(code_streams)
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=batch_size)
    params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        # Exercise the same lossless HTJ2K path and MCT setting as the benchmark.
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(mct_mode=1, ht=True),
    )

    # Exclude one-time plugin/driver resource creation.
    for _ in range(32):
        code_streams = encoder.encode(
            images, codec="jpeg2k", params=params, code_stream_s=code_streams)
    stable_capacities = tuple(stream.capacity for stream in code_streams)
    gc.collect()
    wait_for_process_resources_to_settle()
    before = snapshot_process_resources()

    for _ in range(iterations):
        code_streams = encoder.encode(
            images, codec="jpeg2k", params=params, code_stream_s=code_streams)
        assert len(code_streams) == len(original_streams)
        assert all(actual is expected for actual, expected in zip(code_streams, original_streams))

    # Settle like `before`, so short-lived handles from in-flight work are not counted on
    # one side only. Skip gc.collect() so it cannot hide resources left by the loop.
    wait_for_process_resources_to_settle()
    after = snapshot_process_resources()
    assert tuple(stream.capacity for stream in code_streams) == stable_capacities
    assert all(0 < stream.size <= stream.capacity for stream in code_streams)
    if before.handles is not None and after.handles is not None:
        # Windows and the CUDA driver open and close unrelated handles on their own
        # timers (one run saw 772 -> 771), so exact equality is flaky. A leak in the
        # encode path adds at least one handle per iteration, which this still catches.
        assert after.handles - before.handles < iterations
    if before.rss_bytes is not None and after.rss_bytes is not None:
        assert after.rss_bytes - before.rss_bytes <= 96 * 1024 * 1024

    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(code_streams, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))
    for actual, expected in zip(decoded, arrays):
        np.testing.assert_array_equal(np.asarray(actual.cpu()), expected)


def test_encode_with_code_stream_reuse_batch_insufficient_code_streams():
    """Test that encoding a batch with fewer CodeStreams than images raises an exception."""
    encoder = nvimgcodec.Encoder()

    # Only one CodeStream for two images
    code_streams = [nvimgcodec.CodeStream(1024, pin_memory=True)]

    # Create two test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]

    # Should raise ValueError when fewer CodeStreams than images
    with t.raises(ValueError):
        encoder.encode(nv_imgs, codec="jpeg2k", code_stream_s=code_streams)



def test_encode_with_previous_encode_result_code_stream_reuse():
    """Test reusing a CodeStream for multiple encodings with previous encode result."""
    encoder = nvimgcodec.Encoder()
        
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64

    nv_img1 = nvimgcodec.as_image(arr1)
    nv_img2 = nvimgcodec.as_image(arr2)
    
    # Encode first image
    code_stream = encoder.encode(nv_img1, codec="jpeg2k")
    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 32
    assert code_stream.height == 32
    assert code_stream.size > 0
    assert code_stream.codec_name == "jpeg2k"
    
    # Encode second image to the same code stream previously created
    code_stream = encoder.encode(nv_img2, codec="jpeg", code_stream_s=code_stream)
    assert isinstance(code_stream, nvimgcodec.CodeStream)
    assert code_stream.width == 64
    assert code_stream.height == 64
    assert code_stream.size > 0
    assert code_stream.codec_name == "jpeg"


def test_reused_code_stream_clears_cached_root_tiff_page_count():
    """Reusing a previously counted TIFF root stream must not keep the old page count."""
    root = nvimgcodec.CodeStream(os.path.join(img_dir_path, "tiff/multi_page.tif"))
    assert root.num_images > 1

    encoder = nvimgcodec.Encoder()
    image = nvimgcodec.as_image(np.zeros((18, 24, 3), dtype=np.uint8) + 33)
    reused = encoder.encode(image, codec="jpeg", code_stream_s=root)

    assert reused.num_images == 1
    assert reused.codec_name == "jpeg"
    assert reused.width == 24
    assert reused.height == 18

def test_reused_code_stream_clears_input_view():
    """Reusing a TIFF offset/page-view CodeStream as encoder output must clear the stale input
    view, so it does not leak into `.view` or get_sub_code_stream() on the newly encoded stream."""
    fpath = os.path.join(img_dir_path, "tiff/cat_with_thumbnail.tiff")
    offset = nvimgcodec.CodeStream(fpath).subifd_offsets[0]
    view_cs = nvimgcodec.CodeStream(fpath, bitstream_offset=offset)
    assert view_cs.view is not None and view_cs.view.bitstream_offset == offset

    encoder = nvimgcodec.Encoder()
    image = nvimgcodec.as_image(np.zeros((18, 24, 3), dtype=np.uint8) + 33)
    reused = encoder.encode(image, codec="jpeg", code_stream_s=view_cs)

    assert reused is not None and reused.codec_name == "jpeg"
    assert reused.width == 24 and reused.height == 18
    # Regression: reuse() used to keep the input view, leaking the old IFD offset.
    assert reused.view is None
    # A substream of the result must not inherit the stale offset either.
    assert reused.get_sub_code_stream(0).view.bitstream_offset != offset

def test_encode_with_previous_encode_result_code_stream_reuse_batch():
    """Test reusing a CodeStream for multiple encodings in batch with previous encode result."""
    encoder = nvimgcodec.Encoder()
    
    # Create CodeStream with preallocation
    code_stream = nvimgcodec.CodeStream(1024, pin_memory=True)
    
    # Create test images
    arr1 = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    arr2 = np.zeros((64, 64, 3), dtype=np.uint8) + 64
    nv_imgs_batch1 = [nvimgcodec.as_image(arr1), nvimgcodec.as_image(arr2)]
    
    arr3 = np.zeros((128, 128, 3), dtype=np.uint8) + 128
    arr4 = np.zeros((256, 256, 3), dtype=np.uint8) + 64
    nv_imgs_batch2 = [nvimgcodec.as_image(arr3), nvimgcodec.as_image(arr4)]
    
    # Encode to CodeStream batch
    encoded_code_stream_list = encoder.encode(nv_imgs_batch1, codec="jpeg2k")
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 32
    assert encoded_code_stream_list[0].height == 32
    assert encoded_code_stream_list[1].width == 64
    assert encoded_code_stream_list[1].height == 64
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "jpeg2k"
    assert encoded_code_stream_list[1].codec_name == "jpeg2k"
    
    # Encode to CodeStream batch
    encoded_code_stream_list = encoder.encode(nv_imgs_batch2, codec="tiff", code_stream_s = encoded_code_stream_list)
    assert len(encoded_code_stream_list) == 2
    assert all(isinstance(cs, nvimgcodec.CodeStream) for cs in encoded_code_stream_list)
    assert encoded_code_stream_list[0].width == 128
    assert encoded_code_stream_list[1].width == 256
    assert encoded_code_stream_list[0].height == 128
    assert encoded_code_stream_list[1].height == 256
    assert encoded_code_stream_list[0].size > 0
    assert encoded_code_stream_list[1].size > 0
    assert encoded_code_stream_list[0].codec_name == "tiff"
    assert encoded_code_stream_list[1].codec_name == "tiff"
    
def test_encode_returns_code_stream_with_buffer_protocol():
    """Test that CodeStream implements buffer protocol correctly."""
    encoder = nvimgcodec.Encoder()
    
    # Create test image
    arr = np.zeros((32, 32, 3), dtype=np.uint8) + 128
    nv_img = nvimgcodec.as_image(arr)
    
    # Encode to CodeStream
    encoded_code_stream = encoder.encode(nv_img, codec="jpeg2k")
    
    # Test buffer protocol
    buffer = memoryview(encoded_code_stream)
    assert len(buffer) > 0


def test_get_sub_code_stream_on_encoded_code_stream():
    """get_sub_code_stream(image_idx=0) must work on CodeStream returned by encoder.encode()."""
    encoder = nvimgcodec.Encoder()

    image = np.zeros((426, 640), dtype=np.uint8)
    tiff_encoded = encoder.encode(nvimgcodec.as_image(image), "tiff")

    # Creating sub code stream from encoded CodeStream must succeed (was failing with "image index out of range")
    sub_cs = tiff_encoded.get_sub_code_stream(image_idx=0)
    assert sub_cs is not None
    assert sub_cs.num_images == 1
    assert sub_cs.width == 640
    assert sub_cs.height == 426


@t.mark.parametrize("codec", ["tiff", "jpeg", "jpeg2k", "png"])
def test_encoded_code_stream_metadata_and_get_sub(codec):
    """Encoder CodeStream has correct metadata and buffer; get_sub_code_stream(0) works and matches."""
    encoder = nvimgcodec.Encoder()
    h, w = 32, 48
    image = np.zeros((h, w, 3), dtype=np.uint8) + 100
    enc = encoder.encode(nvimgcodec.as_image(image), codec)

    assert enc.num_images == 1 and enc.codec_name == codec and enc.width == w and enc.height == h
    assert enc.size > 0 and enc.dtype == np.uint8
    assert len(memoryview(enc)) == enc.size and len(bytes(enc)) == enc.size

    sub = enc.get_sub_code_stream(image_idx=0)
    assert sub and sub.num_images == 1 and sub.codec_name == codec and sub.width == w and sub.height == h and sub.size > 0


def test_encoded_code_stream_sub_then_decode_roundtrip():
    """Decode(sub_cs) and decode(encoded) both match original image."""
    encoder, decoder = nvimgcodec.Encoder(), nvimgcodec.Decoder()
    image = np.zeros((40, 60, 3), dtype=np.uint8) + 200
    enc = encoder.encode(nvimgcodec.as_image(image), "tiff")
    sub = enc.get_sub_code_stream(image_idx=0)

    dec_sub = decoder.decode(sub)
    dec_direct = decoder.decode(enc)
    ref = np.asarray(image)
    np.testing.assert_array_equal(np.asarray(dec_sub.cpu()), ref)
    np.testing.assert_array_equal(np.asarray(dec_direct.cpu()), ref)


def test_encoded_code_stream_bytes_roundtrip():
    """bytes(encoded) -> CodeStream -> get_sub_code_stream(0) has correct dimensions."""
    enc = nvimgcodec.Encoder().encode(nvimgcodec.as_image(np.zeros((24, 32), dtype=np.uint8) + 64), "jpeg")
    sub = nvimgcodec.CodeStream(bytes(enc)).get_sub_code_stream(image_idx=0)
    assert sub and sub.num_images == 1 and sub.width == 32 and sub.height == 24


def test_encoded_code_stream_batch_each_get_sub():
    """Batch encode; get_sub_code_stream(0) on each result has correct dimensions."""
    enc_list = nvimgcodec.Encoder().encode(
        [nvimgcodec.as_image(np.zeros((30, 40), dtype=np.uint8) + 10),
         nvimgcodec.as_image(np.zeros((50, 60), dtype=np.uint8) + 20)], codec="tiff")
    for i, enc in enumerate(enc_list):
        sub = enc.get_sub_code_stream(image_idx=0)
        assert sub.num_images == 1 and sub.width == [40, 60][i] and sub.height == [30, 50][i]


def test_encoded_code_stream_reused_then_get_sub():
    """Reusing a preallocated CodeStream for encode: get_sub_code_stream(0) on the returned CodeStream works."""
    enc = nvimgcodec.Encoder().encode(
        nvimgcodec.as_image(np.zeros((20, 25), dtype=np.uint8) + 77), "png",
        code_stream_s=nvimgcodec.CodeStream(8192, pin_memory=True))
    sub = enc.get_sub_code_stream(image_idx=0)
    assert sub and sub.num_images == 1 and sub.width == 25 and sub.height == 20 and sub.codec_name == "png"


def test_encoded_code_stream_single_image_bounds():
    """Single-image encoder output: num_images=1, next_ifd_offset is None; image_idx=1 raises on metadata access."""
    enc = nvimgcodec.Encoder().encode(nvimgcodec.as_image(np.zeros((10, 10), dtype=np.uint8)), "jpeg")
    assert enc.num_images == 1 and enc.next_ifd_offset is None
    with t.raises(RuntimeError):
        _ = enc.get_sub_code_stream(image_idx=1).width


def test_encoded_code_stream_get_sub_invalid_kwarg_raises_type_error():
    """get_sub_code_stream with invalid kwarg (e.g. idx=0) raises TypeError, not assert/abort."""
    enc = nvimgcodec.Encoder().encode(nvimgcodec.as_image(np.zeros((42, 64), dtype=np.uint8)), "jpeg")
    with t.raises(TypeError) as exc_info:
        enc.get_sub_code_stream(idx=0)
    assert "image_idx" in str(exc_info.value) or "incompatible" in str(exc_info.value).lower()


# ----------------------------------------------------------------------
# CodeStream-reuse + precision: read a CodeStream from drive (one
# precision), then encode a NEW image with a different precision into
# the same CodeStream via code_stream_s=. The reused CodeStream's
# .precision must reflect the new image, not the file's original tag.
# ----------------------------------------------------------------------

@t.mark.parametrize(
    "seed_file, seed_precision, encode_dtype, encode_precision",
    [
        # Seed = 8-bit J2K from drive; encode a 12-bit uint16 image into it.
        ("jpeg2k/cat-1046544_640.jp2", 8, np.uint16, 12),
        # Seed = 8-bit J2K from drive; encode a 16-bit uint16 image into it.
        ("jpeg2k/cat-1046544_640.jp2", 8, np.uint16, 16),
        # Seed = 12-bit J2K from drive; encode a sub-8-bit uint8 image into it.
        ("jpeg2k/cat-1245673_640-12bit.jp2", 12, np.uint8, 5),
        # Seed = 16-bit J2K from drive; encode an 8-bit uint8 image into it.
        ("jpeg2k/cat-111793_640-16bit.jp2", 16, np.uint8, 8),
        # Seed = 5-bit J2K from drive; encode a 12-bit uint16 image into it
        # (cross-dtype the other way: uint8 seed -> uint16 target).
        ("jpeg2k/cat-1245673_640-5bit.jp2", 5, np.uint16, 12),
    ]
)
def test_encode_reuse_codestream_from_drive_updates_precision(
    seed_file, seed_precision, encode_dtype, encode_precision):
    """A CodeStream read from disk (with its file's native precision) is reused as the
    output target of a new encode. The reused CodeStream's .precision must reflect the
    newly-encoded image's precision, and the encoded bitstream must round-trip back to
    the original pixel values."""
    seed_path = os.path.join(img_dir_path, seed_file)
    seed_cs = nvimgcodec.CodeStream(seed_path)
    assert seed_cs.precision == seed_precision, (
        f"sanity: {seed_file} on disk should report precision {seed_precision}, "
        f"got {seed_cs.precision}")

    rng = np.random.default_rng(1729)
    max_value = (1 << encode_precision) - 1
    arr = rng.integers(0, max_value + 1, (48, 64, 3), dtype=encode_dtype)
    img = nvimgcodec.as_image(arr, precision=encode_precision)
    assert img.precision == encode_precision

    encoder = nvimgcodec.Encoder()
    encode_params = nvimgcodec.EncodeParams(quality_type=nvimgcodec.QualityType.LOSSLESS)
    reused = encoder.encode(img, codec="jpeg2k", params=encode_params, code_stream_s=seed_cs)

    assert reused is seed_cs, "encode(code_stream_s=...) should return the reused object"
    assert reused.precision == encode_precision, (
        f"after reusing {seed_file} (precision {seed_precision}) for a "
        f"{encode_dtype.__name__} precision={encode_precision} encode, "
        f"expected reused.precision={encode_precision}, got {reused.precision}")
    assert reused.codec_name == "jpeg2k"
    assert reused.width == 64
    assert reused.height == 48

    # Round-trip: decoding the reused CodeStream's bytestream must yield the original pixels.
    decoder = nvimgcodec.Decoder()
    decoded = decoder.decode(reused, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True))
    assert decoded.precision == encode_precision
    assert np.dtype(decoded.dtype) == np.dtype(encode_dtype)
    np.testing.assert_array_equal(np.asarray(decoded.cpu()), arr)
