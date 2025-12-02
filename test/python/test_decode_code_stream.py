# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path
import nvjpeg_test_speedup

try:
    import cupy as cp
    img = cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8) # Force to load necessary libriaries
    CUPY_AVAILABLE = True
except:
    print("CuPy is not available, will skip related tests")
    CUPY_AVAILABLE = False
    cp = None  # Define cp as None so it can be referenced in parametrize decorators

filenames = [
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_440.jpg",
    "jpeg2k/cat-111793_640.jp2",
]

def test_decode_single_file():
    decoder = nvimgcodec.Decoder()
    fpath = os.path.join(img_dir_path, filenames[0])
    raw_bytes = open(fpath, 'rb').read()
    np_arr = np.fromfile(fpath, dtype=np.uint8)

    dec_srcs = [
        fpath,
        raw_bytes,
        np_arr,
        nvimgcodec.CodeStream(raw_bytes),
        nvimgcodec.CodeStream(fpath),
        nvimgcodec.CodeStream(np_arr),
    ]

    imgs = [decoder.decode(src) for src in dec_srcs]
    for img in imgs[1:]:
        if img and imgs[0]:
            np.testing.assert_allclose(imgs[0].cpu(), img.cpu())

def test_decode_roi():
    decoder = nvimgcodec.Decoder()
    fpath = os.path.join(img_dir_path, filenames[0])
    raw_bytes = open(fpath, 'rb').read()
    np_arr = np.fromfile(fpath, dtype=np.uint8)

    code_stream0 = nvimgcodec.CodeStream(fpath)
    code_stream1 = nvimgcodec.CodeStream(raw_bytes)
    code_stream2 = nvimgcodec.CodeStream(np_arr)

    roi = nvimgcodec.Region(10, 20, code_stream0.height-10, code_stream0.width-20)
    imgref = np.array(decoder.decode(code_stream0).cpu())

    dec_srcs = [
        code_stream0.get_sub_code_stream(region = roi),
        code_stream1.get_sub_code_stream(region = roi),
        code_stream2.get_sub_code_stream(region = roi),
    ]
    imgs_roi = [decoder.decode(src).cpu() for src in dec_srcs]
    for img in imgs_roi:
        np.testing.assert_allclose(img, imgref[roi.start[0]:roi.end[0], roi.start[1]:roi.end[1]])

def test_decode_batch():
    decoder = nvimgcodec.Decoder()
    fpaths = [os.path.join(img_dir_path, f) for f in filenames]
    np_arrays = [np.fromfile(fpath, dtype=np.uint8) for fpath in fpaths]
    raw_bytes = [open(fpath, 'rb').read() for fpath in fpaths]

    batch0_srcs = [
        nvimgcodec.CodeStream(fpaths[0]),
        nvimgcodec.CodeStream(raw_bytes[1]),
        np_arrays[2],
    ]

    batch1_srcs = [
        nvimgcodec.CodeStream(raw_bytes[0]),
        fpaths[1],
        nvimgcodec.CodeStream(np_arrays[2]),
    ]

    imgs0 = [img.cpu() for img in decoder.decode(batch0_srcs)]
    imgs1 = [img.cpu() for img in decoder.decode(batch1_srcs)]
    for img0, img1 in zip(imgs0, imgs1):
        np.testing.assert_allclose(img0, img1)

def test_decode_batch_roi():
    decoder = nvimgcodec.Decoder()
    fpaths = [os.path.join(img_dir_path, f) for f in filenames]
    code_streams = [nvimgcodec.CodeStream(fpath) for fpath in fpaths]
    rois = [
        nvimgcodec.Region(10, 20, cs.height-10, cs.width-20)
        for cs in code_streams
    ]
    dec_srcs = [
        cs.get_sub_code_stream(region = roi)
        for cs, roi in zip(code_streams, rois)
    ]
    
    imgs = [np.array(img.cpu()) for img in decoder.decode(code_streams)]
    imgs_roi = [np.array(img.cpu()) for img in decoder.decode(dec_srcs)]

    for img, img_roi, roi in zip(imgs, imgs_roi, rois):
        np.testing.assert_allclose(img_roi, img[roi.start[0]:roi.end[0], roi.start[1]:roi.end[1]])

@t.mark.parametrize("fname", ["jpeg/padlock-406986_640_440.jpg",
                              "jpeg2k/cat-111793_640.jp2"])
def test_decode_repeat_code_stream(fname):
    decoder = nvimgcodec.Decoder(max_num_cpu_threads=1)  # One thread to force a single instance of parsed stream
    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, fname))

    img0 = decoder.decode(code_stream)
    imgs1 = decoder.decode([code_stream, code_stream])

    for img1 in imgs1:
        np.testing.assert_allclose(img0.cpu(), img1.cpu(), atol=3)

@t.mark.parametrize("fname", ["jpeg/padlock-406986_640_440.jpg",
                              "jpeg2k/cat-111793_640.jp2"])
def test_decode_reuses_image_when_provided(fname):
    """Test reusing an Image for multiple decodings."""
    decoder = nvimgcodec.Decoder()
    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, fname))
    
    # First decode to create the original result
    img_ref = decoder.decode(code_stream)

    # Reuse the Image for decoding the same stream
    img_reused = decoder.decode(code_stream, image=img_ref)

    assert img_reused is img_ref


@t.mark.parametrize("fname", ["jpeg/padlock-406986_640_440.jpg",
                              "jpeg2k/cat-111793_640.jp2"])
def test_decode_reuses_images_when_provided_in_batch(fname):
    """Test reusing Images for batch decoding."""
    decoder = nvimgcodec.Decoder()
    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, fname))
    
    # First decode to get reference results
    imgs_ref = decoder.decode([code_stream, code_stream])
    
    # Reuse the Images for decoding the same streams
    imgs_reused = decoder.decode([code_stream, code_stream], images=imgs_ref)
    
    # Verify the results are the same
    assert len(imgs_reused) == len(imgs_ref)
    for i, (img_ref, img_reused) in enumerate(zip(imgs_ref, imgs_reused)):
        assert img_reused is img_ref

def test_decode_with_image_reuse_different_codecs():
    """Test reusing an Image for decoding different codec types."""
    decoder = nvimgcodec.Decoder()
    
    # Use different codec files
    jpeg_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg/padlock-406986_640_440.jpg"))
    jpeg2k_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg2k/cat-111793_640.jp2"))
    
    # First decode JPEG image
    img_jpeg = decoder.decode(jpeg_stream)
    
    # Reuse the Image buffer for JPEG2K (different size image, should resize buffer)
    img_jpeg2k = decoder.decode(jpeg2k_stream, image=img_jpeg)
    
    # Both should be valid Images
    assert img_jpeg is not None
    assert img_jpeg2k is not None
    assert img_jpeg2k is img_jpeg
    
    # Verify the reused image has the correct dimensions for JPEG2K
    assert img_jpeg2k.width == jpeg2k_stream.width
    assert img_jpeg2k.height == jpeg2k_stream.height

def test_decode_with_image_reuse_batch_insufficient_images():
    """Test that decoding a batch with fewer Images than streams raises an exception."""
    decoder = nvimgcodec.Decoder()
    
    # Create two streams but only one reusable Image
    code_streams = [
        nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0])),
        nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[1]))
    ]
    
    # Decode first stream to get a reusable Image
    img = decoder.decode(code_streams[0])
    reusable_imgs = [img]  # Only one Image for two streams
    
    # Should raise an exception due to insufficient Images
    with t.raises(Exception) as e:
        decoder.decode(code_streams, images=reusable_imgs)
    
    assert isinstance(e.value, ValueError)

def test_decode_with_previous_decode_result_image_reuse():
    """Test reusing an Image from a previous decode result."""
    decoder = nvimgcodec.Decoder()
    
    # Use different files
    stream1 = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0]))
    stream2 = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[1]))
    
    # Decode first image
    img1 = decoder.decode(stream1)
    assert img1 is not None
    assert img1.width == stream1.width
    assert img1.height == stream1.height
    
    # Reuse the Image buffer for second image
    img2 = decoder.decode(stream2, image=img1)
    assert img2 is not None
    assert img2 is img1  # Should be the same object (reused)
    assert img2.width == stream2.width
    assert img2.height == stream2.height

def test_decode_with_previous_decode_result_image_reuse_batch():
    """Test reusing Images from previous decode results in batch operations."""
    decoder = nvimgcodec.Decoder()
    
    # Create first batch of streams
    streams_batch1 = [
        nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0])),
        nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[1]))
    ]
    
    # Create second batch of streams  
    streams_batch2 = [
        nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[2])),
        nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[3]))
    ]
    
    # Decode first batch
    imgs_batch1 = decoder.decode(streams_batch1)
    assert len(imgs_batch1) == 2
    assert all(img is not None for img in imgs_batch1)
    
    # Reuse Images from first batch for second batch
    imgs_batch2 = decoder.decode(streams_batch2, images=imgs_batch1)
    assert len(imgs_batch2) == 2
    assert all(img is not None for img in imgs_batch2)
    
    # Verify object reuse
    for img1, img2 in zip(imgs_batch1, imgs_batch2):
        assert img2 is img1  # Should be the same objects (reused)
    
    # Verify dimensions match the new streams
    for img, stream in zip(imgs_batch2, streams_batch2):
        assert img.width == stream.width
        assert img.height == stream.height


BUFFER_SOURCES = [np.random.randint]
if CUPY_AVAILABLE:
    BUFFER_SOURCES.append(cp.random.randint)

@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_continuous_reshape(buffer_source):
    """If buffer used is continuous it can be reshaped to match image shape"""
    decoder = nvimgcodec.Decoder()

    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0]))
    assert code_stream.width > 600 # image width is larger than buffer width
    assert code_stream.height < 500 # height is much smaller than buffer height
    assert code_stream.num_channels == 3

    # Create an Image from external numpy array (externally managed buffer)
    buffer = buffer_source(0, 255, (1000, 600, 3), dtype=np.uint8)
    external_img = nvimgcodec.as_image(buffer) # used buffer is continnous so it can be reshaped to fit image

    assert external_img.size == buffer.nbytes
    assert external_img.capacity == buffer.nbytes

    target_buffer_kind = nvimgcodec.ImageBufferKind.STRIDED_HOST
    if CUPY_AVAILABLE and buffer_source == cp.random.randint:
        target_buffer_kind = nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    assert external_img.buffer_kind == target_buffer_kind 

    result_img = decoder.decode(code_stream, image=external_img)
    ref_img = decoder.decode(code_stream)

    assert result_img is external_img
    assert result_img.capacity == buffer.nbytes
    assert result_img.size == ref_img.size
    assert result_img.shape == ref_img.shape

    assert result_img is not None
    assert ref_img is not None
    np.testing.assert_allclose(ref_img.cpu(), result_img.cpu())


@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_continuous_too_small(buffer_source):
    """Test that Images created from external properly checks for buffer size"""
    decoder = nvimgcodec.Decoder()
    
    # Create an Image from external numpy array (externally managed buffer)
    ref_img = buffer_source(0, 255, (100, 100, 3), dtype=np.uint8)
    external_img = nvimgcodec.as_image(ref_img)
    
    assert external_img.size == ref_img.nbytes
    assert external_img.capacity == ref_img.nbytes

    target_buffer_kind = nvimgcodec.ImageBufferKind.STRIDED_HOST
    if CUPY_AVAILABLE and buffer_source == cp.random.randint:
        target_buffer_kind = nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    assert external_img.buffer_kind == target_buffer_kind 
    
    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0]))
    
    with t.raises(ValueError) as e:
        decoder.decode(code_stream, image=external_img)
    assert str(e.value) == "Existing buffer is too small to fit new image"

@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_non_continuous_fit(buffer_source):
    """For non continuous buffor image must fit inside slice dimensions"""
    decoder = nvimgcodec.Decoder()

    # Create an Image from external numpy array (externally managed buffer)
    buffer = buffer_source(0, 255, (450, 700, 3), dtype=np.uint8)
    slice = buffer[3:-4, 5:-7] # make slice with strides, so it doesn't span across continuous memory
    external_img = nvimgcodec.as_image(slice)

    assert external_img.size == slice.nbytes
    assert external_img.capacity == slice.nbytes

    target_buffer_kind = nvimgcodec.ImageBufferKind.STRIDED_HOST
    if CUPY_AVAILABLE and buffer_source == cp.random.randint:
        target_buffer_kind = nvimgcodec.ImageBufferKind.STRIDED_DEVICE
    assert external_img.buffer_kind == target_buffer_kind 

    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0]))

    result_img = decoder.decode(code_stream, image=external_img)
    ref_img = decoder.decode(code_stream)

    assert result_img is external_img
    assert result_img.capacity == slice.nbytes
    assert result_img.size == result_img.width * result_img.height * 3
    assert result_img.size == ref_img.size
    assert result_img.shape == ref_img.shape

    assert result_img is not None
    assert ref_img is not None
    np.testing.assert_allclose(ref_img.cpu(), result_img.cpu())

@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_non_continuous_reshape_channels(buffer_source):
    """For non continuous buffor image must fit inside slice dimensions, but row can be reshaped"""
    decoder = nvimgcodec.Decoder()

    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0]))
    assert code_stream.height < 440
    assert code_stream.width < 650
    assert code_stream.num_channels == 3
    assert code_stream.width * code_stream.num_channels < 2000 # row size is enough to fit the image

    # Create an Image from external numpy array (externally managed buffer)
    buffer = buffer_source(0, 255, (450, 2100, 1), dtype=np.uint8)
    slice = buffer[3:-4, 5:-7] # make slice with strides, so it doesn't span across continuous memory
    external_img = nvimgcodec.as_image(slice)

    result_img = decoder.decode(code_stream, image=external_img)
    ref_img = decoder.decode(code_stream)

    assert result_img is external_img
    assert result_img.capacity == slice.nbytes
    assert result_img.size == ref_img.size
    assert result_img.shape == ref_img.shape

    assert result_img is not None
    assert ref_img is not None
    np.testing.assert_allclose(ref_img.cpu(), result_img.cpu())

@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_non_continuous_too_small(buffer_source):
    """For non continuous buffor image must fit inside slice dimensions"""
    decoder = nvimgcodec.Decoder()

    # Create an Image from external numpy array (externally managed buffer)
    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, filenames[0]))
    assert code_stream.width > 600 # image width is larger than buffer width
    assert code_stream.height < 500 # height is much smaller than buffer height
    assert code_stream.num_channels == 3

    buffer = buffer_source(0, 255, (1000, 600, 3), dtype=np.uint8) #
    slice = buffer[3:-4, 5:-7] # make slice with strides, so it doesn't span across continuous memory
    external_img = nvimgcodec.as_image(slice) # used buffer is not continnous so it will throw when we try to reuse

    with t.raises(ValueError) as e:
        decoder.decode(code_stream, image=external_img)
    assert str(e.value) == "Existing buffer is not continuous. Row size or height are too small to fit new image." 

@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_non_continuous_reshape_dtype(buffer_source):
    """For non continuous buffor image must fit inside slice dimensions, but row can be reshaped"""
    decoder = nvimgcodec.Decoder()

    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg2k/cat-111793_640-16bit.jp2"))
    assert code_stream.width < 650
    assert code_stream.height < 450
    assert code_stream.num_channels == 3
    assert code_stream.dtype == np.uint16

    # Create an Image from external numpy array (externally managed buffer)
    buffer = buffer_source(0, 255, (500, 700, 6), dtype=np.uint8) # twice as many channels as we have uint8 instead of uint16
    slice = buffer[3:, 5:] # make slice with strides, so it doesn't span across continuous memory
    external_img = nvimgcodec.as_image(slice)

    params = nvimgcodec.DecodeParams(allow_any_depth=True)
    result_img = decoder.decode(code_stream, image=external_img, params=params)
    ref_img = decoder.decode(code_stream, params=params)

    assert result_img is external_img
    assert result_img.capacity == slice.nbytes
    assert result_img.size == ref_img.size
    assert result_img.shape == ref_img.shape

    assert result_img is not None
    assert ref_img is not None
    assert result_img.dtype == np.uint16
    np.testing.assert_allclose(ref_img.cpu(), result_img.cpu())

@t.mark.parametrize("buffer_source", BUFFER_SOURCES)
def test_decode_with_external_buffer_non_continuous_reshape_dtype_not_enough(buffer_source):
    """For non continuous buffor image must fit inside slice dimensions, but row can be reshaped"""
    decoder = nvimgcodec.Decoder()

    code_stream = nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg2k/cat-111793_640-16bit.jp2"))
    assert code_stream.width < 650
    assert code_stream.width > 600
    assert code_stream.height < 450
    assert code_stream.num_channels == 3
    assert code_stream.dtype == np.uint16

    # Create an Image from external numpy array (externally managed buffer)
    buffer = buffer_source(0, 255, (500, 700, 5), dtype=np.uint8) # compared to test above, 5 is not enough to fit uint16
    slice = buffer[3:, 5:] # make slice with strides, so it doesn't span across continuous memory
    external_img = nvimgcodec.as_image(slice)

    params = nvimgcodec.DecodeParams(allow_any_depth=True)

    with t.raises(ValueError) as e:
        decoder.decode(code_stream, image=external_img, params=params)
    assert str(e.value) == "Existing buffer is not continuous. Row size or height are too small to fit new image."

def test_decode_keep_alive_check():
    """Test if get_sub_code_stream properly keeps reference to original CodeStream that contains reference to bytes"""
    files = [os.path.join(img_dir_path, filename) for filename in filenames]

    code_streams_bytes = []
    code_stream_files = []
    for file in files:
        cs_file = nvimgcodec.CodeStream(file)
        code_stream_files.append(cs_file.get_sub_code_stream(image_idx=0))

        with open(file, "rb") as f:
            bytes = f.read()
            cs_bytes = nvimgcodec.CodeStream(bytes)
        code_streams_bytes.append(cs_bytes.get_sub_code_stream(image_idx=0))

    decoder = nvimgcodec.Decoder()
    images = decoder.decode(code_streams_bytes)
    references = decoder.decode(code_stream_files)

    assert len(images) == len(references)
    for img, ref in zip(images, references):
        assert img is not None
        assert ref is not None
        assert img.shape == ref.shape
        np.testing.assert_allclose(img.cpu(), ref.cpu())
