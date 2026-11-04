#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import pytest
import numpy as np
import tempfile
import os
from pathlib import Path
import cv2
from nvidia import nvimgcodec
from utils import compare_host_images, img_dir_path

@pytest.mark.parametrize("backends", [
    [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
])
@pytest.mark.parametrize("dcm_file", [
    "GDCMJ2K_TextGBR.dcm",
])
def test_decode_jpeg2k_from_DICOM(backends, dcm_file):
    """Test decoding JPEG2K streams from DICOM files that pydicom can decode gracefully."""

    # TODO(janton): remove this once nvjpeg2k supports GDCMJ2K_TextGBR.dcm
    if dcm_file == "GDCMJ2K_TextGBR.dcm" and backends[0].backend_kind == nvimgcodec.BackendKind.GPU_ONLY:
        pytest.skip("GDCMJ2K_TextGBR.dcm is not yet supported by nvjpeg2k")

    # Skip when required DICOM codecs stack is not installed
    pytest.importorskip("pylibjpeg")

    import pydicom
    from pydicom.data import get_testdata_file
    dcm_path = get_testdata_file(dcm_file)
    ref = np.asarray(pydicom.dcmread(dcm_path).pixel_array)
    height, width, num_channels = ref.shape

    # Decode with nvimagecodec
    ds = pydicom.dcmread(dcm_path)
    frame = list(pydicom.encaps.generate_frames(ds.PixelData, number_of_frames=1))[0]

    # Check that nvimagecodec parser works as expected
    code_stream = nvimgcodec.CodeStream(frame)
    assert code_stream.height == height
    assert code_stream.width == width
    assert code_stream.num_channels == num_channels

    # Decode with nvimagecodec
    decoder = nvimgcodec.Decoder(backends=backends)
    nvimg_image = decoder.decode(frame)
    assert nvimg_image is not None, "nvimagecodec failed to decode JPEG2K file"

    # Compare reference and nvimagecodec results
    nvimg_array = np.asarray(nvimg_image.cpu())
    np.testing.assert_array_equal(ref, nvimg_array)


@pytest.mark.parametrize("backends", [
    [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
])
@pytest.mark.parametrize("corrupted_file", [
    "jpeg2k/corrupted/hang1.jp2",
    ])
def test_decode_corrupted_jpeg2k(backends, corrupted_file):
    """Test corrupted JPEG2K files that should be handled gracefully."""
    
    corrupted_file_path = os.path.join(img_dir_path, corrupted_file)
    if not os.path.isdir(os.path.dirname(corrupted_file_path)):
        pytest.skip(f"{os.path.dirname(corrupted_file)} samples are not present in {img_dir_path}")
    # Test if OpenCV can decode this file
    ref = cv2.imread(corrupted_file_path, cv2.IMREAD_UNCHANGED)
    assert ref is not None, f"OpenCV failed to read {corrupted_file}"
    ref_array = np.asarray(ref)
    # add trailing channel dimension for grayscale images to match nvimagecodec
    if ref_array.ndim == 2:
        ref_array = ref_array[..., np.newaxis]
    assert ref_array is not None
    height, width, num_channels = ref_array.shape

    # Check that nvimagecodec parser works as expected
    code_stream = nvimgcodec.CodeStream(corrupted_file_path)
    assert code_stream.height == height
    assert code_stream.width == width
    assert code_stream.num_channels == num_channels

    # Decode with nvimagecodec
    params_unchanged = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED)
    decoder = nvimgcodec.Decoder(backends=backends)
    nvimg_image = decoder.read(corrupted_file_path, params=params_unchanged)
    assert nvimg_image is not None, f"nvimagecodec failed to decode {os.path.basename(corrupted_file)}"
    
    # Compare reference and nvimagecodec results
    nvimg_array = np.asarray(nvimg_image.cpu())
    np.testing.assert_array_equal(ref_array, nvimg_array)


@pytest.mark.parametrize("backends", [
    # right now only nvjpeg2k extensions support decoding to ycc or conversion from ycc to rgb
    [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
])
@pytest.mark.parametrize("img_file, chroma_h_factor, chroma_w_factor, expected_dtype", [
    ("jpeg2k/ycc_rgb_conversion/artificial_420_8b3c_uint8.jp2",   2, 2, np.uint8),
    ("jpeg2k/ycc_rgb_conversion/artificial_422_8b3c_uint8.jp2",   1, 2, np.uint8),
    ("jpeg2k/ycc_rgb_conversion/artificial_444_8b3c_uint8.jp2",   1, 1, np.uint8),
    ("jpeg2k/ycc_rgb_conversion/artificial_420_16b3c_uint16.jp2", 2, 2, np.uint16),
    ("jpeg2k/ycc_rgb_conversion/artificial_422_16b3c_uint16.jp2", 1, 2, np.uint16),
    ("jpeg2k/ycc_rgb_conversion/artificial_444_16b3c_uint16.jp2", 1, 1, np.uint16),
    ("jpeg2k/ycc_rgb_conversion/artificial_420_16b3c_int16.jp2",  2, 2, np.int16),
    ("jpeg2k/ycc_rgb_conversion/artificial_422_16b3c_int16.jp2",  1, 2, np.int16),
    ("jpeg2k/ycc_rgb_conversion/artificial_444_16b3c_int16.jp2",  1, 1, np.int16),
])
def test_decode_jpeg2k_subsampled_planar_ycc_to_rgb(
        backends, img_file, chroma_h_factor, chroma_w_factor, expected_dtype):
    """Decode a chroma-subsampled JPEG2K to planar YCbCr, manually upsample the
    chroma planes to full luma resolution, convert to RGB, and compare against a
    direct SRGB decode of the same image.  Covers uint8, uint16, and int16 sources.

    Two conversions are compared:
      1. Manual BT.601 full-range float32 formula.  Thresholds: 1 (uint8), 10 (16-bit).
         The ±9 LSB residual for 16-bit comes from
         nvjpeg2k's internal ycbcr_to_rgb kernel using truncated G coefficients
         (0.344 / 0.714 with (int) cast) instead of the standard 0.344136 / 0.714136.
      2. cv2.COLOR_YCrCb2RGB.  Thresholds: 1 (uint8), 32 (16-bit).  OpenCV
         hard-codes CB2BF=1.773f / CB2BI=29049 (color_yuv.simd.hpp) instead of
         the standard 1.772.  At 8-bit scale the error is < 1 LSB; at 16-bit scale
         it accumulates to ~30 LSBs at maximum chroma deviation.  int16 is not
         natively supported by cv2, so values are shifted into uint16, converted,
         then shifted back."""
    # Per-dtype: (chroma neutral, min clamp value, max clamp value, manual threshold, cv2 threshold)
    dtype_cfg = {
        np.uint8:  (128,   0,      255,   1,  1),
        np.uint16: (32768, 0,      65535, 10, 32),
        np.int16:  (0,     -32768, 32767, 10, 32),
    }
    neutral, clip_min, clip_max, manual_thr, cv2_thr = dtype_cfg[expected_dtype]

    img_path = os.path.join(img_dir_path, img_file)
    cs = nvimgcodec.CodeStream(img_path)
    H, W = cs.height, cs.width

    decoder = nvimgcodec.Decoder(backends=backends)

    # Decode to planar YCbCr.  The decoded buffer has shape (3, H, W) but for a
    # subsampled source the chroma planes only hold valid data in the top-left
    # (H // chroma_h_factor) × (W // chroma_w_factor) region; the remainder of
    # each chroma plane in the (H, W) buffer is unwritten.
    yuv_img = decoder.read(img_path, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SYCC,
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        allow_any_depth=True,
    ))
    assert yuv_img is not None
    assert yuv_img.sample_format == nvimgcodec.SampleFormat.P_YUV
    assert yuv_img.color_spec == nvimgcodec.ColorSpec.SYCC

    yuv_arr = np.asarray(yuv_img.cpu())  # (3, H, W)
    assert yuv_arr.shape == (3, H, W)
    assert yuv_arr.dtype == expected_dtype

    # round up for odd dimension, same as jpeg2000
    chroma_H = (H + chroma_h_factor - 1) // chroma_h_factor
    chroma_W = (W + chroma_w_factor - 1) // chroma_w_factor
    Y        = yuv_arr[0]
    Cb_small = yuv_arr[1, :chroma_H, :chroma_W]
    Cr_small = yuv_arr[2, :chroma_H, :chroma_W]

    # Upsample chroma to luma resolution using nearest-neighbour replication,
    # which matches nvjpeg2k's internal nearest-neighbour chroma upsampling.
    # cv2.resize uses (width, height) order for dsize.
    Cb = cv2.resize(Cb_small, (W, H), interpolation=cv2.INTER_NEAREST)
    Cr = cv2.resize(Cr_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # Reference: decode the same file directly to SRGB (I_RGB layout).
    rgb_img = decoder.read(img_path, params=nvimgcodec.DecodeParams(
        color_spec=nvimgcodec.ColorSpec.SRGB,
        sample_format=nvimgcodec.SampleFormat.I_RGB,
        allow_any_depth=True,
    ))
    assert rgb_img is not None, f"Failed to decode {img_file} to SRGB"
    assert rgb_img.sample_format == nvimgcodec.SampleFormat.I_RGB
    assert rgb_img.color_spec == nvimgcodec.ColorSpec.SRGB
    rgb_arr = np.asarray(rgb_img.cpu())  # (H, W, 3)
    assert rgb_arr.shape == (H, W, 3)
    assert rgb_arr.dtype == expected_dtype

    # 1) Manual BT.601 full-range float32.
    #    For int16: neutral=0, so (Cb - 0) = Cb; this is algebraically equivalent to
    #    shifting int16 values to uint16 (adding 32768), applying the uint16 formula,
    #    and shifting back — the ±32768 cancels in the Cb/Cr difference term.
    Yf  = Y.astype(np.float32)
    Cbf = Cb.astype(np.float32)
    Crf = Cr.astype(np.float32)
    R = np.clip(np.round(Yf + 1.402    * (Crf - neutral)), clip_min, clip_max).astype(expected_dtype)
    G = np.clip(np.round(Yf - 0.344136 * (Cbf - neutral) - 0.714136 * (Crf - neutral)), clip_min, clip_max).astype(expected_dtype)
    B = np.clip(np.round(Yf + 1.772    * (Cbf - neutral)), clip_min, clip_max).astype(expected_dtype)
    compare_host_images([np.stack([R, G, B], axis=-1)], [rgb_arr], threshold=manual_thr)

    # 2) OpenCV COLOR_YCrCb2RGB.  int16 is not natively supported; shift to uint16,
    #    convert, then shift back and clamp to the full int16 range.
    if expected_dtype == np.int16:
        Y_u  = (Y.astype(np.int32)  + 32768).astype(np.uint16)
        Cb_u = (Cb.astype(np.int32) + 32768).astype(np.uint16)
        Cr_u = (Cr.astype(np.int32) + 32768).astype(np.uint16)
        rgb_u = cv2.cvtColor(np.stack([Y_u, Cr_u, Cb_u], axis=-1), cv2.COLOR_YCrCb2RGB)
        rgb_cv2 = np.clip(rgb_u.astype(np.int32) - 32768, -32768, 32767).astype(np.int16)
    else:
        rgb_cv2 = cv2.cvtColor(np.stack([Y, Cr, Cb], axis=-1), cv2.COLOR_YCrCb2RGB)
    compare_host_images([rgb_cv2], [rgb_arr], threshold=cv2_thr)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
