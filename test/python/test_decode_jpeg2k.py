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

# Get path to test resources directory
img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../resources"))

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
