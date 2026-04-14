#!/usr/bin/env python3
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

import os
import numpy as np
import pytest

# Skip when required DICOM codecs stack is not installed
pytest.importorskip("pylibjpeg")

import pydicom
import pydicom.data
import pydicom.examples as examples


from nvidia.nvimgcodec.tools.dicom.convert_htj2k import transcode_datasets_to_htj2k

# HTJ2K Transfer Syntax UIDs
HTJ2K_TRANSFER_SYNTAXES = frozenset([
    "1.2.840.10008.1.2.4.201",  # HTJ2K (Lossless Only)
    "1.2.840.10008.1.2.4.202",  # HTJ2K with RPCL Options (Lossless Only)
    "1.2.840.10008.1.2.4.203",  # HTJ2K
])


@pytest.mark.parametrize("example_name,expected_photo_interp,tolerance", [
    ("ct", None, None),  # CT: grayscale, uncompressed
    ("mr", None, None),  # MR: grayscale, uncompressed
    ("rgb_color", "RGB", None),  # RGB: color, uncompressed
    ("jpeg2k", "RGB", None),  # JPEG2K: YBR_RCT (reversible) -> RGB
    ("ybr_color", "RGB", 5),  # YBR_FULL_422 (lossy JPEG)
])
def test_transcode_lossless(example_name, expected_photo_interp, tolerance):
    """Test lossless transcoding of various DICOM formats.

    Args:
        example_name: Name of pydicom example file (e.g., "ct", "mr", "rgb_color")
        expected_photo_interp: Expected PhotometricInterpretation after transcoding (None = unchanged)
        tolerance: Allowed pixel difference (None = exact match, int = atol for lossy source with decoder differences)
    """
    ds_original = pydicom.dcmread(str(examples.get_path(example_name)))
    ds_original_pixels = ds_original.pixel_array.copy()
    original_photo_interp = ds_original.PhotometricInterpretation

    ds_transcoded = transcode_datasets_to_htj2k([ds_original], skip_transfer_syntaxes=None)[0]

    # Verify HTJ2K transfer syntax
    assert str(ds_transcoded.file_meta.TransferSyntaxUID) in HTJ2K_TRANSFER_SYNTAXES

    # Verify PhotometricInterpretation (either unchanged or as expected)
    if expected_photo_interp is not None:
        assert ds_transcoded.PhotometricInterpretation == expected_photo_interp
    else:
        assert ds_transcoded.PhotometricInterpretation == original_photo_interp

    if tolerance is not None:
        assert np.allclose(ds_original_pixels, ds_transcoded.pixel_array, atol=tolerance, rtol=0)
    else:
        np.testing.assert_array_equal(ds_original_pixels, ds_transcoded.pixel_array)

    # Verify basic metadata preserved
    assert ds_original.Rows == ds_transcoded.Rows
    assert ds_original.Columns == ds_transcoded.Columns

def test_transcode_multiframe_ybr_jpeg():
    """Test transcoding multi-frame JPEG with YCbCr color space to HTJ2K."""
    try:
        ds_original = pydicom.dcmread(pydicom.data.get_testdata_file("examples_ybr_color.dcm"))
    except Exception as e:
        pytest.skip(f"Test data not available: {e}")

    original_pixels = ds_original.pixel_array.copy()
    num_frames = int(ds_original.NumberOfFrames) if hasattr(ds_original, "NumberOfFrames") else 1

    # Override skip list to force transcoding of JPEG
    ds_transcoded = transcode_datasets_to_htj2k([ds_original], skip_transfer_syntaxes=None)[0]

    # Verify transfer syntax and photometric interpretation
    assert str(ds_transcoded.file_meta.TransferSyntaxUID) in HTJ2K_TRANSFER_SYNTAXES
    assert ds_transcoded.PhotometricInterpretation == "RGB"
    
    # Verify metadata
    assert ds_original.Rows == ds_transcoded.Rows
    assert ds_original.Columns == ds_transcoded.Columns
    assert num_frames == int(ds_transcoded.NumberOfFrames)
    
    # Allow tolerance for color space conversion
    assert np.allclose(original_pixels, ds_transcoded.pixel_array, atol=5, rtol=0)

def test_transcode_batch():
    """Test batch transcoding of multiple DICOM files."""
    # Load multiple test files
    datasets = [
        pydicom.dcmread(str(examples.get_path("ct"))),
        pydicom.dcmread(str(examples.get_path("mr"))),
        pydicom.dcmread(str(examples.get_path("rgb_color"))),
    ]
    original_pixels = [ds.pixel_array.copy() for ds in datasets]

    # Transcode all at once
    transcoded = transcode_datasets_to_htj2k(datasets)

    # Verify all were transcoded
    assert len(transcoded) == len(datasets)
    
    for ds_trans, ds_orig, orig_pix in zip(transcoded, datasets, original_pixels):
        assert str(ds_trans.file_meta.TransferSyntaxUID) in HTJ2K_TRANSFER_SYNTAXES
        np.testing.assert_array_equal(orig_pix, ds_trans.pixel_array)

def test_default_progression_order():
    """Test that default progression order is RPCL (1.2.840.10008.1.2.4.202)."""
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    ds_transcoded = transcode_datasets_to_htj2k([ds_original])[0]
    
    # Default should be RPCL
    assert str(ds_transcoded.file_meta.TransferSyntaxUID) == "1.2.840.10008.1.2.4.202"

@pytest.mark.parametrize("prog_order,expected_ts", [
    ("LRCP", "1.2.840.10008.1.2.4.201"),
    ("RLCP", "1.2.840.10008.1.2.4.201"),
    ("RPCL", "1.2.840.10008.1.2.4.202"),
    ("PCRL", "1.2.840.10008.1.2.4.201"),
    ("CPRL", "1.2.840.10008.1.2.4.201"),
])
def test_progression_orders(prog_order, expected_ts):
    """Test all JPEG2000 progression orders produce correct transfer syntaxes and lossless data."""
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    ds_original_pixels = ds_original.pixel_array.copy()
    
    ds_transcoded = transcode_datasets_to_htj2k([ds_original], progression_order=prog_order)[0]
    
    assert str(ds_transcoded.file_meta.TransferSyntaxUID) == expected_ts
    np.testing.assert_array_equal(ds_original_pixels, ds_transcoded.pixel_array)

@pytest.mark.parametrize("invalid_order", [
    "invalid",
    "rpcl",  # lowercase
    "lrcp",  # lowercase
    "ABCD",
    "",
])
def test_invalid_progression_order_raises_error(invalid_order):
    """Test that invalid progression orders raise ValueError."""
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    
    with pytest.raises(ValueError) as ctx:
        transcode_datasets_to_htj2k([ds_original], progression_order=invalid_order)
    
    # Verify error message mentions valid options
    error_msg = str(ctx.value)
    assert "progression_order" in error_msg.lower()
    for valid_order in ["LRCP", "RLCP", "RPCL", "PCRL", "CPRL"]:
        assert valid_order in error_msg

def test_skip_htj2k_by_default():
    """Test that HTJ2K files are skipped by default (returned unchanged)."""
    # First create an HTJ2K file
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    ds_htj2k = transcode_datasets_to_htj2k([ds_original], skip_transfer_syntaxes=None)[0]
    ds_htj2k_pixels = ds_htj2k.pixel_array.copy()

    # Now transcode again with default skip list (should be skipped)
    ds_output = transcode_datasets_to_htj2k([ds_htj2k])[0]

    # Should be returned unchanged (same transfer syntax)
    assert str(ds_htj2k.file_meta.TransferSyntaxUID) == str(ds_output.file_meta.TransferSyntaxUID)
    np.testing.assert_array_equal(ds_htj2k_pixels, ds_output.pixel_array)

def test_skip_transfer_syntaxes_none_forces_retranscoding():
    """Test that skip_transfer_syntaxes=None forces re-transcoding of HTJ2K files."""
    # Create an HTJ2K file with LRCP progression
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    ds_original_pixels = ds_original.pixel_array.copy()
    ds_htj2k_first = transcode_datasets_to_htj2k([ds_original], progression_order="LRCP")[0]
    first_ts = str(ds_htj2k_first.file_meta.TransferSyntaxUID)
    
    # Force re-transcode with different progression order (RPCL)
    # This tests that precision is correctly handled when re-encoding signed INT16
    ds_htj2k_second = transcode_datasets_to_htj2k(
        [ds_htj2k_first], progression_order="RPCL", skip_transfer_syntaxes=None
    )[0]
    second_ts = str(ds_htj2k_second.file_meta.TransferSyntaxUID)
    
    # Verify transfer syntaxes changed (LRCP -> RPCL)
    assert first_ts == "1.2.840.10008.1.2.4.201"  # LRCP
    assert second_ts == "1.2.840.10008.1.2.4.202"  # RPCL
    
    # Verify pixels are still lossless after re-transcoding
    np.testing.assert_array_equal(ds_original_pixels, ds_htj2k_second.pixel_array)

def test_custom_encoding_parameters():
    """Test custom HTJ2K encoding parameters (num_resolutions, code_block_size)."""
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    ds_original_pixels = ds_original.pixel_array.copy()
    ds_transcoded = transcode_datasets_to_htj2k(
        [ds_original],
        num_resolutions=6,
        code_block_size=(64, 64),
        progression_order="RPCL"
    )[0]
    
    assert str(ds_transcoded.file_meta.TransferSyntaxUID) in HTJ2K_TRANSFER_SYNTAXES
    np.testing.assert_array_equal(ds_original_pixels, ds_transcoded.pixel_array)
    #TODO(janton): If possible, verify that the custom encoding parameters were used

def test_empty_dataset_list_returns_empty():
    """Test that empty dataset list returns empty result."""
    result = transcode_datasets_to_htj2k([])
    assert len(result) == 0

def test_transcode_16bit_signed_ct_lossless():
    """Test 16-bit signed CT data (PixelRepresentation=1).
    
    Critical test after nvimagecodec precision bug fix.
    Signed 16-bit data has 15 positive bits (1 bit for sign).
    """
    ds_original = pydicom.dcmread(str(examples.get_path("ct")))
    ds_original_pixels = ds_original.pixel_array.copy()
    
    # CT data should be signed (PixelRepresentation=1)
    if ds_original.PixelRepresentation != 1:
        pytest.skip(f"CT data is not signed (PixelRepresentation={ds_original.PixelRepresentation})")
    
    # Verify it's 16-bit
    assert ds_original.BitsAllocated == 16
    
    ds_transcoded = transcode_datasets_to_htj2k([ds_original])[0]

    # Critical: verify pixel data is exactly preserved
    np.testing.assert_array_equal(ds_original_pixels, ds_transcoded.pixel_array,
                                 err_msg="16-bit signed data not preserved losslessly")
    
    # Verify data type is still signed
    assert ds_transcoded.PixelRepresentation == 1

