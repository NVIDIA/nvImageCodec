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

import copy
import numpy as np
import pytest

pytest.importorskip("highdicom", reason="highdicom required for enhanced multi-frame conversion tests")
pytest.importorskip("pylibjpeg", reason="pylibjpeg required for enhanced multi-frame conversion tests")

import pydicom
import pydicom.data

from nvidia.nvimgcodec.tools.dicom.convert_multiframe import convert_to_enhanced_dicom

# Explicit VR Little Endian
EXPLICIT_VR_LITTLE_ENDIAN = "1.2.840.10008.1.2.1"

# HTJ2K Transfer Syntax UIDs
HTJ2K_LOSSLESS = "1.2.840.10008.1.2.4.201"
HTJ2K_RPCL = "1.2.840.10008.1.2.4.202"


# ============================================================================
# Test Helper Functions
# ============================================================================

def assert_single_multiframe(result, expected_num_frames):
    """Assert that result contains exactly one multiframe dataset with expected frame count."""
    assert len(result) == 1
    ds = result[0]
    assert hasattr(ds, 'NumberOfFrames')
    assert int(ds.NumberOfFrames) == expected_num_frames
    return ds

def assert_all_single_frames(result, expected_count):
    """Assert that result contains only single-frame datasets."""
    assert len(result) == expected_count
    for ds in result:
        num_frames = int(ds.NumberOfFrames) if hasattr(ds, 'NumberOfFrames') else 1
        assert num_frames == 1

def assert_pixels_match(original_series, multiframe_ds):
    """Assert that all frames in multiframe match the original series pixel data."""
    for i, ds_orig in enumerate(original_series):
        np.testing.assert_array_equal(
            ds_orig.pixel_array, 
            multiframe_ds.pixel_array[i],
            err_msg=f"Frame {i} pixel data mismatch"
        )

def modify_series_attributes(series, **kwargs):
    """Modify attributes on all datasets in a series (modifies in-place).
    
    Args:
        series: List of datasets to modify
        **kwargs: Attributes to set on each dataset
        
    Example:
        modify_series_attributes(series, Modality="MR", SOPClassUID="1.2.3.4")
    """
    for ds in series:
        for attr, value in kwargs.items():
            setattr(ds, attr, value)


@pytest.fixture
def ct_series():
    """Create fresh test datasets for each test to avoid pollution."""
    # Create a series of CT slices with different Z positions
    source_file = pydicom.data.get_testdata_file("693_UNCI.dcm")
    ds_base = pydicom.dcmread(source_file)
    
    series = []
    for i in range(5):
        ds = copy.deepcopy(ds_base)
        # Modify each frame so that the pixel values are different
        # (add i to all pixels, which is safe for 693_UNCI.dcm's pixel range)
        ds.PixelData = (ds.pixel_array + i).astype(ds.pixel_array.dtype).tobytes()
        # Set required attributes for enhanced conversion
        ds.Modality = "CT"
        ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.2"  # CT Image Storage
        ds.SeriesInstanceUID = "1.2.3.4.5"
        ds.StudyInstanceUID = "1.2.3.4"
        ds.PatientID = "TEST001"
        ds.ImagePositionPatient = [0.0, 0.0, float(i * 5.0)]
        ds.ImageOrientationPatient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        ds.PixelSpacing = [1.0, 1.0]
        ds.SliceThickness = 5.0
        ds.InstanceNumber = i + 1
        ds.SOPInstanceUID = f"1.2.3.4.5.{i}"  # Unique for each instance
        series.append(ds)
    
    return series


def test_convert_ct_series_to_multiframe(ct_series):
    """Test basic conversion of CT single-frame series to enhanced multiframe."""
    result = convert_to_enhanced_dicom([ct_series], transfer_syntax_uid=EXPLICIT_VR_LITTLE_ENDIAN)
    
    # Should return one multiframe dataset
    ds_enhanced = assert_single_multiframe(result, len(ct_series))
    
    # Verify SOP Class changed to Legacy Converted Enhanced CT
    assert str(ds_enhanced.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.2.2"
    
    # Verify pixel data is lossless for each frame
    assert_pixels_match(ct_series, ds_enhanced)
    
    # Verify metadata preserved
    assert ds_enhanced.SeriesInstanceUID == ct_series[0].SeriesInstanceUID
    assert ds_enhanced.PatientID == ct_series[0].PatientID
    assert ds_enhanced.StudyInstanceUID == ct_series[0].StudyInstanceUID
    
    # Verify new SOPInstanceUID was generated (different from all originals)
    original_sop_uids = {ds.SOPInstanceUID for ds in ct_series}
    assert ds_enhanced.SOPInstanceUID not in original_sop_uids

def test_convert_with_htj2k_transfer_syntax(ct_series):
    """Test conversion with HTJ2K transfer syntax."""
    result = convert_to_enhanced_dicom([ct_series], transfer_syntax_uid=HTJ2K_RPCL)
    
    ds_enhanced = assert_single_multiframe(result, len(ct_series))
    
    # Verify HTJ2K transfer syntax
    assert str(ds_enhanced.file_meta.TransferSyntaxUID) == HTJ2K_RPCL
    
    # Verify lossless - pixel data matches exactly
    assert_pixels_match(ct_series, ds_enhanced)

def test_preserve_series_uid(ct_series):
    """Test that original SeriesInstanceUID is preserved."""
    original_uid = ct_series[0].SeriesInstanceUID
    
    result = convert_to_enhanced_dicom([ct_series])
    assert result[0].SeriesInstanceUID == original_uid


def test_per_frame_functional_groups(ct_series):
    """Test that per-frame functional groups are correctly populated."""
    result = convert_to_enhanced_dicom([ct_series])
    ds_enhanced = result[0]
    
    # Verify PerFrameFunctionalGroupsSequence exists
    assert hasattr(ds_enhanced, 'PerFrameFunctionalGroupsSequence')
    assert len(ds_enhanced.PerFrameFunctionalGroupsSequence) == len(ct_series)
    
    # Verify each frame has PlanePositionSequence with correct values
    for i, frame_item in enumerate(ds_enhanced.PerFrameFunctionalGroupsSequence):
        assert hasattr(frame_item, 'PlanePositionSequence')
        plane_pos = frame_item.PlanePositionSequence[0]
        assert hasattr(plane_pos, 'ImagePositionPatient')
        
        # Verify position matches original
        expected_pos = ct_series[i].ImagePositionPatient
        actual_pos = plane_pos.ImagePositionPatient
        np.testing.assert_array_almost_equal(expected_pos, actual_pos, decimal=2,
                                            err_msg=f"Frame {i} position mismatch")

def test_shared_functional_groups(ct_series):
    """Test that shared functional groups are correctly populated."""
    result = convert_to_enhanced_dicom([ct_series])
    ds_enhanced = result[0]
    
    # Verify SharedFunctionalGroupsSequence exists
    assert hasattr(ds_enhanced, 'SharedFunctionalGroupsSequence')
    shared_fg = ds_enhanced.SharedFunctionalGroupsSequence[0]
    
    # Verify PlaneOrientationSequence in shared groups
    assert hasattr(shared_fg, 'PlaneOrientationSequence')
    plane_orient = shared_fg.PlaneOrientationSequence[0]
    assert hasattr(plane_orient, 'ImageOrientationPatient')
    
    # Verify orientation matches original
    expected_orient = ct_series[0].ImageOrientationPatient
    actual_orient = plane_orient.ImageOrientationPatient
    np.testing.assert_array_almost_equal(expected_orient, actual_orient, decimal=2)

def test_inconsistent_dimensions_returns_single_frames(ct_series):
    """Test that series with inconsistent dimensions returns individual single-frame datasets."""
    # Create datasets with truly different dimensions by creating new pixel arrays
    series_bad = []
    for i in range(3):
        ds = copy.deepcopy(ct_series[i])
        if i == 1:
            # Create a dataset with different dimensions
            # Make it 100x100 instead of 128x128
            new_pixels = ds.pixel_array[:100, :100].copy()
            ds.PixelData = new_pixels.tobytes()
            ds.Rows = 100
            ds.Columns = 100
        series_bad.append(ds)
    
    result = convert_to_enhanced_dicom([series_bad])
    
    # Should return individual single-frame datasets, not multiframe
    assert_all_single_frames(result, len(series_bad))
    
    # Verify pixel data is preserved
    for i, ds in enumerate(result):
        np.testing.assert_array_equal(series_bad[i].pixel_array, ds.pixel_array,
                                        err_msg=f"Dataset {i} pixel data not preserved")

def test_unsupported_modality_returns_single_frames(ct_series):
    """Test that unsupported modalities return individual single-frame datasets."""
    # Create MG (Mammography) series - not supported for enhanced conversion
    mg_series = copy.deepcopy(ct_series[:2])
    modify_series_attributes(mg_series, 
                            Modality="MG", 
                            SOPClassUID="1.2.840.10008.5.1.4.1.1.1.2")
    
    result = convert_to_enhanced_dicom([mg_series])
    
    # Should return individual single-frame datasets
    assert_all_single_frames(result, len(mg_series))
    
    # Verify each dataset is preserved correctly
    for i, ds in enumerate(result):
        assert ds.Modality == "MG"
        np.testing.assert_array_equal(mg_series[i].pixel_array, ds.pixel_array,
                                        err_msg=f"Dataset {i} pixel data not preserved")

def test_unsupported_sop_class_returns_single_frames(ct_series):
    """Test that unsupported SOP Classes return individual single-frame datasets."""
    # Create series with Secondary Capture SOP Class (not supported for enhanced CT)
    series_bad = copy.deepcopy(ct_series[:2])
    modify_series_attributes(series_bad, SOPClassUID="1.2.840.10008.5.1.4.1.1.7")
    
    result = convert_to_enhanced_dicom([series_bad])
    
    # Should return individual single-frame datasets
    assert_all_single_frames(result, len(series_bad))
    
    for i, ds in enumerate(result):
        assert str(ds.SOPClassUID) == "1.2.840.10008.5.1.4.1.1.7"
        np.testing.assert_array_equal(series_bad[i].pixel_array, ds.pixel_array)

def test_already_multiframe_source_single_dataset(ct_series):
    """Test handling of already-multiframe source (single dataset)."""
    # Convert to multiframe first
    multiframe_ds = convert_to_enhanced_dicom([ct_series])[0]
    
    # Try to convert again (should handle gracefully)
    result = convert_to_enhanced_dicom([[multiframe_ds]])
    
    # Should return the multiframe dataset (possibly transcoded)
    assert len(result) == 1
    assert hasattr(result[0], 'NumberOfFrames')
    assert int(result[0].NumberOfFrames) > 1
    
    # Verify pixel data is still lossless
    for i in range(len(ct_series)):
        np.testing.assert_array_equal(ct_series[i].pixel_array, result[0].pixel_array[i])

def test_multiple_multiframe_sources_returned_as_is(ct_series):
    """Test that multiple multiframe datasets in a series are returned as-is."""
    # Convert to multiframe
    multiframe_ds1 = convert_to_enhanced_dicom([ct_series[:2]])[0]
    multiframe_ds2 = convert_to_enhanced_dicom([ct_series[2:4]])[0]
    
    # Multiple multiframe datasets are returned as-is (not re-converted)
    result = convert_to_enhanced_dicom([[multiframe_ds1, multiframe_ds2]])
    
    # Should return both datasets unchanged
    assert len(result) == 2
    assert all(hasattr(ds, 'NumberOfFrames') for ds in result)
    assert int(result[0].NumberOfFrames) == 2
    assert int(result[1].NumberOfFrames) == 2

def test_non_image_dicom_returned_as_is(ct_series):
    """Test that non-image DICOM files are returned unchanged."""
    # Create a mock non-image DICOM (remove PixelData)
    non_image = copy.deepcopy(ct_series[0])
    del non_image.PixelData
    del non_image.Rows
    del non_image.Columns
    non_image.Modality = "SR"  # Structured Report
    
    result = convert_to_enhanced_dicom([[non_image]])
    
    # Should return as-is
    assert len(result) == 1
    assert not hasattr(result[0], 'PixelData')
    assert result[0].Modality == "SR"

def test_mixed_image_non_image_returned_as_is(ct_series):
    """Test that mixed image/non-image series are returned as-is with warning."""
    # Create mixed series
    non_image = copy.deepcopy(ct_series[0])
    del non_image.PixelData
    del non_image.Rows
    non_image.Modality = "SR"
    
    mixed = [ct_series[0], non_image, ct_series[1]]
    
    result = convert_to_enhanced_dicom([mixed])
    
    # Should return all as-is
    assert len(result) == len(mixed)
    
    # Verify each is preserved
    assert hasattr(result[0], 'PixelData')
    assert not hasattr(result[1], 'PixelData')
    assert hasattr(result[2], 'PixelData')


def test_malformed_imagetype_sanitized(ct_series):
    """Test that malformed ImageType attributes are sanitized."""
    # Create series with malformed ImageType
    bad_series = copy.deepcopy(ct_series[:2])
    bad_series[0].ImageType = ""  # Empty string
    bad_series[1].ImageType = ["ORIGINAL"]  # Only one value (needs 2)
    
    # Should not raise error and should convert successfully
    result = convert_to_enhanced_dicom([bad_series])
    
    assert len(result) == 1
    ds_enhanced = result[0]
    
    # ImageType should be fixed (can be list or MultiValue)
    assert hasattr(ds_enhanced, 'ImageType')
    # pydicom MultiValue behaves like list but isn't exactly list
    assert len(ds_enhanced.ImageType) >= 2
    
    # Verify it contains valid values
    valid_first_values = {'ORIGINAL', 'DERIVED'}
    valid_second_values = {'PRIMARY', 'SECONDARY'}
    assert ds_enhanced.ImageType[0] in valid_first_values
    assert ds_enhanced.ImageType[1] in valid_second_values


def test_multiple_series_in_batch(ct_series):
    """Test converting multiple series in one call."""
    # Create fresh copies to avoid test pollution
    series1 = copy.deepcopy(ct_series)
    series2 = copy.deepcopy(ct_series[:3])
    
    # Set different SeriesInstanceUID for series2
    for ds in series2:
        ds.SeriesInstanceUID = "1.2.3.4.6"  # Different from series1
        ds.ImagePositionPatient[2] += 100.0  # Different Z position to distinguish
    
    result = convert_to_enhanced_dicom([series1, series2])
    
    # Should return two multiframe datasets
    assert len(result) == 2
    
    # Verify first series
    assert int(result[0].NumberOfFrames) == len(series1)
    assert result[0].SeriesInstanceUID == series1[0].SeriesInstanceUID
    for i in range(len(series1)):
        np.testing.assert_array_equal(series1[i].pixel_array, result[0].pixel_array[i],
                                        err_msg=f"Series 1, frame {i} mismatch")
    
    # Verify second series
    assert int(result[1].NumberOfFrames) == len(series2)
    assert result[1].SeriesInstanceUID == series2[0].SeriesInstanceUID
    for i in range(len(series2)):
        np.testing.assert_array_equal(series2[i].pixel_array, result[1].pixel_array[i],
                                        err_msg=f"Series 2, frame {i} mismatch")
    
    # Verify series are different
    assert result[0].SeriesInstanceUID != result[1].SeriesInstanceUID

def test_supported_modalities(ct_series):
    """Test that CT, MR, and PT modalities are supported."""
    supported = [
        ("CT", "1.2.840.10008.5.1.4.1.1.2", "1.2.840.10008.5.1.4.1.1.2.2"),  # Legacy Converted Enhanced CT
        ("MR", "1.2.840.10008.5.1.4.1.1.4", "1.2.840.10008.5.1.4.1.1.4.4"),  # Legacy Converted Enhanced MR
        ("PT", "1.2.840.10008.5.1.4.1.1.128", "1.2.840.10008.5.1.4.1.1.128.1"),  # Legacy Converted Enhanced PET
    ]
    
    for modality, sop_class, enhanced_sop_class in supported:
        # Test case: modality=modality
        series = copy.deepcopy(ct_series[:2])
        modify_series_attributes(series, Modality=modality, SOPClassUID=sop_class)
        
        result = convert_to_enhanced_dicom([series])
        
        # Should successfully convert to multiframe
        ds_enhanced = assert_single_multiframe(result, len(series))
        assert str(ds_enhanced.SOPClassUID) == enhanced_sop_class
        
        # Verify pixel data is preserved losslessly
        assert_pixels_match(series, ds_enhanced)

def test_empty_series_list(ct_series):
    """Test that empty series list returns empty result."""
    result = convert_to_enhanced_dicom([])
    assert len(result) == 0

def test_single_frame_series_converts_to_multiframe(ct_series):
    """Test that a series with only one frame still creates a multiframe object."""
    single_frame = [ct_series[0]]
    
    result = convert_to_enhanced_dicom([single_frame])
    
    assert len(result) == 1
    assert hasattr(result[0], 'NumberOfFrames')
    # Single frame should still have NumberOfFrames = 1
    assert int(result[0].NumberOfFrames) == 1
    
    # Verify pixel data preserved
    np.testing.assert_array_equal(single_frame[0].pixel_array, result[0].pixel_array)

def test_invalid_transfer_syntax_raises_error(ct_series):
    """Test that invalid transfer syntax UID raises appropriate error."""
    invalid_ts = "1.2.3.4.5.6.7.8.9"  # Not a valid transfer syntax
    
    with pytest.raises(ValueError) as ctx:
        convert_to_enhanced_dicom([ct_series], transfer_syntax_uid=invalid_ts)
    
    assert "not supported" in str(ctx.value).lower()


def test_unsorted_input_is_sorted_by_spatial_position(ct_series):
    """Regression test: unsorted input datasets must be sorted by spatial position.

    The bug: convert_to_enhanced_dicom accepted pre-loaded datasets in arbitrary
    order (e.g. filesystem order) and passed them directly to highdicom.
    highdicom preserves pixel data in input order but writes correct spatial
    metadata regardless, so frame-by-frame viewers (e.g. OHIF) would display
    slices out of order.

    The fix: sort by ImagePositionPatient projected onto the slice normal before
    conversion. This test passes datasets in reverse Z order and asserts the
    output frames are in ascending Z order, pixel data matches, and
    FrameAcquisitionNumber reflects the sorted order.
    """
    # Pass datasets in reverse spatial order (descending Z)
    reversed_series = list(reversed(ct_series))
    sorted_source = sorted(ct_series, key=lambda ds: float(ds.ImagePositionPatient[2]))

    result = convert_to_enhanced_dicom([reversed_series], transfer_syntax_uid=EXPLICIT_VR_LITTLE_ENDIAN)
    ds_enhanced = assert_single_multiframe(result, len(ct_series))

    # Output frames must be in ascending Z order, regardless of input order
    z_positions = [
        float(ds_enhanced.PerFrameFunctionalGroupsSequence[i].PlanePositionSequence[0].ImagePositionPatient[2])
        for i in range(len(ct_series))
    ]
    assert z_positions == sorted(z_positions), (
        f"Frames are not in ascending spatial order. Z positions: {z_positions}"
    )

    for i, source_ds in enumerate(sorted_source):
        frame_fg = ds_enhanced.PerFrameFunctionalGroupsSequence[i]

        # Pixel data must match the spatially-sorted source frames
        np.testing.assert_array_equal(
            source_ds.pixel_array,
            ds_enhanced.pixel_array[i],
            err_msg=f"Frame {i}: pixel data does not match spatially-sorted source",
        )

        # FrameAcquisitionNumber must reflect the sorted source InstanceNumber
        frame_content = frame_fg.FrameContentSequence[0]
        assert int(frame_content.FrameAcquisitionNumber) == int(source_ds.InstanceNumber), (
            f"Frame {i}: expected FrameAcquisitionNumber {source_ds.InstanceNumber}, "
            f"got {frame_content.FrameAcquisitionNumber}"
        )


def test_unsorted_sagittal_input_is_sorted_by_spatial_position(ct_series):
    """Regression test: spatial sort works for non-axial (sagittal) series.

    For sagittal series the slice-stack normal points along X (not Z), so
    sorting by Z position alone would leave frames in arbitrary order.
    This test verifies that convert_to_enhanced_dicom correctly sorts by
    the projected position along the true slice normal for any imaging plane.
    """
    # Rewrite the series as a sagittal acquisition:
    #   row cosine  = [0, 1, 0]  (phase-encode along Y)
    #   col cosine  = [0, 0, -1] (freq-encode along -Z)
    #   normal      = cross([0,1,0], [0,0,-1]) = [-1, 0, 0]  (slices stack along X)
    sagittal_iop = [0.0, 1.0, 0.0, 0.0, 0.0, -1.0]
    sagittal_series = []
    for i, ds in enumerate(ct_series):
        import copy as _copy
        ds2 = _copy.deepcopy(ds)
        ds2.ImageOrientationPatient = sagittal_iop
        # Slices advance along +X; Z is constant for all frames
        ds2.ImagePositionPatient = [float(i * 5.0), 0.0, 0.0]
        sagittal_series.append(ds2)

    # Pass in reverse spatial order (descending X)
    reversed_series = list(reversed(sagittal_series))

    # Expected order after sort: ascending dot(pos, normal) = ascending -X = descending X.
    # Our sort uses sorted(..., key=dot(pos,normal)) which gives smallest dot first.
    # For normal=[-1,0,0]: dot([x,0,0], [-1,0,0]) = -x → smallest = most negative = largest x.
    # So sorted order is descending X (frame with highest X comes first).
    normal = np.cross(np.array(sagittal_iop[:3]), np.array(sagittal_iop[3:]))
    sorted_source = sorted(sagittal_series,
                           key=lambda ds: np.dot(np.array(ds.ImagePositionPatient), normal))

    result = convert_to_enhanced_dicom([reversed_series], transfer_syntax_uid=EXPLICIT_VR_LITTLE_ENDIAN)
    ds_enhanced = assert_single_multiframe(result, len(sagittal_series))

    # Verify per-frame X positions are in the expected sorted order
    x_positions = [
        float(ds_enhanced.PerFrameFunctionalGroupsSequence[i]
              .PlanePositionSequence[0].ImagePositionPatient[0])
        for i in range(len(sagittal_series))
    ]
    expected_x = [float(ds.ImagePositionPatient[0]) for ds in sorted_source]
    assert x_positions == expected_x, (
        f"Sagittal frames not in correct spatial order. "
        f"Got X={x_positions}, expected X={expected_x}"
    )

    # Pixel data must match the spatially-sorted source frames
    for i, source_ds in enumerate(sorted_source):
        np.testing.assert_array_equal(
            source_ds.pixel_array,
            ds_enhanced.pixel_array[i],
            err_msg=f"Frame {i}: pixel data does not match spatially-sorted sagittal source",
        )


def test_series_with_mixed_transfer_syntaxes_handled(ct_series):
    """Series with different transfer syntaxes should be handled gracefully."""
    # Create series with mixed transfer syntaxes
    mixed_series = copy.deepcopy(ct_series[:3])
    
    # Set different transfer syntaxes
    mixed_series[0].file_meta.TransferSyntaxUID = EXPLICIT_VR_LITTLE_ENDIAN
    mixed_series[1].file_meta.TransferSyntaxUID = "1.2.840.10008.1.2"  # Implicit VR
    mixed_series[2].file_meta.TransferSyntaxUID = EXPLICIT_VR_LITTLE_ENDIAN
    
    # Should handle gracefully (either convert or return as-is)
    try:
        result = convert_to_enhanced_dicom([mixed_series])
        
        # If it succeeds, verify data is preserved
        if len(result) == 1 and hasattr(result[0], 'NumberOfFrames') and int(result[0].NumberOfFrames) > 1:
            # Successfully converted to multiframe
            for i in range(len(mixed_series)):
                np.testing.assert_array_equal(mixed_series[i].pixel_array, result[0].pixel_array[i])
        else:
            # Returned as individual frames - verify all present
            assert len(result) == len(mixed_series)
    except Exception as e:
        # If it fails, should be a clear error message
        assert "transfer syntax" in str(e).lower(), \
                     "Error should mention transfer syntax inconsistency"
