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

"""
MIDI-B Dataset Tests for DICOM Conversion

This test file specifically tests the DICOM conversion pipeline against the
Medical Imaging Data in the Cloud (MIDI-B) dataset from TCIA.

Tests include:
- Downloading MIDI-B series from TCIA
- Converting legacy single-frame series to enhanced multiframe
- Transcoding to HTJ2K (High-Throughput JPEG 2000)
- Verifying pixel-perfect lossless conversion
- Handling various modalities and edge cases from real-world data

Test Modes:
- FAST (default): Runs on 9 hardcoded representative series covering all 8 modalities
  (CR, CT, DX, MG, MR, PT, SR, US) with diverse image counts. Good for regular development.
- SLOW: Runs on the complete MIDI-B dataset (1400+ series). Use for comprehensive
  validation before releases.

Usage:
  # Run fast subset (default - 9 series)
  pytest test_compress_midi_b_dataset.py
  
  # Run comprehensive test on all series (~1400 series)
  pytest test_compress_midi_b_dataset.py -m "slow"
    
  # Run specific series by UID
  pytest -k "1.3.6.1.4.1.14519.5.2.1.8700.9920.275034168943998515927600048873"
  
  # Run with parallelization (recommended for slow mode)
  pytest test_compress_midi_b_dataset.py -m "slow" -n 8
"""

import sys
import tempfile
from pathlib import Path
import zipfile
import numpy as np
import pytest
import logging

pytest.importorskip("highdicom", reason="highdicom required for DICOM conversion tests")
pytest.importorskip("pylibjpeg")
import pydicom

from nvidia.nvimgcodec.tools.dicom.convert_multiframe import convert_to_enhanced_dicom

logger = logging.getLogger(__name__)

# HTJ2K Transfer Syntax UIDs
HTJ2K_TRANSFER_SYNTAXES = frozenset([
    "1.2.840.10008.1.2.4.201",  # HTJ2K (Lossless Only)
    "1.2.840.10008.1.2.4.202",  # HTJ2K with RPCL Options (Lossless Only)
    "1.2.840.10008.1.2.4.203",  # HTJ2K
])


# Hardcoded list of representative series UIDs for fast testing
# These were selected to cover all 8 modalities with diverse image counts and collections
FAST_TEST_SERIES_UIDS = [
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.126449378995880911270289880471",  # CR, 1 image, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.275034168943998515927600048873",  # CT, 97 images, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.543754001074782261949315981043",  # DX, 2 images, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.268606712889231183485099675138",  # MG, 4 images, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.266467002431865277398063957433",  # MR, 12 images, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.187241663860918528303113918662",  # MR, 46 images, Curated-Validation
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.270527999199071377013106821237",  # PT, 83 images, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.789149905765952197628645567193",  # US, 1 image, Curated-Test
    "1.3.6.1.4.1.14519.5.2.1.8700.9920.939883308232520798741231579935",  # SR, 1 image, Curated-Test
]


def _get_midi_b_series_for_testing(use_full_dataset=False):
    """
    Get list of MIDI-B series UIDs for parametrized testing.
    
    Args:
        use_full_dataset: If True, return all series. If False, return hardcoded representative subset.
    
    Returns:
        List of (series_uid, series_info_dict) tuples for testing
    """
    # Test file is now at test/python/integration/dicom/
    try:
        from .download_midi_b_data import get_all_series
        print("[Collection] Fetching MIDI-B series list...")
        all_series = get_all_series()
        
        if not all_series:
            print("[Collection] No series found")
            pytest.fail("No MIDI-B series available for testing")
        
        print(f"[Collection] Found {len(all_series)} total series")
        
        # Build a lookup dictionary for fast access
        series_by_uid = {s.get('SeriesInstanceUID'): s for s in all_series if s.get('SeriesInstanceUID')}
        
        # Select subset or use all based on parameter
        if use_full_dataset:
            selected_series = all_series
            print(f"[Collection] Using all {len(selected_series)} series (slow mode)")
        else:
            # Use hardcoded list of representative series
            selected_series = []
            for uid in FAST_TEST_SERIES_UIDS:
                if uid in series_by_uid:
                    selected_series.append(series_by_uid[uid])
                else:
                    raise ValueError(f"[Collection] Warning: Hardcoded series UID not found: {uid}")
            print(f"[Collection] Using {len(selected_series)} hardcoded representative series (fast mode)")
        
        # Return tuples of (series_uid, series_dict) for parametrization
        result = [(s.get('SeriesInstanceUID'), s) for s in selected_series if s.get('SeriesInstanceUID')]
        print(f"[Collection] Parametrizing with {len(result)} series")
        return result
    except Exception as e:
        print(f"[Collection] Could not get MIDI-B series list: {e}")
        import traceback
        traceback.print_exc()
        pytest.fail(f"Failed to fetch MIDI-B series list: {e}")


def _ensure_series_downloaded(series_uid: str, midi_b_data_dir: Path) -> None:
    try:
        from .download_midi_b_data import download_series
    except ImportError as e:
        pytest.fail(f"Could not import download functions: {e}")

    zip_pattern = f"*{series_uid}*.zip"
    existing_zips = list(midi_b_data_dir.rglob(zip_pattern)) if midi_b_data_dir.exists() else []
    if existing_zips:
        return

    result = download_series(series_uid)
    if result is None:
        pytest.fail(
            f"Could not download series {series_uid} "
            "(download failed or was skipped; check logs for 'Response ended prematurely' or network errors)"
        )


def _find_series_zip(series_uid: str, midi_b_data_dir: Path) -> Path:
    zip_pattern = f"*{series_uid}*.zip"
    zip_files = list(midi_b_data_dir.rglob(zip_pattern))
    if not zip_files:
        pytest.fail(f"Zip file not found for series {series_uid}")
    return zip_files[0]


def _extract_zip(zip_file: Path, extract_dir: Path) -> None:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(str(extract_dir))


def _is_dicom_file(file_path: Path) -> bool:
    try:
        with open(file_path, 'rb') as f:
            f.seek(128)
            return f.read(4) == b'DICM'
    except Exception:
        return False


def _find_dicom_files(extract_dir: Path) -> list:
    dicom_files = []
    for file_path in extract_dir.rglob("*"):
        if file_path.is_file() and _is_dicom_file(file_path):
            dicom_files.append(str(file_path))
    return dicom_files


def _read_datasets(dicom_files: list) -> list:
    return [pydicom.dcmread(file_path) for file_path in dicom_files]


def _compute_series_normal(datasets: list):
    """
    Compute the slice-stack normal for a series using cross(row_cosine, col_cosine).

    Returns a unit normal vector if all frames share a consistent ImageOrientationPatient,
    otherwise returns None.
    """
    if not all(hasattr(ds, 'ImageOrientationPatient') for ds in datasets):
        return None
    first_iop = list(datasets[0].ImageOrientationPatient)
    if not all(list(ds.ImageOrientationPatient) == first_iop for ds in datasets):
        return None
    orientation = np.array(first_iop, dtype=float).reshape(2, 3)
    return np.cross(orientation[0], orientation[1])


def _sort_datasets(datasets: list) -> list:
    """Sort datasets by spatial position along the series normal.

    Uses the same normal-projection sort as convert_to_enhanced_dicom so that
    both the pre-sort and the comparison agree on frame ordering across all
    acquisition planes (axial, sagittal, coronal, oblique).
    """
    normal = _compute_series_normal(datasets)
    if normal is not None:
        return sorted(
            datasets,
            key=lambda ds: (
                np.dot(np.array(list(ds.ImagePositionPatient), dtype=float), normal)
                if hasattr(ds, 'ImagePositionPatient') else 0.0,
                int(getattr(ds, 'InstanceNumber', 0)),
            ),
        )
    return sorted(
        datasets,
        key=lambda ds: (
            float(ds.ImagePositionPatient[2]) if hasattr(ds, 'ImagePositionPatient') else 0.0,
            int(getattr(ds, 'InstanceNumber', 0)),
        ),
    )


def _check_encoding_attributes_match(ds_orig, ds_enhanced):
    """
    Check if critical encoding attributes match between original and enhanced datasets.
    
    Args:
        ds_orig: Original pydicom.Dataset
        ds_enhanced: Enhanced pydicom.Dataset
    
    Raises:
        AssertionError: If encoding attributes don't match
    """
    attrs = [
        ('BitsAllocated', 'BitsAllocated'),
        ('BitsStored', 'BitsStored'),
        ('HighBit', 'HighBit'),
        ('PixelRepresentation', 'PixelRepresentation'),
        ('PhotometricInterpretation', 'PhotometricInterpretation'),
        ('SamplesPerPixel', 'SamplesPerPixel'),
    ]

    differences = []
    for display_name, attr_name in attrs:
        orig_val = getattr(ds_orig, attr_name, 'N/A')
        enh_val = getattr(ds_enhanced, attr_name, 'N/A')
        if orig_val != 'N/A' and orig_val != enh_val:
            differences.append((display_name, orig_val, enh_val))

    if differences:
        error_msg = "\nEncoding attribute mismatch detected!\n"
        error_msg += "The following critical attributes differ between original and enhanced:\n"
        for attr, orig_val, enh_val in differences:
            error_msg += f"  - {attr}: {orig_val} → {enh_val}\n"
        raise AssertionError(error_msg)


def compare_enhanced_dicom_with_originals(
    enhanced_dicom_path: str,
    original_datasets: list,
    series_uid: str = "unknown",
    expected_transfer_syntaxes: frozenset = HTJ2K_TRANSFER_SYNTAXES,
    require_exact_match: bool = True,
    require_encoding_match: bool = False,
):
    """
    Compare an enhanced/multiframe DICOM file with original single-frame datasets.
    
    This function validates that the pixel data in the enhanced DICOM matches
    the original datasets exactly (or within tolerance). It performs frame-by-frame
    comparison.
    
    Args:
        enhanced_dicom_path: Path to the enhanced/multiframe DICOM file to verify
        original_datasets: List of pydicom.Dataset objects representing the original
            single-frame DICOMs, or list of tuples (z_pos, dataset, file_path)
        series_uid: Series UID for identification in error messages
        expected_transfer_syntaxes: Set of expected transfer syntax UIDs (default: HTJ2K)
        require_exact_match: If True, require exact pixel match. If False, allow small differences
        require_encoding_match: If True, fail test if critical encoding attributes differ
    
    Returns:
        dict: Comparison results containing:
            - 'frames_matched': Number of frames that matched
            - 'total_frames': Total number of frames compared
            - 'max_differences': List of max pixel differences per frame
            - 'transfer_syntax': Transfer syntax UID of the enhanced file
            - 'all_match': Boolean indicating if all frames matched
            - 'encoding_comparison': Dict with encoding comparison details (if require_encoding_match=True)
    
    Raises:
        AssertionError: If validation fails (frame counts mismatch, shape mismatch, encoding mismatch, etc.)
        ValueError: If pixel values don't match (when require_exact_match=True)
    """
    # Read the enhanced DICOM file
    ds_enhanced = pydicom.dcmread(enhanced_dicom_path)
    
    # Verify transfer syntax
    enhanced_ts = str(ds_enhanced.file_meta.TransferSyntaxUID)
    if expected_transfer_syntaxes and enhanced_ts not in expected_transfer_syntaxes:
        raise AssertionError(
            f"Transfer syntax mismatch: expected one of {expected_transfer_syntaxes}, "
            f"got {enhanced_ts}"
        )
    
    # Get number of frames
    num_frames = int(ds_enhanced.NumberOfFrames) if hasattr(ds_enhanced, 'NumberOfFrames') else 1
    
    # Get pixel data
    enhanced_pixels = ds_enhanced.pixel_array
    
    # Normalize original_datasets format
    normalized_originals = []
    
    # First pass: collect all items and detect multiframe
    temp_items = []
    for item in original_datasets:
        if isinstance(item, tuple) and len(item) == 3:
            temp_items.append(item)
        elif isinstance(item, pydicom.Dataset):
            z_pos = (
                float(item.ImagePositionPatient[2])
                if hasattr(item, "ImagePositionPatient")
                else 0.0
            )
            file_path = getattr(item, 'filename', 'unknown')
            temp_items.append((z_pos, item, file_path))
        else:
            raise ValueError(f"Invalid original dataset format: {type(item)}")
    
    # Check if any original datasets are multiframe and expand if needed
    if len(temp_items) == 1:
        z_pos, ds, file_path = temp_items[0]
        num_frames_orig = int(ds.NumberOfFrames) if hasattr(ds, 'NumberOfFrames') else 1
        
        if num_frames_orig > 1:
            # Original is multiframe - expand into individual frames
            print(f"  ℹ️  Original dataset is multiframe with {num_frames_orig} frames - expanding for comparison")
            
            all_pixels = ds.pixel_array
            
            for frame_idx in range(num_frames_orig):
                frame_wrapper = type('FrameWrapper', (), {})()
                
                # Extract frame pixel data
                if all_pixels.ndim == 3:  # Grayscale (N, H, W)
                    frame_pixels = all_pixels[frame_idx]
                elif all_pixels.ndim == 4:  # Color (N, H, W, C)
                    frame_pixels = all_pixels[frame_idx]
                else:
                    frame_pixels = all_pixels
                
                frame_wrapper.pixel_array = frame_pixels
                
                for attr in ['PhotometricInterpretation', 'file_meta']:
                    if hasattr(ds, attr):
                        setattr(frame_wrapper, attr, getattr(ds, attr))
                
                frame_wrapper.InstanceNumber = frame_idx + 1
                
                frame_z_pos = z_pos
                if hasattr(ds, 'PerFrameFunctionalGroupsSequence') and frame_idx < len(ds.PerFrameFunctionalGroupsSequence):
                    frame_item = ds.PerFrameFunctionalGroupsSequence[frame_idx]
                    if hasattr(frame_item, 'PlanePositionSequence'):
                        plane_pos = frame_item.PlanePositionSequence[0]
                        if hasattr(plane_pos, 'ImagePositionPatient'):
                            frame_wrapper.ImagePositionPatient = plane_pos.ImagePositionPatient
                            frame_z_pos = float(plane_pos.ImagePositionPatient[2])
                elif hasattr(ds, 'ImagePositionPatient'):
                    frame_wrapper.ImagePositionPatient = ds.ImagePositionPatient
                
                frame_file_path = f"{file_path}_frame_{frame_idx}"
                normalized_originals.append((frame_z_pos, frame_wrapper, frame_file_path))
        else:
            # Single-frame dataset
            normalized_originals.append((z_pos, ds, file_path))
    else:
        # Multiple datasets - use as-is (should all be single-frame)
        for z_pos, ds, file_path in temp_items:
            num_frames_orig = int(ds.NumberOfFrames) if hasattr(ds, 'NumberOfFrames') else 1
            if num_frames_orig > 1:
                raise ValueError(
                    f"Cannot have multiframe datasets when multiple datasets are provided. "
                    f"Found multiframe dataset with {num_frames_orig} frames at {file_path}"
                )
            normalized_originals.append((z_pos, ds, file_path))
    
    # Sort by projected spatial position along the series normal (same sort as convert_to_enhanced_dicom).
    # For axial series the normal is [0,0,1] so this equals ascending Z; for sagittal/coronal
    # series the normal points along X or Y, ensuring consistent ordering regardless of plane.
    orig_datasets = [x[1] for x in normalized_originals]
    normal = _compute_series_normal(orig_datasets)
    if normal is not None:
        normalized_originals.sort(key=lambda x: (
            np.dot(np.array(list(x[1].ImagePositionPatient), dtype=float), normal)
            if hasattr(x[1], 'ImagePositionPatient') else x[0],
            int(getattr(x[1], 'InstanceNumber', 0)),
        ))
    else:
        normalized_originals.sort(key=lambda x: (
            x[0],  # Primary: Z position
            int(getattr(x[1], 'InstanceNumber', 0))  # Secondary: InstanceNumber
        ))
    
    # Verify frame count matches
    if len(normalized_originals) != num_frames:
        raise AssertionError(
            f"Frame count mismatch: {len(normalized_originals)} original frames "
            f"vs {num_frames} enhanced frames"
        )
    
    # Compare frames
    frames_matched = 0
    max_differences = []
    all_match = True
    
    for frame_idx, (z_pos, ds_orig, orig_file_path) in enumerate(normalized_originals):
        # Get original pixel data
        orig_pixels = ds_orig.pixel_array
        
        # Extract frame from enhanced DICOM
        if enhanced_pixels.ndim == 3:  # Grayscale multiframe (N, H, W)
            enhanced_frame = enhanced_pixels[frame_idx]
        elif enhanced_pixels.ndim == 4:  # Color multiframe (N, H, W, C)
            enhanced_frame = enhanced_pixels[frame_idx]
        else:  # Single frame
            enhanced_frame = enhanced_pixels
        
        # Verify shapes match
        if orig_pixels.shape != enhanced_frame.shape:
            raise AssertionError(
                f"Frame {frame_idx} shape mismatch: original {orig_pixels.shape} "
                f"vs enhanced {enhanced_frame.shape}"
            )
        
        # Compare pixel values
        max_diff = np.abs(orig_pixels.astype(np.float32) - enhanced_frame.astype(np.float32)).max()
        max_differences.append(max_diff)
        
        # Check if pixels match
        if require_exact_match:
            pixels_match = np.array_equal(orig_pixels, enhanced_frame)
        else:
            pixels_match = np.allclose(orig_pixels, enhanced_frame, rtol=1e-5, atol=1e-8)
        
        if not pixels_match:
            all_match = False
        
        if pixels_match:
            frames_matched += 1
    
    # Prepare results
    results = {
        'frames_matched': frames_matched,
        'total_frames': num_frames,
        'max_differences': max_differences,
        'transfer_syntax': enhanced_ts,
        'all_match': all_match,
    }
    
    # Validate encoding if requested
    if require_encoding_match:
        if not normalized_originals or not hasattr(ds_enhanced, 'PixelData'):
            raise AssertionError("No image data available for encoding validation")
        _, ds_orig, _ = normalized_originals[0]
        _check_encoding_attributes_match(ds_orig, ds_enhanced)

    
    # Raise error if pixels don't match and exact match is required
    if not all_match and require_exact_match:
        error_msg = f"\nPixel data mismatch detected!\n"
        error_msg += f"  Frames matched: {frames_matched}/{num_frames}\n"
        error_msg += f"  Max difference: {max(max_differences):.2f}\n"
        error_msg += f"  Mean difference: {np.mean(max_differences):.4f}\n"
        raise ValueError(error_msg)
    
    return results


class TestMIDIBDataset:
    """Tests for MIDI-B dataset conversion to enhanced multiframe HTJ2K."""

    def _test_single_series(self, series_uid, series_info):
        """Test a single MIDI-B series for lossless conversion to HTJ2K."""
        midi_b_data_dir = Path(__file__).resolve().parent / ".midi_b_data"

        _ensure_series_downloaded(series_uid, midi_b_data_dir)
        zip_file = _find_series_zip(series_uid, midi_b_data_dir)

        with tempfile.TemporaryDirectory(prefix=f"midi_b_{series_uid[:20]}_extracted_") as extract_dir, \
             tempfile.TemporaryDirectory(prefix=f"midi_b_{series_uid[:20]}_htj2k_") as output_dir:
            extract_path = Path(extract_dir)
            output_path = Path(output_dir)

            _extract_zip(zip_file, extract_path)
            dicom_files = _find_dicom_files(extract_path)
            if not dicom_files:
                pytest.fail("No valid DICOM files found in extracted archive")

            datasets = _sort_datasets(_read_datasets(dicom_files))

            transfer_syntax = "1.2.840.10008.1.2.4.202"  # HTJ2K with RPCL
            enhanced_datasets = convert_to_enhanced_dicom(
                series_datasets=[datasets],
                transfer_syntax_uid=transfer_syntax,
            )

            if len(enhanced_datasets) == 1:
                enhanced_ds = enhanced_datasets[0]
                if not hasattr(enhanced_ds, 'PixelData'):
                    return

                enhanced_path = output_path / f"{series_uid}_enhanced.dcm"
                enhanced_ds.save_as(str(enhanced_path), enforce_file_format=True)

                original_for_comparison = [
                    (float(ds.ImagePositionPatient[2]) if hasattr(ds, "ImagePositionPatient") else 0.0, ds, getattr(ds, 'filename', 'unknown'))
                    for ds in datasets
                ]

                compare_enhanced_dicom_with_originals(
                    enhanced_dicom_path=str(enhanced_path),
                    original_datasets=original_for_comparison,
                    series_uid=series_uid,
                    expected_transfer_syntaxes=HTJ2K_TRANSFER_SYNTAXES,
                    require_exact_match=True,
                    require_encoding_match=True,
                )
                return

            # Inconsistent dimensions: multiple single-frame outputs
            original_images = [ds for ds in datasets if hasattr(ds, 'PixelData')]
            original_images = _sort_datasets(original_images)

            assert len(original_images) == len(enhanced_datasets), (
                f"Frame count mismatch: {len(original_images)} original frames vs "
                f"{len(enhanced_datasets)} enhanced frames"
            )

            if original_images:
                _check_encoding_attributes_match(original_images[0], enhanced_datasets[0])

            for idx, (ds_orig, ds_enhanced) in enumerate(zip(original_images, enhanced_datasets)):
                enhanced_ts = str(ds_enhanced.file_meta.TransferSyntaxUID)
                assert enhanced_ts in HTJ2K_TRANSFER_SYNTAXES, \
                    f"Frame {idx}: Should be HTJ2K, got {enhanced_ts}"

                np.testing.assert_array_equal(
                    ds_orig.pixel_array,
                    ds_enhanced.pixel_array,
                    err_msg=f"Frame {idx} pixels do not match!",
                )


# Parametrized pytest-style tests
# We provide two versions: a fast subset (default) and full dataset (slow)

@pytest.mark.integration
@pytest.mark.parametrize("series_uid,series_info", _get_midi_b_series_for_testing(use_full_dataset=False))
def test_midi_b_series_representative_subset(series_uid, series_info):
    """
    Parametrized pytest test for representative subset of MIDI-B series.
    
    This test runs on 9 hardcoded series that cover:
    - All 8 modalities: CR, CT, DX, MG, MR, PT, SR, US
    - Diverse image counts: 1 to 97 images per series
    - Multiple collections: Curated-Test and Curated-Validation
    
    This provides good coverage while keeping test times reasonable.
    To run the full dataset (1400+ series), use: pytest -m "slow"
    
    Args:
        series_uid: SeriesInstanceUID to test
        series_info: Dictionary with series metadata from TCIA
    """
    if series_uid is None:
        pytest.fail("No MIDI-B series available")
    
    # Create a test case instance to use its helper method
    test_case = TestMIDIBDataset()
    test_case._test_single_series(series_uid, series_info)


@pytest.mark.slow
@pytest.mark.integration
@pytest.mark.parametrize("series_uid,series_info", _get_midi_b_series_for_testing(use_full_dataset=True))
def test_midi_b_series_full_dataset(series_uid, series_info):
    """
    Parametrized pytest test for ALL MIDI-B series (slow, comprehensive testing).
    
    This test runs on the complete MIDI-B dataset. It's marked as 'slow' and should
    be run in CI/CD or before major releases to ensure comprehensive coverage.
    
    To run only this comprehensive test: pytest -m "slow"
    To skip slow tests (default): pytest -m "not slow"
    
    Args:
        series_uid: SeriesInstanceUID to test
        series_info: Dictionary with series metadata from TCIA
    """
    if series_uid is None:
        pytest.fail("No MIDI-B series available")
    
    # Create a test case instance to use its helper method
    test_case = TestMIDIBDataset()
    test_case._test_single_series(series_uid, series_info)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Run MIDI-B DICOM conversion tests',
        epilog='Recommended: Use pytest for better control over test selection'
    )
    parser.add_argument('--series-uid', type=str, 
                       help='Run test for specific SeriesInstanceUID')
    parser.add_argument('--all', action='store_true', 
                       help='Run tests for all available MIDI-B series (~1400 series, slow)')
    parser.add_argument('--subset', action='store_true', 
                       help='Run tests for 9 hardcoded representative series (fast)')

    args = parser.parse_args()
    test_case = TestMIDIBDataset()

    if args.all:
        # Run all series tests
        all_series = _get_midi_b_series_for_testing(use_full_dataset=True)
        print(f"Running tests for all {len(all_series)} MIDI-B series...")
        for series_uid, series_info in all_series:
            test_case._test_single_series(series_uid, series_info)

        print("\n" + "="*70)
        print("✅ ALL TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
    elif args.subset:
        # Run subset tests
        subset_series = _get_midi_b_series_for_testing(use_full_dataset=False)
        print(f"Running tests for {len(subset_series)} hardcoded representative series...")
        for series_uid, series_info in subset_series:
            test_case._test_single_series(series_uid, series_info)

        print("\n" + "="*70)
        print("✅ SUBSET TESTS COMPLETED SUCCESSFULLY")
        print("="*70)
    elif args.series_uid:
        # Run specific series test
        print(f"Running test for series: {args.series_uid}")
        
        # Get all series info (need full list to find the specific one)
        all_series = _get_midi_b_series_for_testing(use_full_dataset=True)
        
        # Find the specific series
        series_info = None
        for uid, info in all_series:
            if uid == args.series_uid:
                series_info = info
                break
        
        if series_info is None:
            print(f"ERROR: Series UID not found: {args.series_uid}")
            sys.exit(1)
        
        # Run the test
        test_case._test_single_series(args.series_uid, series_info)
        
        print("\n" + "="*70)
        print("✅ TEST COMPLETED SUCCESSFULLY")
        print("="*70)
    else:
        print("\n" + "="*70)
        print("MIDI-B Dataset Test Suite")
        print("="*70)
        print("\nDirect Python Usage:")
        print("  1. Run specific series:          python test_compress_midi_b_dataset.py --series-uid <UID>")
        print("  2. Run 9 representative series:  python test_compress_midi_b_dataset.py --subset")
        print("  3. Run all ~1400 series (slow):  python test_compress_midi_b_dataset.py --all")
        print("\nRecommended Pytest Usage:")
        print("  4. Run fast subset (9 series):   pytest test_compress_midi_b_dataset.py")
        print("  5. Run all series (~1400):       pytest test_compress_midi_b_dataset.py -m 'slow'")
        print("  6. Exclude slow tests:           pytest test_compress_midi_b_dataset.py -m 'not slow'")
        print("  7. Run specific series:          pytest -k '<series_uid>'")
        print("  8. Parallel execution:           pytest test_compress_midi_b_dataset.py -m 'slow' -n 8")
        print("\nFast subset covers all 8 modalities: CR, CT, DX, MG, MR, PT, SR, US")
        print("\nFor more options, use --help")
        print("="*70)
