# SPDX-FileCopyrightText: Copyright (c) 2025-2026 MONAI Consortium
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pytest
from PIL import Image as PILImage

# Skip when optional codec stack is not installed (e.g. not installed on Windows with Python >=3.14)
pytest.importorskip("pylibjpeg", reason="pylibjpeg required for pydicom plugin tests")
from pydicom import dcmread
from pydicom.data import get_testdata_files

from nvidia.nvimgcodec.tools.dicom.pydicom_plugin import (
    SUPPORTED_DECODER_CLASSES,
    SUPPORTED_TRANSFER_SYNTAXES,
    register,
    unregister,
)

# Files that fail to decode with any decoder
_PROBLEMATIC_FILES_STEMS = [
    "un_sequence",  # The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Doub...
    "jpeg2000-embedded-sequence-delimiter",  # Unable to decode as exceptions were raised by all available ...
    "emri_small_jpeg_2k_lossless_too_short",  # The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Doub...
]

_logger = logging.getLogger(__name__)


class NvimgcodecPlugin:
    """Context manager for nvimgcodec plugin registration/unregistration.
    
    This class helps isolate plugin testing by temporarily removing all other
    decoders and registering only the nvimgcodec plugin, then restoring
    the original state when done.
    """

    def __init__(self):
        self._prev_decoders = dict[str, Any]()

    def __enter__(self):
        for decoder_class in SUPPORTED_DECODER_CLASSES:
            self._prev_decoders[decoder_class.UID.name] = decoder_class._available
            decoder_class._available = {}  # remove all plugins
        register()  # register nvimgcodec plugin
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            unregister()
        except Exception:
            pass
        for decoder_class in SUPPORTED_DECODER_CLASSES:
            decoder_class._available = self._prev_decoders[decoder_class.UID.name]
        self._prev_decoders = {}
        return False


def _iter_frames(pixel_array: np.ndarray) -> Iterator[tuple[int, np.ndarray, bool]]:
    """Yield per-frame arrays and whether they represent color data.
    
    Args:
        pixel_array: Input pixel array that may be 2D, 3D, or 4D
        
    Yields:
        tuple[int, np.ndarray, bool]: Frame index, frame data, and whether it's color data
        
    Raises:
        ValueError: If pixel array has unsupported shape for PNG export
    """
    arr = np.asarray(pixel_array)
    if arr.ndim == 2:
        yield 0, arr, False
        return

    if arr.ndim == 3:
        if arr.shape[-1] in (3, 4):
            yield 0, arr, True
        else:
            for index in range(arr.shape[0]):
                frame = arr[index]
                yield index, frame, False  # grayscale multi-frame images
        return

    if arr.ndim == 4:
        for index in range(arr.shape[0]):
            frame = arr[index]
            is_color = frame.shape[-1] in (3, 4)
            yield index, frame, is_color
        return

    raise ValueError(f"Unsupported pixel array shape {arr.shape!r} for PNG export")


def _prepare_frame_for_png(frame: np.ndarray, is_color: bool) -> np.ndarray:
    """Convert a decoded frame into a dtype supported by PNG writers."""
    arr = np.nan_to_num(np.asarray(frame), copy=False)

    # Remove singleton channel dimension for grayscale data.
    if not is_color and arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[..., 0]

    arr_float = arr.astype(np.float64, copy=False)
    if np.issubdtype(arr.dtype, np.integer):
        arr_min = float(arr.min())
        arr_max = float(arr.max())
    else:
        arr_min = float(arr_float.min())
        arr_max = float(arr_float.max())

    if is_color:
        if arr.dtype == np.uint8:
            return arr
        if arr_max == arr_min:
            return np.zeros_like(arr, dtype=np.uint8)
        scaled = (arr_float - arr_min) / (arr_max - arr_min)
        return np.clip(np.round(scaled * 255.0), 0, 255).astype(np.uint8)  # type: ignore[no-any-return]

    # Grayscale path
    if np.issubdtype(arr.dtype, np.integer):
        if arr_min >= 0 and arr_max <= 255:
            return arr.astype(np.uint8, copy=False)
        if arr_min >= 0 and arr_max <= 65535:
            return arr.astype(np.uint16, copy=False)

    if arr_max == arr_min:
        return np.zeros_like(arr_float, dtype=np.uint8)

    use_uint16 = arr_max - arr_min > 255.0
    scale = 65535.0 if use_uint16 else 255.0
    scaled = (arr_float - arr_min) / (arr_max - arr_min)
    scaled = np.clip(np.round(scaled * scale), 0, scale)
    target_dtype = np.uint16 if use_uint16 else np.uint8
    return scaled.astype(target_dtype)  # type: ignore[no-any-return]


def _save_frames_as_png(pixel_array: np.ndarray, output_dir: Path, file_stem: str) -> None:
    """Persist each frame as a PNG image in the specified directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for frame_index, frame, is_color in _iter_frames(pixel_array):
        frame_for_png = _prepare_frame_for_png(frame, is_color)
        image = PILImage.fromarray(frame_for_png)
        filename = output_dir / f"{file_stem}_frame_{frame_index:04d}.png"
        image.save(filename)


def get_test_dicoms(folder_path: str | None = None):
    """Get testable DICOM files (supported transfer syntax, not problematic)."""
    if folder_path:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        dcm_paths = sorted(folder.glob("*.dcm"))
    else:
        dcm_paths = [Path(x) for x in get_testdata_files("*.dcm")]

    for path in dcm_paths:
        if path.stem.lower() in _PROBLEMATIC_FILES_STEMS:
            continue
        try:
            ds = dcmread(str(path), stop_before_pixels=True)
            if ds.file_meta.TransferSyntaxUID in SUPPORTED_TRANSFER_SYNTAXES:
                yield str(path)
        except Exception:
            pass


def _get_problematic_file_paths() -> list[str]:
    """Get full paths for problematic files from pydicom test data."""
    return [
        str(p) for p in [Path(x) for x in get_testdata_files("*.dcm")]
        if p.stem.lower() in _PROBLEMATIC_FILES_STEMS
    ]


@pytest.mark.parametrize("path", list(get_test_dicoms()))
def test_nvimgcodec_decoder_matches_default(path: str) -> None:
    """Verify nvimgcodec decoder produces same output as default decoder."""
    # Decode with default decoder
    baseline_pixels = dcmread(path).pixel_array
    
    # Get transfer syntax to determine appropriate tolerances
    ds = dcmread(path, stop_before_pixels=True)
    transfer_syntax = ds.file_meta.TransferSyntaxUID

    # Decode with nvimgcodec
    with NvimgcodecPlugin():
        nv_pixels = dcmread(path).pixel_array

    # Verify they match
    assert baseline_pixels.shape == nv_pixels.shape
    assert baseline_pixels.dtype == nv_pixels.dtype
    
    # Set tolerances based on transfer syntax (data-driven from error distribution analysis)
    from pydicom.pixels.decoders import (
        JPEGBaseline8BitDecoder,
        JPEG2000Decoder,
        HTJ2KDecoder,
    )
    
    if transfer_syntax == JPEGBaseline8BitDecoder.UID:
        # JPEG Baseline (lossy): measured max_diff=5.0, P99.99=3.0
        strict_atol = 3.0
        relaxed_atol = 5.0
        min_strict_fraction = 0.9999  # 99.99%
    elif transfer_syntax == JPEG2000Decoder.UID:
        # JPEG 2000 (can be lossy or lossless): measured max_diff=1.0, P99.99=1.0
        strict_atol = 1.0
        relaxed_atol = 1.0
        min_strict_fraction = 0.9999  # 99.99%
    elif transfer_syntax == HTJ2KDecoder.UID:
        # HTJ2K (lossy): measured max_diff=1.0, P99.99=0.0
        strict_atol = 0.0
        relaxed_atol = 1.0
        min_strict_fraction = 0.9999  # 99.99%
    else:
        # Lossless formats (JPEG Lossless, JPEG 2000 Lossless, HTJ2K Lossless): measured max_diff=0.0
        strict_atol = 0.0
        relaxed_atol = 0.0
        min_strict_fraction = 1.0  # 100% - perfect match expected
    
    # Custom tolerance check: most pixels within strict tolerance, allow small % outliers
    baseline_flat = baseline_pixels.astype(np.float64).flatten()
    nv_flat = nv_pixels.astype(np.float64).flatten()
    abs_diff = np.abs(baseline_flat - nv_flat)
    
    # Check that required fraction of pixels are within strict tolerance
    pixels_within_strict = np.sum(abs_diff <= strict_atol)
    strict_fraction = pixels_within_strict / abs_diff.size
    
    assert strict_fraction >= min_strict_fraction, (
        f"Only {strict_fraction:.5%} of pixels within atol={strict_atol}, "
        f"expected >= {min_strict_fraction:.5%}. Max diff: {abs_diff.max()}, "
        f"Transfer Syntax: {transfer_syntax.name}"
    )
    
    # Check that all pixels are within relaxed tolerance
    max_diff = abs_diff.max()
    assert max_diff <= relaxed_atol, (
        f"Max absolute difference {max_diff} exceeds relaxed tolerance {relaxed_atol}. "
        f"Transfer Syntax: {transfer_syntax.name}"
    )


@pytest.mark.parametrize("path", _get_problematic_file_paths())
def test_problematic_files_fail_with_all_decoders(path: str) -> None:
    """Verify problematic files fail with both decoders."""
    # Baseline must fail
    with pytest.raises(Exception):
        dcmread(path).pixel_array

    # nvimgcodec must fail
    with NvimgcodecPlugin():
        with pytest.raises(Exception):
            dcmread(path).pixel_array



def performance_test_nvimgcodec_decoder_against_defaults(
    file_paths: list[str] | None = None,
    png_output_dir: str | None = None,
    num_warmup_runs: int = 3,
    num_test_runs: int = 10,
) -> None:
    """Benchmark nvimgcodec vs default decoders with proper warmup and multiple runs.

    Args:
        file_paths: List of paths to DICOM files to benchmark. If None, uses pydicom test data.
        png_output_dir: Optional directory to write PNG exports.
        num_warmup_runs: Number of warmup runs per file.
        num_test_runs: Number of timed runs per file.
    """
    paths = file_paths if file_paths is not None else list(get_test_dicoms())
    png_root = Path(png_output_dir).expanduser() if png_output_dir else None
    results = []
    errors = []

    for path in paths:
        _logger.debug(f"Testing {path}")
        try:
            # Warmup baseline
            for _ in range(num_warmup_runs):
                dcmread(path).pixel_array

            # Measure baseline
            baseline_times = []
            baseline_pixels = None
            for _ in range(num_test_runs):
                start = time.perf_counter()
                baseline_pixels = dcmread(path).pixel_array
                baseline_times.append(time.perf_counter() - start)

            # Warmup nvimgcodec
            with NvimgcodecPlugin():
                # Warmup nvimgcodec
                for _ in range(num_warmup_runs):
                    dcmread(path).pixel_array

                # Measure nvimgcodec
                nv_times = []
                nv_pixels = None
                for _ in range(num_test_runs):
                    start = time.perf_counter()
                    nv_pixels = dcmread(path).pixel_array
                    nv_times.append(time.perf_counter() - start)

            # Collect results
            baseline_mean = statistics.mean(baseline_times)
            nv_mean = statistics.mean(nv_times)
            image_shape = baseline_pixels.shape if baseline_pixels is not None else ()
            results.append({
                "file": Path(path).name,
                "syntax": str(dcmread(path, stop_before_pixels=True).file_meta.TransferSyntaxUID),
                "shape": "x".join(str(d) for d in image_shape),
                "baseline_mean": baseline_mean,
                "baseline_std": statistics.stdev(baseline_times) if len(baseline_times) > 1 else 0.0,
                "nv_mean": nv_mean,
                "nv_std": statistics.stdev(nv_times) if len(nv_times) > 1 else 0.0,
                "speedup": baseline_mean / nv_mean if nv_mean > 0 else 0.0,
            })

            # Optional PNG export
            if png_root and baseline_pixels is not None and nv_pixels is not None:
                stem = Path(path).stem
                _save_frames_as_png(baseline_pixels, png_root / stem / "default", stem)
                _save_frames_as_png(nv_pixels, png_root / stem / "nvimgcodec", stem)

        except Exception as e:
            errors.append(Path(path).name)
            _logger.error(f"Error testing {Path(path).name}: {e}")

    # Print results
    print(f"\n## Performance Results ({num_warmup_runs} warmup, {num_test_runs} test runs)\n")
    print("| Transfer Syntax | Shape | Baseline (s) | Std | nvimgcodec (s) | Std | Speedup | File |")
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")

    total_baseline = sum(r["baseline_mean"] for r in results)
    total_nv = sum(r["nv_mean"] for r in results)

    for r in results:
        print(f"| {r['syntax']} | {r['shape']} | {r['baseline_mean']:.4f} | {r['baseline_std']:.4f} | "
              f"{r['nv_mean']:.4f} | {r['nv_std']:.4f} | {r['speedup']:.2f}x | {r['file']} |")

    if total_nv > 0:
        print(f"| **TOTAL** | - | {total_baseline:.4f} | - | {total_nv:.4f} | - | "
              f"{total_baseline/total_nv:.2f}x | - |")

    if errors:
        print(f"\n__Errors__: {errors}")


if __name__ == "__main__":
    try:
        import pylibjpeg
    except ImportError:
        sys.exit("pylibjpeg not available; cannot run performance test")
    performance_test_nvimgcodec_decoder_against_defaults(
        num_warmup_runs=2,
        num_test_runs=3,
    )
