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
import struct
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from utils import get_nvjpeg_ver

import numpy as np
import pytest
from PIL import Image as PILImage

# Skip when optional codec stack is not installed (e.g. not installed on Windows with Python >=3.14)
pytest.importorskip("pylibjpeg", reason="pylibjpeg required for pydicom plugin tests")
from pydicom import dcmread
from pydicom.data import get_testdata_file
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import UID

from nvidia.nvimgcodec.tools.dicom.pydicom_plugin import (
    SUPPORTED_DECODER_CLASSES,
    SUPPORTED_TRANSFER_SYNTAXES,
    register,
    unregister,
)

_PYDICOM_DECODER_TEST_FILES = (
    "693_J2KI.dcm",
    "693_J2KR.dcm",
    "GDCMJ2K_TextGBR.dcm",
    "J2K_pixelrep_mismatch.dcm",
    "JPEG-LL.dcm",
    "JPEG2000.dcm",
    "JPGLosslessP14SV1_1s_1f_8b.dcm",
    "MR2_J2KI.dcm",
    "MR2_J2KR.dcm",
    "MR_small_jp2klossless.dcm",
    "RG1_J2KI.dcm",
    "RG1_J2KR.dcm",
    "RG3_J2KI.dcm",
    "RG3_J2KR.dcm",
    "SC_jpeg_no_color_transform.dcm",
    "SC_jpeg_no_color_transform_2.dcm",
    "SC_rgb_dcmtk_+eb+cr.dcm",
    "SC_rgb_dcmtk_+eb+cy+n1.dcm",
    "SC_rgb_dcmtk_+eb+cy+n2.dcm",
    "SC_rgb_dcmtk_+eb+cy+np.dcm",
    "SC_rgb_dcmtk_+eb+cy+s2.dcm",
    "SC_rgb_dcmtk_+eb+cy+s4.dcm",
    "SC_rgb_gdcm_KY.dcm",
    "SC_rgb_jpeg.dcm",
    "SC_rgb_jpeg_app14_dcmd.dcm",
    "SC_rgb_jpeg_dcmtk.dcm",
    "SC_rgb_jpeg_gdcm.dcm",
    "SC_rgb_jpeg_lossy_gdcm.dcm",
    "SC_rgb_small_odd_jpeg.dcm",
    "US1_J2KI.dcm",
    "US1_J2KR.dcm",
    "bad_sequence.dcm",
    "color3d_jpeg_baseline.dcm",
    "emri_small_jpeg_2k_lossless.dcm",
    "examples_jpeg2k.dcm",
    "examples_ybr_color.dcm",
    "explicit_VR-UN.dcm",
)

_PYDICOM_PROBLEMATIC_TEST_FILES = (
    "UN_sequence.dcm",
    "JPEG2000-embedded-sequence-delimiter.dcm",
    "emri_small_jpeg_2k_lossless_too_short.dcm",
)

_PROBLEMATIC_FILES_STEMS = {Path(name).stem.lower() for name in _PYDICOM_PROBLEMATIC_TEST_FILES}

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


def _local_pydicom_dcm(filename: str) -> str | None:
    """Return a local pydicom test-data file without triggering downloads."""
    return get_testdata_file(filename, download=False)


def _require_local_pydicom_dcm(filename: str) -> str:
    path = _local_pydicom_dcm(filename)
    if path is None:
        pytest.fail(
            f"Required pydicom test file {filename!r} is unavailable; "
            "verify the pinned pydicom and pydicom-data packages"
        )
    return path


def get_test_dicoms(folder_path: str | None = None):
    """Get testable DICOM files (supported transfer syntax, not problematic)."""
    if folder_path:
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        dcm_paths = sorted(folder.glob("*.dcm"))
    else:
        dcm_paths = [
            Path(path)
            for filename in _PYDICOM_DECODER_TEST_FILES
            if (path := _local_pydicom_dcm(filename)) is not None
        ]

    for path in dcm_paths:
        if path.stem.lower() in _PROBLEMATIC_FILES_STEMS:
            continue
        try:
            ds = dcmread(str(path), stop_before_pixels=True)
            if ds.file_meta.TransferSyntaxUID in SUPPORTED_TRANSFER_SYNTAXES:
                yield str(path)
        except Exception:
            pass


@pytest.mark.parametrize("filename", _PYDICOM_DECODER_TEST_FILES)
def test_nvimgcodec_decoder_matches_default(filename: str) -> None:
    """Verify nvimgcodec decoder produces same output as default decoder."""
    path = _require_local_pydicom_dcm(filename)

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


@pytest.mark.parametrize("filename", _PYDICOM_PROBLEMATIC_TEST_FILES)
def test_problematic_files_fail_with_all_decoders(filename: str) -> None:
    """Verify problematic files fail with both decoders."""
    path = _require_local_pydicom_dcm(filename)

    # Baseline must fail
    with pytest.raises(Exception):
        dcmread(path).pixel_array

    # nvimgcodec must fail
    with NvimgcodecPlugin():
        with pytest.raises(Exception):
            dcmread(path).pixel_array



# ── Per-stream bug coverage of JPEG lossless marker handling ────────────────
#
# Prior to nvjpeg 13.0.2, the GPU parser failed to properly parse streams that contained
# unexpected markers before the SOF3 (Start of Frame) segment. Notably, when
# encountering a DHT (Define Huffman Table) or COM (Comment) marker before SOF3,
# it would decode the image to an array of all zeros, silently producing invalid output.
#
# The tests below verify that our nvimgcodec integration now rejects these problematic
# marker-ordering cases on old nvJPEG instead of returning zeroed output, while
# preserving valid standard ordering such as SOF3-before-DHT P=12 streams.
# The tested files/scenarios:
#
#   bad_sequence / Siemens  – P=16, SOI→SOF3→DHT→SOS   → nvimgcodec OK (valid)
#   CSS P=12 stream        – P=12, SOI→SOF3→DHT→SOS   → nvimgcodec OK (valid)
#   DHT before SOF3        – P=12, SOI→DHT→SOF3→SOS   → old nvJPEG rejects; fixed nvJPEG decodes
#   COM before SOF3        – P=12, SOI→COM→SOF3→DHT→SOS → old nvJPEG rejects; fixed nvJPEG decodes
#
# Helper utilities and tests below construct each code path to ensure expected behavior.

_RESOURCES_DIR = Path(__file__).parents[4] / "resources"
_CSS_12BIT_JPEG = _RESOURCES_DIR / "jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg"
_LOSSLESS_JPEG_SV1_UID = UID("1.2.840.10008.1.2.4.70")


def _get_pydicom_dcm(stem: str) -> str | None:
    """Return the path to a pydicom test DICOM file by stem, or None if not found."""
    return _local_pydicom_dcm(f"{stem}.dcm")


def _sof3_dimensions(data: bytes) -> tuple[int, int]:
    """Extract (rows, cols) from the first SOF3 marker in a JPEG byte stream."""
    pos = 2  # past SOI
    while pos + 3 < len(data):
        if data[pos] != 0xFF:
            break
        pos += 1
        while pos < len(data) and data[pos] == 0xFF:
            pos += 1
        marker = data[pos]
        pos += 1
        if marker in (0xD9, 0xDA):
            break
        if marker == 0xC3:  # SOF3: [len 2B][P 1B][rows 2B][cols 2B]...
            rows = struct.unpack_from(">H", data, pos + 3)[0]
            cols = struct.unpack_from(">H", data, pos + 5)[0]
            return rows, cols
        seg_len = struct.unpack_from(">H", data, pos)[0]
        pos += seg_len
    raise ValueError("SOF3 marker not found in JPEG stream")


def _make_jpeg_lossless_dicom(jpeg_bytes: bytes, bits_stored: int) -> FileDataset:
    """Wrap raw JPEG lossless bytes in a minimal DICOM FileDataset."""
    rows, cols = _sof3_dimensions(jpeg_bytes)

    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6"
    file_meta.TransferSyntaxUID = _LOSSLESS_JPEG_SV1_UID

    import pydicom.encaps
    ds = FileDataset(None, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.is_implicit_VR = False
    ds.is_little_endian = True
    ds.SOPClassUID = "1.2.840.10008.5.1.4.1.1.4"
    ds.SOPInstanceUID = "1.2.3.4.5.6"
    ds.Rows = rows
    ds.Columns = cols
    ds.BitsAllocated = 16
    ds.BitsStored = bits_stored
    ds.HighBit = bits_stored - 1
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelData = pydicom.encaps.encapsulate([jpeg_bytes])
    return ds


def _jpeg_parse_segments(data: bytes) -> list[tuple[int, bytes]]:
    """Parse a JPEG byte stream into a list of (marker, raw_bytes) pairs.
    Each raw_bytes includes the 0xFF marker byte and length field (where applicable).
    The compressed scan data after SOS is stored under marker 0x00."""
    segs = []
    pos = 2  # past SOI
    while pos < len(data):
        assert data[pos] == 0xFF, f"expected 0xFF at offset {pos}"
        marker = data[pos + 1]
        if marker == 0xD9:  # EOI
            segs.append((0xD9, b"\xff\xd9"))
            break
        if marker == 0xDA:  # SOS: rest of file is SOS header + compressed bitstream
            seg_len = struct.unpack_from(">H", data, pos + 2)[0]
            segs.append((0xDA, data[pos : pos + 2 + seg_len]))
            segs.append((0x00, data[pos + 2 + seg_len :]))  # bitstream + EOI
            break
        seg_len = struct.unpack_from(">H", data, pos + 2)[0]
        segs.append((marker, data[pos : pos + 2 + seg_len]))
        pos += 2 + seg_len
    return segs


def _jpeg_rearrange_dht_before_sof3(jpeg_bytes: bytes) -> bytes:
    """Return a copy of jpeg_bytes with the DHT segment moved before SOF3."""
    segs = _jpeg_parse_segments(jpeg_bytes)
    dht_i = next(i for i, (m, _) in enumerate(segs) if m == 0xC4)
    sof3_i = next(i for i, (m, _) in enumerate(segs) if m == 0xC3)
    if dht_i > sof3_i:
        dht = segs.pop(dht_i)
        segs.insert(sof3_i, dht)
    return b"\xff\xd8" + b"".join(d for _, d in segs)


def _jpeg_insert_com_before_sof3(jpeg_bytes: bytes) -> bytes:
    """Return a copy of jpeg_bytes with a COMMENT segment inserted before SOF3."""
    comment = b"test"
    seg_len = 2 + len(comment)
    com_seg = bytes([0xFF, 0xFE, seg_len >> 8, seg_len & 0xFF]) + comment
    segs = _jpeg_parse_segments(jpeg_bytes)
    sof3_i = next(i for i, (m, _) in enumerate(segs) if m == 0xC3)
    segs.insert(sof3_i, (0xFE, com_seg))
    return b"\xff\xd8" + b"".join(d for _, d in segs)


def _assert_nvimgcodec_matches_default_pydicom(path: str) -> None:
    baseline_pixels = dcmread(path).pixel_array
    with NvimgcodecPlugin():
        nvimgcodec_pixels = dcmread(path).pixel_array
    np.testing.assert_array_equal(nvimgcodec_pixels, baseline_pixels)


def test_lossless_p16_standard_decodes_correctly():
    """
    Siemens-style: BitsStored=12 in the DICOM tag but SOF3 P=16 with standard
    marker ordering (no unexpected marker before SOF3). nvimgcodec should decode
    exactly like native pydicom decoders, confirming the valid case works.
    """
    path = _get_pydicom_dcm("bad_sequence")
    assert path is not None, "bad_sequence.dcm not found in pydicom test data"
    _assert_nvimgcodec_matches_default_pydicom(path)


def test_lossless_p12_decodes_correctly(tmp_path):
    """
    SOF3 P=12 must decode exactly like native pydicom decoders.
    nvimgcodec handles arbitrary bit depths; rejection is not acceptable.
    """
    assert _CSS_12BIT_JPEG.exists(), f"CSS 12-bit test resource not found: {_CSS_12BIT_JPEG}"
    jpeg_bytes = _CSS_12BIT_JPEG.read_bytes()
    ds = _make_jpeg_lossless_dicom(jpeg_bytes, bits_stored=12)
    dcm_path = tmp_path / "p12_mr_style.dcm"
    ds.save_as(str(dcm_path))
    _assert_nvimgcodec_matches_default_pydicom(str(dcm_path))


def test_lossless_dht_before_sof3(tmp_path):
    """
    Philips CT-style — DHT marker before SOF3, using the real CSS 12-bit JPEG with
    DHT and SOF3 segments swapped.
    - nvJPEG < 13.0.2: HasProblematicMarkerBeforeSof3 returns CODESTREAM_UNSUPPORTED -> exception.
    - nvJPEG >= 13.0.2: both CPU and GPU parsers handle DHT-before-SOF3 → non-zero pixels.
    """
    assert _CSS_12BIT_JPEG.exists(), f"CSS 12-bit test resource not found: {_CSS_12BIT_JPEG}"
    jpeg_bytes = _jpeg_rearrange_dht_before_sof3(_CSS_12BIT_JPEG.read_bytes())
    ds = _make_jpeg_lossless_dicom(jpeg_bytes, bits_stored=12)
    dcm_path = tmp_path / "dht_before_sof3.dcm"
    ds.save_as(str(dcm_path))

    baseline_pixels = dcmread(str(dcm_path)).pixel_array
    assert baseline_pixels.max() > 0, "DHT-before-SOF3: native pydicom decoder returned all-zero output"
    with NvimgcodecPlugin():
        if get_nvjpeg_ver() >= (13, 0, 2):
            pixels = dcmread(str(dcm_path)).pixel_array
            np.testing.assert_array_equal(pixels, baseline_pixels)
        else:
            with pytest.raises(Exception):
                dcmread(str(dcm_path)).pixel_array


def test_lossless_com_before_sof3(tmp_path):
    """
    COM marker before SOF3, using the real CSS 12-bit JPEG with a COMMENT segment
    inserted before SOF3.
    - nvJPEG < 13.0.2: HasProblematicMarkerBeforeSof3 returns CODESTREAM_UNSUPPORTED -> exception.
    - nvJPEG >= 13.0.2: GPU kernel handles COMMENT markers → non-zero pixels.
    """
    assert _CSS_12BIT_JPEG.exists(), f"CSS 12-bit test resource not found: {_CSS_12BIT_JPEG}"
    jpeg_bytes = _jpeg_insert_com_before_sof3(_CSS_12BIT_JPEG.read_bytes())
    ds = _make_jpeg_lossless_dicom(jpeg_bytes, bits_stored=12)
    dcm_path = tmp_path / "com_before_sof3.dcm"
    ds.save_as(str(dcm_path))

    baseline_pixels = dcmread(str(dcm_path)).pixel_array
    assert baseline_pixels.max() > 0, "COM-before-SOF3: native pydicom decoder returned all-zero output"
    with NvimgcodecPlugin():
        if get_nvjpeg_ver() >= (13, 0, 2):
            pixels = dcmread(str(dcm_path)).pixel_array
            np.testing.assert_array_equal(pixels, baseline_pixels)
        else:
            with pytest.raises(Exception):
                dcmread(str(dcm_path)).pixel_array


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
    # Fail on an empty list rather than print a header with no rows and no TOTAL, which
    # reads as a successful benchmark that measured nothing.
    if file_paths is not None:
        paths = file_paths
        if not paths:
            raise RuntimeError("No DICOM files to benchmark: the given file_paths is empty.")
    else:
        # get_test_dicoms() resolves pydicom sample files without downloading them, so an
        # environment missing the pinned pydicom-data package yields nothing.
        paths = list(get_test_dicoms())
        if not paths:
            raise RuntimeError(
                "No DICOM files to benchmark. When file_paths is not given the files come "
                "from get_test_dicoms(), which never downloads; install the pinned "
                "pydicom-data package (see test/requirements_lnx_cu12.txt)."
            )
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
