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

"""End-to-end tests for the documented coordinate-system contract of
nvimgcodecRegion_t. The contract (see include/nvimgcodec.h:nvimgcodecRegion_t):

    When DecodeParams.apply_exif_orientation == True the region is interpreted in
    *display* (post-orientation) coordinates and the decoded output is the displayed
    image cropped to the region. When apply_exif_orientation == False the region is
    interpreted in *codestream* coordinates and matches CodeStream.height/width.

The reference image set under resources/jpeg/exif/ covers all eight EXIF
orientation values, so we can verify the contract against a numpy crop of the
full-image decode for every backend that can read each variant.
"""

from __future__ import annotations
import os
import numpy as np
import pytest as t

from nvidia import nvimgcodec
from utils import img_dir_path, get_nvjpeg_ver, NVJPEG_WITH_FIXED_UPSAMPLING_VERSION


# if there are problems detecting nvJPEG version, use the stricter test
SKIP_NVJPEG_ROI_EDGE_COMPARISON = get_nvjpeg_ver() != (0,0,0) and get_nvjpeg_ver() < NVJPEG_WITH_FIXED_UPSAMPLING_VERSION

EXIF_FILES = [
    "padlock-406986_640_horizontal.jpg",
    "padlock-406986_640_mirror_horizontal.jpg",
    "padlock-406986_640_mirror_horizontal_rotate_270.jpg",
    "padlock-406986_640_mirror_horizontal_rotate_90.jpg",
    "padlock-406986_640_mirror_vertical.jpg",
    "padlock-406986_640_no_orientation.jpg",
    "padlock-406986_640_rotate_180.jpg",
    "padlock-406986_640_rotate_270.jpg",
    "padlock-406986_640_rotate_90.jpg",
]

# Backends to exercise. HW_GPU_ONLY is skipped at runtime if the platform doesn't
# expose a hardware JPEG decoder.
BACKENDS = [
    ("cpu", nvimgcodec.BackendKind.CPU_ONLY),
    ("gpu", nvimgcodec.BackendKind.GPU_ONLY),
    ("hw_gpu", nvimgcodec.BackendKind.HW_GPU_ONLY),
    ("hybrid", nvimgcodec.BackendKind.HYBRID_CPU_GPU),
]


def _make_decoder(backend_kind):
    # The older nvjpeg had bugged roi decode, so for older version
    # pin :fancy_upsampling=0 explicitly so this test stays pixel-exact regardless of
    # whether the platform's nvjpeg default is on or off. With fancy upsampling on,
    # nvjpeg's ROI mode and full-decode-plus-crop disagree by a few LSB at MCU
    # boundaries, which is unrelated to the coord-system contract under test.
    try:
        if get_nvjpeg_ver() != (0,0,0) and get_nvjpeg_ver() < NVJPEG_WITH_FIXED_UPSAMPLING_VERSION:
            options = ":fancy_upsampling=0"
        else:
            options = ""
        return nvimgcodec.Decoder(backends=[nvimgcodec.Backend(backend_kind)],
                                  options=options)
    except RuntimeError as e:
        t.skip(f"Backend {backend_kind} not available: {e}")


def _full_decode(decoder, path, apply_exif_orientation):
    params = nvimgcodec.DecodeParams(apply_exif_orientation=apply_exif_orientation)
    img = decoder.read(path, params=params)
    if img is None:
        t.skip("Decoder unavailable for this image (no backend accepted it).")
    return np.asarray(img.cpu())


def _roi_decode(decoder, path, region, apply_exif_orientation):
    params = nvimgcodec.DecodeParams(apply_exif_orientation=apply_exif_orientation)
    cs = nvimgcodec.CodeStream(path)
    sub = cs.get_sub_code_stream(region=region)
    img = decoder.read(sub, params=params)
    if img is None:
        t.skip("Decoder unavailable for this ROI (no backend accepted it).")
    return np.asarray(img.cpu())


# ROIs covering each corner, the center, and the full image. We pick numbers that fit
# inside the smaller of the two image dimensions (426) so the parametrisation works
# for both landscape (640x426) and portrait (426x640) orientations.
ROI_BOXES = [
    ("top_left",     (0,   0,   80,  120)),
    ("top_right",    (0,   200, 80,  420)),
    ("bottom_left",  (250, 0,   400, 120)),
    ("bottom_right", (250, 200, 400, 420)),
    ("center",       (100, 100, 300, 380)),
    # Full-image ROI. The display dims depend on orientation, so we use a marker
    # tuple and resolve at test time.
    ("full",         None),
]


@t.fixture(scope="module", params=[bn for bn, _ in BACKENDS], ids=[bn for bn, _ in BACKENDS])
def decoder(request):
    name = request.param
    backend_kind = dict(BACKENDS)[name]
    dec = _make_decoder(backend_kind)
    yield dec


_PIXEL_TOLERANCE_BY_BACKEND = {
    "cpu": 0,
    "gpu": 0,
    "hw_gpu": 0,
    "hybrid": 0,
}


@t.mark.parametrize("image_name", EXIF_FILES)
@t.mark.parametrize("roi_name,roi_coords", ROI_BOXES, ids=[r[0] for r in ROI_BOXES])
def test_display_coord_roi_matches_numpy_crop(decoder, image_name, roi_name, roi_coords, request):
    """With apply_exif_orientation=True, a region in display coords must produce
    pixels that equal numpy-cropping the full-image oriented decode by the same
    region. This is the core contract the change enforces.
    """
    backend_id = request.node.callspec.id.split("-")[0]
    tol = _PIXEL_TOLERANCE_BY_BACKEND[backend_id]

    path = os.path.join(img_dir_path, "jpeg", "exif", image_name)
    full = _full_decode(decoder, path, apply_exif_orientation=True)
    H, W = full.shape[:2]

    if roi_coords is None:
        y0, x0, y1, x1 = 0, 0, H, W
    else:
        y0, x0, y1, x1 = roi_coords
        if y1 > H or x1 > W:
            t.skip(f"ROI {(y0, x0, y1, x1)} does not fit in display dims {(H, W)}")

    region = nvimgcodec.Region(y0, x0, y1, x1)
    out = _roi_decode(decoder, path, region, apply_exif_orientation=True)
    expected = full[y0:y1, x0:x1]

    assert out.shape == expected.shape, (
        f"shape mismatch: got {out.shape}, expected {expected.shape}")
    if tol == 0:
        assert np.array_equal(out, expected), "pixel mismatch (CPU path should be exact)"
    else:
        max_diff = int(np.abs(out.astype(int) - expected.astype(int)).max())
        assert max_diff <= tol, (
            f"max pixel diff {max_diff} exceeds tolerance {tol} for backend {backend_id}")


@t.mark.parametrize("image_name", [
    "padlock-406986_640_rotate_90.jpg",   # display dims swapped vs raw
    "padlock-406986_640_rotate_270.jpg",  # display dims swapped vs raw
])
def test_codestream_dims_are_raw(image_name):
    """CodeStream.height/width are documented to be in raw codestream coordinates,
    independent of EXIF orientation."""
    path = os.path.join(img_dir_path, "jpeg", "exif", image_name)
    cs = nvimgcodec.CodeStream(path)
    # The raw codestream of these images is 640 wide x 426 tall.
    assert cs.width == 640
    assert cs.height == 426


@t.mark.parametrize("image_name", [
    "padlock-406986_640_rotate_90.jpg",
    "padlock-406986_640_rotate_270.jpg",
])
def test_apply_exif_orientation_swaps_dims(image_name):
    """Sanity check that apply_exif_orientation actually transposes the image when
    EXIF orientation rotates by 90/270. This is the precondition for the ROI contract."""
    dec = _make_decoder(nvimgcodec.BackendKind.CPU_ONLY)
    path = os.path.join(img_dir_path, "jpeg", "exif", image_name)
    oriented = _full_decode(dec, path, apply_exif_orientation=True)
    raw = _full_decode(dec, path, apply_exif_orientation=False)
    # Raw is landscape 426x640; oriented is portrait 640x426.
    assert raw.shape[:2] == (426, 640)
    assert oriented.shape[:2] == (640, 426)


@t.mark.parametrize("image_name", EXIF_FILES)
def test_raw_coord_roi_still_works_when_orientation_off(image_name):
    """Regression guard: with apply_exif_orientation=False the region is interpreted in
    codestream coordinates (raw dims). This must match numpy-cropping the raw decode."""
    dec = _make_decoder(nvimgcodec.BackendKind.CPU_ONLY)
    path = os.path.join(img_dir_path, "jpeg", "exif", image_name)
    raw = _full_decode(dec, path, apply_exif_orientation=False)
    H, W = raw.shape[:2]
    y0, x0, y1, x1 = 50, 100, 150, 300
    if y1 > H or x1 > W:
        t.skip(f"ROI does not fit in raw dims {(H, W)}")
    region = nvimgcodec.Region(y0, x0, y1, x1)
    out = _roi_decode(dec, path, region, apply_exif_orientation=False)
    expected = raw[y0:y1, x0:x1]
    assert out.shape == expected.shape
    assert np.array_equal(out, expected)


@t.mark.parametrize("backend_name,backend_kind", BACKENDS,
                    ids=[bn for bn, _ in BACKENDS])
def test_dali_repro_case(backend_name, backend_kind):
    """The original DALI failure case: an EXIF-rotated image where a display-coord
    region's y extent exceeds the raw codestream's height. Pre-fix this was rejected
    with PROCESSING_STATUS_ROI_UNSUPPORTED because the bounds check used raw dims.
    Post-fix the bounds check uses display dims when apply_exif_orientation is on.

    Parametrised across every backend so that the contract is enforced on CPU as well
    as on the nvjpeg CUDA / HW / HYBRID paths."""
    path = os.path.join(img_dir_path, "jpeg", "exif", "padlock-406986_640_rotate_270.jpg")
    cs = nvimgcodec.CodeStream(path)
    # Raw codestream is 640 wide x 426 tall. After EXIF correction display is 426 x 640.
    assert cs.width == 640 and cs.height == 426
    # Pick a region whose y extent (500) exceeds raw_H=426 but fits in display H=640.
    region = nvimgcodec.Region(0, 0, 500, 300)
    dec = _make_decoder(backend_kind)
    params = nvimgcodec.DecodeParams(apply_exif_orientation=True)
    sub = cs.get_sub_code_stream(region=region)
    img = dec.read(sub, params=params)
    if img is None:
        t.skip(f"Backend {backend_name} did not accept the ROI (no fallback configured).")
    out = np.asarray(img.cpu())
    assert out.shape == (500, 300, 3), (
        f"Backend {backend_name} returned wrong shape {out.shape} — coord-system regression?")


def test_full_image_roi_matches_full_decode():
    """An ROI that covers the entire image should decode pixel-identically to a
    no-ROI decode."""
    dec = _make_decoder(nvimgcodec.BackendKind.CPU_ONLY)
    path = os.path.join(img_dir_path, "jpeg", "exif", "padlock-406986_640_rotate_90.jpg")
    full = _full_decode(dec, path, apply_exif_orientation=True)
    H, W = full.shape[:2]
    region = nvimgcodec.Region(0, 0, H, W)
    out = _roi_decode(dec, path, region, apply_exif_orientation=True)
    assert out.shape == full.shape
    assert np.array_equal(out, full)


# ---------------------------------------------------------------------------
# OOB-after-EXIF tests. The decode-then-fill path is only implemented for
# device-buffer backends (nvjpeg cuda / HW / hybrid). CPU-only opencv and
# libjpeg_turbo reject OOB ROIs in canDecode, so the cpu backend skips here.
# ---------------------------------------------------------------------------

# Backends that actually run the OOB fill code path. The cpu backend is
# excluded — opencv rejects OOB ROIs and libjpeg_turbo cannot apply EXIF
# orientation at all, so on cpu no decoder accepts these test cases.
_OOB_BACKENDS = [(name, kind) for (name, kind) in BACKENDS if name != "cpu"]


@t.mark.parametrize("backend_name,backend_kind", _OOB_BACKENDS,
                    ids=[bn for bn, _ in _OOB_BACKENDS])
def test_oob_roi_symmetric_frame_on_rotated_image(backend_name, backend_kind):
    """ROI extends past the display image on every edge; the output buffer must
    contain a real-pixel inner region (matching the full display decode) surrounded
    by a fill (zero) frame.

    Scenario: image with 90-degree EXIF, ROI starts negative and
    ends past display dims. The fill must be computed in display coords. If the
    OOB fill mistakenly used raw codestream coords (640w x 426h for this image)
    instead of display coords (426w x 640h), the rows in [426, 640) on the y axis
    would land in the fill region and the *real* image pixels there would be
    zeroed out — i.e. the inner region would not match the display decode.
    """
    path = os.path.join(img_dir_path, "jpeg", "exif", "padlock-406986_640_rotate_90.jpg")
    dec = _make_decoder(backend_kind)
    full = _full_decode(dec, path, apply_exif_orientation=True)
    H, W = full.shape[:2]
    # The image is rotated 90 degrees: display dims swap raw codestream dims.
    assert (H, W) == (640, 426), f"unexpected display dims {(H, W)}"

    pad = 50
    region = nvimgcodec.Region(-pad, -pad, H + pad, W + pad)
    out = _roi_decode(dec, path, region, apply_exif_orientation=True)

    assert out.shape == (H + 2 * pad, W + 2 * pad, 3), f"got {out.shape}"

    # Inner real-pixel region must match the full display decode exactly. This is
    # the assertion that catches a "fill used raw dims" regression on the y axis:
    # rows in [426, 640) would otherwise have been zeroed in display coords.
    inner = out[pad:pad + H, pad:pad + W]
    assert np.array_equal(inner, full), (
        f"inner region differs from full display decode on backend {backend_name} — "
        f"OOB fill likely ran in the wrong coordinate system")

    # The 50-pixel frame on each side must be zero (default fill).
    assert (out[:pad, :] == 0).all(), f"top border not zero on backend {backend_name}"
    assert (out[H + pad:, :] == 0).all(), f"bottom border not zero on backend {backend_name}"
    assert (out[pad:H + pad, :pad] == 0).all(), f"left border not zero on backend {backend_name}"
    assert (out[pad:H + pad, W + pad:] == 0).all(), f"right border not zero on backend {backend_name}"


@t.mark.parametrize("backend_name,backend_kind", _OOB_BACKENDS,
                    ids=[bn for bn, _ in _OOB_BACKENDS])
def test_oob_roi_extends_past_display_h_only(backend_name, backend_kind):
    """Regression guard for a ROI that is
    OOB in display coords but whose y-extent (640+60=700) is also greater than the
    raw codestream height (426). If the OOB fill used raw dims, it would mark rows
    [426, 700) as OOB and zero out the real-pixel rows [426, 640) inside the
    display image. With the correct (display-coord) fill, only rows [640, 700) are
    OOB and rows [0, 640) survive intact.
    """
    path = os.path.join(img_dir_path, "jpeg", "exif", "padlock-406986_640_rotate_90.jpg")
    dec = _make_decoder(backend_kind)
    full = _full_decode(dec, path, apply_exif_orientation=True)
    H, W = full.shape[:2]
    assert (H, W) == (640, 426)

    pad_bottom = 60
    region = nvimgcodec.Region(0, 0, H + pad_bottom, W)
    out = _roi_decode(dec, path, region, apply_exif_orientation=True)
    assert out.shape == (H + pad_bottom, W, 3)

    # In-bounds rows must match the full display decode. This fails if the fill
    # used raw_H=426 instead of display_H=640.
    assert np.array_equal(out[:H], full), (
        f"rows [0, {H}) differ from full display decode on backend {backend_name} — "
        f"OOB fill likely ran in raw codestream coords")
    # OOB rows must be zero.
    assert (out[H:] == 0).all(), (
        f"bottom OOB rows are not zero on backend {backend_name}")


@t.mark.parametrize("backend_name,backend_kind", _OOB_BACKENDS,
                    ids=[bn for bn, _ in _OOB_BACKENDS])
def test_oob_roi_extends_past_display_w_only(backend_name, backend_kind):
    """Mirror of the previous case on the x axis. Display W=426 < raw W=640 for
    this image, so a ROI x-extent of 500 is OOB in display coords but in-bounds in
    raw coords. If the fill used raw dims, no x-OOB fill would happen and the
    right-edge fill region would contain whatever garbage the decoder left there.
    """
    path = os.path.join(img_dir_path, "jpeg", "exif", "padlock-406986_640_rotate_90.jpg")
    dec = _make_decoder(backend_kind)
    full = _full_decode(dec, path, apply_exif_orientation=True)
    H, W = full.shape[:2]
    assert (H, W) == (640, 426)

    pad_right = 74  # 426 + 74 = 500
    region = nvimgcodec.Region(0, 0, H, W + pad_right)
    out = _roi_decode(dec, path, region, apply_exif_orientation=True)
    assert out.shape == (H, W + pad_right, 3)

    # In-bounds columns must match the full display decode.
    assert np.array_equal(out[:, :W], full), (
        f"cols [0, {W}) differ from full display decode on backend {backend_name} — "
        f"OOB fill likely ran in raw codestream coords")
    # OOB cols must be zero.
    assert (out[:, W:] == 0).all(), (
        f"right OOB cols are not zero on backend {backend_name}")


@t.mark.parametrize("backend_name,backend_kind", _OOB_BACKENDS,
                    ids=[bn for bn, _ in _OOB_BACKENDS])
def test_oob_roi_mixed_top_only_partial(backend_name, backend_kind):
    """ROI starts above the image (negative y) but ends well inside display dims.
    Covers the negative-start half of the OOB-fill machinery, which lives on a
    different branch from the past-end case in fill_out_of_bounds_region_device.
    """
    path = os.path.join(img_dir_path, "jpeg", "exif", "padlock-406986_640_rotate_90.jpg")
    dec = _make_decoder(backend_kind)
    full = _full_decode(dec, path, apply_exif_orientation=True)
    H, W = full.shape[:2]
    assert (H, W) == (640, 426)

    pad_top = 40
    y_end_in_display = 200
    x0, x_end_in_display = 50, 300
    region = nvimgcodec.Region(-pad_top, x0, y_end_in_display, x_end_in_display)
    out = _roi_decode(dec, path, region, apply_exif_orientation=True)

    crop_h = pad_top + y_end_in_display
    crop_w = x_end_in_display - x0
    assert out.shape == (crop_h, crop_w, 3)

    # Top pad_top rows must be zero.
    assert (out[:pad_top] == 0).all(), (
        f"top OOB rows are not zero on backend {backend_name}")
    # The rest must match the corresponding display-coord crop.
    assert np.array_equal(out[pad_top:], full[:y_end_in_display, x0:x_end_in_display]), (
        f"in-bounds region differs from display crop on backend {backend_name}")
