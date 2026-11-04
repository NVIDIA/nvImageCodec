# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0

"""Tests for the ``DecodeParams.chroma_subsampling`` knob.

Verifies that:
  * Omitting ``chroma_subsampling`` reproduces the legacy behaviour exactly
    (GRAY for luma-only outputs, NONE otherwise).
  * Requesting CSS_420 on a planar SYCC output yields half-sized chroma planes.
  * Requesting a subsampled chroma on an interleaved output is rejected.
"""

import os

import pytest as t
from nvidia import nvimgcodec
from utils import img_dir_path

JPEG_420 = "jpeg/padlock-406986_640_420.jpg"

try:
    import cupy as cp
    # Force CUDA + libnvrtc to load eagerly. cp.zeros is a plain allocation that does not trigger
    # kernel compilation, so libnvrtc / CUDA header problems would only surface mid-test;
    # cp.random.randint compiles a kernel. Probing here makes the CuPy tests skip, like in the
    # other test modules, in environments where CuPy cannot compile kernels (e.g. no CUDA headers).
    cp.random.randint(0, 255, (100, 100, 3), dtype=cp.uint8)
    CUPY_AVAILABLE = True
except Exception:
    CUPY_AVAILABLE = False


def _planes(img):
    # Image.planes builds CuPy arrays for a device image, so without CuPy the plane
    # geometry is checked on a host copy instead.
    return img.planes if CUPY_AVAILABLE else img.cpu().planes


def test_default_output_is_not_subsampled():
    # Guards the promise that an unset parameter leaves today's behaviour untouched.
    dec = nvimgcodec.Decoder()
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)))
    assert img.chroma_subsampling in (
        nvimgcodec.ChromaSubsampling.CSS_444,  # NVIMGCODEC_SAMPLING_NONE is an alias of CSS_444
        nvimgcodec.ChromaSubsampling.CSS_GRAY,
    )


def test_decode_to_420_yields_half_sized_chroma_planes():
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420))
    img = dec.read(cs, params=params)

    assert img.chroma_subsampling == nvimgcodec.ChromaSubsampling.CSS_420
    planes = _planes(img)
    assert len(planes) == 3
    y, u, v = planes
    assert y.shape[:2] == (cs.height, cs.width)
    assert u.shape[:2] == ((cs.height + 1) // 2, (cs.width + 1) // 2)
    assert v.shape[:2] == u.shape[:2]


def test_subsampling_with_interleaved_rgb_is_rejected():
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.I_RGB,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    with t.raises(Exception) as excinfo:
        dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)
    assert "planar" in str(excinfo.value).lower()


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_planes_are_zero_copy_views_of_the_image_buffer():
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)
    planes = img.planes
    # Views, not copies: the planes must sit inside one allocation, luma first.
    base = planes[0].__cuda_array_interface__["data"][0]
    luma_bytes = planes[0].shape[0] * planes[0].strides[0]
    assert planes[1].__cuda_array_interface__["data"][0] == base + luma_bytes


def test_planes_on_a_non_subsampled_image_are_all_full_size():
    dec = nvimgcodec.Decoder()
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)))
    for p in _planes(img):
        assert p.shape[0] == img.height


def test_flat_array_interface_rejects_a_subsampled_image():
    import numpy as np
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)
    cpu_img = img.cpu()  # must NOT raise - .cpu() copies per-plane and stays valid
    assert len(cpu_img.planes) == 3
    with t.raises(Exception) as excinfo:
        np.asarray(cpu_img)
    assert "Use Image.planes" in str(excinfo.value)


@t.mark.parametrize("fixture,css,fx,fy", [
    ("jpeg/padlock-406986_640_422.jpg", nvimgcodec.ChromaSubsampling.CSS_422, 2, 1),
    ("jpeg/padlock-406986_640_440.jpg", nvimgcodec.ChromaSubsampling.CSS_440, 1, 2),
    ("jpeg/padlock-406986_640_420.jpg", nvimgcodec.ChromaSubsampling.CSS_420, 2, 2),
    # Divisor 4 is its own rounding path: 640 / 4 = 160 wide chroma.
    ("jpeg/padlock-406986_640_411.jpg", nvimgcodec.ChromaSubsampling.CSS_411, 4, 1),
    ("jpeg/padlock-406986_640_410.jpg", nvimgcodec.ChromaSubsampling.CSS_410, 4, 2),
])
def test_chroma_plane_extents_follow_the_subsampling_ratio(fixture, css, fx, fy):
    # Each fixture's own sampling factors match the requested ratio, so this exercises
    # a genuine subsampled decode rather than a transcode across ratios.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=css,
    )
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, fixture))
    img = dec.read(cs, params=params)
    y, u, v = _planes(img)
    assert u.shape[:2] == ((cs.height + fy - 1) // fy, (cs.width + fx - 1) // fx)
    assert v.shape[:2] == u.shape[:2]


def test_odd_height_rounds_chroma_plane_up():
    # 200x213: the only odd-dimension subsampled fixture in the repo. Height 213 is odd,
    # so a 4:2:0 decode's chroma planes must be ceil(213/2) = 107 tall, not 106.
    # Caveat: no fixture has an odd WIDTH with subsampling, so the horizontal ceiling
    # path remains unexercised. This fixture is also progressive, which may route to a
    # different decoder path than the baseline fixtures used elsewhere in this file.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    cs = nvimgcodec.CodeStream(
        os.path.join(img_dir_path, "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg"))
    assert cs.height == 213
    img = dec.read(cs, params=params)
    y, u, v = _planes(img)
    assert u.shape[:2] == (107, (cs.width + 1) // 2)
    assert v.shape[:2] == u.shape[:2]


def test_roi_with_odd_extents_rounds_chroma_plane_up():
    # An ROI with odd extents on a 4:2:0 grid is where the ceil arithmetic that sizes
    # chroma planes gets exercised on the *ROI's* dimensions rather than the full
    # image's - nothing else in this file combines ROI with subsampling. (An odd start
    # offset was tried too, but nvJPEG rejects that ROI outright with "Invalid
    # parameter" - unrelated to what this test targets, so the origin is kept at (0, 0).)
    # enable_roi_fancy_upsampling=1: without it nvJPEG < 13.2 declines ROI decode of a
    # subsampled image outright (see test_decode_roi_fancy_upsampling.py); it is
    # unrelated to the chroma-plane sizing this test targets.
    dec = nvimgcodec.Decoder(options=":enable_roi_fancy_upsampling=1")
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420))
    roi_height, roi_width = 101, 51  # both odd
    region = nvimgcodec.Region(0, 0, roi_height, roi_width)
    sub_cs = cs.get_sub_code_stream(region=region)
    img = dec.read(sub_cs, params=params)
    y, u, v = _planes(img)
    assert y.shape[:2] == (roi_height, roi_width)
    assert u.shape[:2] == ((roi_height + 1) // 2, (roi_width + 1) // 2)
    assert v.shape[:2] == u.shape[:2]


def test_buffer_is_resized_when_subsampling_changes_across_reuse():
    dec = nvimgcodec.Decoder()
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420))
    planar = dict(sample_format=nvimgcodec.SampleFormat.P_YUV, color_spec=nvimgcodec.ColorSpec.SYCC)

    img = dec.read(cs, params=nvimgcodec.DecodeParams(**planar))
    full_shapes = [p.shape for p in _planes(img)]

    img = dec.read(cs, image=img, params=nvimgcodec.DecodeParams(
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420, **planar))
    assert _planes(img)[1].shape[0] < full_shapes[1][0]

    img = dec.read(cs, image=img, params=nvimgcodec.DecodeParams(**planar))
    assert [p.shape for p in _planes(img)] == full_shapes


@t.mark.skipif(not CUPY_AVAILABLE, reason="cupy is not available")
def test_host_and_device_planes_agree():
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)
    host = img.cpu()
    assert [p.shape for p in host.planes] == [p.shape for p in img.planes]


def test_codec_that_cannot_subsample_fails_clearly():
    # A PNG carries no chroma subsampling at all, so a CSS_420 request cannot be honoured.
    # This must surface as a raised error naming the mismatch - not a crash, not a silently
    # non-subsampled result, and not a None that the caller has to notice.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, "png/with_alpha_16bit/4ch16bpp.png"))
    with t.raises(ValueError) as excinfo:
        dec.read(cs, params=params)
    assert "does not match" in str(excinfo.value)


def test_subsampling_that_differs_from_the_source_is_rejected():
    # A 4:4:4 source decoded with a CSS_420 request: decoders emit chroma at the code
    # stream's rate, so the output planes would be sized for 4:2:0 while three full-size
    # planes get written into them. The request must be rejected outright.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, "jpeg/padlock-406986_640_444.jpg"))
    assert cs.height == 426 and cs.width == 640
    # Must raise, not return None: a swallowed error would leave callers with no signal.
    with t.raises(ValueError) as excinfo:
        dec.read(cs, params=params)
    assert "does not match" in str(excinfo.value)


def test_cpu_only_decoder_rejects_subsampled_output():
    # extensions/libjpeg_turbo/libjpeg_turbo_decoder.cpp refuses P_YUV for any code stream
    # that is not 4:4:4, so on a CPU-only backend every subsampled request fails -
    # including the ones the rest of this file proves work on GPU (nvJPEG). This is a
    # different rejection layer than the ValueErrors above (a decode failure that yields
    # None, not a raised exception), so it is worth pinning what that looks like too.
    dec = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)])
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    cs = nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420))
    assert dec.read(cs, params=params) is None


@t.mark.parametrize("fixture,css", [
    ("jpeg/padlock-406986_640_420.jpg", nvimgcodec.ChromaSubsampling.CSS_422),
    ("jpeg/padlock-406986_640_422.jpg", nvimgcodec.ChromaSubsampling.CSS_420),
    ("jpeg/padlock-406986_640_444.jpg", nvimgcodec.ChromaSubsampling.CSS_440),
    ("jpeg/padlock-406986_640_gray.jpg", nvimgcodec.ChromaSubsampling.CSS_420),
])
def test_every_cross_ratio_request_is_rejected(fixture, css):
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=css,
    )
    with t.raises(ValueError):
        dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, fixture)), params=params)


def test_css_gray_on_a_multi_component_output_is_rejected():
    # CSS_GRAY claims the output has no chroma. On a three-plane YUV output that would
    # allocate full-size U/V planes, report the image as "not subsampled" to the flat
    # array interfaces, and expose whatever a luma-only decode left in them.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_GRAY,
    )
    with t.raises(ValueError) as excinfo:
        dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)
    assert "CSS_GRAY" in str(excinfo.value)


def test_css_410v_raises_instead_of_returning_none():
    # CSS_410V is exported to Python but has no plane geometry we can derive. It must
    # raise so the caller sees the reason, rather than being swallowed into a None.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_410V,
    )
    with t.raises(ValueError):
        dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)


def _subsampled_image(host=True):
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_420,
    )
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)), params=params)
    return img.cpu() if host else img


def test_dlpack_rejects_a_subsampled_image():
    # A DLPack tensor is dense by construction: it would advertise
    # (num_planes, H, W) full-size elements over a buffer that only holds
    # H*W + 2*(H/2)*(W/2), so any consumer would read past the allocation.
    img = _subsampled_image()
    with t.raises(RuntimeError) as excinfo:
        img.__dlpack__()
    assert "Use Image.planes" in str(excinfo.value)
    with t.raises(RuntimeError):
        img.to_dlpack()


def test_shape_and_strides_reject_a_subsampled_image():
    # `shape` would claim num_planes * H * W elements while `size` reports the real
    # (smaller) allocation - the same object contradicting itself.
    img = _subsampled_image()
    with t.raises(RuntimeError):
        img.shape
    with t.raises(RuntimeError):
        img.strides


@t.mark.parametrize("fixture,css,fx,fy", [
    (JPEG_420, nvimgcodec.ChromaSubsampling.CSS_420, 2, 2),
    ("jpeg/padlock-406986_640_411.jpg", nvimgcodec.ChromaSubsampling.CSS_411, 4, 1),
    ("jpeg/padlock-406986_640_410.jpg", nvimgcodec.ChromaSubsampling.CSS_410, 4, 2),
])
def test_subsampled_plane_contents_match_an_rgb_reference(fixture, css, fx, fy):
    # Shape assertions alone would still pass if the chroma planes held uninitialised
    # memory. Compare the actual samples against the same image decoded to RGB and
    # converted to YCbCr, with the chroma averaged down by the subsampling factors.
    import numpy as np
    dec = nvimgcodec.Decoder()
    path = os.path.join(img_dir_path, fixture)

    yuv = dec.read(nvimgcodec.CodeStream(path), params=nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=css,
    )).cpu()
    y, u, v = (np.squeeze(np.asarray(p)) for p in yuv.planes)

    rgb = np.asarray(dec.read(nvimgcodec.CodeStream(path), params=nvimgcodec.DecodeParams(
        sample_format=nvimgcodec.SampleFormat.I_RGB)).cpu()).astype(np.float64)
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]

    # JPEG (full-range BT.601) RGB -> YCbCr.
    ref_y = 0.299 * r + 0.587 * g + 0.114 * b
    ref_cb = 128.0 - 0.168736 * r - 0.331264 * g + 0.5 * b
    ref_cr = 128.0 + 0.5 * r - 0.418688 * g - 0.081312 * b

    def downsample(plane):
        h, w = plane.shape[0] // fy * fy, plane.shape[1] // fx * fx
        return plane[:h, :w].reshape(h // fy, fy, w // fx, fx).mean(axis=(1, 3))

    assert y.shape == ref_y.shape
    assert np.abs(y.astype(np.float64) - ref_y).mean() < 2.0
    for got, ref in ((u, ref_cb), (v, ref_cr)):
        ref_ds = downsample(ref)
        got_f = got.astype(np.float64)[:ref_ds.shape[0], :ref_ds.shape[1]]
        assert np.abs(got_f - ref_ds).mean() < 3.0
        # And they must carry real content, not a constant fill.
        assert got_f.std() > 1.0
        assert 1 < got_f.mean() < 254


def test_plane_views_outlive_the_image():
    # Each plane view holds a reference to the Image, so dropping the Image must not
    # free the buffer the view points at.
    import gc
    import numpy as np
    img = _subsampled_image()
    u = img.planes[1]
    expected = np.array(np.asarray(u), copy=True)
    del img
    gc.collect()
    assert np.array_equal(np.asarray(u), expected)
    assert float(np.asarray(u).astype(np.float64).mean()) > 0.0


def test_rotated_asymmetric_subsampling_transposes_chroma_extents():
    # A 4:2:2 source has chroma at half width, full height. A 90-degree EXIF rotation
    # turns that into half height, full width in display coordinates - the sampling axes
    # rotate with the image. The chroma divisors must transpose along with the extents.
    input_img_path = os.path.join(img_dir_path, "jpeg/exif/padlock-406986_640_422_rotate_90.jpg")
    cs = nvimgcodec.CodeStream(input_img_path)
    decoder = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        apply_exif_orientation=True,
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=nvimgcodec.ChromaSubsampling.CSS_422,
    )
    img = decoder.read(cs, params=params)

    # The reported label must transpose along with the extents: a 90-degree rotation of
    # 4:2:2 (half width, full height) is 4:4:0 (full width, half height), not 4:2:2 -
    # otherwise a caller sizing buffers from the property (or a subsequent encode, which
    # reads this same field) gets it wrong even though the plane shapes below are correct.
    assert img.chroma_subsampling == nvimgcodec.ChromaSubsampling.CSS_440

    y, u, v = _planes(img)
    # Codestream is width x height; after a 90-degree rotation the output luma is
    # height x width, and the chroma is half of the *rotated* vertical axis.
    assert y.shape[:2] == (cs.width, cs.height)
    assert u.shape[:2] == ((cs.width + 1) // 2, cs.height)
    assert v.shape[:2] == u.shape[:2]


@t.mark.parametrize("fixture,css", [
    ("jpeg/exif/padlock-406986_640_411_rotate_90.jpg", nvimgcodec.ChromaSubsampling.CSS_411),
    ("jpeg/exif/padlock-406986_640_410_rotate_90.jpg", nvimgcodec.ChromaSubsampling.CSS_410),
])
def test_rotated_411_and_410_are_rejected(fixture, css):
    # Transposing 4:1:1 (4x1) or 4:1:0 (4x2) for a 90-degree rotation gives a 1x4 or 2x4
    # ratio that ChromaSubsampling cannot name, so the decode must raise rather than
    # mislabel the output. These fixtures are the plain 4:1:1 / 4:1:0 ones with only an
    # orientation tag added; the codestream itself is unchanged.
    dec = nvimgcodec.Decoder()
    params = nvimgcodec.DecodeParams(
        apply_exif_orientation=True,
        sample_format=nvimgcodec.SampleFormat.P_YUV,
        color_spec=nvimgcodec.ColorSpec.SYCC,
        chroma_subsampling=css,
    )
    with t.raises(ValueError) as excinfo:
        dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, fixture)), params=params)
    assert "rotated" in str(excinfo.value)

    # Without applying the orientation there is nothing to transpose, so the same
    # request decodes.
    params.apply_exif_orientation = False
    img = dec.read(nvimgcodec.CodeStream(os.path.join(img_dir_path, fixture)), params=params)
    assert img.chroma_subsampling == css


@t.mark.parametrize("codec", ["jpeg", "png", "tiff"])
def test_encoding_a_subsampled_image_is_rejected(codec):
    # Encoders read every input plane as full size, so encoding from subsampled planes
    # is not supported yet. It must fail loudly instead of reading the smaller chroma
    # planes as full-size ones.
    img = _subsampled_image(host=False)
    enc = nvimgcodec.Encoder()
    with t.raises(ValueError) as excinfo:
        enc.encode(img, codec)
    assert "chroma-subsampled" in str(excinfo.value)
    # Batch form: the whole batch is rejected, naming the offending image.
    plain = nvimgcodec.Decoder().read(nvimgcodec.CodeStream(os.path.join(img_dir_path, JPEG_420)))
    with t.raises(ValueError) as excinfo:
        enc.encode([plain, img], codec)
    assert "#1" in str(excinfo.value)
