# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import struct
import tempfile
import numpy as np
from nvidia import nvimgcodec
import pytest as t
from utils import is_nvcomp_supported, img_dir_path

backends_list=[
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
    [nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY)],
    None, # use default backend
]

@t.mark.parametrize("backends", backends_list)
@t.mark.parametrize("full_precision", [True, False])
def test_decode_tiff_palette(backends, full_precision):
    if not is_nvcomp_supported():
        if (backends is not None and backends[0].backend_kind == nvimgcodec.BackendKind.GPU_ONLY):
            t.skip("nvCOMP is not supported on this platform")

    path_regular = os.path.join(img_dir_path, "tiff/cat-300572_640.tiff")
    path_palette = os.path.join(img_dir_path, "tiff/cat-300572_640_palette.tiff")

    decoder = nvimgcodec.Decoder(backends=backends)
    decode_params = nvimgcodec.DecodeParams(allow_any_depth=full_precision)
    
    nv_img_regular = decoder.read(path_regular)
    nv_img_palette = decoder.read(path_palette, params=decode_params)

    assert nv_img_regular is not None
    assert nv_img_palette is not None

    img_regular = np.array(nv_img_regular.cpu())
    img_palette = np.array(nv_img_palette.cpu())

    if full_precision:
        assert img_palette.dtype.itemsize == 2
        precision = 16
    else:
        assert img_palette.dtype.itemsize == 1
        precision = 8

    delta = np.abs(img_regular / 256 - img_palette / 2 ** precision)
    assert np.quantile(delta, 0.9) < 0.05, "Original and palette TIFF differ significantly"

@t.mark.parametrize(
    "other_image_path, other_image_precision",
    [
        ("tiff/cat-300572_640_uint16.tiff", 16),
        ("tiff/cat-300572_640_uint32.tiff", 32),
        ("tiff/cat-300572_640_fp32.tiff", 32),
    ]
)
@t.mark.parametrize("backends", backends_list)
def test_decode_tiff_cross_precision_validation(other_image_path, other_image_precision, backends):
    if not is_nvcomp_supported():
        if (backends is not None and backends[0].backend_kind == nvimgcodec.BackendKind.GPU_ONLY):
            t.skip("nvCOMP is not supported on this platform")

    path_regular = os.path.join(img_dir_path, "tiff/cat-300572_640.tiff")
    path_other = os.path.join(img_dir_path, other_image_path)

    decode_params=nvimgcodec.DecodeParams(allow_any_depth=True)
    decoder = nvimgcodec.Decoder()
    img_regular = np.array(decoder.read(path_regular).cpu())
    
    other_image = decoder.read(path_other, params=decode_params).cpu()
    assert other_image.precision == other_image_precision
    img_other = np.asarray(other_image.cpu())

    if "fp32" in other_image_path:
        delta = np.abs(img_regular / 256 - img_other)
    else:
        delta = np.abs(img_regular / 256 - img_other / 2 ** other_image_precision)
    assert np.max(delta) < 1.1 / 256, "Images differ significantly"

@t.mark.parametrize("backends", backends_list)
def test_decode_tiff_uint16_reference(backends):
    path_u16 = os.path.join(img_dir_path, "tiff/uint16.tiff")
    path_u16_npy = os.path.join(img_dir_path, "tiff/uint16.npy")
    params = nvimgcodec.DecodeParams(color_spec=nvimgcodec.ColorSpec.UNCHANGED, allow_any_depth=True)
    dec = nvimgcodec.Decoder(backends=backends)
    img_decoded = dec.read(path_u16, params = params)
    decoded = np.array(img_decoded.cpu())[: , :, 0]  # nvImageCodec gives an extra dimension
    reference = np.load(path_u16_npy)
    np.testing.assert_array_equal(decoded, reference)

# This tests used a crafted TIFF to provoke an OOM error,
# to check that we don't crash and gracefully raise an error
# See:
# https://gitlab.com/libtiff/libtiff/-/issues/621
# https://bugzilla.redhat.com/show_bug.cgi?id=2251326
# https://access.redhat.com/security/cve/CVE-2023-52355

# First the original test from https://gitlab.com/libtiff/libtiff/-/issues/621
def test_decode_tiff_too_many_planes():
    assert None == nvimgcodec.Decoder().read(
        os.path.join(img_dir_path, "tiff/error/too_many_planes.tiff"))

# Now reduce the number of planes to the maximum allowed (32) so that nvimagecodec
# doesn't throw an error early
def test_decode_tiff_oom():
    assert None == nvimgcodec.Decoder().read(
        os.path.join(img_dir_path, "tiff/error/oom.tiff"))


def _create_geotiff(path, width=4, height=4):
    """Build a minimal GeoTIFF from scratch with all eight GeoTIFF/GDAL metadata tags.

    Returns the expected pixel array (HxWx1 uint8) so callers can compare against decoded output.
    The file is constructed with struct so no third-party TIFF library is required.
    """
    # Image: row-major, values 0..(H*W-1)
    image_data = bytes(i % 256 for i in range(width * height))

    # Extra tag payloads (appended after the IFD)
    model_pixel_scale = struct.pack("<3d", 1.0, 1.0, 0.0)  # ModelPixelScaleTag  (3 doubles)
    model_tiepoint = struct.pack("<6d", *([0.0] * 6))  # ModelTiepointTag    (6 doubles)
    model_transform = struct.pack("<16d", *([0.0] * 16))  # ModelTransformationTag (16 doubles)
    geo_key_dir = struct.pack("<4H", 1, 1, 0, 0)  # GeoKeyDirectoryTag  (4 shorts)
    geo_double = struct.pack("<d", 0.0)  # GeoDoubleParamsTag  (1 double)
    geo_ascii = b"WGS 84|\x00"  # GeoAsciiParamsTag   (ASCII)
    gdal_metadata = b"<GDALMetadata/>\x00"  # GDAL_METADATA      (ASCII)
    gdal_nodata = b"-9999\x00"  # GDAL_NODATA         (ASCII)

    # Compute layout offsets
    # 0..7:   header
    # 8..8+len(image_data)-1: pixel data
    # ifd_offset: IFD
    num_entries = 17
    image_offset = 8
    ifd_offset = image_offset + len(image_data)
    entries_offset = ifd_offset + 2  # skip num_entries field
    extra_start = entries_offset + num_entries * 12 + 4  # after entries + next-IFD pointer

    off_mps = extra_start
    off_mt = off_mps + len(model_pixel_scale)
    off_mtx = off_mt + len(model_tiepoint)
    off_gkd = off_mtx + len(model_transform)
    off_gdp = off_gkd + len(geo_key_dir)
    off_gas = off_gdp + len(geo_double)
    off_gmd = off_gas + len(geo_ascii)
    off_gnd = off_gmd + len(gdal_metadata)

    SHORT, LONG, DOUBLE, ASCII = 3, 4, 12, 2

    def ifd_entry(tag, ttype, count, val):
        return struct.pack("<HHII", tag, ttype, count, val)

    entries = b"".join(
        [
            ifd_entry(256, SHORT, 1, width),  # ImageWidth
            ifd_entry(257, SHORT, 1, height),  # ImageLength
            ifd_entry(258, SHORT, 1, 8),  # BitsPerSample
            ifd_entry(259, SHORT, 1, 1),  # Compression = None
            ifd_entry(262, SHORT, 1, 1),  # PhotometricInterpretation
            ifd_entry(273, LONG, 1, image_offset),  # StripOffsets
            ifd_entry(277, SHORT, 1, 1),  # SamplesPerPixel
            ifd_entry(278, SHORT, 1, height),  # RowsPerStrip
            ifd_entry(279, LONG, 1, len(image_data)),  # StripByteCounts
            ifd_entry(33550, DOUBLE, 3, off_mps),  # ModelPixelScaleTag
            ifd_entry(33922, DOUBLE, 6, off_mt),  # ModelTiepointTag
            ifd_entry(34264, DOUBLE, 16, off_mtx),  # ModelTransformationTag
            ifd_entry(34735, SHORT, 4, off_gkd),  # GeoKeyDirectoryTag
            ifd_entry(34736, DOUBLE, 1, off_gdp),  # GeoDoubleParamsTag
            ifd_entry(34737, ASCII, len(geo_ascii), off_gas),  # GeoAsciiParamsTag
            ifd_entry(42112, ASCII, len(gdal_metadata), off_gmd),  # GDAL_METADATA
            ifd_entry(42113, ASCII, len(gdal_nodata), off_gnd),  # GDAL_NODATA
        ]
    )
    assert len(entries) == num_entries * 12

    tiff_bytes = (
        struct.pack("<HHI", 0x4949, 42, ifd_offset)  # header: LE + magic + IFD offset
        + image_data
        + struct.pack("<H", num_entries)
        + entries
        + struct.pack("<I", 0)  # next IFD = none
        + model_pixel_scale
        + model_tiepoint
        + model_transform
        + geo_key_dir
        + geo_double
        + geo_ascii
        + gdal_metadata
        + gdal_nodata
    )
    with open(path, "wb") as f:
        f.write(tiff_bytes)

    return np.frombuffer(image_data, dtype=np.uint8).reshape(height, width, 1)


@t.mark.parametrize("backends", [
    [nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
    None,  # default backend
])
def test_decode_tiff_geotiff(backends):
    """GeoTIFF files with standard geographic metadata tags must decode without errors.

    GeoTIFF/GDAL tags are not natively known to libtiff, which emits
    "Unknown field with tag X" warnings for them. The fix installs a custom libtiff
    warning handler that suppresses those specific warnings while passing through all others.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        geo_path = os.path.join(tmpdir, "geo.tif")
        expected = _create_geotiff(geo_path)

        decoder = nvimgcodec.Decoder(backends=backends)
        result = decoder.read(geo_path)
        assert result is not None
        np.testing.assert_array_equal(np.array(result.cpu()), expected)
