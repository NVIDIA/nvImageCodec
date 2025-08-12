# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import numpy as np
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path
import nvjpeg_test_speedup

filenames = [
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_440.jpg",
    "jpeg2k/cat-111793_640.jp2",
]

@t.mark.parametrize("filename,height,width,channels,dtype,precision",
                    [
                        ("jpeg/padlock-406986_640_420.jpg", 426, 640, 3, np.uint8, 8),
                        ("jpeg/lossless/cat-1245673_640_grayscale_16bit.jpg", 423, 640, 1, np.uint16, 16),
                        ("jpeg/lossless/cat-3449999_640_grayscale_8bit.jpg", 426, 640, 1, np.uint8, 8),
                        ("jpeg/lossless/cat-3449999_640_grayscale_12bit.jpg", 426, 640, 1, np.uint16, 12),
                        ("jpeg/lossless/cat-3449999_640_grayscale_16bit.jpg", 426, 640, 1, np.uint16, 16),
                        ("jpeg2k/cat-111793_640.jp2", 426, 640, 3, np.uint8, 8),
                        ("tiff/multichannel/cat-1245673_640_multichannel.tif", 423, 640, 6, np.uint8, 8),
                        ("tiff/with_alpha_16bit/4ch16bpp.tiff", 497, 497, 4, np.uint16, 16),
                        ("png/with_alpha_16bit/4ch16bpp.png", 497, 497, 4, np.uint16, 16),
                        ("png/with_alpha/cat-111793_640-alpha.png", 426, 640, 4, np.uint8, 8),
                        ("jpeg2k/with_alpha/cat-111793_640-alpha.jp2", 426, 640, 4, np.uint8, 8),
                        ("jpeg2k/with_alpha/cat-111793_640-alpha.jp2", 426, 640, 4, np.uint8, 8),
                        ("jpeg2k/with_alpha_16bit/4ch16bpp.jp2", 497, 497, 4, np.uint16, 16),
                        ("jpeg2k/with_alpha_16bit/4ch16bpp.jp2", 497, 497, 4, np.uint16, 16),
                        ("jpeg2k/tiled-cat-1046544_640_gray.jp2", 475, 640, 1, np.uint8, 8),
                        ("jpeg2k/tiled-cat-1046544_640_gray.jp2", 475, 640, 1, np.uint8, 8),
                        ("bmp/cat-111793_640.bmp", 426, 640, 3, np.uint8, 8),
                        ("pnm/cat-1245673_640.pgm", 423, 640, 1, np.uint8, 8),
                        ("webp/lossy/cat-3113513_640.webp", 299, 640, 3, np.uint8, 8),
                    ]
                    )
def test_code_stream(filename, height, width, channels, dtype, precision):
    fpath = os.path.join(img_dir_path, filename)
    with open(fpath, 'rb') as in_file:
        data = in_file.read()
    nparr = np.fromfile(fpath, dtype=np.uint8)
    for stream in [nvimgcodec.CodeStream(fpath), nvimgcodec.CodeStream(data), nvimgcodec.CodeStream(nparr)]:
        assert stream.height == height
        assert stream.width == width
        assert stream.channels == channels
        assert stream.dtype == dtype
        assert stream.precision == precision

def test_code_stream_constructor_sets_all_properties_correctly():
    fpath = os.path.join(img_dir_path, "jpeg2k/tiled-cat-111793_640.jp2")
    with open(fpath, 'rb') as in_file:
        data = in_file.read()
    nparr = np.fromfile(fpath, dtype=np.uint8)
    for stream in [nvimgcodec.CodeStream(fpath), nvimgcodec.CodeStream(data), nvimgcodec.CodeStream(nparr)]:
        assert stream.height == 426
        assert stream.width == 640
        assert stream.channels == 3
        assert stream.dtype == np.uint8
        assert stream.precision == 8
        assert stream.codec_name == 'jpeg2k'
        assert stream.num_tiles_y == 5
        assert stream.num_tiles_x == 7
        assert stream.tile_height == 100
        assert stream.tile_width == 100

        print(stream)

    fpath = os.path.join(img_dir_path, "jpeg2k/cat-111793_640.jp2")
    with open(fpath, 'rb') as in_file:
        data = in_file.read()
    nparr = np.fromfile(fpath, dtype=np.uint8)
    for stream in [nvimgcodec.CodeStream(fpath), nvimgcodec.CodeStream(data), nvimgcodec.CodeStream(nparr)]:
        assert stream.height == 426
        assert stream.width == 640
        assert stream.channels == 3
        assert stream.dtype == np.uint8
        assert stream.precision == 8
        assert stream.codec_name == 'jpeg2k'
        assert stream.num_tiles_y == 1
        assert stream.num_tiles_x == 1
        assert stream.tile_height == 426
        assert stream.tile_width == 640

        print(stream)

    fpath = os.path.join(img_dir_path, "tiff/cat-300572_640.tiff")
    with open(fpath, 'rb') as in_file:
        data = in_file.read()
    nparr = np.fromfile(fpath, dtype=np.uint8)
    for stream in [nvimgcodec.CodeStream(fpath), nvimgcodec.CodeStream(data), nvimgcodec.CodeStream(nparr)]:
        assert stream.height == 536
        assert stream.width == 640
        assert stream.channels == 3
        assert stream.dtype == np.uint8
        assert stream.precision == 8
        assert stream.codec_name == 'tiff'
        assert stream.num_tiles_y == 1
        assert stream.num_tiles_x == 1
        assert stream.tile_height == 536
        assert stream.tile_width == 640

        print(stream)

    fpath = os.path.join(img_dir_path, "bmp/cat-111793_640.bmp")
    with open(fpath, 'rb') as in_file:
        data = in_file.read()
    nparr = np.fromfile(fpath, dtype=np.uint8)
    for stream in [nvimgcodec.CodeStream(fpath), nvimgcodec.CodeStream(data), nvimgcodec.CodeStream(nparr)]:
        assert stream.height == 426
        assert stream.width == 640
        assert stream.channels == 3
        assert stream.dtype == np.uint8
        assert stream.precision == 8
        assert stream.codec_name == 'bmp'
        assert stream.num_tiles_y == None
        assert stream.num_tiles_x == None
        assert stream.tile_height == None
        assert stream.tile_width == None

        print(stream)

@t.mark.parametrize("fname", ["jpeg/padlock-406986_640_440.jpg"])
def test_code_stream_get_sub_code_stream_with_nested_roi_throws_exception(fname):
    fpath = os.path.join(img_dir_path, fname)
    code_stream = nvimgcodec.CodeStream(fpath)
    roi1 = nvimgcodec.Region(10, 20, code_stream.height - 10, code_stream.width - 20)
    roi1_height = roi1.end[0] - roi1.start[0]
    roi1_width = roi1.end[1] - roi1.start[1]
    roi2 = nvimgcodec.Region(5, 5, roi1_height - 5, roi1_width - 5)
    sub_cs = code_stream.getSubCodeStream(region=roi1)
    with t.raises(Exception) as excinfo:
        nested_cs = sub_cs.getSubCodeStream(region=roi2)
    assert (isinstance(excinfo.value, RuntimeError) )
    assert (str(excinfo.value) == "Cannot create a sub code stream with nested regions. This is not supported.")
