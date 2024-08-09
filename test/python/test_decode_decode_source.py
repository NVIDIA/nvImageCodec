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

img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../resources"))

filenames = [
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_440.jpg",
    "jpeg2k/cat-111793_640.jp2",
]

def test_decode_single_file():
    decoder = nvimgcodec.Decoder()
    fpath = os.path.join(img_dir_path, filenames[0])
    raw_bytes = open(fpath, 'rb').read()
    np_arr = np.fromfile(fpath, dtype=np.uint8)

    dec_srcs = [
        fpath,
        raw_bytes,
        np_arr,
        nvimgcodec.CodeStream(raw_bytes),
        nvimgcodec.CodeStream(fpath),
        nvimgcodec.CodeStream(np_arr),
        nvimgcodec.DecodeSource(raw_bytes),
        nvimgcodec.DecodeSource(fpath),
        nvimgcodec.DecodeSource(np_arr),
        nvimgcodec.DecodeSource(nvimgcodec.CodeStream(fpath)),
        nvimgcodec.DecodeSource(nvimgcodec.CodeStream(raw_bytes)),
        nvimgcodec.DecodeSource(nvimgcodec.CodeStream(np_arr)),
    ]

    imgs = [decoder.decode(src) for src in dec_srcs]
    for img in imgs[1:]:
        np.testing.assert_allclose(imgs[0].cpu(), img.cpu())

def test_decode_roi():
    decoder = nvimgcodec.Decoder()
    fpath = os.path.join(img_dir_path, filenames[0])
    raw_bytes = open(fpath, 'rb').read()
    np_arr = np.fromfile(fpath, dtype=np.uint8)

    code_stream0 = nvimgcodec.CodeStream(fpath)
    code_stream1 = nvimgcodec.CodeStream(raw_bytes)
    code_stream2 = nvimgcodec.CodeStream(np_arr)

    roi = nvimgcodec.Region(10, 20, code_stream0.height-10, code_stream0.width-20)
    imgref = np.array(decoder.decode(code_stream0).cpu())

    dec_srcs = [
        nvimgcodec.DecodeSource(raw_bytes, roi),
        nvimgcodec.DecodeSource(fpath, roi),
        nvimgcodec.DecodeSource(np_arr, roi),
        nvimgcodec.DecodeSource(code_stream0, roi),
        nvimgcodec.DecodeSource(code_stream1, roi),
        nvimgcodec.DecodeSource(code_stream2, roi),
    ]
    imgs_roi = [decoder.decode(src).cpu() for src in dec_srcs]
    for img in imgs_roi:
        np.testing.assert_allclose(img, imgref[roi.start[0]:roi.end[0], roi.start[1]:roi.end[1]])

def test_decode_batch():
    decoder = nvimgcodec.Decoder()
    fpaths = [os.path.join(img_dir_path, f) for f in filenames]
    np_arrays = [np.fromfile(fpath, dtype=np.uint8) for fpath in fpaths]
    raw_bytes = [open(fpath, 'rb').read() for fpath in fpaths]

    batch0_srcs = [
        nvimgcodec.CodeStream(fpaths[0]),
        nvimgcodec.DecodeSource(raw_bytes[1]),
        np_arrays[2],
    ]

    batch1_srcs = [
        nvimgcodec.DecodeSource(raw_bytes[0]),
        fpaths[1],
        nvimgcodec.CodeStream(np_arrays[2]),
    ]

    imgs0 = [img.cpu() for img in decoder.decode(batch0_srcs)]
    imgs1 = [img.cpu() for img in decoder.decode(batch1_srcs)]
    for img0, img1 in zip(imgs0, imgs1):
        np.testing.assert_allclose(img0, img1)

def test_decode_batch_roi():
    decoder = nvimgcodec.Decoder()
    fpaths = [os.path.join(img_dir_path, f) for f in filenames]
    code_streams = [nvimgcodec.CodeStream(fpath) for fpath in fpaths]
    rois = [
        nvimgcodec.Region(10, 20, cs.height-10, cs.width-20)
        for cs in code_streams
    ]
    dec_srcs = [
        nvimgcodec.DecodeSource(cs, roi)
        for cs, roi in zip(code_streams, rois)
    ]
    
    imgs = [np.array(img.cpu()) for img in decoder.decode(code_streams)]
    imgs_roi = [np.array(img.cpu()) for img in decoder.decode(dec_srcs)]

    for img, img_roi, roi in zip(imgs, imgs_roi, rois):
        np.testing.assert_allclose(img_roi, img[roi.start[0]:roi.end[0], roi.start[1]:roi.end[1]])

@t.mark.parametrize("fname", ["jpeg/padlock-406986_640_440.jpg",
                              "jpeg2k/cat-111793_640.jp2"])
def test_decode_repeat_code_stream(fname):
    decoder = nvimgcodec.Decoder(max_num_cpu_threads=1)  # One thread to force a single instance of parsed stream
    fpaths = [os.path.join(img_dir_path, f) for f in filenames]
    code_stream = nvimgcodec.CodeStream(fpaths[0])

    img0 = decoder.decode(code_stream)
    imgs1 = decoder.decode([code_stream, code_stream])

    for img1 in imgs1:
        np.testing.assert_allclose(img0.cpu(), img1.cpu())
