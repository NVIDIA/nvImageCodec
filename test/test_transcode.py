# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import pytest as t
import hashlib
import subprocess


img_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../resources"))
exec_dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../bin"))
transcode_exec="nvimtrans"

def file_md5(file_name):
    hash_md5 = hashlib.md5()
    with open(file_name, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def imtrans_test(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    os.chdir(exec_dir_path)
    input_img_path = os.path.join(img_dir_path, input_img_file)
    output_img_path = os.path.join(tmp_path, output_img_file)
    cmd = ".{}{} -i {} -c {} {} -o {} --skip_hw_gpu_backend true".format(os.sep, transcode_exec,
                                              str(input_img_path), codec, params, str(output_img_path))
    print('Running command: '+cmd)
    subprocess.run(cmd, shell=True)
    assert check_sum == file_md5(output_img_path)


@t.mark.parametrize(
    "input_img_file,codec,output_img_file,params,check_sum",
    [
    ("bmp/cat-111793_640.bmp", "bmp", "cat-111793_640-bmp.bmp", "", "f16bcbf2c29d7e861ebea368ec455786"),
    ("bmp/cat-111793_640.bmp", "jpeg2k", "cat-111793_640-bmp.jp2", "--enc_color_trans true", "c5c1d9400c095ae5b56284c777063e4f"),
    
    ("jpeg2k/cat-1046544_640.jp2", "bmp", "cat-1046544_640-jp2.bmp","", "4db891cabea6873df5aedd73799befb4"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg2k", "cat-1046544_640-jp2.jp2", "--enc_color_trans true", "150e1eb7929680a3d0c2e1647e51e525"),
    
    ("jpeg/padlock-406986_640_444.jpg", "bmp", "padlock-406986_640_444-jpg.bmp","", "b8400d754f8e7f06da0e67db19fb4760"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg2k", "padlock-406986_640_444-jpg.jp2", "--enc_color_trans true", "7c6a2375a8b1387036bd7a02c19e8f8e"),

    
    #jpeg various input chroma subsampling
    ("jpeg/padlock-406986_640_410.jpg", "bmp",    "padlock-406986_640_410-jpg.bmp", "", "117703b2bc7d29a833bd486d4ed042eb"),
    ("jpeg/padlock-406986_640_410.jpg", "jpeg2k", "padlock-406986_640_410-jpg.jp2", "--enc_color_trans true", "06d36693b45d45214ea672a0461b9540"),
    
    ("jpeg/padlock-406986_640_411.jpg", "bmp",    "padlock-406986_640_411-jpg.bmp", "", "5c9c72dbbde0083da5d24dc27075f692"),
    ("jpeg/padlock-406986_640_411.jpg", "jpeg2k", "padlock-406986_640_411-jpg.jp2", "--enc_color_trans true", "08c7921bf34adb4ea180fa1359fe4cca"),
    
    ("jpeg/padlock-406986_640_420.jpg", "bmp",    "padlock-406986_640_420-jpg.bmp", "", "9667004a27b47f4964e7e02771b98f84"),
    ("jpeg/padlock-406986_640_420.jpg", "jpeg2k", "padlock-406986_640_420-jpg.jp2", "--enc_color_trans true", "10d7bf042a3fc42c90c762eebce16f09"),

    ("jpeg/padlock-406986_640_422.jpg", "bmp",    "padlock-406986_640_422-jpg.bmp", "", "86e8a4fd2ec34a3318202704d0052d4b"),
    ("jpeg/padlock-406986_640_422.jpg", "jpeg2k", "padlock-406986_640_422-jpg.jp2", "--enc_color_trans true", "a9e445c5df6ef56f1f7f695bae42b807"),

    ("jpeg/padlock-406986_640_440.jpg", "bmp",    "padlock-406986_640_440-jpg.bmp", "", "e6f228079c60bfb3aea756c340bb4402"),
    ("jpeg/padlock-406986_640_440.jpg", "jpeg2k", "padlock-406986_640_440-jpg.jp2", "--enc_color_trans true", "a90b221752545b5f4640494e9d67ff6c"),

    ("jpeg/padlock-406986_640_444.jpg", "bmp",    "padlock-406986_640_444-jpg.bmp", "", "b8400d754f8e7f06da0e67db19fb4760"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg2k", "padlock-406986_640_444-jpg.jp2", "--enc_color_trans true", "7c6a2375a8b1387036bd7a02c19e8f8e"),

    #test pnm
    ("bmp/cat-111793_640.bmp", "pnm", "cat-111793_640-bmp.ppm","", "50592b9f875a5468f1f7585a4eefadf1"),
    ("jpeg/padlock-406986_640_444.jpg", "pnm", "padlock-406986_640_444-jpg.ppm","", "bee24d5011a4b554ec80f124541ac15a"),
    ("jpeg2k/cat-1046544_640.jp2", "pnm", "cat-1046544_640-jp2.ppm", "", "dbffde83c8d7dbbdc517c05280a74468"),
    
    #test orientation
    ("jpeg/exif/padlock-406986_640_rotate_270.jpg", "bmp", "padlock-406986_640_rotate_270-exif_orientation_enabled.bmp", "", "1fc2f66a4865c5377161d77d78d1851e"),
    ("jpeg/exif/padlock-406986_640_rotate_270.jpg", "bmp", "padlock-406986_640_rotate_270-exif_orientation_disabled.bmp", "--ignore_orientation true", "9667004a27b47f4964e7e02771b98f84"),

    # png to pnm/webp/png
    ("png/cat-1245673_640.png", "pnm", "cat-1245673_640.ppm", "", "f2b8d1255372cc02272cc3f788b509aa"),
    ("png/cat-1245673_640.png", "webp", "cat-1245673_640.webp", "", "742bb7965d34e076cfe13e6fff51eddd"),
    ("png/cat-1245673_640.png", "png", "cat-1245673_640.png", "", "e22894e5874b0c3c4dd6c263c06252a7"),

    # webp to pnm/webp/png
    ("webp/lossless/cat-3113513_640.webp", "pnm", "cat-3113513_640.ppm", "", "c8f1e7f5d0b199d3f8f7c34e3b0ee21c"),
    ("webp/lossless/cat-3113513_640.webp", "webp", "cat-3113513_640.webp", "", "60ecfd733acb88c68a9600103a95e065"),
    ("webp/lossless/cat-3113513_640.webp", "png", "cat-3113513_640.png", "", "91cfc923c3bcb8a7038a3b74f56963a5"),

    # pnm to pnm/webp/png
    ("pnm/cat-111793_640.ppm", "pnm", "cat-111793_640.ppm", "", "50592b9f875a5468f1f7585a4eefadf1"),
    ("pnm/cat-111793_640.ppm", "webp", "cat-111793_640.webp", "", "a75a23975e5fe274f1454157392092e5"),
    ("pnm/cat-111793_640.ppm", "png", "cat-111793_640.png", "", "21a45892b0487939b452e9f81ddbb01a"),
    ("pnm/cat-1245673_640.pgm", "pnm", "cat-1245673_640.ppm", "", "28e305ddfea67cc8aa2fd6b7f6512cb6"),
    ("pnm/cat-1245673_640.pgm", "webp", "cat-1245673_640.webp", "", "ecfde58b493e963256a7e0204350941e"),
    ("pnm/cat-1245673_640.pgm", "png", "cat-1245673_640.png", "", "5a95f58c19b64d9f96da4cabb29a9f36"),
    ("pnm/cat-2184682_640.pbm", "pnm", "cat-2184682_640.ppm", "", "1e17cda0d8627203ccf9dbe3ad8cbd9d"),
    ("pnm/cat-2184682_640.pbm", "webp", "cat-2184682_640.webp", "", "4a88d94d74b00c41d8eb24b09c0c42f6"),
    ("pnm/cat-2184682_640.pbm", "png", "cat-2184682_640.png", "", "a96bd079bf5ceaf1371450842cc0c1a7"),
    ]
)
def test_imtrans_common(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    imtrans_test(tmp_path, input_img_file, codec, output_img_file, params, check_sum)

# Encoding with nvjpeg images with unaligned height did not correctly before CTK 12 
@t.mark.skipif(int(os.getenv(("CUDA_VERSION_MAJOR"), 12)) < 12,  reason="requires CUDA >= 12")
@t.mark.parametrize(
    "input_img_file,codec,output_img_file,params,check_sum",
    [
    ("bmp/cat-111793_640.bmp", "jpeg",  "cat-111793_640-bmp.jpg","", "2b9b14c674aa3a5d9384a5dffcddd1e5"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg", "cat-1046544_640-jp2.jpg","", "4d283fb05a2edf443fdf8fa6481da8a8"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg", "padlock-406986_640_444-jpg.jpg","", "8d7e0c2b8a458f6ec4b6af03cc1c9f16"),

    #encoding with chroma subsampling
    ("bmp/cat-111793_640.bmp", "jpeg",  "cat-111793_640-bmp-420.jpg","--chroma_subsampling 420", "68dad28a9c8dcd3b28f426441ab229ab"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg", "cat-1046544_640-jp2-420.jpg","--chroma_subsampling 420", "a682802dabbeebffbbed9e74f9756ec2"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg", "cat-1046544_640-jp2-422.jpg","--chroma_subsampling 422", "1d587d174b5ce5640c74bd549e3a7145"),
    
    #jpeg various input chroma subsampling
    ("jpeg/padlock-406986_640_410.jpg", "jpeg",   "padlock-406986_640_410-jpg.jpg", "", "31f72adf6f8553118927df00a1a8adc3"),
    ("jpeg/padlock-406986_640_411.jpg", "jpeg",   "padlock-406986_640_411-jpg.jpg", "", "31fcb68b4641c6db3e0464d06dffbbda"),
    ("jpeg/padlock-406986_640_420.jpg", "jpeg",   "padlock-406986_640_420-jpg.jpg", "", "740d7a4ea7237793033a1de88f77e610"),
    ("jpeg/padlock-406986_640_422.jpg", "jpeg",   "padlock-406986_640_422-jpg.jpg", "", "b225ca4dcc19412d9c18caf524f9080f"),
    ("jpeg/padlock-406986_640_440.jpg", "jpeg",   "padlock-406986_640_440-jpg.jpg", "", "bd29ce0b525a33e8919362c96aee7aa6"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg",   "padlock-406986_640_444-jpg.jpg", "", "8d7e0c2b8a458f6ec4b6af03cc1c9f16"),
    ]
)
def test_imtrans_encoding_to_jpeg_cuda12_and_above(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    imtrans_test(tmp_path, input_img_file, codec, output_img_file, params, check_sum)
    