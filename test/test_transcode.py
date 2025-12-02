# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# nvjpeg from cuda 13 produces different outputs
cuda13_checksum_map = {
    "b8400d754f8e7f06da0e67db19fb4760" : "79746801b0cacae6e7593c7a9a52293a",
    "1689eaa9b5c613fed8b940dd8e3be5bd" : "10d93af5b72736552a1dc0fe635dc929",
    "117703b2bc7d29a833bd486d4ed042eb" : "23a6e8fac6df4a3320cb3f77b5e93559",
    "468034fa400b5c55a5ca4b1749f8628f" : "7a2f4140c1bf56462fa406b6263765ac",
    "5c9c72dbbde0083da5d24dc27075f692" : "3e79a8ac196df602db3cc4e4a77d3dfc",
    "d3aa73b2650847bc1e5bdada616e4ac7" : "cd569e69d34899d26d504865acf96d7d",
    "64d413f583388a9dd46fc4ecc87f292c" : "fabddc43594d95232ef072e6a3d79dca",
    "86e8a4fd2ec34a3318202704d0052d4b" : "4194dabe31d13c6cc2774690c969abc1",
    "6632ad75a777c6542b6fa2cc32759182" : "3e106b06cb8719971f69fc2e668dbd00",
    "e6f228079c60bfb3aea756c340bb4402" : "7ae4bc147f75f968184bb1d190fbddba",
    "f79e12bb078b3923767070631526a27a" : "038986edc3c1615551c815a1d90fdf68",
    "bee24d5011a4b554ec80f124541ac15a" : "3b2382cc97e4c3ca98ad380a91f0867c",
    "1fc2f66a4865c5377161d77d78d1851e" : "fa2fa954e4c3b9618a6df519dd3eea0b",
    "9667004a27b47f4964e7e02771b98f84" : "fc19af28b011bd4ae01288375af4d6c6",
    "665828af06d865114efbefc14f457738" : "3f25db4d7cc800d70c054dfcf197765d",
    "3ed087e532ae5405eb864070cd8d7f26" : "023678c471c1d65b0da226138f3dd53f",
    "9ecc2908eb9a76042f9bd1176b9025b6" : "389a9f1cbc5c17fb91877b4799160fbc",
    "8a8e2d821f0ef66a02c56021d1d0021e" : "0d5562f3402291ae373f1df785412a37",
    "e221e880b967c5483b5f55794d302c3c" : "87be8a49ce71e270c3877ed4cbaceca1",
    "068042e9b2c752746304847e585e8832" : "dd6042560a2750c5383c4e28db9e1c6c",
    "488dbc9bbf262cc094dbda47c652ce18" : "a0de2ac6b88a0f4903a818cf5aa3a286",
    "211d6fd3c62539cecfe0a93548fe7e64" : "cba6c23ed00d92c234c72a36d625715b",
    "cf45f2de52006fe895491d9f728412eb" : "e65899811b6cfdd64f197c81c2ba652b",
    "400467d818acb8666b85ce05f84d2c09" : "508041a1f870b9c05c8a7e806547aaaf",
    "46b68c074cd8989b3d5ac027ce0b75bd" : "991f5239af68d65369169db6455a7716"
}

def imtrans_test(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    if int(os.getenv(("CUDA_VERSION_MAJOR"), 12)) == 13:
        # if check_sum is not in the map (meaning it is the same for cuda 12 and 13) just keep the value unchanged
        check_sum = cuda13_checksum_map.get(check_sum, check_sum)

    os.chdir(exec_dir_path)
    input_img_path = os.path.join(img_dir_path, input_img_file)
    output_img_path = os.path.join(tmp_path, output_img_file)
    cmd = ".{}{} -i {} -c {} {} -o {} --skip_hw_gpu_backend true".format(os.sep, transcode_exec,
                                              str(input_img_path), codec, params, str(output_img_path))
    print('Running command: '+cmd)
    subprocess.run(cmd, shell=True, check=True)
    assert check_sum == file_md5(output_img_path)


@t.mark.parametrize(
    "input_img_file,codec,output_img_file,params,check_sum",
    [
    ("bmp/cat-111793_640.bmp", "bmp", "cat-111793_640-bmp.bmp", "", "f16bcbf2c29d7e861ebea368ec455786"),
    ("bmp/cat-111793_640.bmp", "jpeg2k", "cat-111793_640-bmp.jp2", "--enc_color_trans true", "69da14c76b829a0911fac3df745f95e6"),
    
    ("jpeg2k/cat-1046544_640.jp2", "bmp", "cat-1046544_640-jp2.bmp","", "4db891cabea6873df5aedd73799befb4"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg2k", "cat-1046544_640-jp2.jp2", "--enc_color_trans true", "d518da5b21585f347673f7657b42a00c"),
    
    ("jpeg/padlock-406986_640_444.jpg", "bmp", "padlock-406986_640_444-jpg.bmp","", "b8400d754f8e7f06da0e67db19fb4760"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg2k", "padlock-406986_640_444-jpg.jp2", "--enc_color_trans true", "1689eaa9b5c613fed8b940dd8e3be5bd"),

    
    #jpeg various input chroma subsampling
    ("jpeg/padlock-406986_640_410.jpg", "bmp",    "padlock-406986_640_410-jpg.bmp", "", "117703b2bc7d29a833bd486d4ed042eb"),
    ("jpeg/padlock-406986_640_410.jpg", "jpeg2k", "padlock-406986_640_410-jpg.jp2", "--enc_color_trans true", "468034fa400b5c55a5ca4b1749f8628f"),
    
    ("jpeg/padlock-406986_640_411.jpg", "bmp",    "padlock-406986_640_411-jpg.bmp", "", "5c9c72dbbde0083da5d24dc27075f692"),
    ("jpeg/padlock-406986_640_411.jpg", "jpeg2k", "padlock-406986_640_411-jpg.jp2", "--enc_color_trans true", "d3aa73b2650847bc1e5bdada616e4ac7"),
    
    ("jpeg/padlock-406986_640_420.jpg", "bmp",    "padlock-406986_640_420-jpg.bmp", "", "9667004a27b47f4964e7e02771b98f84"),
    ("jpeg/padlock-406986_640_420.jpg", "jpeg2k", "padlock-406986_640_420-jpg.jp2", "--enc_color_trans true", "64d413f583388a9dd46fc4ecc87f292c"),

    ("jpeg/padlock-406986_640_422.jpg", "bmp",    "padlock-406986_640_422-jpg.bmp", "", "86e8a4fd2ec34a3318202704d0052d4b"),
    ("jpeg/padlock-406986_640_422.jpg", "jpeg2k", "padlock-406986_640_422-jpg.jp2", "--enc_color_trans true", "6632ad75a777c6542b6fa2cc32759182"),

    ("jpeg/padlock-406986_640_440.jpg", "bmp",    "padlock-406986_640_440-jpg.bmp", "", "e6f228079c60bfb3aea756c340bb4402"),
    ("jpeg/padlock-406986_640_440.jpg", "jpeg2k", "padlock-406986_640_440-jpg.jp2", "--enc_color_trans true", "f79e12bb078b3923767070631526a27a"),

    ("jpeg/padlock-406986_640_444.jpg", "bmp",    "padlock-406986_640_444-jpg.bmp", "", "b8400d754f8e7f06da0e67db19fb4760"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg2k", "padlock-406986_640_444-jpg.jp2", "--enc_color_trans true", "1689eaa9b5c613fed8b940dd8e3be5bd"),

    #test pnm
    ("bmp/cat-111793_640.bmp", "pnm", "cat-111793_640-bmp.ppm","", "50592b9f875a5468f1f7585a4eefadf1"),
    ("jpeg/padlock-406986_640_444.jpg", "pnm", "padlock-406986_640_444-jpg.ppm","", "bee24d5011a4b554ec80f124541ac15a"),
    ("jpeg2k/cat-1046544_640.jp2", "pnm", "cat-1046544_640-jp2.ppm", "", "dbffde83c8d7dbbdc517c05280a74468"),
    
    #test orientation
    ("jpeg/exif/padlock-406986_640_rotate_270.jpg", "bmp", "padlock-406986_640_rotate_270-exif_orientation_enabled.bmp", "", "1fc2f66a4865c5377161d77d78d1851e"),
    ("jpeg/exif/padlock-406986_640_rotate_270.jpg", "bmp", "padlock-406986_640_rotate_270-exif_orientation_disabled.bmp", "--ignore_orientation true", "9667004a27b47f4964e7e02771b98f84"),

    # png to pnm/webp/png
    ("png/cat-1245673_640.png", "pnm", "cat-1245673_640.ppm", "", "f2b8d1255372cc02272cc3f788b509aa"),
    ("png/cat-1245673_640.png", "webp", "cat-1245673_640.webp", "", "c8907431e6a6b5850d7c0aabbacfbba8"),
    ("png/cat-1245673_640.png", "png", "cat-1245673_640.png", "", "e22894e5874b0c3c4dd6c263c06252a7"),

    # webp to pnm/webp/png
    ("webp/lossless/cat-3113513_640.webp", "pnm", "cat-3113513_640.ppm", "", "c8f1e7f5d0b199d3f8f7c34e3b0ee21c"),
    ("webp/lossless/cat-3113513_640.webp", "webp", "cat-3113513_640.webp", "", "fccf0dac6439c71dd6e31568195d673e"),
    ("webp/lossless/cat-3113513_640.webp", "png", "cat-3113513_640.png", "", "91cfc923c3bcb8a7038a3b74f56963a5"),

    # pnm to pnm/webp/png
    ("pnm/cat-111793_640.ppm", "pnm", "cat-111793_640.ppm", "", "50592b9f875a5468f1f7585a4eefadf1"),
    ("pnm/cat-111793_640.ppm", "webp", "cat-111793_640.webp", "", "d341a89d5862e12187b67bcf2a46a6c5"),
    ("pnm/cat-111793_640.ppm", "png", "cat-111793_640.png", "", "21a45892b0487939b452e9f81ddbb01a"),
    ("pnm/cat-1245673_640.pgm", "pnm", "cat-1245673_640.ppm", "", "28e305ddfea67cc8aa2fd6b7f6512cb6"),
    ("pnm/cat-1245673_640.pgm", "webp", "cat-1245673_640.webp", "", "7368fcbcbfd2ed98596d594807a46179"),
    ("pnm/cat-1245673_640.pgm", "png", "cat-1245673_640.png", "", "5a95f58c19b64d9f96da4cabb29a9f36"),
    ("pnm/cat-2184682_640.pbm", "pnm", "cat-2184682_640.ppm", "", "1e17cda0d8627203ccf9dbe3ad8cbd9d"),
    ("pnm/cat-2184682_640.pbm", "webp", "cat-2184682_640.webp", "", "585a5897d16d7ef6f0e699b375f678fd"),
    ("pnm/cat-2184682_640.pbm", "png", "cat-2184682_640.png", "", "a96bd079bf5ceaf1371450842cc0c1a7"),
    ]
)
def test_imtrans_common_lossless(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    imtrans_test(tmp_path, input_img_file, codec, output_img_file, params + " --quality_type LOSSLESS", check_sum)

# Encoding with nvjpeg images with unaligned height did not correctly before CTK 12 
@t.mark.skipif(int(os.getenv(("CUDA_VERSION_MAJOR"), 12)) < 12,  reason="requires CUDA >= 12")
@t.mark.parametrize(
    "input_img_file,codec,output_img_file,params,check_sum",
    [
    ("bmp/cat-111793_640.bmp", "jpeg",  "cat-111793_640-bmp.jpg","", "665828af06d865114efbefc14f457738"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg", "cat-1046544_640-jp2.jpg","", "3ed087e532ae5405eb864070cd8d7f26"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg", "padlock-406986_640_444-jpg.jpg","", "9ecc2908eb9a76042f9bd1176b9025b6"),

    #encoding with chroma subsampling
    ("bmp/cat-111793_640.bmp", "jpeg",  "cat-111793_640-bmp-420.jpg","--chroma_subsampling 420", "8a8e2d821f0ef66a02c56021d1d0021e"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg", "cat-1046544_640-jp2-420.jpg","--chroma_subsampling 420", "e221e880b967c5483b5f55794d302c3c"),
    ("jpeg2k/cat-1046544_640.jp2", "jpeg", "cat-1046544_640-jp2-422.jpg","--chroma_subsampling 422", "068042e9b2c752746304847e585e8832"),
    
    #jpeg various input chroma subsampling
    ("jpeg/padlock-406986_640_410.jpg", "jpeg",   "padlock-406986_640_410-jpg.jpg", "", "488dbc9bbf262cc094dbda47c652ce18"),
    ("jpeg/padlock-406986_640_411.jpg", "jpeg",   "padlock-406986_640_411-jpg.jpg", "", "211d6fd3c62539cecfe0a93548fe7e64"),
    ("jpeg/padlock-406986_640_420.jpg", "jpeg",   "padlock-406986_640_420-jpg.jpg", "", "cf45f2de52006fe895491d9f728412eb"),
    ("jpeg/padlock-406986_640_422.jpg", "jpeg",   "padlock-406986_640_422-jpg.jpg", "", "400467d818acb8666b85ce05f84d2c09"),
    ("jpeg/padlock-406986_640_440.jpg", "jpeg",   "padlock-406986_640_440-jpg.jpg", "", "46b68c074cd8989b3d5ac027ce0b75bd"),
    ("jpeg/padlock-406986_640_444.jpg", "jpeg",   "padlock-406986_640_444-jpg.jpg", "", "9ecc2908eb9a76042f9bd1176b9025b6"),
    ]
)
def test_imtrans_encoding_to_jpeg_cuda12_and_above(tmp_path, input_img_file, codec, output_img_file, params, check_sum):
    imtrans_test(tmp_path, input_img_file, codec, output_img_file, params, check_sum)
    