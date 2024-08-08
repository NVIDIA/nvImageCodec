# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import ctypes as c
from nvidia import nvimgcodec
import os
import numpy as np
import cupy as cp
import cv2

def get_nvjpeg_ver():
    nvjpeg_ver_major, nvjpeg_ver_minor, nvjpeg_ver_patch = (c.c_int(), c.c_int(), c.c_int())
    try:
        nvjpeg_libname = f'libnvjpeg.so'
        nvjpeg_lib = c.CDLL(nvjpeg_libname)
        nvjpeg_lib.nvjpegGetProperty(0, c.byref(nvjpeg_ver_major))
        nvjpeg_lib.nvjpegGetProperty(1, c.byref(nvjpeg_ver_minor))
        nvjpeg_lib.nvjpegGetProperty(2, c.byref(nvjpeg_ver_patch))
    except:
        for file in os.listdir("/usr/local/cuda/lib64/"):
            try:
                if file.startswith("libnvjpeg.so"):
                    nvjpeg_lib = c.CDLL(file)
                    nvjpeg_lib.nvjpegGetProperty(0, c.byref(nvjpeg_ver_major))
                    nvjpeg_lib.nvjpegGetProperty(1, c.byref(nvjpeg_ver_minor))
                    nvjpeg_lib.nvjpegGetProperty(2, c.byref(nvjpeg_ver_patch))
                    break
            except:
                continue
    return nvjpeg_ver_major.value, nvjpeg_ver_minor.value, nvjpeg_ver_patch.value

def get_cuda_compute_capability(device_id=0):
    compute_cap = 0
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(device_id)
        compute_cap = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
        compute_cap = compute_cap[0] + compute_cap[1] / 10.
    except Exception as e:
        print(f"Error: {e}")
    return compute_cap

img_dir_path = os.path.abspath(os.path.join(
    os.path.dirname(__file__), "../../resources"))

def is_fancy_upsampling_available():
    return nvimgcodec.__cuda_version__ >= 12010

def get_default_decoder_options():
    return ":fancy_upsampling=1" if is_fancy_upsampling_available() else ":fancy_upsampling=0"

def get_max_diff_threshold():
    return 4 if is_fancy_upsampling_available() else 44

def compare_image(test_img, ref_img):
    diff = ref_img.astype(np.int32) - test_img.astype(np.int32)
    diff = np.absolute(diff)

    assert test_img.shape == ref_img.shape
    assert diff.max() <= get_max_diff_threshold()

def compare_device_with_host_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        cp_test_img = cp.asarray(test_images[i])
        np_test_img = np.asarray(cp.asnumpy(cp_test_img))
        ref_img = cv2.cvtColor(ref_images[i], cv2.COLOR_BGR2RGB)
        ref_img = np.asarray(ref_img)
        compare_image(np_test_img, ref_img)


def compare_host_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        test_img = np.asarray(test_images[i])
        ref_img = np.asarray(ref_images[i])
        compare_image(test_img, ref_img)

def is_nvjpeg2k_supported():
    return True