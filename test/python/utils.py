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
import platform
import ctypes


def get_nvjpeg_ver():
    nvjpeg_ver_major, nvjpeg_ver_minor, nvjpeg_ver_patch = (c.c_int(), c.c_int(), c.c_int())
    nvjpeg_found = False
    cuda_major_version = str(nvimgcodec.__cuda_version__ // 1000)

    # Try standard paths by CDLL first (this should include PATH, LD_LIBRARY_PATH, etc)
    libnames = [
        'libnvjpeg.so',
        f'libnvjpeg.so.{cuda_major_version}',
        f'nvjpeg64_{cuda_major_version}.dll'
    ]
    for libname in libnames:
        try:
            nvjpeg_lib = c.CDLL(libname)
            nvjpeg_lib.nvjpegGetProperty(0, c.byref(nvjpeg_ver_major))
            nvjpeg_lib.nvjpegGetProperty(1, c.byref(nvjpeg_ver_minor))
            nvjpeg_lib.nvjpegGetProperty(2, c.byref(nvjpeg_ver_patch))
            nvjpeg_found = True
            break
        except:
            continue
    
    if not nvjpeg_found:
        # If standard approach fails, search in python site-packages then CTK default path
        nvimgcodec_dir = os.path.dirname(nvimgcodec.__file__)
        search_paths = []
        if platform.system() == "Windows":
            search_paths.append(os.path.join(os.path.dirname(nvimgcodec_dir), "nvjpeg/bin"))
            for minor_version in range(9, -1, -1): # try newer versions first
                search_paths.append(f"C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v{cuda_major_version}.{minor_version}/bin")
        else: # Linux
            search_paths.append(os.path.join(os.path.dirname(nvimgcodec_dir), "nvjpeg/lib"))
            search_paths.append("/usr/local/cuda/lib64/")
        
        for path in search_paths:
            if not os.path.exists(path):
                continue
                
            for file in os.listdir(path):
                if file.startswith("libnvjpeg.so") or (file.startswith("nvjpeg64_") and file.endswith(".dll")):
                    try:
                        nvjpeg_lib = c.CDLL(os.path.join(path, file))
                        nvjpeg_lib.nvjpegGetProperty(0, c.byref(nvjpeg_ver_major))
                        nvjpeg_lib.nvjpegGetProperty(1, c.byref(nvjpeg_ver_minor))
                        nvjpeg_lib.nvjpegGetProperty(2, c.byref(nvjpeg_ver_patch))
                        nvjpeg_found = True
                        break
                    except:
                        continue
            if nvjpeg_found:
                break

    if not nvjpeg_found:
        return 0, 0, 0
        
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


def get_default_decoder_options():
    return ":fancy_upsampling=1" # this option is always available since cuda 12.1

def get_max_diff_threshold(threshold=None):
    if threshold is not None:
        return threshold
    return 6

def compare_image(test_img, ref_img, threshold=None):
    diff = ref_img.astype(np.int32) - test_img.astype(np.int32)
    diff = np.absolute(diff)

    assert test_img.shape == ref_img.shape
    assert diff.max() <= get_max_diff_threshold(threshold)

def compare_device_with_host_images(test_images, ref_images, threshold=None):
    for i in range(0, len(test_images)):
        np_test_img = np.asarray(test_images[i].cpu())
        ref_img = np.asarray(ref_images[i])
        compare_image(np_test_img, ref_img, threshold)

def compare_host_images(test_images, ref_images, threshold=None):
    for i in range(0, len(test_images)):
        test_img = np.asarray(test_images[i])
        ref_img = np.asarray(ref_images[i])
        compare_image(test_img, ref_img, threshold)

# nvTIFF requires nvCOMP to decode images with compression
def is_nvcomp_supported(device_id=0):
    if platform.system() == "Linux" and platform.machine() == 'x86_64':
        return True 
    if platform.system() == "Windows":
        return get_cuda_compute_capability(device_id) >= 7.0 # Deflate requires sm70 on Windows

    # it is aarch system, nvCOMP is supported for sbsa, but not for tegra,
    # so just check if nvCOMP is installed in the system, by trying to load it
    try:
        ctypes.CDLL('libnvcomp.so')
        return True
    except:
        return False

def is_nvjpeg_lossless_supported(device_id=0):
    min_cuda_compute_capability = 6.0 if platform.system() == "Linux" else 7.0
    return get_cuda_compute_capability(device_id) >= min_cuda_compute_capability and get_nvjpeg_ver() >= (12, 2, 0)

def load_single_image(file_path: str, load_mode: str | None = None):
    """
    Loads a single image to de decoded.
    :param file_path: Path to file with the image to be loaded.
    :param load_mode: In what format the image shall be loaded:
                        "numpy"  - loading using `np.fromfile`,
                        "python" - loading using Python's `open`,
                        "path"   - loading skipped, image path will be returned.
    :return: Encoded image.
    """
    if load_mode == "numpy":
        return np.fromfile(file_path, dtype=np.uint8)
    elif load_mode == "python":
        with open(file_path, 'rb') as in_file:
            return in_file.read()
    elif load_mode == "path":
        return file_path
    else:
        raise RuntimeError(f"Unknown load mode: {load_mode}")

def load_batch(file_paths: list[str], load_mode: str | None = None):
    return [load_single_image(f, load_mode) for f in file_paths]

def get_opencv_reference(input_img_path, color_spec=nvimgcodec.ColorSpec.SRGB, any_depth=False):
    import cv2
    flags = None
    if color_spec == nvimgcodec.ColorSpec.SRGB:
        flags = cv2.IMREAD_COLOR
    elif color_spec == nvimgcodec.ColorSpec.UNCHANGED:
        # UNCHANGED actually implies ANYDEPTH and we don't want that
        # we only want 'UNCHANGED' behavior to get the alpha channel, so limiting to that case
        if any_depth or 'alpha' in input_img_path:
            flags = cv2.IMREAD_UNCHANGED
        elif 'gray' in input_img_path:
            # We want grayscale samples to be decoded to grayscale when "unchanged"
            flags = cv2.IMREAD_GRAYSCALE
        else:
            flags = cv2.IMREAD_COLOR
    elif color_spec == nvimgcodec.ColorSpec.GRAY:
        flags = cv2.IMREAD_GRAYSCALE
    if any_depth:
        flags = flags | cv2.IMREAD_ANYDEPTH
    if isinstance(input_img_path, str):
        ref_img = cv2.imread(input_img_path, flags)
    elif isinstance(input_img_path, np.ndarray):
        ref_img = cv2.imdecode(input_img_path, flags)
    else:
        raise ValueError("Input must be a file path (string) or a numpy array.")
    if ref_img is None:
        return ref_img

    has_alpha = len(ref_img.shape) == 3 and ref_img.shape[-1] == 4
    if has_alpha:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGRA2RGBA)
    elif len(ref_img.shape) == 3 and ref_img.shape[-1] == 3:
        ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2RGB)
    elif len(ref_img.shape) == 2:
        ref_img = ref_img.reshape(ref_img.__array_interface__['shape'][0:2] + (1,))
    return np.asarray(ref_img)
