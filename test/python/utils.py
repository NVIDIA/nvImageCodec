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
    try:
        if platform.system() == "Linux":
            nvjpeg_libname = f'libnvjpeg.so'
        else:
            cuda_major_version = os.getenv("CUDA_VERSION_MAJOR", "12")
            nvjpeg_libname = f'nvjpeg64_{cuda_major_version}.dll'

        nvjpeg_lib = c.CDLL(nvjpeg_libname)
        nvjpeg_lib.nvjpegGetProperty(0, c.byref(nvjpeg_ver_major))
        nvjpeg_lib.nvjpegGetProperty(1, c.byref(nvjpeg_ver_minor))
        nvjpeg_lib.nvjpegGetProperty(2, c.byref(nvjpeg_ver_patch))
    except:
        if os.path.exists("/usr/local/cuda/lib64/"):
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
        np_test_img = np.asarray(test_images[i].cpu())
        ref_img = np.asarray(ref_images[i])
        compare_image(np_test_img, ref_img)

def compare_host_images(test_images, ref_images):
    for i in range(0, len(test_images)):
        test_img = np.asarray(test_images[i])
        ref_img = np.asarray(ref_images[i])
        compare_image(test_img, ref_img)

def is_nvjpeg2k_supported():
    return True

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

def get_opencv_reference(input_img_path, color_spec=nvimgcodec.ColorSpec.RGB, any_depth=False):
    import cv2
    flags = None
    if color_spec == nvimgcodec.ColorSpec.RGB:
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
