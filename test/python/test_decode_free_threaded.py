# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import random
import numpy as np
import pytest as t
import threading
import sysconfig
from nvidia import nvimgcodec
from utils import get_default_decoder_options, img_dir_path, compare_image
from concurrent.futures import ThreadPoolExecutor
files = {
    "jpeg" : [
        "jpeg/padlock-406986_640_410.jpg",
        "jpeg/padlock-406986_640_411.jpg",
        "jpeg/padlock-406986_640_420.jpg",
        "jpeg/padlock-406986_640_422.jpg",
        "jpeg/padlock-406986_640_440.jpg",
        "jpeg/padlock-406986_640_444.jpg",
        "jpeg/padlock-406986_640_gray.jpg",
        "jpeg/cmyk-dali.jpg",
        "jpeg/progressive-subsampled-imagenet-n02089973_1957.jpg",
        "jpeg/exif/padlock-406986_640_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_mirror_horizontal_rotate_90.jpg",
        "jpeg/exif/padlock-406986_640_mirror_vertical.jpg",
        "jpeg/exif/padlock-406986_640_no_orientation.jpg",
        "jpeg/exif/padlock-406986_640_rotate_180.jpg",
        "jpeg/exif/padlock-406986_640_rotate_270.jpg",
        "jpeg/exif/padlock-406986_640_rotate_90.jpg",
    ],
    "jpeg2k" : [
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-1046544_640.jp2",
        "jpeg2k/cat-111793_640.jp2",
        "jpeg2k/tiled-cat-1046544_640.jp2",
        "jpeg2k/tiled-cat-111793_640.jp2",
        "jpeg2k/cat-111793_640-16bit.jp2",
        "jpeg2k/cat-1245673_640-12bit.jp2",
    ],
}

@t.fixture(scope="module")
def shared_tmp_path(tmp_path_factory):
    return tmp_path_factory.mktemp("test_decode_free_threaded_tmp_path")

@t.mark.parametrize("input_format", ["path"])
@t.mark.parametrize("image_format", ["jpeg", "jpeg2k", "all"])
@t.mark.parametrize("backends", [None, [nvimgcodec.Backend(nvimgcodec.CPU_ONLY)]])
@t.mark.parametrize("num_threads", [1, 16, 64])
@t.mark.parametrize("num_images", [100])
@t.mark.skipif(not sysconfig.get_config_var("Py_GIL_DISABLED"), reason="Skip for not free-threaded Python")
def test_decode_free_threaded(shared_tmp_path, input_format, image_format, backends, num_threads, num_images):
    thread_local = threading.local()

    # A different decoder per thread
    def decoder():
        if not hasattr(thread_local, "decoder"):
            thread_local.decoder = nvimgcodec.Decoder(
                max_num_cpu_threads=1, backends=backends, options=get_default_decoder_options())
        return thread_local.decoder

    # A different decoder per thread
    def decoder_ref():
        if not hasattr(thread_local, "decoder_ref"):
            thread_local.decoder_ref = nvimgcodec.Decoder(
                max_num_cpu_threads=1, backends=[nvimgcodec.Backend(nvimgcodec.CPU_ONLY)],
                options=get_default_decoder_options())
        return thread_local.decoder_ref

    assert input_format == 'path'  # for now

    # Generate full paths for input images
    if image_format == 'all':
        input_img_files = []
        for f in files.values():
            input_img_files += f
    else:
        input_img_files = files[image_format]
    input_img_paths = [os.path.join(img_dir_path, img_file) for img_file in input_img_files]

    if len(input_img_paths) < num_images:
        input_img_paths *= (num_images // len(input_img_paths)) + 1  # Duplicate to exceed num_images
    random.shuffle(input_img_paths)  # Shuffle the list
    input_img_paths = input_img_paths[:num_images]  # Trim excess

    images = []
    images_ref = []

    def reference_decode(input_img_path, cache_dir, debug=False):
        # Generate a unique cache file name based on the image path
        cache_file = os.path.join(cache_dir, f"{os.path.basename(input_img_path)}.npy")
        if os.path.exists(cache_file):
            if debug:
                print(f"Reusing {cache_file}")
            # If cache file exists, load the NumPy array
            return np.load(cache_file)
        else:
            # Decode image and save as NumPy array to cache
            if debug:
                print(f"Decoding {cache_file}")
            img = decoder_ref().decode(input_img_path)
            array = np.asarray(img.cpu())
            np.save(cache_file, array)
            return array

    # Create a temporary directory for caching
    cache_dir = os.path.join(shared_tmp_path, "cache")
    os.makedirs(cache_dir, exist_ok=True)

    for path in input_img_paths:
        images_ref.append(reference_decode(path, cache_dir))

    # Decode function to be executed by the thread pool
    def decode_image(input_img_path):
        return decoder().decode(input_img_path)

    # Use ThreadPoolExecutor for managing threads
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = [executor.submit(decode_image, img_path) for img_path in input_img_paths]
        for future in futures:
            img = future.result()
            images.append(np.asarray(img.cpu()))

    for image, image_ref in zip(images, images_ref):
        compare_image(image, image_ref)
