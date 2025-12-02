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

from nvidia import nvimgcodec
from utils import img_dir_path
import os

# create dummy decoder, which is alive for all tests and speedups creation of nvJPEG extension
# make sure utils are imported in each test that is using nvJPEG to see the speedup
try:
    dummy = nvimgcodec.Decoder(backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY)])
    # decode an image to load HW JPEG decoder
    dummy.decode(os.path.join(img_dir_path, "jpeg/padlock-406986_640_420.jpg"))
except: # may fail, if image is not present, but we just need to load decoder, so we don't care
    pass
