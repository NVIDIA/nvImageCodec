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
from nvidia import nvimgcodec
import pytest as t
from utils import img_dir_path

@t.mark.parametrize( "test_sample", [
    {"input_img_file": "tiff/JP2K-33003-1.svs", 
            "num_images": 6,
            "image_with_icc_profile": 0,
            "icc_profile_index": 1
    },
])
def test_icc_profile_extraction(test_sample):
    decoder = nvimgcodec.Decoder()
    fpath = os.path.join(img_dir_path, test_sample["input_img_file"])
    code_stream = nvimgcodec.CodeStream(fpath)
    assert code_stream.num_images == test_sample["num_images"]

    sub_cs = code_stream.get_sub_code_stream(test_sample["image_with_icc_profile"])
    metadata = decoder.get_metadata(sub_cs)
    icc_profile_index = test_sample["icc_profile_index"]
  
    icc_profile = metadata[icc_profile_index].buffer
    assert icc_profile is not None
    assert len(icc_profile) > 0
    
    # Basic ICC profile validation - check for minimum size and signature
    assert len(icc_profile) >= 128, "ICC profile should be at least 128 bytes"

    # Check ICC profile signature (should be 'acsp')
    profile_signature = icc_profile[36:40]
    assert profile_signature == b'acsp', f"Invalid ICC profile signature: {profile_signature}"

    # Check profile size from header matches actual size
    profile_size = int.from_bytes(icc_profile[0:4], byteorder='big')
    assert profile_size == len(icc_profile), f"Profile size mismatch: header={profile_size}, actual={len(icc_profile)}"
