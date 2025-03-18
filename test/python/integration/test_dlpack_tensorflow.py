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
import sys
import pytest as t
pytestmark = t.mark.skipif(sys.version_info >= (3, 13), reason="Requires Python version lower than 3.13")

import numpy as np
try:
    import cupy as cp
except:
    print("CuPy is not available, will skip related tests")

from nvidia import nvimgcodec

try:
    import tensorflow as tf
    has_tf = len(tf.config.list_physical_devices('GPU')) > 0
except:
    print("Tensorflow is not available, will skip related tests")
    has_tf = False

@t.mark.skipif(not has_tf, reason="Tensorflow with GPU is not available")
@t.mark.parametrize("shape,dtype",
                    [
                        ((5, 23, 65), np.uint8),
                        ((5, 23, 65), np.int16),
                        ((5, 23, 65), np.int32),
                        ((65, 3, 3), np.int64),
                    ],
                    )
def test_dlpack_import_from_tensorflow(shape, dtype):
    #TODO this is temporary workaround - create tf.tensor indirectly from cupy  
    #code below cause some problem in following tests
    # with tf.device('/GPU:0'):
        #a = tf.random.uniform(shape, 0, 128, dtype)
        #ref = a.numpy()

    rng = np.random.default_rng()
    ref = rng.integers(0, 128, shape, dtype)
    cp_img = cp.asarray(ref)

    # tensorflow needs to have all data in the device, as it didn't expose stream which we can synchronize with
    cp.cuda.stream.get_current_stream().synchronize()
    a = tf.experimental.dlpack.from_dlpack(cp_img.toDlpack())

    cap = tf.experimental.dlpack.to_dlpack(a)
    img = nvimgcodec.from_dlpack(cap)
    
    assert (np.asarray(img.cpu()) == ref).all()

@t.mark.skipif(not has_tf, reason="Tensorflow with GPU is not available")
@t.mark.parametrize("shape,dtype",
                    [
                        ((640, 480, 3), np.int8),
                        ((640, 480, 3), np.uint8),
                        ((640, 480, 3), np.int16),
                    ],
                    )
def test_dlpack_export_to_tensorflow(shape, dtype):
    rng = np.random.default_rng()
    host_array = rng.integers(0, 128, shape, dtype)
    dev_array = cp.asarray(host_array)

    img = nvimgcodec.as_image(dev_array)

    # tensorflow needs to have all data in the device, as it didn't expose stream which we can synchronize with
    cp.cuda.stream.get_current_stream().synchronize()
    tf_tensor = tf.experimental.dlpack.from_dlpack(img.to_dlpack())
    converted = cp.asnumpy(tf_tensor)

    assert (converted == host_array).all()
