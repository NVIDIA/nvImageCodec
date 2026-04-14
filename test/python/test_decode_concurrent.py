# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
Concurrent decode tests for nvimgcodec Python bindings.

Covers three scenarios (mirroring test/api/concurrent_decode_test.cpp):
  1. N threads each decoding simultaneously with their own per-thread Decoder.
  2. Batch decode (list of CodeStreams submitted in one call).
  3. ThreadPoolExecutor dispatching single-image decode calls.

Also contains a regression test for a pybind11 implicit-conversion thread-safety
bug: in pybind11 < 3.x the recursion guard inside py::implicitly_convertible was a
shared static bool (not thread-local), so concurrent calls to decode(bytes) could
race — one thread would see the guard set by another and abort the conversion,
causing "TypeError: decode(): incompatible function arguments".
pybind11 3.x fixed this by switching to thread_specific_storage (TLS) for the
guard.  We additionally add explicit decode(py::bytes) and
decode(py::array_t<uint8_t>) overloads in Decoder (python/decoder.cpp) so that
the implicit-conversion path is never taken for decode() calls at all, making the
fix independent of the pybind11 version in use.
"""

from __future__ import annotations

import io
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from PIL import Image

from nvidia import nvimgcodec
from utils import img_dir_path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_concurrent(n_threads, setup, work):
    """Run *n_threads* threads through a barrier, collecting exceptions.

    Each thread calls ``ctx = setup(tid)`` before the barrier and
    ``work(tid, ctx)`` after.  A pre-barrier failure calls ``barrier.abort()``
    so no thread blocks indefinitely.  Raises AssertionError if any thread
    failed.
    """
    errors = {}
    barrier = threading.Barrier(n_threads)

    def worker(tid):
        try:
            ctx = setup(tid)
        except Exception as exc:
            errors[tid] = exc
            barrier.abort()
            return
        barrier.wait()
        try:
            work(tid, ctx)
        except Exception as exc:
            errors[tid] = exc

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, (
        f"{len(errors)} thread(s) failed:\n"
        + "\n".join(f"  thread {tid}: {type(exc).__name__}: {exc}" for tid, exc in sorted(errors.items()))
    )


def make_jpeg_bytes(width: int = 256, height: int = 256, quality: int = 85) -> bytes:
    rng = np.random.RandomState(42)
    arr = rng.randint(0, 255, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


JPEG_FILES = [
    "jpeg/padlock-406986_640_410.jpg",
    "jpeg/padlock-406986_640_420.jpg",
    "jpeg/padlock-406986_640_422.jpg",
    "jpeg/padlock-406986_640_444.jpg",
]


# ---------------------------------------------------------------------------
# Regression: decode(bytes) thread-safety
#
# In pybind11 < 3.x, py::implicitly_convertible used a shared static bool as a
# recursion guard.  Concurrent threads racing through the guard would each find
# it set by the other thread, return nullptr from the converter, and cause
# pybind11's overload resolution to fail with:
#   TypeError: decode(): incompatible function arguments
# pybind11 3.x fixed this with thread_specific_storage (TLS) for the guard.
# We also add explicit decode(py::bytes) overloads in Decoder (python/decoder.cpp)
# so the implicit-conversion path is never taken for decode() calls.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_threads,n_iters", [(16, 50)])
def test_decode_bytes_concurrent(n_threads, n_iters):
    """decode(bytes) must not raise TypeError when called from multiple threads simultaneously."""
    jpeg_bytes = make_jpeg_bytes()

    def setup(_tid):
        dec = nvimgcodec.Decoder()
        dec.decode(jpeg_bytes)  # prewarm
        return dec

    def work(_tid, dec):
        for _ in range(n_iters):
            dec.decode(jpeg_bytes)

    run_concurrent(n_threads, setup, work)


def test_decode_bytes_matches_codestream():
    """decode(bytes) and decode(CodeStream(bytes)) must produce identical pixels."""
    jpeg_bytes = make_jpeg_bytes()
    dec = nvimgcodec.Decoder()
    img_bytes = np.asarray(dec.decode(jpeg_bytes).cpu())
    img_cs = np.asarray(dec.decode(nvimgcodec.CodeStream(jpeg_bytes)).cpu())
    np.testing.assert_array_equal(img_bytes, img_cs)


def test_decode_numpy_array_concurrent():
    """decode(np.array of uint8) must work correctly from multiple threads."""
    jpeg_arr = np.frombuffer(make_jpeg_bytes(), dtype=np.uint8)

    def setup(_tid):
        dec = nvimgcodec.Decoder()
        dec.decode(jpeg_arr)  # prewarm
        return dec

    def work(_tid, dec):
        for _ in range(20):
            assert dec.decode(jpeg_arr) is not None

    run_concurrent(8, setup, work)


# ---------------------------------------------------------------------------
# Concurrent decode: N threads, per-thread Decoder
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_threads,n_iters", [(8, 20)])
def test_concurrent_threads_per_thread_decoder(n_threads, n_iters):
    """N threads decoding simultaneously, each with its own Decoder instance."""
    img_paths = [os.path.join(img_dir_path, f) for f in JPEG_FILES]

    def setup(_tid):
        return nvimgcodec.Decoder()

    def work(_tid, dec):
        for i in range(n_iters):
            assert dec.decode(nvimgcodec.CodeStream(img_paths[i % len(img_paths)])) is not None

    run_concurrent(n_threads, setup, work)


# ---------------------------------------------------------------------------
# Concurrent decode: batch API
# ---------------------------------------------------------------------------

def test_batch_decode():
    """Batch decode (list input) returns one result per input."""
    dec = nvimgcodec.Decoder()
    img_paths = [os.path.join(img_dir_path, f) for f in JPEG_FILES]
    code_streams = [nvimgcodec.CodeStream(p) for p in img_paths]

    results = dec.decode(code_streams)

    assert len(results) == len(code_streams)
    for r in results:
        assert r is not None


# ---------------------------------------------------------------------------
# Concurrent decode: ThreadPoolExecutor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("n_workers", [4, 8])
def test_threadpool_decode(n_workers):
    """ThreadPoolExecutor dispatching decode calls, each worker with its own Decoder."""
    img_paths = [os.path.join(img_dir_path, f) for f in JPEG_FILES] * 4  # 16 total
    thread_local = threading.local()

    def decode_one(path):
        if not hasattr(thread_local, "dec"):
            thread_local.dec = nvimgcodec.Decoder()
        return thread_local.dec.decode(nvimgcodec.CodeStream(path))

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(decode_one, img_paths))

    assert all(r is not None for r in results)
