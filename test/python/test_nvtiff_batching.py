# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native TIFF batching: mixed IFDs, output conversion, streams, and decoder image limits."""

import importlib
import os
import re
import subprocess
import sys

import numpy as np
import pytest
from nvidia import nvimgcodec as nic
from utils import img_dir_path, is_nvcomp_supported

# nvTIFF needs nvCOMP to decode Deflate. nvCOMP is not available on Tegra.
NVCOMP = is_nvcomp_supported()
# Use Deflate where the codec is incidental to the test, if nvCOMP is available.
INCIDENTAL_DEFLATE = "deflate" if NVCOMP else "lzw"
COMPRESSIONS = [
    pytest.param("deflate", marks=pytest.mark.skipif(not NVCOMP, reason="nvCOMP is needed for Deflate decode")),
    "lzw",
]


def _dependency(name):
    # Python 3.14 currently has no CuPy wheel in these test environments.
    if os.environ.get("NVIMGCODEC_REQUIRE_TIFF_TESTS") == "1" and sys.version_info < (3, 14):
        return importlib.import_module(name)
    return pytest.importorskip(name)


@pytest.fixture
def mixed_tiff(tmp_path):
    tifffile = _dependency("tifffile")
    _dependency("imagecodecs")
    rng = np.random.default_rng(42)
    pixels = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    path = tmp_path / "mixed.tiff"
    with tifffile.TiffWriter(path) as writer:
        for i, (compression, tile) in enumerate([
            ("lzw", 64), (INCIDENTAL_DEFLATE, 64), ("jpeg", 64), ("lzw", 64), ("lzw", 32)
        ]):
            writer.write(np.roll(pixels, i * 7, axis=0), photometric="rgb",
                         compression=compression, tile=(tile, tile), metadata=None)
    return path


@pytest.mark.parametrize("threads", [1, 4])
@pytest.mark.parametrize("max_images", [0, 1, 3])
@pytest.mark.parametrize("sample_format", [nic.SampleFormat.I_RGB, nic.SampleFormat.P_RGB])
def test_mixed_tiff_subbatches(mixed_tiff, threads, max_images, sample_format):
    cp = _dependency("cupy")
    decoder = nic.Decoder(max_num_cpu_threads=threads,
                          backends=[nic.Backend(nic.BackendKind.GPU_ONLY)],
                          options=f"nvtiff_cuda_decoder:max_subbatch_images={max_images}")
    source = nic.CodeStream(str(mixed_tiff))
    # Matching IFDs are deliberately nonadjacent. Two independent opens also
    # exercise the one-backing-source-per-native-call rule.
    other = nic.CodeStream(str(mixed_tiff))
    inputs = [parent.get_sub_code_stream(i, nic.Region(16, 16, 80, 80))
              for parent in (source, other) for i in [3, 1, 0, 2, 4, 0, 3]]
    params = nic.DecodeParams(sample_format=sample_format)
    reference = []
    for src in inputs:
        image = decoder.decode(src, params=params)
        assert image is not None
        reference.append(cp.asarray(image).get())
    stream = cp.cuda.Stream(non_blocking=True)
    outputs = None
    for _ in range(2):
        outputs = decoder.decode(inputs, images=outputs, params=params,
                                 cuda_stream=stream.ptr)
        stream.synchronize()
        assert len(outputs) == len(reference)
        for image, expected in zip(outputs, reference):
            assert image is not None
            np.testing.assert_array_equal(cp.asarray(image).get(), expected)


def test_damaged_jpeg_ifd_does_not_fail_matching_valid_ifd(tmp_path):
    cp = _dependency("cupy")
    tifffile = _dependency("tifffile")
    _dependency("imagecodecs")
    path = tmp_path / "damaged.tiff"
    pixels = np.random.default_rng(7).integers(0, 256, (64, 64, 3), dtype=np.uint8)
    with tifffile.TiffWriter(path) as writer:
        for _ in range(2):
            writer.write(pixels, compression="jpeg", photometric="rgb", tile=(64, 64), metadata=None)
    with tifffile.TiffFile(path) as file:
        offset = file.pages[0].dataoffsets[0]
        size = file.pages[0].databytecounts[0]
    with path.open("r+b") as file:
        file.seek(offset)
        file.write(bytes(size))
    decoder = nic.Decoder(max_num_cpu_threads=4,
                          backends=[nic.Backend(nic.BackendKind.GPU_ONLY)])
    source = nic.CodeStream(str(path))
    valid = source.get_sub_code_stream(1)
    reference = decoder.decode(valid)
    assert reference is not None
    expected = cp.asarray(reference).get()
    result = decoder.decode([valid, source.get_sub_code_stream(0), valid])
    assert len(result) == 3
    assert result[1] is None
    for i in [0, 2]:
        assert result[i] is not None
        np.testing.assert_array_equal(cp.asarray(result[i]).get(), expected)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
@pytest.mark.parametrize("max_images", [0, 1, 3])
@pytest.mark.parametrize("layout", [nic.SampleFormat.P_AUTO_COMPONENTS, nic.SampleFormat.I_AUTO_COMPONENTS])
def test_native_depth_and_out_of_bounds(tmp_path, dtype, max_images, layout, compression):
    cp = _dependency("cupy")
    tifffile = _dependency("tifffile")
    _dependency("imagecodecs")
    pixels = np.arange(128 * 128, dtype=dtype).reshape(128, 128)
    path = tmp_path / "native-depth.tiff"
    tifffile.imwrite(path, pixels, compression=compression, tile=(64, 64), metadata=None)
    source = nic.CodeStream(str(path))
    regions = [(0, 0, 64, 64), (-8, -8, 56, 56), (96, 96, 160, 160), (16, 32, 80, 96)]
    inputs = [source.get_sub_code_stream(0, nic.Region(*region, out_of_bounds_sample=0)) for region in regions]
    decoder = nic.Decoder(max_num_cpu_threads=4, backends=[nic.Backend(nic.BackendKind.GPU_ONLY)],
                          options=f"nvtiff_cuda_decoder:max_subbatch_images={max_images}")
    params = nic.DecodeParams(color_spec=nic.ColorSpec.UNCHANGED, allow_any_depth=True,
                              sample_format=layout)
    outputs = decoder.decode(inputs, params=params)
    assert len(outputs) == len(regions)
    padded = np.pad(pixels, ((8, 32), (8, 32)))
    for image, (y0, x0, y1, x1) in zip(outputs, regions):
        assert image is not None
        actual = cp.asarray(image).get().reshape(y1 - y0, x1 - x0)
        assert actual.dtype == pixels.dtype
        np.testing.assert_array_equal(actual, padded[y0 + 8:y1 + 8, x0 + 8:x1 + 8])


@pytest.mark.parametrize("from_file", [False, True])
def test_sources_can_be_released_between_submissions(mixed_tiff, from_file):
    cp = _dependency("cupy")
    decoder = nic.Decoder(max_num_cpu_threads=4, backends=[nic.Backend(nic.BackendKind.GPU_ONLY)])
    expected = None
    for _ in range(4):
        # Each submission owns independent encoded storage. Retain the decoder
        # while releasing every source/view and output from the previous call.
        source = nic.CodeStream(str(mixed_tiff) if from_file else mixed_tiff.read_bytes())
        inputs = [source.get_sub_code_stream(i, nic.Region(16, 16, 80, 80)) for i in [0, 3, 1, 4]]
        images = decoder.decode(inputs)
        assert len(images) == len(inputs)
        assert all(image is not None for image in images)
        actual = [cp.asarray(image).get() for image in images]
        if expected is not None:
            for value, reference in zip(actual, expected):
                np.testing.assert_array_equal(value, reference)
        expected = actual
        del images, inputs, source


@pytest.mark.parametrize("max_images", [0, 1, 8, 16, 64])
@pytest.mark.parametrize("ifd", [0, 2])
def test_repeated_rois_with_pending_output_work(mixed_tiff, max_images, ifd):
    """Producer work must precede decode even when many ROIs share each stream."""
    cp = _dependency("cupy")
    decoder = nic.Decoder(max_num_cpu_threads=4,
                          backends=[nic.Backend(nic.BackendKind.GPU_ONLY)],
                          options=f"nvtiff_cuda_decoder:max_subbatch_images={max_images}")
    source = nic.CodeStream(str(mixed_tiff))
    inputs = [source.get_sub_code_stream(ifd, nic.Region(16, 16, 80, 80)) for _ in range(64)]
    expected = cp.asarray(decoder.decode(inputs[0])).get()
    outputs = decoder.decode(inputs)
    assert len(outputs) == len(inputs)
    assert all(image is not None for image in outputs)
    arrays = [cp.asarray(image) for image in outputs]
    delay = cp.RawKernel(r'''
        extern "C" __global__ void delay(unsigned long long cycles) {
            unsigned long long start = clock64();
            while (clock64() - start < cycles) {}
        }
    ''', 'delay')
    # Alternate caller streams while the framework distributes the outputs
    # over its own four worker streams. Overwriting the outputs is deliberately
    # delayed so missing producer dependencies cannot hide behind host setup.
    streams = [cp.cuda.Stream(non_blocking=True) for _ in range(3)]
    for count, stream in zip((len(inputs), 1, len(inputs)), streams):
        with stream:
            delay((1,), (1,), (np.uint64(50_000_000),))
            for array in arrays[:count]:
                array.fill(0)
        decoded = decoder.decode(inputs[:count], images=outputs[:count], cuda_stream=stream.ptr)
        stream.synchronize()
        assert len(decoded) == count
        for image in decoded:
            assert image is not None
            np.testing.assert_array_equal(cp.asarray(image).get(), expected)


@pytest.mark.parametrize("threads", [1, 4])
def test_mixed_sources_survive_reuse_and_replacement(tmp_path, threads):
    """Reuse owned/borrowed sources, then replace the batch's source set."""
    cp = _dependency("cupy")
    tifffile = _dependency("tifffile")
    _dependency("imagecodecs")
    pixels = [np.random.default_rng(i).integers(0, 256, (128, 128, 3), dtype=np.uint8)
              for i in range(4)]
    sources = []
    for i, data in enumerate(pixels):
        path = tmp_path / f"source-{i}.tiff"
        tifffile.imwrite(path, data, photometric="rgb", compression="lzw" if i % 2 else INCIDENTAL_DEFLATE,
                         tile=(64, 64), metadata=None)
        sources.append(nic.CodeStream(str(path) if i % 2 else path.read_bytes()))
    decoder = nic.Decoder(max_num_cpu_threads=threads,
                          backends=[nic.Backend(nic.BackendKind.GPU_ONLY)])
    outputs = None
    for selected in ([0, 1, 2], [2, 0, 1], [1, 2, 0], [2, 3], [3, 2], [3]):
        inputs = [sources[i].get_sub_code_stream(0, nic.Region(16, 16, 80, 80))
                  for i in selected for _ in range(4)]
        # The output-list length changes when a source leaves the working set.
        if outputs is not None and len(outputs) != len(inputs):
            outputs = None
        outputs = decoder.decode(inputs, images=outputs)
        assert len(outputs) == len(inputs)
        assert all(image is not None for image in outputs)
        for image, i in zip(outputs, [i for i in selected for _ in range(4)]):
            np.testing.assert_array_equal(cp.asarray(image).get(), pixels[i][16:80, 16:80])
        del inputs
        if selected == [1, 2, 0]:
            sources[0] = sources[1] = None
        elif selected == [3, 2]:
            sources[2] = None


@pytest.mark.parametrize("compression", COMPRESSIONS)
def test_mixed_ifd_types_preserve_output(tmp_path, compression):
    """Compatibility grouping must preserve each IFD's channels and sample type."""
    cp = _dependency("cupy")
    tifffile = _dependency("tifffile")
    _dependency("imagecodecs")
    rng = np.random.default_rng(91)
    pixels = [
        rng.integers(0, 256, (64, 64, 3), dtype=np.uint8),
        rng.integers(0, 65536, (64, 64, 3), dtype=np.uint16),
        rng.random((64, 64), dtype=np.float32),
    ]
    path = tmp_path / "mixed-types.tiff"
    with tifffile.TiffWriter(path) as writer:
        for data in pixels:
            writer.write(data, compression=compression, tile=(32, 32),
                         photometric="rgb" if data.ndim == 3 else "minisblack", metadata=None)
    decoder = nic.Decoder(max_num_cpu_threads=4,
                          backends=[nic.Backend(nic.BackendKind.GPU_ONLY)])
    source = nic.CodeStream(str(path))
    order = [2, 0, 1, 0, 2, 1]
    inputs = [source.get_sub_code_stream(i) for i in order]
    params = nic.DecodeParams(color_spec=nic.ColorSpec.UNCHANGED, allow_any_depth=True,
                              sample_format=nic.SampleFormat.I_AUTO_COMPONENTS)
    outputs = None
    for _ in range(2):
        outputs = decoder.decode(inputs, images=outputs, params=params)
        assert len(outputs) == len(order)
        for image, i in zip(outputs, order):
            assert image is not None
            actual = cp.asarray(image).get()
            expected = pixels[i] if pixels[i].ndim == 3 else pixels[i][..., None]
            assert actual.dtype == expected.dtype
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("value", ["x", "-1", ""])
def test_invalid_option_does_not_disable_decoder(mixed_tiff, value):
    decoder = nic.Decoder(backends=[nic.Backend(nic.BackendKind.GPU_ONLY)],
                          options="nvtiff_cuda_decoder:max_subbatch_images=" + value)
    assert decoder.decode(nic.CodeStream(str(mixed_tiff))) is not None


@pytest.mark.parametrize("name", ["max_subbatch_images", "trace_subbatches"])
@pytest.mark.parametrize("invalid_first", [True, False])
def test_duplicate_option_warning(mixed_tiff, name, invalid_first):
    prefix = "nvtiff_cuda_decoder:" + name + "="
    tokens = [prefix + "bogus", prefix + "8"]
    if not invalid_first:
        tokens.reverse()
    code = r"""
import sys
from nvidia import nvimgcodec as n
d = n.Decoder(backends=[n.Backend(n.BackendKind.GPU_ONLY)], options=sys.argv[2])
assert d.decode(n.CodeStream(sys.argv[1])) is not None
"""
    result = subprocess.run([sys.executable, "-c", code, str(mixed_tiff), " ".join(tokens)],
                            env=dict(os.environ, PYNVIMGCODEC_VERBOSITY="3"), capture_output=True, text=True)
    log = result.stdout + result.stderr
    assert result.returncode == 0, log
    max_images = 8 if invalid_first and name == "max_subbatch_images" else 0
    tracing = "enabled" if invalid_first and name == "trace_subbatches" else "disabled"
    assert (f"Malformed nvTIFF option(s): {prefix}bogus; "
            f"effective max_subbatch_images={max_images}, tracing={tracing}") in log
    assert "using defaults" not in log


def test_released_singleton_outputs(mixed_tiff):
    """Decode completion must precede immediate output deallocation."""
    cp = _dependency("cupy")
    decoder = nic.Decoder(max_num_cpu_threads=1, backends=[nic.Backend(nic.BackendKind.GPU_ONLY)])
    source = nic.CodeStream(str(mixed_tiff))
    # Do not copy to host or otherwise synchronize before dropping the image.
    for _ in range(40):
        image = decoder.decode(source)
        assert image is not None
        del image
    cp.cuda.runtime.deviceSynchronize()


@pytest.mark.parametrize("threads", [1, 4, 8])
@pytest.mark.parametrize("load_hint", [1.0, 0.5, 0.0])
@pytest.mark.parametrize("from_file", [True, False])
def test_growing_ifd_batches(threads, load_hint, from_file):
    """Shared IFD views must decode correctly even when load hints send them to CPU."""
    cp = _dependency("cupy")
    path = os.path.join(img_dir_path, "tiff", "multi_page.tif")
    source = nic.CodeStream(path if from_file else np.fromfile(path, dtype=np.uint8))
    inputs = [source.get_sub_code_stream(i) for i in range(source.num_images)]
    # Preserve native-depth coverage for GPU-only routing. CPU fallback needs
    # RGB output because libtiff does not support the four-channel last IFD as-is.
    params = (nic.DecodeParams(allow_any_depth=True, color_spec=nic.ColorSpec.UNCHANGED)
              if load_hint == 1.0 else nic.DecodeParams())
    reference_decoder = nic.Decoder(max_num_cpu_threads=1,
                                    backends=[nic.Backend(nic.BackendKind.GPU_ONLY)])
    expected = [cp.asarray(reference_decoder.decode(item, params=params)).get() for item in inputs]
    decoder = nic.Decoder(max_num_cpu_threads=threads, backends=[
        nic.Backend(nic.BackendKind.GPU_ONLY, load_hint=load_hint),
        nic.Backend(nic.BackendKind.CPU_ONLY)])
    for _ in range(10):
        for count in range(1, len(inputs) + 1):
            images = decoder.decode(inputs[:count], params=params)
            assert len(images) == count
            for image, pixels in zip(images, expected[:count]):
                assert image is not None
                np.testing.assert_array_equal(cp.asarray(image).get(), pixels)


@pytest.mark.parametrize("max_images", [0, 1, 3, 8, 64])
@pytest.mark.parametrize("threads", [1, 4])
def test_native_call_limit_and_singleton_routing(tmp_path, max_images, threads):
    """The decoder option limits actual native calls, not the application batch."""
    _dependency("cupy")
    tifffile = _dependency("tifffile")
    path = tmp_path / "native.tiff"
    tifffile.imwrite(path, np.zeros((64, 64, 3), dtype=np.uint8), photometric="rgb", metadata=None)
    code = r"""
import sys
from nvidia import nvimgcodec as n
path, max_images, threads = sys.argv[1:]
d = n.Decoder(max_num_cpu_threads=int(threads),
              options="nvtiff_cuda_decoder:trace_subbatches=1 "
                      "nvtiff_cuda_decoder:max_subbatch_images=" + max_images)
s = n.CodeStream(path)
assert d.decode(s) is not None
for _ in range(2):
    images = d.decode([s] * 8)
    assert len(images) == 8 and all(x is not None for x in images)
    assert d.decode(s) is not None
"""
    result = subprocess.run([sys.executable, "-c", code, str(path), str(max_images), str(threads)],
                            env=dict(os.environ, PYNVIMGCODEC_VERBOSITY="3"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    log = result.stdout + result.stderr
    calls = [(int(n), int(b)) for n, b in re.findall(r"TIFF_NATIVE_CALL images=(\d+) bytes=(\d+)", log)]
    assert calls, log
    assert sum(n for n, _ in calls) == 16  # Every requested image stayed on nvTIFF.
    for count, size in calls:
        assert size == count * 64 * 64 * 3
        assert not max_images or count <= max_images
    if not max_images or max_images >= 8:
        assert calls == [(8, 8 * 64 * 64 * 3)] * 2
    assert "Recreating nvtiff decoder instance" not in log
    assert "libtiff_decoder process" not in log


def test_conversion_cache_reclaims_unused_slots(tmp_path):
    _dependency("cupy")
    tifffile = _dependency("tifffile")
    path = tmp_path / "conversion.tiff"
    tifffile.imwrite(path, np.zeros((64, 64, 3), dtype=np.uint8), photometric="rgb", metadata=None)
    code = r"""
import sys
from nvidia import nvimgcodec as n
d = n.Decoder(max_num_cpu_threads=1, backends=[n.Backend(n.BackendKind.GPU_ONLY)],
              options="nvtiff_cuda_decoder:trace_subbatches=1")
s = n.CodeStream(sys.argv[1])
p = n.DecodeParams(sample_format=n.SampleFormat.P_RGB)
for count in [128] + [4] * 8:
    images = d.decode([s] * count, params=p)
    assert len(images) == count and all(x is not None for x in images)
"""
    result = subprocess.run([sys.executable, "-c", code, str(path)],
                            env=dict(os.environ, PYNVIMGCODEC_VERBOSITY="3"), capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    cache = [(int(n), int(b)) for n, b in re.findall(r"TIFF_BUFFER_CACHE slots=(\d+) bytes=(\d+)", result.stdout + result.stderr)]
    assert len(cache) == 9
    assert cache[0][0] == 128 and cache[0][1] > 0
    assert cache[7] == cache[0]  # Short fluctuations keep reusable scratch.
    assert cache[8][0] == 4 and cache[8][1] < cache[0][1] / 4
