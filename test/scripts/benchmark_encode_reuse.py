#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Benchmark sustained nvJPEG2K encoding with reusable CodeStream outputs.

Run against an installed package or set PYTHONPATH to a build's ``python``
directory. Example:

    python test/scripts/benchmark_encode_reuse.py --batch-size 8 --iterations 200

The input is uploaded once by default. Timed work starts with input resident on
the GPU and ends after compressed streams are ready in host memory.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

TEST_PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
sys.path.insert(0, str(TEST_PYTHON_DIR))
os.environ.setdefault("PYNVIMGCODEC_VERBOSITY", "0")

from nvidia import nvimgcodec
from resource_utils import snapshot_process_resources, wait_for_process_resources_to_settle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--channels", type=int, choices=(1, 3), default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--capacity-factor", type=float, default=2.0)
    parser.add_argument("--max-num-cpu-threads", type=int, default=0)
    parser.add_argument("--mode", choices=("reuse", "fresh"), default="reuse")
    parser.add_argument("--pageable-output", action="store_true")
    parser.add_argument("--host-input", action="store_true")
    parser.add_argument("--no-ht", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.width, args.height, args.batch_size, args.iterations) <= 0:
        raise ValueError("width, height, batch size, and iterations must be positive")
    if args.warmup < 0 or args.capacity_factor <= 0:
        raise ValueError("warmup must be nonnegative and capacity factor must be positive")
    if args.mode == "fresh" and args.pageable_output:
        raise ValueError("--pageable-output applies only to --mode reuse")

    shape = (args.height, args.width, args.channels) if args.channels == 3 else (args.height, args.width)
    array = np.random.default_rng(12345).integers(0, 256, shape, dtype=np.uint8)
    image = nvimgcodec.as_image(array)
    if not args.host_input:
        image = image.cuda()
    images = [image] * args.batch_size
    outputs = []
    if args.mode == "reuse":
        capacity = max(1, int(array.nbytes * args.capacity_factor))
        outputs = [
            nvimgcodec.CodeStream(capacity, pin_memory=not args.pageable_output)
            for _ in range(args.batch_size)
        ]
    original_outputs = tuple(outputs)
    encoder = nvimgcodec.Encoder(max_num_cpu_threads=args.max_num_cpu_threads)
    params = nvimgcodec.EncodeParams(
        quality_type=nvimgcodec.QualityType.LOSSLESS,
        jpeg2k_encode_params=nvimgcodec.Jpeg2kEncodeParams(
            mct_mode=1 if args.channels == 3 else 0,
            ht=not args.no_ht,
        ),
    )

    def encode_once():
        nonlocal outputs
        if args.mode == "reuse":
            outputs = encoder.encode(
                images, codec="jpeg2k", params=params, code_stream_s=outputs)
        else:
            outputs = encoder.encode(images, codec="jpeg2k", params=params)

    for _ in range(args.warmup):
        encode_once()
    if args.mode == "reuse" and (
            len(outputs) != len(original_outputs) or not all(
                actual is expected for actual, expected in zip(outputs, original_outputs))):
        raise RuntimeError("Encoder did not retain the supplied CodeStream objects")

    gc.collect()
    # Driver/plugin initialization may continue briefly after construction.
    # Settle it before measuring resources, then restore warm GPU clocks.
    wait_for_process_resources_to_settle()
    for _ in range(min(8, args.warmup)):
        encode_once()
    gc.collect()
    before = snapshot_process_resources()
    # encode() is synchronous here: input is already on the GPU by default and
    # each iteration returns only after compressed output is ready on the host.
    start = time.perf_counter()
    for _ in range(args.iterations):
        encode_once()
    elapsed = time.perf_counter() - start
    gc.collect()
    after = snapshot_process_resources()

    image_count = args.batch_size * args.iterations
    input_bytes = array.nbytes * image_count
    compressed_bytes = sum(stream.size for stream in outputs)
    result = {
        "mode": args.mode,
        "width": args.width,
        "height": args.height,
        "channels": args.channels,
        "batch_size": args.batch_size,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "htj2k": not args.no_ht,
        "device_input": not args.host_input,
        "pinned_output": not args.pageable_output,
        "elapsed_seconds": elapsed,
        "mean_batch_ms": elapsed * 1000 / args.iterations,
        "images_per_second": image_count / elapsed,
        "input_gb_per_second": input_bytes / elapsed / 1e9,
        "last_batch_compressed_bytes": compressed_bytes,
        "last_batch_compression_ratio": compressed_bytes / (array.nbytes * args.batch_size),
        "resources_before": asdict(before),
        "resources_after": asdict(after),
        "rss_growth_bytes": None if before.rss_bytes is None or after.rss_bytes is None
            else after.rss_bytes - before.rss_bytes,
        "handle_growth": None if before.handles is None or after.handles is None
            else after.handles - before.handles,
    }
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        for key, value in result.items():
            print(f"{key}={value}")


if __name__ == "__main__":
    main()
