# Experimental baseline JPEG CUDA decoding core

This opt-in standalone CUDA component is proposed as the core of a future
nvImageCodec decoder extension. It is not registered as a framework decoder and
does not change default codec selection. The component is ready for review of
the standalone core and its API; a framework adapter is separate follow-up work.

The host parses marker segments, validates tables, and stages encoded bytes.
Entropy decoding, DC prediction, dequantization, IDCT, and RGB conversion run on
the GPU. The core has no Torch dependency or CPU entropy fallback.

## Algorithm and files

1. Remove byte stuffing with CUB selection and identify the entropy boundary.
2. Speculatively decode 1024-bit chunks with 3072 bits of overlapping context.
3. Reconcile chunk states and repair remaining breaks to recover exact block
   positions. Some remaining dependencies can be serial; self-synchronization
   is not assumed to succeed for every chunk.
4. Decode exact coefficients, then recover per-component DC prediction by scans.
5. Dequantize, apply IDCT, convert YCbCr to RGB, and write independently pitched
   caller-owned planes. Explicit rounding and disabled FMA contraction preserve
   the prototype reference arithmetic.

`jpeg_header.cpp` contains CPU-only bounded parsing and layout planning;
`kernels.cuh` contains device code; `decoder.cu` manages allocation, CUDA streams,
and completion. `LICENSE.cuda-samples` retains the IDCT butterfly's source notice.

## Initial contract

- Baseline 8-bit sequential, one interleaved scan, gray or YCbCr 4:4:4.
- At most 4096x4096 pixels and 8 MiB encoded data.
- Planar RGB uint8 output with caller-provided row strides and storage.
- No progressive, arithmetic, lossless, restart intervals, multi-scan, CMYK/YCCK,
  Adobe RGB, ROI, EXIF transform, or fancy chroma upsampling in this component.
- Device with 96 KiB opt-in shared memory. Default allocation uses CUDA memory
  pools; custom stream-aware device and pinned allocators are supported.
- One in-flight image per decoder. `Decode` enqueues work on the supplied stream;
  `Wait` checks GPU corruption flags and completion. Reuse waits for the previous
  image. Successful header probing does not validate the entropy payload.
- A CUDA submission exception drains work already submitted on that stream and
  leaves the decoder idle. Recoverable submission failures do not leave an
  unwritten completion status that could be mistaken for an entropy error.
- Allocator context, CUDA context, streams, and output must retain the documented
  lifetimes. Allocation/growth and explicit completion checks can block the host.

Unsupported input returns a status. A framework adapter must implement the
`canDecode`/orientation/output contracts and check GPU completion before reporting
success. The prototype does not replace the existing nvJPEG format coverage.

## Build and checks

The main build enables the component with `-DBUILD_EXPERIMENTAL_JPEG_CUDA=ON`
(default OFF). It also builds independently:

```bash
cmake -S extensions/jpeg_cuda_experimental -B build/jpeg-cuda \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_ARCHITECTURES=120 \
  -DJPEG_CUDA_BUILD_GPU_TESTS=ON
cmake --build build/jpeg-cuda -j2
ctest --test-dir build/jpeg-cuda --output-on-failure
python extensions/jpeg_cuda_experimental/tests/test_decoder.py \
  --library build/jpeg-cuda/libjpeg_cuda_test_bridge.so
```

The optional Python GPU test uses Torch/torchvision/Pillow/NumPy as references
only. It covers generated gray/RGB images, small/odd sizes, several qualities,
row padding, and an alternate CUDA stream. `--images DIR --limit N` adds inputs.

`JPEG_CUDA_BUILD_GPU_TESTS` defaults to OFF so ordinary test collection does not
require a GPU. Enabling it adds a native lifecycle regression for a submission
failure after work has been queued, recovery on another stream, corrupt entropy,
pitched outputs, and restoring the caller's current CUDA device. It needs no
Torch dependency and can run under CUDA memory checking:

```bash
compute-sanitizer --tool memcheck --leak-check full --report-api-errors no \
  --error-exitcode 99 \
  build/jpeg-cuda/jpeg_cuda_lifecycle_test
```

The lifecycle test deliberately makes an invalid-pitch CUDA API call. API error
reporting is disabled for that expected failure; device memory and leak checks
remain enabled, and unexpected API failures still fail the test executable.

CPU parser checks include every truncation of a valid tiny stream, malformed
lengths, oversubscribed Huffman tables, invalid selectors, unsupported sampling,
and 10,000 deterministic mutations. It can run without a CUDA compiler:

```bash
c++ -std=c++17 -O1 -g -fsanitize=address,undefined \
  -I extensions/jpeg_cuda_experimental \
  extensions/jpeg_cuda_experimental/jpeg_header.cpp \
  extensions/jpeg_cuda_experimental/tests/header_test.cpp -o jpeg-header-test
./jpeg-header-test
```

## Validation of this component

On 2026-11-04, the extracted component was rebuilt with CUDA 12.8.93 and GCC 11.4
for SM120 and tested on an RTX 5090. Both the standalone and parent-project
component builds succeeded. The optional GPU comparison checked 36 generated
JPEGs against torchvision 0.23.0+cu128/PyTorch 2.8.0+cu128; every decoded output was
bit-identical, including pitched rows and an alternate caller stream.

The native lifecycle regression reproduced an unwritten completion status after
a failed asynchronous submission before the fix and passed after the fix. It
also checks corrupt entropy, reuse across streams, and caller-device restoration.
CUDA Compute Sanitizer memcheck reported zero device-memory errors and zero
leaked allocations. The CPU header regression passed ASan/UBSan, all tiny-stream
truncations, malformed-table cases, and 10,000 deterministic mutations.

## Motivation and scope of measurements

On 2026-10-22, the earlier Torch-bound prototype was checked on an RTX 5090 with
PyTorch 2.8.0+cu128 and torchvision 0.23.0+cu128. Decoded images and downstream
preprocessing tensors were bit-identical for 5000 images: 4990 baseline 4:4:4 and
10 gray. Its 200-image sequential loading benchmark was 2335.05 versus 425.75
microseconds/image; GPU busy time increased from 57.79 to 335.80 microseconds/image.
These are historical prototype/load-path measurements, not performance numbers
for this extracted component or model end-to-end speedups.

nvJPEG already offers GPU-assisted Huffman decoding. The proposal asks whether
less host work for small batches justifies another backend. Follow-up work is a
matched comparison against current default/forced GPU hybrid paths, the framework
decoder adapter, broader GPU malformed-input tests, and concurrency validation.
