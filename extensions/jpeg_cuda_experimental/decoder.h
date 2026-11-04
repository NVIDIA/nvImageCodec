// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cuda_runtime_api.h>
#include <memory>
#include "jpeg_header.h"

namespace nvimgcodec::jpeg_cuda_experimental {

struct Output
{
    uint8_t* plane[3] = {};
    size_t row_stride[3] = {};
    int width = 0;
    int height = 0;
};

// Stream-aware callbacks use the same calling convention as nvImageCodec allocators.
// Supply both callbacks or neither. The context and streams must outlive this decoder.
struct MemoryResource
{
    void* context = nullptr;
    int (*allocate)(void*, void**, size_t, cudaStream_t) = nullptr;
    int (*release)(void*, void*, size_t, cudaStream_t) = nullptr;
};

class Decoder
{
  public:
    explicit Decoder(int device, MemoryResource device_memory = {}, MemoryResource pinned_memory = {});
    ~Decoder();
    Decoder(const Decoder&) = delete;
    Decoder& operator=(const Decoder&) = delete;

    // Enqueues GPU entropy decoding through RGB conversion. Host reads marker
    // segments, builds lookup tables, and stages input; it never decodes entropy.
    // Input bytes may be released on return. Output remains caller-owned and
    // must stay alive until Wait completes. Reuse waits for this decoder's prior image.
    // CUDA failures throw std::runtime_error after draining submitted work;
    // recoverable submission failures leave the decoder idle for reuse.
    // Header failures return a Status.
    Status Decode(const uint8_t* encoded, size_t size, const Output& output, cudaStream_t stream);

    // Waits on this decoder's completion event and checks GPU corruption flags.
    // A framework adapter must check this before reporting successful processing.
    Status Wait();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace nvimgcodec::jpeg_cuda_experimental
