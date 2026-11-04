// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>

namespace nvimgcodec::jpeg_cuda_experimental {

enum class Status
{
    kSuccess = 0,
    kNotJpeg,
    kProgressive,
    kLossless,
    kArithmetic,
    kPrecision,
    kComponents,
    kMultiscan,
    kRestart,
    kSampling,
    kTables,
    kTruncated,
    kResourceLimit,
    kCorrupt
};

struct ImageInfo
{
    Status status = Status::kNotJpeg;
    int width = 0;
    int height = 0;
    int components = 0;
};

// Header/table inspection only. Successful inspection does not validate entropy data.
// Initial scope: baseline 8-bit, one interleaved scan, gray or YCbCr 4:4:4,
// no restart markers, at most 4096x4096 pixels and 8 MiB encoded data.
ImageInfo Probe(const uint8_t* encoded, size_t size);

} // namespace nvimgcodec::jpeg_cuda_experimental
