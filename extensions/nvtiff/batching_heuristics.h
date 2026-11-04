// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nvtiff.h>
#include <algorithm>
#include <charconv>
#include <cstddef>
#include <sstream>
#include <string>

namespace nvtiff {

// Parse a nonnegative integer option for nvtiff_cuda_decoder. The last value wins.
// A malformed value sets the option to zero and is added to *invalid.
inline size_t parseUIntOption(const char* options, const char* name, std::string* invalid = nullptr)
{
    const std::string prefix = std::string("nvtiff_cuda_decoder:") + name + "=";
    std::istringstream input(options ? options : "");
    std::string token;
    size_t value = 0;
    while (input >> token) {
        if (token.compare(0, prefix.size(), prefix) != 0)
            continue;
        const char* begin = token.data() + prefix.size();
        const char* end = token.data() + token.size();
        auto result = std::from_chars(begin, end, value);
        if (result.ec != std::errc{} || result.ptr != end) {
            if (invalid) {
                if (!invalid->empty())
                    *invalid += ", ";
                *invalid += token;
            }
            value = 0;
        }
    }
    return value;
}

// Number of subbatches for one compatible group. This affects only performance.
inline size_t preferredSubbatchCount(
    nvtiffCompression_t compression, bool is_hamamatsu, size_t samples, size_t workers)
{
    size_t parts = 1;
    // Tuned on RTX 6000 Ada ROI workloads. Hamamatsu NDPI spends more time
    // preparing synthetic JPEGs on the CPU; LZW/Deflate favor one large call.
    if (compression == NVTIFF_COMPRESSION_JPEG)
        parts = is_hamamatsu ? 4 : 2;
    // Split only if each subbatch gets at least 16 outputs. Each native call has a setup cost.
    return std::max<size_t>(1, std::min({parts, workers, samples / 16}));
}

inline size_t subbatchSize(nvtiffCompression_t compression, bool is_hamamatsu,
    size_t samples, size_t workers, size_t max_images)
{
    const size_t parts = preferredSubbatchCount(compression, is_hamamatsu, samples, workers);
    const size_t automatic = samples / parts + (samples % parts != 0);
    return max_images ? std::min(automatic, max_images) : automatic;
}

} // namespace nvtiff
