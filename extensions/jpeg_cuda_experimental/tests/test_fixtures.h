// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace jpeg_cuda_test {
using Bytes = std::vector<uint8_t>;

inline void Segment(Bytes& bytes, int marker, const Bytes& payload)
{
    size_t length = payload.size() + 2;
    bytes.insert(bytes.end(), {0xff, static_cast<uint8_t>(marker), static_cast<uint8_t>(length >> 8), static_cast<uint8_t>(length)});
    bytes.insert(bytes.end(), payload.begin(), payload.end());
}

inline Bytes Gray(uint16_t width = 1, uint16_t height = 1)
{
    Bytes bytes{0xff, 0xd8}, table(65, 1);
    table[0] = 0;
    Segment(bytes, 0xdb, table);
    Segment(bytes, 0xc0,
        {8, static_cast<uint8_t>(height >> 8), static_cast<uint8_t>(height), static_cast<uint8_t>(width >> 8), static_cast<uint8_t>(width),
            1, 1, 0x11, 0});
    Bytes dc(18, 0), ac(18, 0);
    dc[1] = 1;
    ac[0] = 0x10;
    ac[1] = 1;
    Segment(bytes, 0xc4, dc);
    Segment(bytes, 0xc4, ac);
    Segment(bytes, 0xda, {1, 1, 0, 0, 63, 0});
    bytes.insert(bytes.end(), {0x3f, 0xff, 0xd9});
    return bytes;
}
} // namespace jpeg_cuda_test
