// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#include "jpeg_header.h"
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

using nvimgcodec::jpeg_cuda_experimental::Probe;
using nvimgcodec::jpeg_cuda_experimental::Status;
using Bytes = std::vector<uint8_t>;

static void Require(bool condition) {
    if (!condition) { std::cerr << "header regression failed\n"; std::exit(1); }
}
static void Segment(Bytes& bytes, int marker, const Bytes& payload) {
    size_t length = payload.size() + 2;
    bytes.insert(bytes.end(), {0xff, static_cast<uint8_t>(marker), static_cast<uint8_t>(length >> 8), static_cast<uint8_t>(length)});
    bytes.insert(bytes.end(), payload.begin(), payload.end());
}
static Bytes Gray() {
    Bytes bytes{0xff, 0xd8}, table(65, 1);
    table[0] = 0;
    Segment(bytes, 0xdb, table);
    Segment(bytes, 0xc0, {8, 0, 1, 0, 1, 1, 1, 0x11, 0});
    Bytes dc(18, 0), ac(18, 0);
    dc[1] = 1;
    ac[0] = 0x10; ac[1] = 1;
    Segment(bytes, 0xc4, dc); Segment(bytes, 0xc4, ac);
    Segment(bytes, 0xda, {1, 1, 0, 0, 63, 0});
    bytes.insert(bytes.end(), {0x3f, 0xff, 0xd9});
    return bytes;
}
static size_t Find(const Bytes& bytes, int marker) {
    for (size_t i = 2; i + 1 < bytes.size(); ++i)
        if (bytes[i] == 0xff && bytes[i + 1] == marker) return i;
    std::abort();
}
int main() {
    auto good = Gray();
    auto info = Probe(good.data(), good.size());
    Require(info.status == Status::kSuccess && info.width == 1 && info.height == 1 && info.components == 1);
    Require(Probe(nullptr, 0).status == Status::kNotJpeg);
    for (size_t n = 0; n < good.size(); ++n) Require(Probe(good.data(), n).status != Status::kSuccess);
    auto bad = good;
    bad[Find(bad, 0xc4) + 5] = 3;  // oversubscribed one-bit Huffman codes
    Require(Probe(bad.data(), bad.size()).status == Status::kTables);
    bad = good; bad[Find(bad, 0xda) + 6] = 0xff;  // table selector out of range
    Require(Probe(bad.data(), bad.size()).status == Status::kTables);
    bad = good; bad[Find(bad, 0xc0) + 1] = 0xc2;
    Require(Probe(bad.data(), bad.size()).status == Status::kProgressive);
    bad = good; bad[Find(bad, 0xc0) + 11] = 0x21;
    Require(Probe(bad.data(), bad.size()).status == Status::kSampling);
    bad = good; bad[Find(bad, 0xdb) + 5] = 0;
    Require(Probe(bad.data(), bad.size()).status == Status::kTables);
    bad = good;
    size_t sos = Find(bad, 0xda);
    bad.insert(bad.begin() + sos, {0xff, 0xdd, 0, 2});  // DRI with no payload
    Require(Probe(bad.data(), bad.size()).status != Status::kSuccess);
    bad = {0xff, 0xd8, 0xff, 0xda, 0, 2, 0xff, 0xd9};
    Require(Probe(bad.data(), bad.size()).status != Status::kSuccess);
    std::mt19937 random(2026);
    for (int i = 0; i < 10000; ++i) {
        bad = good;
        for (int k = 0; k < 1 + (i % 4); ++k) bad[random() % bad.size()] = random() & 255;
        (void)Probe(bad.data(), bad.size());
    }
    std::cout << "header cases, all truncations and 10000 mutations passed\n";
}
