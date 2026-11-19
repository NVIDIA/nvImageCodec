// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>
#include "jpeg_header.h"
#include "test_fixtures.h"

using jpeg_cuda_test::Bytes;
using jpeg_cuda_test::Gray;
using nvimgcodec::jpeg_cuda_experimental::Probe;
using nvimgcodec::jpeg_cuda_experimental::Status;

static void Require(bool condition)
{
    if (!condition) {
        std::cerr << "header regression failed\n";
        std::exit(1);
    }
}
static size_t Find(const Bytes& bytes, int marker)
{
    for (size_t i = 2; i + 1 < bytes.size(); ++i)
        if (bytes[i] == 0xff && bytes[i + 1] == marker)
            return i;
    std::abort();
}
int main()
{
    auto good = Gray();
    auto info = Probe(good.data(), good.size());
    Require(info.status == Status::kSuccess && info.width == 1 && info.height == 1 && info.components == 1);
    Require(Probe(nullptr, 0).status == Status::kNotJpeg);
    for (size_t n = 0; n < good.size(); ++n)
        Require(Probe(good.data(), n).status != Status::kSuccess);
    auto bad = good;
    bad[Find(bad, 0xc4) + 5] = 3; // oversubscribed one-bit Huffman codes
    Require(Probe(bad.data(), bad.size()).status == Status::kTables);
    bad = good;
    bad[Find(bad, 0xda) + 6] = 0xff; // table selector out of range
    Require(Probe(bad.data(), bad.size()).status == Status::kTables);
    bad = good;
    bad[Find(bad, 0xc0) + 1] = 0xc2;
    Require(Probe(bad.data(), bad.size()).status == Status::kProgressive);
    bad = good;
    bad[Find(bad, 0xc0) + 11] = 0x21;
    Require(Probe(bad.data(), bad.size()).status == Status::kSampling);
    bad = good;
    bad[Find(bad, 0xdb) + 5] = 0;
    Require(Probe(bad.data(), bad.size()).status == Status::kTables);
    bad = good;
    size_t sos = Find(bad, 0xda);
    bad.insert(bad.begin() + sos, {0xff, 0xdd, 0, 2}); // DRI with no payload
    Require(Probe(bad.data(), bad.size()).status != Status::kSuccess);
    bad = {0xff, 0xd8, 0xff, 0xda, 0, 2, 0xff, 0xd9};
    Require(Probe(bad.data(), bad.size()).status != Status::kSuccess);
    std::mt19937 random(2026);
    for (int i = 0; i < 10000; ++i) {
        bad = good;
        for (int k = 0; k < 1 + (i % 4); ++k)
            bad[random() % bad.size()] = random() & 255;
        (void)Probe(bad.data(), bad.size());
    }
    std::cout << "header cases, all truncations and 10000 mutations passed\n";
}
