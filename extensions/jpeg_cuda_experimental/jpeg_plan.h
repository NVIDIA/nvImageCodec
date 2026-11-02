// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#pragma once
#include "jpeg_header.h"
#include <cstdint>
namespace nvimgcodec::jpeg_cuda_experimental {
constexpr int MAXC = 4;          // components
constexpr int MAXB = 10;         // blocks per MCU (JPEG limit)
constexpr int LUTB = 11;         // bits resolved by one table lookup
constexpr int NT = 2 * MAXC;     // Huffman tables: [component][dc, ac]

struct alignas(16) Params {      // uploaded with the entropy-coded bytes, one per image
  uint16_t lut[NT][1 << LUTB];   // first LUTB bits -> (code length << 8) | symbol; 0: longer code (or invalid)
  int32_t maxcode[NT][18];       // canonical decoding of longer codes
  int32_t valoff[NT][18];
  uint8_t vals[NT][256];
  uint16_t qt[MAXC][64];         // per component, zigzag order
  int64_t plane_off[MAXC];       // byte offset of component c's padded plane
  int32_t ncomp, B, interleaved;
  int32_t blk_comp[MAXB], blk_dy[MAXB], blk_dx[MAXB];
  int32_t comp_h[MAXC], comp_v[MAXC];
  int32_t comp_bw[MAXC], comp_bh[MAXC];   // stored block grid per component
  int32_t comp_off[MAXC];                 // first block of component c in the coefficient array
  int32_t comp_nblk[MAXC];                // blocks of component c in the scan
  int32_t mcux, mcuy;
  int32_t total_blocks;                   // blocks in the scan, in decode order
  int32_t width, height, hmax, vmax;
  int32_t raw_bytes;                      // uploaded entropy-coded bytes (scan start .. end of file)
  int32_t chunk_bits, nchunks;            // nchunks: upper bound from raw_bytes
  int32_t overlap_bits;                   // warm-up of the speculative decode before each chunk
  int32_t list_cap;                       // block-start records per chunk
  int32_t out_mode;                       // 0: RGB from YCbCr, 1: gray -> RGB, 2: raw planes (4 components)
};

struct Header {
  int sof = -1, precision = 0, width = 0, height = 0, ncomp = 0;
  int cid[MAXC], h[MAXC], v[MAXC], tq[MAXC];
  uint16_t qt[4][64];
  bool qt_ok[4] = {false, false, false, false};
  uint8_t hcount[2][4][16];
  uint8_t hsym[2][4][256];
  bool ht_ok[2][4] = {{false, false, false, false}, {false, false, false, false}};
  int dri = 0;
  int scan_n = 0, scan_c[MAXC], scan_td[MAXC], scan_ta[MAXC], ss = 0, se = 63, ah = 0, al = 0;
  size_t scan_start = 0;
  int adobe = -1;
};

// kind codes returned to Python
enum Kind { OK = 0, NOT_JPEG = 1, PROGRESSIVE = 2, LOSSLESS = 3, ARITHMETIC = 4, PRECISION = 5, COMPONENTS = 6,
            MULTISCAN = 7, RESTART = 8, SAMPLING = 9, TABLES = 10, TRUNCATED = 11 };


int MakePlan(const uint8_t*, size_t, Params&, size_t&);
}
