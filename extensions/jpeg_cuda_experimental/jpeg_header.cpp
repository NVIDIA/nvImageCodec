// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <cstring>
#include "jpeg_plan.h"
namespace nvimgcodec::jpeg_cuda_experimental {
static int parse_header(const uint8_t* d, size_t n, Header& H)
{
    if (n < 4 || d[0] != 0xFF || d[1] != 0xD8)
        return NOT_JPEG;
    size_t i = 2;
    while (i + 4 <= n) {
        if (d[i] != 0xFF)
            return NOT_JPEG;
        uint8_t m = d[i + 1];
        if (m == 0xFF) {
            ++i;
            continue;
        }
        if (m == 0xD8 || (m >= 0xD0 && m <= 0xD7) || m == 0x01) {
            i += 2;
            continue;
        }
        if (m == 0xD9)
            return TRUNCATED;
        size_t L = ((size_t)d[i + 2] << 8) | d[i + 3];
        if (L < 2 || i + 2 + L > n)
            return TRUNCATED;
        const uint8_t* s = d + i + 4;
        size_t sl = L - 2;
        if (m == 0xDB) {
            size_t j = 0;
            while (j < sl) {
                int pq = s[j] >> 4, t = s[j] & 15;
                if (pq > 1)
                    return TABLES;
                ++j;
                if (t > 3 || j + (pq ? 128 : 64) > sl)
                    return TABLES;
                for (int k = 0; k < 64; ++k)
                    H.qt[t][k] = pq ? (uint16_t)((s[j + 2 * k] << 8) | s[j + 2 * k + 1]) : s[j + k];
                j += pq ? 128 : 64;
                H.qt_ok[t] = true;
            }
        } else if (m == 0xC4) {
            size_t j = 0;
            while (j + 17 <= sl) {
                int tc = s[j] >> 4, th = s[j] & 15;
                if (tc > 1 || th > 3)
                    return TABLES;
                int tot = 0;
                for (int k = 0; k < 16; ++k) {
                    H.hcount[tc][th][k] = s[j + 1 + k];
                    tot += s[j + 1 + k];
                }
                if (tot > 256 || j + 17 + tot > sl)
                    return TABLES;
                std::memset(H.hsym[tc][th], 0, 256);
                std::memcpy(H.hsym[tc][th], s + j + 17, tot);
                H.ht_ok[tc][th] = true;
                j += 17 + tot;
            }
            if (j != sl)
                return TABLES;
        } else if ((m >= 0xC0 && m <= 0xCF) && m != 0xC4 && m != 0xC8 && m != 0xCC) {
            if (sl < 6)
                return TRUNCATED;
            if (H.sof != -1)
                return COMPONENTS;
            H.sof = m;
            H.precision = s[0];
            H.height = (s[1] << 8) | s[2];
            H.width = (s[3] << 8) | s[4];
            H.ncomp = s[5];
            if (H.ncomp < 1 || H.ncomp > MAXC || sl < 6 + 3 * (size_t)H.ncomp)
                return COMPONENTS;
            for (int c = 0; c < H.ncomp; ++c) {
                H.cid[c] = s[6 + 3 * c];
                H.h[c] = s[7 + 3 * c] >> 4;
                H.v[c] = s[7 + 3 * c] & 15;
                H.tq[c] = s[8 + 3 * c];
            }
        } else if (m == 0xDD) {
            if (sl != 2)
                return TRUNCATED;
            H.dri = (s[0] << 8) | s[1];
        } else if (m == 0xEE) {
            if (sl >= 12 && std::memcmp(s, "Adobe", 5) == 0)
                H.adobe = s[11];
        } else if (m == 0xDA) {
            if (sl < 1)
                return TRUNCATED;
            H.scan_n = s[0];
            if (H.scan_n < 1 || H.scan_n > MAXC || sl < 4 + 2 * (size_t)H.scan_n)
                return COMPONENTS;
            for (int k = 0; k < H.scan_n; ++k) {
                H.scan_c[k] = s[1 + 2 * k];
                H.scan_td[k] = s[2 + 2 * k] >> 4;
                H.scan_ta[k] = s[2 + 2 * k] & 15;
                if (H.scan_td[k] > 3 || H.scan_ta[k] > 3)
                    return TABLES;
            }
            H.ss = s[1 + 2 * H.scan_n];
            H.se = s[2 + 2 * H.scan_n];
            H.ah = s[3 + 2 * H.scan_n] >> 4;
            H.al = s[3 + 2 * H.scan_n] & 15;
            H.scan_start = i + 2 + L;
            return OK;
        }
        i += 2 + L;
    }
    return TRUNCATED;
}

static bool build_huff(const uint8_t* counts, const uint8_t* syms, Params& P, int t)
{
    std::memset(P.lut[t], 0, sizeof(P.lut[t]));
    std::memset(P.vals[t], 0, sizeof(P.vals[t]));
    int code = 0, k = 0;
    for (int l = 1; l <= 16; ++l) {
        int nl = counts[l - 1];
        if (code + nl > (1 << l) || k + nl > 256)
            return false;
        P.valoff[t][l] = k - code;
        for (int j = 0; j < nl; ++j) {
            int sym = syms[k];
            if ((t % 2 == 0 && sym > 11) || (t % 2 == 1 && ((sym & 15) > 10 || ((sym & 15) == 0 && sym != 0 && sym != 0xf0))))
                return false;
            if (l <= LUTB) {
                int lo = code << (LUTB - l), hi = (code + 1) << (LUTB - l);
                for (int x = lo; x < hi; ++x)
                    P.lut[t][x] = (uint16_t)((l << 8) | syms[k]);
            }
            P.vals[t][k] = syms[k];
            ++code;
            ++k;
        }
        P.maxcode[t][l] = nl ? code - 1 : -1;
        if (code > (1 << l))
            return false;
        code <<= 1;
    }
    P.maxcode[t][0] = -1;
    P.maxcode[t][17] = 0x7fffffff;
    return true;
}

// Fills Params; returns a Kind code.
static int plan(const Header& H, size_t file_bytes, int chunk_bits, int overlap_bits, Params& P)
{
    if (H.sof == 0xC2 || H.sof == 0xC6 || H.sof == 0xCA || H.sof == 0xCE)
        return PROGRESSIVE;
    if (H.sof == 0xC3 || H.sof == 0xC7 || H.sof == 0xCB || H.sof == 0xCF)
        return LOSSLESS;
    if (H.sof >= 0xC9)
        return ARITHMETIC;
    if (H.sof != 0xC0 && H.sof != 0xC1)
        return NOT_JPEG;
    if (H.precision != 8)
        return PRECISION;
    if (H.ncomp != 1 && H.ncomp != 3 && H.ncomp != 4)
        return COMPONENTS;
    if (H.scan_n != H.ncomp)
        return MULTISCAN;
    if (H.ss != 0 || H.se != 63 || H.ah != 0 || H.al != 0)
        return MULTISCAN;
    if (H.dri != 0)
        return RESTART;
    if (H.width < 1 || H.height < 1)
        return NOT_JPEG;
    for (int c = 0; c < H.ncomp; ++c)
        for (int j = 0; j < c; ++j)
            if (H.cid[c] == H.cid[j])
                return COMPONENTS;
    std::memset(&P, 0, sizeof(P));
    P.ncomp = H.ncomp;
    P.width = H.width;
    P.height = H.height;
    int hmax = 1, vmax = 1;
    for (int c = 0; c < H.ncomp; ++c) {
        if (H.h[c] < 1 || H.h[c] > 4 || H.v[c] < 1 || H.v[c] > 4)
            return SAMPLING;
        hmax = std::max(hmax, H.h[c]);
        vmax = std::max(vmax, H.v[c]);
    }
    for (int c = 0; c < H.ncomp; ++c)
        if (hmax % H.h[c] || vmax % H.v[c])
            return SAMPLING;
    P.hmax = hmax;
    P.vmax = vmax;
    int order[MAXC];
    for (int k = 0; k < H.scan_n; ++k) {
        order[k] = -1;
        for (int c = 0; c < H.ncomp; ++c)
            if (H.cid[c] == H.scan_c[k])
                order[k] = c;
        if (order[k] < 0)
            return COMPONENTS;
        for (int j = 0; j < k; ++j)
            if (order[k] == order[j])
                return COMPONENTS;
    }
    for (int k = 0; k < H.scan_n; ++k) {
        int c = order[k];
        if (!H.ht_ok[0][H.scan_td[k]] || !H.ht_ok[1][H.scan_ta[k]] || H.tq[c] > 3 || !H.qt_ok[H.tq[c]])
            return TABLES;
        if (!build_huff(H.hcount[0][H.scan_td[k]], H.hsym[0][H.scan_td[k]], P, 2 * c))
            return TABLES;
        if (!build_huff(H.hcount[1][H.scan_ta[k]], H.hsym[1][H.scan_ta[k]], P, 2 * c + 1))
            return TABLES;
        for (int j = 0; j < 64; ++j)
            P.qt[c][j] = H.qt[H.tq[c]][j];
    }
    P.interleaved = H.scan_n > 1 ? 1 : 0;
    int off = 0;
    if (P.interleaved) {
        P.mcux = (H.width + 8 * hmax - 1) / (8 * hmax);
        P.mcuy = (H.height + 8 * vmax - 1) / (8 * vmax);
        int B = 0;
        for (int k = 0; k < H.scan_n; ++k) {
            int c = order[k];
            for (int dy = 0; dy < H.v[c]; ++dy)
                for (int dx = 0; dx < H.h[c]; ++dx) {
                    if (B >= MAXB)
                        return SAMPLING;
                    P.blk_comp[B] = c;
                    P.blk_dy[B] = dy;
                    P.blk_dx[B] = dx;
                    ++B;
                }
        }
        P.B = B;
        for (int c = 0; c < H.ncomp; ++c) {
            P.comp_h[c] = H.h[c];
            P.comp_v[c] = H.v[c];
            P.comp_bw[c] = P.mcux * H.h[c];
            P.comp_bh[c] = P.mcuy * H.v[c];
            P.comp_off[c] = off;
            P.comp_nblk[c] = P.comp_bw[c] * P.comp_bh[c];
            off += P.comp_nblk[c];
        }
        P.total_blocks = P.mcux * P.mcuy * B;
    } else {
        int c = 0;
        int cw = (H.width * H.h[c] + hmax - 1) / hmax, ch = (H.height * H.v[c] + vmax - 1) / vmax;
        P.B = 1;
        P.blk_comp[0] = 0;
        P.comp_h[0] = 1;
        P.comp_v[0] = 1;
        P.comp_bw[0] = (cw + 7) / 8;
        P.comp_bh[0] = (ch + 7) / 8;
        P.comp_off[0] = 0;
        P.comp_nblk[0] = P.comp_bw[0] * P.comp_bh[0];
        P.mcux = P.comp_bw[0];
        P.mcuy = P.comp_bh[0];
        P.total_blocks = P.comp_nblk[0];
        P.hmax = P.vmax = 1;
    }
    int64_t poff = 0;
    for (int c = 0; c < H.ncomp; ++c) {
        P.plane_off[c] = poff;
        poff += (int64_t)P.comp_bw[c] * 8 * P.comp_bh[c] * 8;
    }
    P.raw_bytes = (int32_t)(file_bytes - H.scan_start);
    P.chunk_bits = chunk_bits;
    P.overlap_bits = overlap_bits;
    P.list_cap = chunk_bits / 2 + 2; // a block takes at least 2 bits
    P.nchunks = (int32_t)(((int64_t)P.raw_bytes * 8 + chunk_bits - 1) / chunk_bits);
    if (P.nchunks < 1)
        P.nchunks = 1;
    P.out_mode = H.ncomp == 1 ? 1 : (H.ncomp == 3 ? 0 : 2);
    return OK;
}

int MakePlan(const uint8_t* encoded, size_t size, Params& params, size_t& scan_start)
{
    if (!encoded || size < 4)
        return NOT_JPEG;
    if (size > 8u * 1024 * 1024)
        return static_cast<int>(Status::kResourceLimit);
    Header header;
    int result = parse_header(encoded, size, header);
    if (result != OK)
        return result;
    if (header.sof != 0xc0)
        return header.sof == 0xc2 ? PROGRESSIVE : NOT_JPEG;
    if (header.width > 4096 || header.height > 4096)
        return static_cast<int>(Status::kResourceLimit);
    if (header.ncomp != 1 && header.ncomp != 3)
        return COMPONENTS;
    if (header.ncomp == 3 && (header.adobe == 0 || header.adobe > 1))
        return COMPONENTS;
    for (int c = 0; c < header.ncomp; ++c) {
        if (header.h[c] != 1 || header.v[c] != 1)
            return SAMPLING;
        if (header.ncomp == 3 && header.cid[c] != c + 1)
            return COMPONENTS;
    }
    if (header.scan_start >= size || encoded[size - 2] != 0xff || encoded[size - 1] != 0xd9)
        return TRUNCATED;
    result = plan(header, size, 1024, 3072, params);
    if (result != OK)
        return result;
    for (int c = 0; c < header.ncomp; ++c)
        for (int i = 0; i < 64; ++i)
            if (!params.qt[c][i])
                return TABLES;
    scan_start = header.scan_start;
    return OK;
}

ImageInfo Probe(const uint8_t* encoded, size_t size)
{
    Params params{};
    size_t scan_start = 0;
    int kind = MakePlan(encoded, size, params, scan_start);
    return {static_cast<Status>(kind), params.width, params.height, params.ncomp};
}

} // namespace nvimgcodec::jpeg_cuda_experimental
