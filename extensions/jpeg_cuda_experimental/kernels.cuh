// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
// IDCT butterfly derived from NVIDIA cuda-samples dct8x8.
// See LICENSE.cuda-samples for the retained BSD-3-Clause notice.
// shared-memory copy used by the decoding kernels: tables of the frame's components + layout
struct alignas(16) Shared
{
    uint16_t lut[NT][1 << LUTB];
    int32_t maxcode[NT][18];
    int32_t valoff[NT][18];
    uint8_t vals[NT][256];
    int32_t B, mcux, total_blocks, chunk_bits, overlap_bits, list_cap, interleaved, nchunks;
    int32_t blk_comp[MAXB], blk_dy[MAXB], blk_dx[MAXB];
    int32_t comp_h[MAXC], comp_v[MAXC], comp_bw[MAXC], comp_off[MAXC];
};

__device__ void load_shared(const Params* gp, Shared* sp, int ntables)
{
    const int lut_words = ntables * (1 << LUTB) * 2 / 16; // the first 2*ncomp lookup tables, int4 granularity
    const int4* src = reinterpret_cast<const int4*>(gp->lut);
    int4* dst = reinterpret_cast<int4*>(sp->lut);
    for (int i = threadIdx.x; i < lut_words; i += blockDim.x)
        dst[i] = src[i];
    for (int i = threadIdx.x; i < NT * 18; i += blockDim.x) {
        (&sp->maxcode[0][0])[i] = (&gp->maxcode[0][0])[i];
        (&sp->valoff[0][0])[i] = (&gp->valoff[0][0])[i];
    }
    for (int i = threadIdx.x; i < NT * 256 / 4; i += blockDim.x)
        reinterpret_cast<uint32_t*>(&sp->vals[0][0])[i] = reinterpret_cast<const uint32_t*>(&gp->vals[0][0])[i];
    if (threadIdx.x < MAXB) {
        sp->blk_comp[threadIdx.x] = gp->blk_comp[threadIdx.x];
        sp->blk_dy[threadIdx.x] = gp->blk_dy[threadIdx.x];
        sp->blk_dx[threadIdx.x] = gp->blk_dx[threadIdx.x];
    }
    if (threadIdx.x < MAXC) {
        sp->comp_h[threadIdx.x] = gp->comp_h[threadIdx.x];
        sp->comp_v[threadIdx.x] = gp->comp_v[threadIdx.x];
        sp->comp_bw[threadIdx.x] = gp->comp_bw[threadIdx.x];
        sp->comp_off[threadIdx.x] = gp->comp_off[threadIdx.x];
    }
    if (threadIdx.x == 0) {
        sp->B = gp->B;
        sp->mcux = gp->mcux;
        sp->total_blocks = gp->total_blocks;
        sp->chunk_bits = gp->chunk_bits;
        sp->overlap_bits = gp->overlap_bits;
        sp->list_cap = gp->list_cap;
        sp->interleaved = gp->interleaved;
        sp->nchunks = gp->nchunks;
    }
}

// ---------------------------------------------------------------------------------------------------- bit access
// The destuffed stream is kept as big-endian 32-bit words (bw), zero padded. A decoding block stages the words it
// needs in shared memory; window(p) returns 64 bits whose top bits are the stream at bit p (>= 33 valid bits).
struct Bits
{
    const uint32_t* w; // words; w[0] is word index `base`
    uint32_t base;
    __device__ __forceinline__ uint64_t window(uint32_t p) const
    {
        uint32_t q = (p >> 5) - base;
        uint64_t v = ((uint64_t)w[q] << 32) | w[q + 1];
        return v << (p & 31);
    }
};

__device__ __forceinline__ Bits stage(const uint32_t* bw, uint32_t* sw, uint32_t first_bit, uint32_t last_bit)
{
    uint32_t w0 = first_bit >> 5, w1 = (last_bit >> 5) + 3;
    for (uint32_t i = w0 + threadIdx.x; i < w1; i += blockDim.x)
        sw[i - w0] = bw[i];
    __syncthreads();
    return Bits{sw, w0};
}

// Huffman symbol at the top of `v` with table t: (len << 8) | sym, or 0 if invalid.
__device__ __forceinline__ uint32_t huff(const Shared& S, int t, uint64_t v)
{
    uint32_t e = S.lut[t][(uint32_t)(v >> (64 - LUTB))];
    if (e)
        return e;
    uint32_t w16 = (uint32_t)(v >> 48);
#pragma unroll 1
    for (int l = LUTB + 1; l <= 16; ++l) {
        int code = (int)(w16 >> (16 - l));
        if (code <= S.maxcode[t][l]) {
            int index = S.valoff[t][l] + code;
            return index >= 0 && index < 256 ? ((uint32_t)l << 8) | S.vals[t][index] : 0;
        }
    }
    return 0;
}

// One parse step (no values). State: p (bit), z (zigzag index, 0 = DC next), b (block in MCU), c (its component).
// Returns false on an invalid code (then p advances by one bit and z = 0). *ended = a block was completed.
__device__ __forceinline__ bool pstep(const Shared& S, const Bits& bits, uint32_t& p, int& z, int& b, int& c, bool& ended)
{
    uint64_t v = bits.window(p);
    uint32_t e = huff(S, 2 * c + (z != 0), v);
    int len = (int)(e >> 8), sym = (int)(e & 0xFF);
    int size = z ? (sym & 15) : sym;
    int zinc = z ? (sym == 0 ? 64 : (sym == 0xF0 ? 16 : (sym >> 4) + 1)) : 1;
    bool ok = e != 0 && size <= 15;
    p += ok ? (uint32_t)(len + size) : 1u;
    z = ok ? z + zinc : 0;
    ended = z >= 64;
    if (ended) {
        z = 0;
        b = (b + 1 == S.B) ? 0 : b + 1;
        c = S.blk_comp[b];
    }
    return ok;
}

// ---------------------------------------------------------------------------------------------------- kernels
__global__ void k_find_end(const uint8_t* raw, int n, int* end)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    if (raw[i] == 0xFF) {
        uint8_t nx = (i + 1 < n) ? raw[i + 1] : 0xD9;
        if (nx != 0x00)
            atomicMin(end, i);
    }
}

__global__ void k_mark(const uint8_t* raw, int n, const int* end, uint8_t* keep)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    int e = *end;
    bool stuffed = i > 0 && raw[i] == 0x00 && raw[i - 1] == 0xFF;
    keep[i] = (i < e && !stuffed) ? 1 : 0;
}

__global__ void k_words(const uint8_t* bytes, int nwords, uint32_t* bw)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nwords)
        return;
    bw[i] = __byte_perm(reinterpret_cast<const uint32_t*>(bytes)[i], 0, 0x0123);
}

__device__ __forceinline__ int nchunks_used(int nchunks, int chunk_bits, uint32_t L)
{
    return min(nchunks, (int)((L + chunk_bits - 1) / chunk_bits));
}

// P1: speculative decode of every chunk, one thread per chunk. The decode for chunk i starts overlap_bits before the
// chunk with (z=0, b=0) (chunk 0: at bit 0 in the true state), so by the chunk start it has almost always fallen onto
// the true trajectory. Inside the chunk it records every block start (bit, b, block counter) in its list; E[i] = exit
// state (first codeword start at or after the chunk end) with the block counter. Counters only enter as differences.
__global__ void k_spec(const Params* gp, const uint32_t* bw, const int* nbytes, int2* lists, int* nlist, int4* E)
{
    extern __shared__ int4 smem_[];
    Shared& S = *reinterpret_cast<Shared*>(smem_);
    uint32_t* sw = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_) + sizeof(Shared));
    load_shared(gp, &S, 2 * gp->ncomp);
    const uint32_t L = (uint32_t)(*nbytes) * 8u;
    const uint32_t s = gp->chunk_bits, o = gp->overlap_bits;
    const int first = blockIdx.x * blockDim.x;
    uint32_t lo = (uint32_t)first * s;
    lo = lo > o ? lo - o : 0u;
    Bits bits = stage(bw, sw, lo, min((uint32_t)(first + blockDim.x) * s, L) + 64);
    int i = first + threadIdx.x;
    if (i >= gp->nchunks)
        return;
    uint32_t p0 = (uint32_t)i * s;
    if (p0 >= L) {
        E[i] = make_int4(-1, 0, 0, 0);
        nlist[i] = 0;
        return;
    }
    uint32_t end = min(p0 + s, L);
    uint32_t p = (i == 0) ? 0u : (p0 > o ? p0 - o : 0u);
    int z = 0, b = 0, c = S.blk_comp[0], cnt = 0, n = 0;
    int2* lst = lists + (size_t)i * S.list_cap;
    const int cap = S.list_cap;
    bool ended;
    while (p < end) {
        if (z == 0 && p >= p0 && n < cap)
            lst[n++] = make_int2((int)p, (b << 20) | cnt);
        pstep(S, bits, p, z, b, c, ended); // an invalid code restarts one bit later with z = 0
        cnt += ended;
    }
    E[i] = make_int4((int)p, z, b, cnt);
    nlist[i] = n;
}

// Decodes chunk i from state `in` (.w = valid) until it reaches a block start recorded on the chunk's speculative
// trajectory in the same state (the rest is known from E[i]) or the chunk end. Returns the exit (p, z, b, 1), or
// (-1, 0, 0, 0) after an invalid code; nb = blocks completed between `in` and the exit.
__device__ int4 run_chunk(
    const Shared& S, const Bits& bits, const int2* lists, const int* nlist, const int4* E, int i, uint32_t L, int4 in, int& nb)
{
    nb = 0;
    if (!in.w)
        return make_int4(-1, 0, 0, 0);
    const uint32_t chunk_end = min(((uint32_t)i + 1) * (uint32_t)S.chunk_bits, L);
    uint32_t p = (uint32_t)in.x;
    int z = in.y, b = in.z, c = S.blk_comp[b], cnt = 0;
    const int2* lst = lists + (size_t)i * S.list_cap;
    const int n = nlist[i];
    int k = 0;
    bool ended;
    while (true) {
        if (p >= chunk_end) {
            nb = cnt;
            return make_int4((int)p, z, b, 1);
        }
        if (z == 0) {
            while (k < n && (uint32_t)lst[k].x < p)
                ++k;
            if (k < n && (uint32_t)lst[k].x == p && (lst[k].y >> 20) == b) {
                int4 e = E[i];
                nb = cnt + e.w - (lst[k].y & 0xFFFFF);
                return make_int4(e.x, e.y, e.z, 1);
            }
        }
        if (!pstep(S, bits, p, z, b, c, ended)) {
            nb = cnt;
            return make_int4(-1, 0, 0, 0);
        }
        cnt += ended;
    }
}

__device__ __forceinline__ int4 spec_input(const int4* E, int i)
{ // E[i-1] as an input state
    if (i == 0)
        return make_int4(0, 0, 0, 1);
    int4 e = E[i - 1];
    return e.x >= 0 ? make_int4(e.x, e.y, e.z, 1) : make_int4(-1, 0, 0, 0);
}

__device__ __forceinline__ bool same(int4 a, int4 b)
{
    return a.w && b.w && a.x == b.x && a.y == b.y && a.z == b.z;
}

// P2a (mode 0): every chunk decoded from its speculative input E[i-1]; correct whenever E[i-1] is the true entry.
// P2b (mode 1): chunks whose speculative input differs from the exit their predecessor reached in P2a, decoded from that
// exit (the true entry whenever the predecessor itself had the true input).
__global__ void k_relink(const Params* gp, const uint32_t* bw, const int* nbytes, const int2* lists, const int* nlist, const int4* E,
    int4* R1, int* R1nb, int4* R2, int* R2nb, int mode)
{
    extern __shared__ int4 smem_[];
    Shared& S = *reinterpret_cast<Shared*>(smem_);
    uint32_t* sw = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_) + sizeof(Shared));
    load_shared(gp, &S, 2 * gp->ncomp);
    const uint32_t L = (uint32_t)(*nbytes) * 8u;
    const uint32_t s = gp->chunk_bits;
    const int first = blockIdx.x * blockDim.x;
    Bits bits = stage(bw, sw, (uint32_t)first * s, min((uint32_t)(first + blockDim.x) * s, L) + 64);
    int i = first + threadIdx.x;
    if (i >= gp->nchunks)
        return;
    const int nch = nchunks_used(gp->nchunks, gp->chunk_bits, L);
    if (mode == 0) {
        if (i >= nch) {
            R1[i] = make_int4(-1, 0, 0, 0);
            R1nb[i] = 0;
            return;
        }
        int nb;
        R1[i] = run_chunk(S, bits, lists, nlist, E, i, L, spec_input(E, i), nb);
        R1nb[i] = nb;
    } else {
        if (i == 0 || i >= nch)
            return;
        int4 a = R1[i - 1];
        if (same(a, spec_input(E, i)))
            return;
        int nb;
        R2[i] = run_chunk(S, bits, lists, nlist, E, i, L, a, nb);
        R2nb[i] = nb;
    }
}

// P2c (one thread block): the true entry T[i] of every chunk and G[i], the decode-order index of the block in progress
// there. Chunk i is consistent when its speculative input is its true entry; consistency propagates through chunks
// whose P2a exit equals the next speculative input (bitmask `ok`). Thread 0 walks only the breaks: an isolated break
// takes P2b's result, consecutive breaks are decoded sequentially (from global memory).
__global__ void k_walk(const Params* gp, const uint32_t* bw, const int* nbytes, const int2* lists, const int* nlist, const int4* E,
    const int4* R1, const int* R1nb, const int4* R2, const int* R2nb, int4* fixT, int* fixNb, int4* T, int* G, int* stat)
{
    extern __shared__ int4 smem_[];
    Shared& S = *reinterpret_cast<Shared*>(smem_);
    uint32_t* ok = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_) + sizeof(Shared));
    load_shared(gp, &S, 2 * gp->ncomp);
    __shared__ int warp_sums[32];
    __shared__ int invalid_from;
    const uint32_t L = (uint32_t)(*nbytes) * 8u;
    const int nch = nchunks_used(gp->nchunks, gp->chunk_bits, L);
    const int nw = (nch + 31) / 32;
    for (int base = 0; base < nw * 32; base += blockDim.x) {
        int i = base + threadIdx.x;
        bool good = i >= 1 && i < nch && same(R1[i - 1], spec_input(E, i));
        unsigned bal = __ballot_sync(0xffffffffu, good);
        if ((threadIdx.x & 31) == 0 && (i >> 5) < nw)
            ok[i >> 5] = bal;
    }
    if (threadIdx.x == 0)
        invalid_from = nch;
    __syncthreads();
    if (threadIdx.x == 0) {
        int isolated = 0, sequential = 0;
        int i = 1;
        bool prev_fixed = false;
        int4 prevX = R1[0];
        Bits gbits{bw, 0};
        while (i < nch) {
            if (!prev_fixed) {
                int w = i >> 5;
                uint32_t m = ~ok[w] & (0xffffffffu << (i & 31));
                while (!m && ++w < nw)
                    m = ~ok[w];
                if (w >= nw)
                    break;
                i = (w << 5) + __ffs(m) - 1;
                if (i >= nch)
                    break;
                prevX = R1[i - 1];
                if (!prevX.w) {
                    invalid_from = i;
                    break;
                }
                fixT[i] = prevX;
                fixNb[i] = R2nb[i];
                prevX = R2[i];
                prev_fixed = true;
                isolated++;
                ++i;
                continue;
            }
            if (!prevX.w) {
                invalid_from = i;
                break;
            }
            if (same(prevX, spec_input(E, i))) { // chunk i is consistent again
                prev_fixed = false;
                ++i;
                continue;
            }
            int nb;
            int4 x = run_chunk(S, gbits, lists, nlist, E, i, L, prevX, nb);
            fixT[i] = prevX;
            fixNb[i] = nb;
            prevX = x;
            sequential++;
            ++i;
        }
        *stat = isolated + (sequential << 16);
    }
    __syncthreads();
    const int lim = invalid_from;
    const int per = (nch + blockDim.x - 1) / blockDim.x;
    const int lo = threadIdx.x * per, hi = min(lo + per, nch);
    int sum = 0;
    for (int i = lo; i < hi; ++i) {
        int nb = 0;
        if (i < lim)
            nb = fixNb[i] >= 0 ? fixNb[i] : R1nb[i];
        sum += nb;
    }
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = sum;
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o)
            x += y;
    }
    if (lane == 31)
        warp_sums[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int nwarps = (blockDim.x + 31) >> 5;
        int y = lane < nwarps ? warp_sums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            int t2 = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= o)
                y += t2;
        }
        warp_sums[lane] = y;
    }
    __syncthreads();
    int base = x - sum + (wid > 0 ? warp_sums[wid - 1] : 0);
    for (int i = lo; i < hi; ++i) {
        G[i] = base;
        if (i >= lim) {
            T[i] = make_int4(-1, 0, 0, 0);
            continue;
        }
        bool fixed = fixNb[i] >= 0;
        base += fixed ? fixNb[i] : R1nb[i];
        T[i] = fixed ? fixT[i] : spec_input(E, i);
    }
    for (int i = nch + threadIdx.x; i < gp->nchunks; i += blockDim.x)
        T[i] = make_int4(-1, 0, 0, 0);
}

__constant__ int ZZ[64] = {0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6, 7, 14, 21,
    28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31, 39, 46, 53, 60, 61, 54, 47, 55, 62, 63};

// P3: decode every chunk from its true entry to the next chunk's entry and write coefficients (DC as differences).
__global__ void k_final(const Params* gp, const uint32_t* bw, const int* nbytes, const int4* T, const int* G, int16_t* coef, int* status)
{
    extern __shared__ int4 smem_[];
    Shared& S = *reinterpret_cast<Shared*>(smem_);
    uint32_t* sw = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(smem_) + sizeof(Shared));
    load_shared(gp, &S, 2 * gp->ncomp);
    const uint32_t L = (uint32_t)(*nbytes) * 8u;
    const uint32_t s = gp->chunk_bits;
    const int first = blockIdx.x * blockDim.x;
    Bits bits = stage(bw, sw, (uint32_t)first * s, min((uint32_t)(first + blockDim.x) * s, L) + 64);
    int i = first + threadIdx.x;
    if (i >= gp->nchunks)
        return;
    int4 t = T[i];
    if (!t.w)
        return;
    int g = G[i];
    if (g >= S.total_blocks)
        return;
    uint32_t stop = L;
    if (i + 1 < gp->nchunks && T[i + 1].w)
        stop = min(stop, (uint32_t)T[i + 1].x);
    uint32_t p = (uint32_t)t.x;
    int z = t.y, b = t.z, c = S.blk_comp[b];
    int m = S.interleaved ? g / S.B : g; // block position from g once, then incremental
    int my = m / S.mcux, mx = m - my * S.mcux;
    auto blk_ptr = [&]() -> int16_t* {
        int cc = S.interleaved ? S.blk_comp[b] : 0;
        int by = S.interleaved ? my * S.comp_v[cc] + S.blk_dy[b] : my;
        int bx = S.interleaved ? mx * S.comp_h[cc] + S.blk_dx[b] : mx;
        return coef + ((size_t)S.comp_off[cc] + (size_t)by * S.comp_bw[cc] + bx) * 64;
    };
    int16_t* blkp = blk_ptr();
    while (p < stop) {
        uint64_t v = bits.window(p);
        uint32_t e = huff(S, 2 * c + (z != 0), v);
        int len = (int)(e >> 8), sym = (int)(e & 0xFF);
        int size = z ? (sym & 15) : sym;
        if (!e || size > 15) {
            atomicOr(status, 1);
            return;
        }
        uint64_t a = v << len;
        uint32_t u = size ? (uint32_t)(a >> (64 - size)) : 0u;
        int val = (size && u < (1u << (size - 1))) ? (int)u - (1 << size) + 1 : (int)u;
        p += (uint32_t)(len + size);
        if (z == 0) {
            blkp[0] = (int16_t)val;
            z = 1;
        } else {
            int run = sym >> 4;
            if (size == 0) {
                z = (run == 15) ? z + 16 : 64;
            } else {
                z += run;
                if (z < 64)
                    blkp[z] = (int16_t)val;
                else
                    atomicOr(status, 4);
                z += 1;
            }
        }
        if (z >= 64) {
            z = 0;
            if (++g >= S.total_blocks) {
                atomicOr(status, 8); // the scan is complete
                return;
            }
            if (S.interleaved) {
                if (++b == S.B) {
                    b = 0;
                    if (++mx == S.mcux) {
                        mx = 0;
                        ++my;
                    }
                }
            } else if (++mx == S.mcux) {
                mx = 0;
                ++my;
            }
            c = S.blk_comp[b];
            blkp = blk_ptr();
        }
    }
}

// DC prediction: per component, prefix sum of the DC differences in decode order (one thread block per component).
__global__ void k_dcscan(const Params* gp, int16_t* coef, int* status)
{
    __shared__ int warp_sums[32];
    const Params& P = *gp;
    const int c = blockIdx.x;
    if (c >= P.ncomp)
        return;
    if (c == 0 && threadIdx.x == 0) {
        if (!(*status & 8))
            atomicOr(status, 2); // truncated: the last block was never reached
    }
    const int n = P.comp_nblk[c];
    const int hc = P.interleaved ? P.comp_h[c] : 1, vc = P.interleaved ? P.comp_v[c] : 1, bc = hc * vc;
    const int bw = P.comp_bw[c], off = P.comp_off[c], mcux = P.mcux, inter = P.interleaved;
    auto pos = [&](int k) -> int { // k-th block of component c in decode order -> coefficient index
        if (!inter)
            return off + k;
        int m = k / bc, j = k - m * bc;
        int my = m / mcux, mx = m - my * mcux;
        int by = my * vc + j / hc, bx = mx * hc + (j % hc);
        return off + by * bw + bx;
    };
    const int per = (n + blockDim.x - 1) / blockDim.x;
    const int lo = threadIdx.x * per, hi = min(lo + per, n);
    int sum = 0;
    for (int k = lo; k < hi; ++k)
        sum += coef[(size_t)pos(k) * 64];
    int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    int x = sum;
    for (int o = 1; o < 32; o <<= 1) {
        int y = __shfl_up_sync(0xffffffffu, x, o);
        if (lane >= o)
            x += y;
    }
    if (lane == 31)
        warp_sums[wid] = x;
    __syncthreads();
    if (wid == 0) {
        int nw = (blockDim.x + 31) >> 5;
        int y = lane < nw ? warp_sums[lane] : 0;
        for (int o = 1; o < 32; o <<= 1) {
            int t2 = __shfl_up_sync(0xffffffffu, y, o);
            if (lane >= o)
                y += t2;
        }
        warp_sums[lane] = y;
    }
    __syncthreads();
    int acc = x - sum + (wid > 0 ? warp_sums[wid - 1] : 0);
    for (int k = lo; k < hi; ++k) {
        int16_t* d = coef + (size_t)pos(k) * 64;
        acc += d[0];
        d[0] = (int16_t)acc;
    }
}

// IDCT exactly as nvJPEG's hybrid path (identified black-box): the CUDA-samples dct8x8 inverse butterfly, columns
// first, the level shift fused into the last multiply of the second pass, round half to even, clamp.
#define C_a 1.387039845322148f
#define C_b 1.306562964876377f
#define C_c 1.175875602419359f
#define C_d 0.785694958387102f
#define C_e 0.541196100146197f
#define C_f 0.275899379282943f
#define C_norm 0.3535533905932737f

// Every rounding step is explicit (the extension is compiled with -fmad=false): this is the operation sequence nvcc
// produces for the CUDA-samples butterfly, which reproduces nvJPEG bit for bit.
__device__ __forceinline__ void idct8(float* v, int s, float L)
{
    float v0 = v[0], v1 = v[s], v2 = v[2 * s], v3 = v[3 * s], v4 = v[4 * s], v5 = v[5 * s], v6 = v[6 * s], v7 = v[7 * s];
    float Y04P = __fadd_rn(v0, v4);
    float Y2b6eP = fmaf(C_b, v2, __fmul_rn(C_e, v6));
    float Y04P2b6ePP = __fadd_rn(Y04P, Y2b6eP);
    float Y04P2b6ePM = __fsub_rn(Y04P, Y2b6eP);
    float Y7f1aP3c5dPP = fmaf(C_d, v5, fmaf(C_c, v3, fmaf(C_a, v1, __fmul_rn(C_f, v7))));
    float Y7a1fM3d5cMP = fmaf(-C_c, v5, fmaf(C_d, v3, fmaf(C_a, v7, -__fmul_rn(C_f, v1))));
    float Y04M = __fsub_rn(v0, v4);
    float Y2e6bM = fmaf(C_e, v2, -__fmul_rn(C_b, v6));
    float Y04M2e6bMP = __fadd_rn(Y04M, Y2e6bM);
    float Y04M2e6bMM = __fsub_rn(Y04M, Y2e6bM);
    float Y1c7dM3f5aPM = fmaf(-C_a, v5, fmaf(-C_f, v3, fmaf(C_c, v1, -__fmul_rn(C_d, v7))));
    float Y1d7cP3a5fMM = fmaf(C_f, v5, fmaf(-C_a, v3, fmaf(C_d, v1, __fmul_rn(C_c, v7))));
    v[0] = fmaf(C_norm, __fadd_rn(Y04P2b6ePP, Y7f1aP3c5dPP), L);
    v[7 * s] = fmaf(C_norm, __fsub_rn(Y04P2b6ePP, Y7f1aP3c5dPP), L);
    v[4 * s] = fmaf(C_norm, __fadd_rn(Y04P2b6ePM, Y7a1fM3d5cMP), L);
    v[3 * s] = fmaf(C_norm, __fsub_rn(Y04P2b6ePM, Y7a1fM3d5cMP), L);
    v[1 * s] = fmaf(C_norm, __fadd_rn(Y04M2e6bMP, Y1c7dM3f5aPM), L);
    v[5 * s] = fmaf(C_norm, __fsub_rn(Y04M2e6bMM, Y1d7cP3a5fMM), L);
    v[2 * s] = fmaf(C_norm, __fadd_rn(Y04M2e6bMM, Y1d7cP3a5fMM), L);
    v[6 * s] = fmaf(C_norm, __fsub_rn(Y04M2e6bMP, Y1c7dM3f5aPM), L);
}

__global__ void k_idct(const Params* gp, const int16_t* coef, int nblocks, uint8_t* planes)
{
    __shared__ uint16_t qt[MAXC][64];
    __shared__ int32_t off[MAXC + 1], bwid[MAXC];
    __shared__ int64_t poff[MAXC];
    const Params& P = *gp;
    for (int i = threadIdx.x; i < MAXC * 64; i += blockDim.x)
        (&qt[0][0])[i] = (&P.qt[0][0])[i];
    if (threadIdx.x < MAXC) {
        off[threadIdx.x] = threadIdx.x < P.ncomp ? P.comp_off[threadIdx.x] : 0x7fffffff;
        bwid[threadIdx.x] = P.comp_bw[threadIdx.x];
        poff[threadIdx.x] = P.plane_off[threadIdx.x];
    }
    if (threadIdx.x == 0)
        off[MAXC] = 0x7fffffff;
    __syncthreads();
    int blk = blockIdx.x * blockDim.x + threadIdx.x;
    if (blk >= nblocks)
        return;
    int c = 0;
    while (c + 1 < MAXC && blk >= off[c + 1])
        ++c;
    int local = blk - off[c];
    int by = local / bwid[c], bx = local - by * bwid[c];
    float v[64];
    int16_t cf[64];
    const int4* cp4 = reinterpret_cast<const int4*>(coef + (size_t)blk * 64);
#pragma unroll
    for (int k = 0; k < 8; ++k)
        reinterpret_cast<int4*>(cf)[k] = cp4[k];
#pragma unroll
    for (int k = 0; k < 64; ++k)
        v[ZZ[k]] = (float)((int)cf[k] * (int)qt[c][k]);
#pragma unroll
    for (int i = 0; i < 8; ++i)
        idct8(v + i, 8, 0.f); // columns
#pragma unroll
    for (int i = 0; i < 8; ++i)
        idct8(v + i * 8, 1, 128.f); // rows, level shift fused
    const int pitch = bwid[c] * 8;
    uint8_t* dst = planes + poff[c] + (size_t)(by * 8) * pitch + bx * 8;
#pragma unroll
    for (int y = 0; y < 8; ++y) {
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int x = 0; x < 4; ++x) {
            int a = __float2int_rn(v[y * 8 + x]);
            int bb = __float2int_rn(v[y * 8 + 4 + x]);
            a = min(max(a, 0), 255);
            bb = min(max(bb, 0), 255);
            w0 |= (uint32_t)a << (8 * x);
            w1 |= (uint32_t)bb << (8 * x);
        }
        uint32_t* d32 = reinterpret_cast<uint32_t*>(dst + (size_t)y * pitch);
        d32[0] = w0;
        d32[1] = w1;
    }
}

// colour conversion as nvJPEG (identified black-box, exhaustively over all 2^24 YCbCr triples): BT.601 full range with
// G's constants 0.114*1.772/0.587 and 0.299*1.402/0.587 in float, explicit FMAs, truncation, clamp; chroma replicated.
__global__ void k_color(const Params* gp, const uint8_t* planes, uint8_t* out, int mode)
{
    const Params& P = *gp;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y;
    if (x >= P.width || y >= P.height)
        return;
    const size_t hw = (size_t)P.width * P.height;
    const int p0 = P.comp_bw[0] * 8;
    int ys0 = y * P.comp_v[0] / P.vmax, xs0 = x * P.comp_h[0] / P.hmax;
    float Y = planes[P.plane_off[0] + (size_t)ys0 * p0 + xs0];
    if (mode == 1) {
        uint8_t g = (uint8_t)Y;
        out[(size_t)y * P.width + x] = g;
        out[hw + (size_t)y * P.width + x] = g;
        out[2 * hw + (size_t)y * P.width + x] = g;
        return;
    }
    if (mode == 2) {
        for (int c = 0; c < P.ncomp; ++c) {
            int pc = P.comp_bw[c] * 8;
            int yc = y * P.comp_v[c] / P.vmax, xc = x * P.comp_h[c] / P.hmax;
            out[c * hw + (size_t)y * P.width + x] = planes[P.plane_off[c] + (size_t)yc * pc + xc];
        }
        return;
    }
    const int p1 = P.comp_bw[1] * 8, p2 = P.comp_bw[2] * 8;
    int y1 = y * P.comp_v[1] / P.vmax, x1 = x * P.comp_h[1] / P.hmax;
    int y2 = y * P.comp_v[2] / P.vmax, x2 = x * P.comp_h[2] / P.hmax;
    float cb = __fsub_rn((float)planes[P.plane_off[1] + (size_t)y1 * p1 + x1], 128.f);
    float cr = __fsub_rn((float)planes[P.plane_off[2] + (size_t)y2 * p2 + x2], 128.f);
    float R = fmaf(1.402f, cr, Y);
    float G = fmaf(-0.7141362862f, cr, fmaf(-0.3441362862f, cb, Y));
    float B = fmaf(1.772f, cb, Y);
    int r = min(max(__float2int_rz(R), 0), 255), g = min(max(__float2int_rz(G), 0), 255), b = min(max(__float2int_rz(B), 0), 255);
    out[(size_t)y * P.width + x] = (uint8_t)r;
    out[hw + (size_t)y * P.width + x] = (uint8_t)g;
    out[2 * hw + (size_t)y * P.width + x] = (uint8_t)b;
}

// ---------------------------------------------------------------------------------------------------- host side
