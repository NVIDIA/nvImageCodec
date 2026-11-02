// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#include "decoder.h"
#include "jpeg_plan.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

namespace nvimgcodec::jpeg_cuda_experimental {
namespace {
void Check(cudaError_t status, const char* operation) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
}
#define CU(call) Check((call), #call)

class DeviceGuard {
 public:
    explicit DeviceGuard(int device) {
        CU(cudaGetDevice(&previous_));
        if (previous_ != device) CU(cudaSetDevice(device));
    }
    ~DeviceGuard() { cudaSetDevice(previous_); }
 private:
    int previous_ = 0;
};

#include "kernels.cuh"

int DeviceAllocate(void*, void** pointer, size_t size, cudaStream_t stream) {
    return static_cast<int>(cudaMallocAsync(pointer, size, stream));
}
int DeviceRelease(void*, void* pointer, size_t, cudaStream_t stream) {
    return static_cast<int>(cudaFreeAsync(pointer, stream));
}
int PinnedAllocate(void*, void** pointer, size_t size, cudaStream_t) {
    return static_cast<int>(cudaHostAlloc(pointer, size, cudaHostAllocDefault));
}
int PinnedRelease(void*, void* pointer, size_t, cudaStream_t) {
    return static_cast<int>(cudaFreeHost(pointer));
}
MemoryResource Resolve(MemoryResource resource, bool pinned) {
    if (static_cast<bool>(resource.allocate) != static_cast<bool>(resource.release))
        throw std::invalid_argument("Supply both allocator callbacks or neither");
    if (!resource.allocate) {
        resource.allocate = pinned ? PinnedAllocate : DeviceAllocate;
        resource.release = pinned ? PinnedRelease : DeviceRelease;
    }
    return resource;
}
void Allocate(const MemoryResource& resource, void** pointer, size_t size, cudaStream_t stream) {
    if (resource.allocate(resource.context, pointer, size, stream) != 0 || !*pointer)
        throw std::runtime_error("JPEG allocation failed");
}
void Release(const MemoryResource& resource, void* pointer, size_t size, cudaStream_t stream) {
    if (pointer && resource.release(resource.context, pointer, size, stream) != 0)
        throw std::runtime_error("JPEG deallocation failed");
}
}  // namespace

struct Decoder::Impl {
    int device;
    MemoryResource device_memory, pinned_memory;
    uint8_t* staging = nullptr;
    size_t staging_capacity = 0;
    uint8_t* workspace = nullptr;
    size_t workspace_capacity = 0;
    int* status = nullptr;
    cudaEvent_t done = nullptr;
    cudaStream_t last_stream = nullptr;
    bool pending = false;
    std::mutex mutex;

    Impl(int device_id, MemoryResource device_resource, MemoryResource pinned_resource)
        : device(device_id), device_memory(Resolve(device_resource, false)),
          pinned_memory(Resolve(pinned_resource, true)) {
        DeviceGuard guard(device);
        const size_t shared = sizeof(Shared) + (64 * 1024 + 3072 + 256) / 8 + 64;
        int available = 0;
        CU(cudaDeviceGetAttribute(&available, cudaDevAttrMaxSharedMemoryPerBlockOptin, device));
        if (available < 96 * 1024 || shared > static_cast<size_t>(available))
            throw std::runtime_error("JPEG prototype requires 96 KiB opt-in shared memory");
        for (const void* kernel : {reinterpret_cast<const void*>(k_spec), reinterpret_cast<const void*>(k_relink),
                                   reinterpret_cast<const void*>(k_final)})
            CU(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        CU(cudaFuncSetAttribute(reinterpret_cast<const void*>(k_walk), cudaFuncAttributeMaxDynamicSharedMemorySize, 96 * 1024));
        Allocate(pinned_memory, reinterpret_cast<void**>(&status), 2 * sizeof(int), nullptr);
        try {
            CU(cudaEventCreateWithFlags(&done, cudaEventDisableTiming));
        } catch (...) {
            Release(pinned_memory, status, 2 * sizeof(int), nullptr);
            throw;
        }
    }

    Status Finish() {
        if (!pending) return Status::kSuccess;
        CU(cudaEventSynchronize(done));
        pending = false;
        return status[0] == 8 ? Status::kSuccess : Status::kCorrupt;
    }

    void GrowHost(size_t size, cudaStream_t stream) {
        if (size <= staging_capacity) return;
        uint8_t* replacement = nullptr;
        size_t capacity = std::max(size, staging_capacity * 2);
        Allocate(pinned_memory, reinterpret_cast<void**>(&replacement), capacity, stream);
        Release(pinned_memory, staging, staging_capacity, stream);
        staging = replacement;
        staging_capacity = capacity;
    }

    void GrowDevice(size_t size, cudaStream_t stream) {
        if (size <= workspace_capacity) return;
        uint8_t* replacement = nullptr;
        size_t capacity = std::max(size, workspace_capacity * 2);
        Allocate(device_memory, reinterpret_cast<void**>(&replacement), capacity, stream);
        Release(device_memory, workspace, workspace_capacity, stream);
        workspace = replacement;
        workspace_capacity = capacity;
    }

    ~Impl() {
        // The owning application must check Wait; destruction only releases resources.
        try {
            DeviceGuard guard(device);
            if (pending) cudaEventSynchronize(done);
            if (workspace) device_memory.release(device_memory.context, workspace, workspace_capacity, last_stream);
            if (staging) pinned_memory.release(pinned_memory.context, staging, staging_capacity, last_stream);
            if (status) pinned_memory.release(pinned_memory.context, status, 2 * sizeof(int), last_stream);
            if (done) cudaEventDestroy(done);
        } catch (...) {
            // CUDA contexts can already be unavailable during process teardown.
        }
    }
};

Decoder::Decoder(int device, MemoryResource device_memory, MemoryResource pinned_memory)
    : impl_(std::make_unique<Impl>(device, device_memory, pinned_memory)) {}
Decoder::~Decoder() = default;

Status Decoder::Wait() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    DeviceGuard guard(impl_->device);
    return impl_->Finish();
}

Status Decoder::Decode(const uint8_t* encoded, size_t size, const Output& output, cudaStream_t stream) {
    Impl& D = *impl_;
    std::lock_guard<std::mutex> lock(D.mutex);
    DeviceGuard guard(D.device);
    Status previous = D.Finish();
    if (previous != Status::kSuccess) return previous;
    Params params{};
    size_t scan_start = 0;
    int kind = MakePlan(encoded, size, params, scan_start);
    if (kind != OK) return static_cast<Status>(kind);
    if (output.width != params.width || output.height != params.height)
        throw std::invalid_argument("Output dimensions differ from JPEG header");
    for (int c = 0; c < 3; ++c)
        if (!output.plane[c] || output.row_stride[c] < static_cast<size_t>(params.width))
            throw std::invalid_argument("Invalid planar RGB destination");
    D.GrowHost(sizeof(Params) + size, stream);
    std::memcpy(D.staging, &params, sizeof(params));
    const Params& P = *reinterpret_cast<const Params*>(D.staging);
    cudaStream_t s = stream;
    D.last_stream = stream;
    try {
// Included by decoder.cu inside Decoder::Decode.
  const size_t raw = (size_t)P.raw_bytes;
  const size_t up = sizeof(Params) + raw;
  std::memcpy(D.staging + sizeof(Params), encoded + scan_start, raw);
  const int nch = P.nchunks;
  const size_t rawpad = (raw + 128 + 15) & ~(size_t)15;
  const size_t nwords = rawpad / 4;
  int64_t ncoef_blocks = 0;
  for (int c = 0; c < P.ncomp; ++c) ncoef_blocks += P.comp_nblk[c];
  int64_t plane_bytes = 0;
  for (int c = 0; c < P.ncomp; ++c) plane_bytes += (int64_t)P.comp_bw[c] * 8 * P.comp_bh[c] * 8;
  size_t temp_bytes = 0;
  CU(cub::DeviceSelect::Flagged(nullptr, temp_bytes, (const uint8_t*)nullptr, (const uint8_t*)nullptr, (uint8_t*)nullptr,
                                (int*)nullptr, (int)raw, s));
  auto align = [](size_t x) { return (x + 255) & ~(size_t)255; };
  const size_t o_up = 0, o_small = align(up), o_keep = o_small + 256, o_bytes = o_keep + align(raw),
               o_words = o_bytes + align(rawpad), o_temp = o_words + align(nwords * 4), o_lists = o_temp + align(temp_bytes),
               o_nlist = o_lists + align((size_t)nch * P.list_cap * sizeof(int2)), o_E = o_nlist + align((size_t)nch * 4),
               o_fix = o_E + align((size_t)nch * 16), o_coef = o_fix + align((size_t)nch * (4 * 16 + 5 * 4)),
               o_planes = o_coef + align((size_t)ncoef_blocks * 128), total = o_planes + align((size_t)plane_bytes);
  const size_t rgb_bytes = (size_t)3 * P.height * P.width;
  D.GrowDevice(total + rgb_bytes, s);
  uint8_t* W = D.workspace;
  CU(cudaMemcpyAsync(W + o_up, D.staging, up, cudaMemcpyHostToDevice, s));
  const Params* dP = reinterpret_cast<const Params*>(W + o_up);
  const uint8_t* draw = W + sizeof(Params);
  int* d_end = reinterpret_cast<int*>(W + o_small);
  int* d_nbytes = d_end + 1;
  int* d_status = d_end + 2;
  int* d_stat = d_end + 3;
  CU(cudaMemsetAsync(d_end, 0x7f, sizeof(int), s));
  CU(cudaMemsetAsync(d_nbytes, 0, 3 * sizeof(int), s));
  uint8_t* keep = W + o_keep;
  uint8_t* dbytes = W + o_bytes;
  uint32_t* bw = reinterpret_cast<uint32_t*>(W + o_words);
  CU(cudaMemsetAsync(dbytes, 0, rawpad, s));
  const int tb = 256;
  k_find_end<<<(int)((raw + tb - 1) / tb), tb, 0, s>>>(draw, (int)raw, d_end);
  k_mark<<<(int)((raw + tb - 1) / tb), tb, 0, s>>>(draw, (int)raw, d_end, keep);
  CU(cub::DeviceSelect::Flagged(W + o_temp, temp_bytes, draw, keep, dbytes, d_nbytes, (int)raw, s));
  k_words<<<(int)((nwords + tb - 1) / tb), tb, 0, s>>>(dbytes, (int)nwords, bw);
  int2* lists = reinterpret_cast<int2*>(W + o_lists);
  int* nlist = reinterpret_cast<int*>(W + o_nlist);
  int4* E = reinterpret_cast<int4*>(W + o_E);
  int4* R1 = reinterpret_cast<int4*>(W + o_fix);
  int4* R2 = R1 + nch;
  int4* fixT = R2 + nch;
  int4* T = fixT + nch;
  int* R1nb = reinterpret_cast<int*>(T + nch);
  int* R2nb = R1nb + nch;
  int* fixNb = R2nb + nch;
  int* G = fixNb + nch;
  CU(cudaMemsetAsync(fixNb, 0xff, (size_t)nch * sizeof(int), s));
  const int nt = 64;
  const int nblk = (nch + nt - 1) / nt;
  const size_t shm_spec = sizeof(Shared) + (size_t)(nt * P.chunk_bits + P.overlap_bits + 256) / 8 + 64;
  const size_t shm_rel = sizeof(Shared) + (size_t)(nt * P.chunk_bits + 256) / 8 + 64;
  k_spec<<<nblk, nt, shm_spec, s>>>(dP, bw, d_nbytes, lists, nlist, E);
  k_relink<<<nblk, nt, shm_rel, s>>>(dP, bw, d_nbytes, lists, nlist, E, R1, R1nb, R2, R2nb, 0);
  k_relink<<<nblk, nt, shm_rel, s>>>(dP, bw, d_nbytes, lists, nlist, E, R1, R1nb, R2, R2nb, 1);
  const size_t shm_walk = sizeof(Shared) + (size_t)((nch + 31) / 32 + 1) * sizeof(uint32_t);
  if (shm_walk > 96 * 1024) throw std::runtime_error("JPEG walk workspace exceeds shared-memory limit");
  k_walk<<<1, 1024, shm_walk, s>>>(dP, bw, d_nbytes, lists, nlist, E, R1, R1nb, R2, R2nb, fixT, fixNb, T, G, d_stat);
  int16_t* coef = reinterpret_cast<int16_t*>(W + o_coef);
  CU(cudaMemsetAsync(coef, 0, (size_t)ncoef_blocks * 128, s));
  k_final<<<nblk, nt, shm_rel, s>>>(dP, bw, d_nbytes, T, G, coef, d_status);
  k_dcscan<<<P.ncomp, 1024, 0, s>>>(dP, coef, d_status);
  uint8_t* planes = W + o_planes;
  k_idct<<<(int)((ncoef_blocks + 127) / 128), 128, 0, s>>>(dP, coef, (int)ncoef_blocks, planes);
  uint8_t* out = W + total;
  dim3 grid((P.width + 127) / 128, P.height);
  k_color<<<grid, 128, 0, s>>>(dP, planes, out, P.out_mode);
  CU(cudaGetLastError());
  for (int c = 0; c < 3; ++c)
    CU(cudaMemcpy2DAsync(output.plane[c], output.row_stride[c], out + (size_t)c * P.width * P.height,
                         P.width, P.width, P.height, cudaMemcpyDeviceToDevice, s));
  CU(cudaMemcpyAsync(D.status, d_status, 2 * sizeof(int), cudaMemcpyDeviceToHost, s));
  CU(cudaEventRecord(D.done, s));
  D.pending = true;
    } catch (...) {
        // Protect staging if submission failed after an asynchronous H2D copy.
        if (cudaEventRecord(D.done, s) == cudaSuccess) D.pending = true;
        else cudaStreamSynchronize(s);
        throw;
    }
    return Status::kSuccess;
}

}  // namespace nvimgcodec::jpeg_cuda_experimental
