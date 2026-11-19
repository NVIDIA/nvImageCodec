// Copyright (c) 2026 wangyi. SPDX-License-Identifier: Apache-2.0
#include "decoder.h"
#include "test_fixtures.h"

#include <array>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

using namespace nvimgcodec::jpeg_cuda_experimental;

namespace {
void Check(cudaError_t status)
{
    if (status != cudaSuccess)
        throw std::runtime_error(cudaGetErrorString(status));
}
void Require(bool condition, const char* message)
{
    if (!condition)
        throw std::runtime_error(message);
}
int ZeroPinnedAllocate(void*, void** pointer, size_t size, cudaStream_t)
{
    cudaError_t result = cudaHostAlloc(pointer, size, cudaHostAllocDefault);
    if (result == cudaSuccess)
        std::memset(*pointer, 0, size);
    return static_cast<int>(result);
}
int PinnedRelease(void*, void* pointer, size_t, cudaStream_t)
{
    return static_cast<int>(cudaFreeHost(pointer));
}

struct Destination
{
    static constexpr size_t pitch = 8;
    static constexpr size_t height = 2;
    static constexpr size_t plane_stride = pitch * height;
    cudaStream_t stream = nullptr;
    uint8_t* data = nullptr;

    Destination()
    {
        Check(cudaStreamCreate(&stream));
        try {
            Check(cudaMalloc(reinterpret_cast<void**>(&data), 3 * plane_stride));
            Check(cudaMemset(data, 203, 3 * plane_stride));
        } catch (...) {
            if (data)
                cudaFree(data);
            cudaStreamDestroy(stream);
            throw;
        }
    }
    ~Destination()
    {
        cudaFree(data);
        cudaStreamDestroy(stream);
    }
    Output View() const
    {
        Output output;
        output.width = 1;
        output.height = height;
        for (int c = 0; c < 3; ++c) {
            output.plane[c] = data + c * plane_stride;
            output.row_stride[c] = pitch;
        }
        return output;
    }
    void CheckGray() const
    {
        std::array<uint8_t, 3 * plane_stride> host{};
        Check(cudaMemcpy(host.data(), data, host.size(), cudaMemcpyDeviceToHost));
        for (int c = 0; c < 3; ++c) {
            for (size_t y = 0; y < height; ++y) {
                size_t offset = c * plane_stride + y * pitch;
                Require(host[offset] == 128, "Gray sample differs from expected value");
                for (size_t x = 1; x < pitch; ++x)
                    Require(host[offset + x] == 203, "Row padding was overwritten");
            }
        }
    }
};
} // namespace

int main()
{
    try {
        Check(cudaSetDevice(0));
        // Both streams outlive Decoder, which retains the last submission stream.
        Destination first, second;
        MemoryResource pinned{nullptr, ZeroPinnedAllocate, PinnedRelease};
        Decoder decoder(0, {}, pinned);
        auto good = jpeg_cuda_test::Gray(1, Destination::height);

        // An invalid pitch fails after kernels have already been submitted. A
        // zero-initialized pinned status makes a stale/unwritten status visible.
        Output invalid = first.View();
        invalid.row_stride[0] = std::numeric_limits<size_t>::max();
        bool threw = false;
        try {
            decoder.Decode(good.data(), good.size(), invalid, first.stream);
        } catch (const std::runtime_error&) {
            threw = true;
        }
        Require(threw, "Invalid destination pitch did not fail submission");
        Require(decoder.Wait() == Status::kSuccess, "Failed submission left an unwritten completion status");

        Require(decoder.Decode(good.data(), good.size(), second.View(), second.stream) == Status::kSuccess,
            "Decoder could not be reused on another stream after submission failure");
        Require(decoder.Wait() == Status::kSuccess, "Valid sample failed after submission failure");
        second.CheckGray();

        auto corrupt = good;
        corrupt.erase(corrupt.end() - 3); // Header and EOI remain; entropy is empty.
        Require(Probe(corrupt.data(), corrupt.size()).status == Status::kSuccess, "Malformed test did not reach GPU validation");
        Require(decoder.Decode(corrupt.data(), corrupt.size(), first.View(), first.stream) == Status::kSuccess,
            "Corrupt entropy was not submitted");
        Require(decoder.Wait() == Status::kCorrupt, "Empty entropy was not reported as corrupt");
        Require(decoder.Decode(good.data(), good.size(), first.View(), first.stream) == Status::kSuccess,
            "Decoder could not recover after corruption");
        Require(decoder.Wait() == Status::kSuccess, "Valid sample failed after corruption");
        first.CheckGray();

        int devices = 0;
        Check(cudaGetDeviceCount(&devices));
        if (devices > 1) {
            Check(cudaSetDevice(1));
            Require(decoder.Decode(good.data(), good.size(), second.View(), second.stream) == Status::kSuccess,
                "Submission failed with another caller device current");
            int current = -1;
            Check(cudaGetDevice(&current));
            Require(current == 1, "Decode did not restore the caller device");
            Require(decoder.Wait() == Status::kSuccess, "Completion failed with another caller device current");
            Check(cudaGetDevice(&current));
            Require(current == 1, "Wait did not restore the caller device");
            Check(cudaSetDevice(0));
            second.CheckGray();
        }
        std::cout << "submission recovery, corrupt entropy, pitched output, stream reuse and device restoration passed\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
