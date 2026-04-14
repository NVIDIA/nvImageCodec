/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * Tests for concurrent nvimgcodecDecoderDecode calls.
 * Tests are parametrised by backend (TypeParam):
 *   CpuTraits — libjpeg_turbo, STRIDED_HOST
 *   GpuTraits — nvjpeg,        STRIDED_DEVICE
 *
 * Scenarios:
 *   submit_n_then_wait       — submit N futures before waiting on any (pipelining)
 *   submit_batches_then_wait — same with batch decode calls
 *   concurrent_threads       — N threads each decode with their own decoder simultaneously
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <nvimgcodec.h>

#if defined(BUILD_LIBJPEG_TURBO_EXT)
#  include <extensions/libjpeg_turbo/libjpeg_turbo_ext.h>
#endif
#if defined(BUILD_NVJPEG_EXT)
#  include <extensions/nvjpeg/nvjpeg_ext.h>
#endif
#include <parsers/jpeg.h>
#include "imgproc/device_buffer.h"
#include "imgproc/pinned_buffer.h"

#include <cstring>
#include <latch>
#include <string>
#include <thread>
#include <vector>

#include "nvimgcodec_tests.h"

namespace nvimgcodec { namespace test {


// ---------------------------------------------------------------------------
// Buffer wrappers (host / device)
// ---------------------------------------------------------------------------

struct HostBuf {
    PinnedBuffer buf;
    HostBuf() = default;
    explicit HostBuf(size_t n) { buf.resize(n, 0); std::memset(buf.data, 0, n); }
    void* ptr() { return buf.data; }
    std::vector<uint8_t> host_bytes() const {
        const auto* p = static_cast<const uint8_t*>(buf.data);
        return std::vector<uint8_t>(p, p + buf.size);
    }
};

struct DevBuf {
    DeviceBuffer buf;
    DevBuf() = default;
    explicit DevBuf(size_t n) {
        buf.resize(n, 0);
        if (cudaError_t rc = cudaMemset(buf.data, 0, n); rc != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(rc));
    }
    void* ptr() { return buf.data; }
    std::vector<uint8_t> host_bytes() const {
        if (!buf.data) return {};
        std::vector<uint8_t> h(buf.size);
        if (cudaError_t rc = cudaMemcpy(h.data(), buf.data, buf.size, cudaMemcpyDeviceToHost); rc != cudaSuccess)
            throw std::runtime_error(cudaGetErrorString(rc));
        return h;
    }
};

// ---------------------------------------------------------------------------
// Traits: backend-specific extension registration and buffer allocation
// ---------------------------------------------------------------------------

using ExtGetter = nvimgcodecStatus_t (*)(nvimgcodecExtensionDesc_t*);

static void RegisterExt(nvimgcodecInstance_t inst, std::vector<nvimgcodecExtension_t>& out, ExtGetter fn)
{
    nvimgcodecExtensionDesc_t d{NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(d), 0};
    fn(&d);
    nvimgcodecExtension_t e = nullptr;
    nvimgcodecStatus_t rc = nvimgcodecExtensionCreate(inst, &e, &d);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, rc) << "nvimgcodecExtensionCreate failed: " << rc;
    out.push_back(e);
}

struct CpuTraits {
    using Buf = HostBuf;
    static constexpr auto kKind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    static constexpr bool kAvailable =
#if defined(BUILD_LIBJPEG_TURBO_EXT)
        true;
#else
        false;
#endif
    static void RegisterExts(nvimgcodecInstance_t inst, std::vector<nvimgcodecExtension_t>& out) {
        ASSERT_NO_FATAL_FAILURE(RegisterExt(inst, out, get_jpeg_parser_extension_desc));
#if defined(BUILD_LIBJPEG_TURBO_EXT)
        ASSERT_NO_FATAL_FAILURE(RegisterExt(inst, out, get_libjpeg_turbo_extension_desc));
#endif
    }
    static Buf Alloc(uint32_t w, uint32_t h) { return Buf(h * w * 3); }
    static constexpr int kNumBackends = 0;
    static constexpr const nvimgcodecBackend_t* kBackends = nullptr;
};

// Use only the software CUDA decoder (HYBRID_CPU_GPU) to get reproducible pixel output.
// The hardware JPEG engine (HW_GPU_ONLY) may produce slightly different results across
// concurrent invocations, causing non-deterministic pixel mismatches in correctness tests.
static const nvimgcodecBackend_t kHybridBackend = {
    NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr,
    NVIMGCODEC_BACKEND_KIND_HYBRID_CPU_GPU,
    {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f, {}}};

struct GpuTraits {
    using Buf = DevBuf;
    static constexpr auto kKind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
    static constexpr bool kAvailable =
#if defined(BUILD_NVJPEG_EXT)
        true;
#else
        false;
#endif
    static void RegisterExts(nvimgcodecInstance_t inst, std::vector<nvimgcodecExtension_t>& out) {
        ASSERT_NO_FATAL_FAILURE(RegisterExt(inst, out, get_jpeg_parser_extension_desc));
#if defined(BUILD_NVJPEG_EXT)
        ASSERT_NO_FATAL_FAILURE(RegisterExt(inst, out, get_nvjpeg_extension_desc));
#endif
    }
    static Buf Alloc(uint32_t w, uint32_t h) { return Buf(h * w * 3); }
    static constexpr int kNumBackends = 1;
    static constexpr const nvimgcodecBackend_t* kBackends = &kHybridBackend;
};

// ---------------------------------------------------------------------------
// Fixture
// ---------------------------------------------------------------------------

static const std::vector<std::string> kImages = {
    "/jpeg/padlock-406986_640_420.jpg",
    "/jpeg/padlock-406986_640_444.jpg",
    "/jpeg/padlock-406986_640_gray.jpg",
    "/jpeg/padlock-406986_640_410.jpg",
    "/jpeg/padlock-406986_640_422.jpg",
};

template<typename T>
class ConcurrentDecodeTest : public ::testing::Test {
  protected:
    using Buf = typename T::Buf;

    void SetUp() override {
        if (!T::kAvailable)
            GTEST_SKIP() << "Required extension not built; skipping test";
        nvimgcodecInstanceCreateInfo_t ci{
            NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(ci), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance_, &ci));
        ASSERT_NO_FATAL_FAILURE(T::RegisterExts(instance_, exts_));
        decoder_ = MakeDecoder();
    }

    void TearDown() override {
        if (!instance_) return;  // SetUp was skipped
        if (decoder_) {
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder_));
        }
        for (auto e : exts_) ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(e));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance_));
    }

    nvimgcodecDecoder_t MakeDecoder() {
        nvimgcodecExecutionParams_t ep{
            NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(ep), 0};
        ep.device_id           = NVIMGCODEC_DEVICE_CURRENT;
        ep.max_num_cpu_threads = 1;
        ep.num_backends        = T::kNumBackends;
        ep.backends            = T::kBackends;
        nvimgcodecDecoder_t d  = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderCreate(instance_, &d, &ep, nullptr));
        return d;
    }

    struct Entry {
        nvimgcodecCodeStream_t cs    = nullptr;
        nvimgcodecImage_t      image = nullptr;
        Buf                    buf;
    };

    void Prepare(Entry& e, const std::string& path) {
        nvimgcodecImageInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(info), 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecCodeStreamCreateFromFile(instance_, &e.cs, path.c_str(), nullptr));
        if (nvimgcodecCodeStreamGetImageInfo(e.cs, &info) != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecCodeStreamDestroy(e.cs);
            e.cs = nullptr;
            FAIL() << "nvimgcodecCodeStreamGetImageInfo failed for " << path;
        }
        uint32_t w = info.plane_info[0].width, h = info.plane_info[0].height;
        info.sample_format              = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        info.color_spec                 = NVIMGCODEC_COLORSPEC_SRGB;
        info.chroma_subsampling         = NVIMGCODEC_SAMPLING_NONE;
        info.num_planes                 = 1;
        info.plane_info[0].num_channels = 3;
        info.plane_info[0].sample_type  = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
        info.plane_info[0].row_stride   = w * 3;
        info.plane_info[0].width        = w;
        info.plane_info[0].height       = h;
        info.buffer_kind                = T::kKind;
        e.buf                           = T::Alloc(w, h);
        info.buffer                     = e.buf.ptr();
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance_, &e.image, &info));
    }

    void Destroy(Entry& e) {
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(e.image));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(e.cs));
    }

    void WaitAndCheck(nvimgcodecFuture_t f, nvimgcodecProcessingStatus_t* s, size_t n) {
        // Always destroy the future, even if intermediate steps fail
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(f));
        size_t sz = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(f, s, &sz));
        EXPECT_EQ(n, sz);
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureDestroy(f));
    }

    // Decode a single image sequentially and return its pixels as host bytes.
    // Used to build per-image reference data for pixel-level correctness checks.
    std::vector<uint8_t> DecodeSeq(const std::string& path) {
        Entry e;
        SCOPED_TRACE("DecodeSeq: " + path);
        EXPECT_NO_FATAL_FAILURE(Prepare(e, path));
        if (HasFatalFailure()) return {};
        nvimgcodecFuture_t f = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderDecode(decoder_, &e.cs, &e.image, 1, &dec_params_, &f));
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(f));
        nvimgcodecProcessingStatus_t st = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        size_t sz = 0;
        nvimgcodecFutureGetProcessingStatus(f, &st, &sz);
        nvimgcodecFutureDestroy(f);
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, st) << "DecodeSeq ref failed for " << path;
        auto bytes = e.buf.host_bytes();
        Destroy(e);
        return bytes;
    }

    nvimgcodecInstance_t instance_ = nullptr;
    nvimgcodecDecoder_t  decoder_  = nullptr;
    std::vector<nvimgcodecExtension_t> exts_;
    nvimgcodecDecodeParams_t dec_params_{
        NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(dec_params_), 0};
};

using TestTypes = ::testing::Types<CpuTraits, GpuTraits>;
TYPED_TEST_SUITE(ConcurrentDecodeTest, TestTypes);

// ---------------------------------------------------------------------------
// Test 1: Submit N single-image futures before waiting on any
// ---------------------------------------------------------------------------

TYPED_TEST(ConcurrentDecodeTest, submit_n_then_wait)
{
    const int N = static_cast<int>(kImages.size());

    std::vector<std::vector<uint8_t>> refs(N);
    for (int i = 0; i < N; ++i)
        refs[i] = this->DecodeSeq(resources_dir + kImages[i]);

    std::vector<typename TestFixture::Entry> entries(N);
    std::vector<nvimgcodecFuture_t>          futures(N, nullptr);

    for (int i = 0; i < N; ++i)
        ASSERT_NO_FATAL_FAILURE(this->Prepare(entries[i], resources_dir + kImages[i]));

    for (int i = 0; i < N; ++i)
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderDecode(this->decoder_,
                &entries[i].cs, &entries[i].image, 1, &this->dec_params_, &futures[i]));

    for (int i = 0; i < N; ++i) {
        nvimgcodecProcessingStatus_t status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
        this->WaitAndCheck(futures[i], &status, 1);
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, status) << kImages[i];
        EXPECT_EQ(refs[i], entries[i].buf.host_bytes()) << "pixel mismatch for " << kImages[i];
        this->Destroy(entries[i]);
    }
}

// ---------------------------------------------------------------------------
// Test 2: Submit N batch futures before waiting on any
// ---------------------------------------------------------------------------

TYPED_TEST(ConcurrentDecodeTest, submit_batches_then_wait)
{
    const std::vector<std::vector<std::string>> batches = {
        {resources_dir + kImages[0], resources_dir + kImages[1]},
        {resources_dir + kImages[2], resources_dir + kImages[3]},
    };

    // Build per-image reference pixels via sequential decode
    std::vector<std::vector<std::vector<uint8_t>>> refs(batches.size());
    for (size_t b = 0; b < batches.size(); ++b) {
        refs[b].resize(batches[b].size());
        for (size_t i = 0; i < batches[b].size(); ++i)
            refs[b][i] = this->DecodeSeq(batches[b][i]);
    }

    struct Batch {
        std::vector<typename TestFixture::Entry> entries;
        std::vector<nvimgcodecCodeStream_t>      cs_ptrs;
        std::vector<nvimgcodecImage_t>           img_ptrs;
        nvimgcodecFuture_t                       future = nullptr;
    };

    std::vector<Batch> bdata(batches.size());
    for (size_t b = 0; b < batches.size(); ++b) {
        bdata[b].entries.resize(batches[b].size());
        for (size_t i = 0; i < batches[b].size(); ++i)
            ASSERT_NO_FATAL_FAILURE(this->Prepare(bdata[b].entries[i], batches[b][i]));
        for (auto& e : bdata[b].entries) {
            bdata[b].cs_ptrs.push_back(e.cs);
            bdata[b].img_ptrs.push_back(e.image);
        }
    }

    for (size_t b = 0; b < batches.size(); ++b)
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
            nvimgcodecDecoderDecode(this->decoder_,
                bdata[b].cs_ptrs.data(), bdata[b].img_ptrs.data(),
                static_cast<int>(batches[b].size()), &this->dec_params_, &bdata[b].future));

    for (size_t b = 0; b < batches.size(); ++b) {
        std::vector<nvimgcodecProcessingStatus_t> statuses(
            batches[b].size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
        this->WaitAndCheck(bdata[b].future, statuses.data(), batches[b].size());
        for (size_t i = 0; i < batches[b].size(); ++i) {
            EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << batches[b][i];
            EXPECT_EQ(refs[b][i], bdata[b].entries[i].buf.host_bytes())
                << "pixel mismatch for " << batches[b][i];
            this->Destroy(bdata[b].entries[i]);
        }
    }
}

// ---------------------------------------------------------------------------
// Test 3: N threads each decode with their own decoder simultaneously
// ---------------------------------------------------------------------------

TYPED_TEST(ConcurrentDecodeTest, concurrent_threads)
{
    const int N = static_cast<int>(kImages.size());

    // Build per-image reference pixels via sequential decode
    std::vector<std::vector<uint8_t>> refs(N);
    for (int i = 0; i < N; ++i)
        refs[i] = this->DecodeSeq(resources_dir + kImages[i]);

    // Create one decoder per thread (decoders are not thread-safe to share)
    std::vector<nvimgcodecDecoder_t> decoders(N);
    for (int i = 0; i < N; ++i)
        decoders[i] = this->MakeDecoder();

    std::vector<typename TestFixture::Entry> entries(N);
    for (int i = 0; i < N; ++i)
        ASSERT_NO_FATAL_FAILURE(this->Prepare(entries[i], resources_dir + kImages[i]));

    std::latch barrier(N);
    std::vector<std::string>                  errors(N);
    std::vector<nvimgcodecProcessingStatus_t> statuses(N, NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);

    std::vector<std::thread> threads;
    threads.reserve(N);
    for (int i = 0; i < N; ++i) {
        threads.emplace_back([&, i]() {
            barrier.arrive_and_wait();
            nvimgcodecFuture_t f = nullptr;
            auto ret = nvimgcodecDecoderDecode(
                decoders[i], &entries[i].cs, &entries[i].image, 1, &this->dec_params_, &f);
            if (ret != NVIMGCODEC_STATUS_SUCCESS) {
                errors[i] = "Decode submit failed: " + std::to_string(ret);
                return;
            }
            if (nvimgcodecFutureWaitForAll(f) != NVIMGCODEC_STATUS_SUCCESS) {
                errors[i] = "FutureWait failed";
                nvimgcodecFutureDestroy(f);
                return;
            }
            size_t sz = 0;
            nvimgcodecFutureGetProcessingStatus(f, &statuses[i], &sz);
            nvimgcodecFutureDestroy(f);
        });
    }
    for (auto& t : threads) t.join();
    for (auto d : decoders) EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(d));

    for (int i = 0; i < N; ++i) {
        EXPECT_TRUE(errors[i].empty()) << "Thread " << i << ": " << errors[i];
        EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, statuses[i]) << kImages[i];
        EXPECT_EQ(refs[i], entries[i].buf.host_bytes()) << "pixel mismatch for " << kImages[i];
        this->Destroy(entries[i]);
    }
}

}} // namespace nvimgcodec::test
