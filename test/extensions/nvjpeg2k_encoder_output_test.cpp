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

#include <extensions/nvjpeg2k/encoder_output.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace nvimgcodec { namespace test {
namespace {

struct FakeOutputStream
{
    explicit FakeOutputStream(size_t capacity)
        : storage(capacity)
        , logical_size(capacity)
    {
        desc = {};
        desc.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC;
        desc.struct_size = sizeof(desc);
        desc.instance = this;
        desc.seek = seek;
        desc.write = write;
        desc.flush = flush;
        desc.unmap = unmap;
    }

    static nvimgcodecStatus_t seek(void* instance, ptrdiff_t offset, int whence)
    {
        auto& self = *static_cast<FakeOutputStream*>(instance);
        self.calls.emplace_back("seek");
        if (self.seek_status != NVIMGCODEC_STATUS_SUCCESS)
            return self.seek_status;
        if (whence != SEEK_SET || offset < 0 || static_cast<size_t>(offset) > self.logical_size)
            return NVIMGCODEC_STATUS_INVALID_PARAMETER;
        self.position = static_cast<size_t>(offset);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t write(void* instance, size_t* output_size, void* data, size_t bytes)
    {
        auto& self = *static_cast<FakeOutputStream*>(instance);
        self.calls.emplace_back("write");
        if (self.write_status != NVIMGCODEC_STATUS_SUCCESS)
            return self.write_status;
        size_t available = self.logical_size - self.position;
        size_t written = std::min(bytes, available);
        if (self.short_write && written)
            --written;
        std::memcpy(self.storage.data() + self.position, data, written);
        self.position += written;
        *output_size = written;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t flush(void* instance)
    {
        auto& self = *static_cast<FakeOutputStream*>(instance);
        self.calls.emplace_back("flush");
        if (self.flush_status != NVIMGCODEC_STATUS_SUCCESS)
            return self.flush_status;
        self.logical_size = self.position;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t unmap(void* instance, void* data, size_t size)
    {
        auto& self = *static_cast<FakeOutputStream*>(instance);
        self.calls.emplace_back("unmap");
        self.unmapped_data = data;
        self.unmapped_size = size;
        return self.unmap_status;
    }

    nvimgcodecIoStreamDesc_t desc{};
    std::vector<unsigned char> storage;
    std::vector<std::string> calls;
    size_t logical_size = 0;
    size_t position = 0;
    void* unmapped_data = nullptr;
    size_t unmapped_size = 0;
    bool short_write = false;
    nvimgcodecStatus_t seek_status = NVIMGCODEC_STATUS_SUCCESS;
    nvimgcodecStatus_t write_status = NVIMGCODEC_STATUS_SUCCESS;
    nvimgcodecStatus_t flush_status = NVIMGCODEC_STATUS_SUCCESS;
    nvimgcodecStatus_t unmap_status = NVIMGCODEC_STATUS_SUCCESS;
};

TEST(NvJpeg2kExtEncoderOutputTest, DirectOutputUnmapsCapacityAndCommitsProducedLength)
{
    FakeOutputStream stream(64);
    void* mapped_data = stream.storage.data();
    void* original_mapping = mapped_data;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvjpeg2k::detail::finalize_output(&stream.desc, mapped_data, 64, 17, nullptr));

    EXPECT_EQ(nullptr, mapped_data);
    EXPECT_EQ(original_mapping, stream.unmapped_data);
    EXPECT_EQ(64u, stream.unmapped_size);
    EXPECT_EQ(17u, stream.logical_size);
    EXPECT_EQ((std::vector<std::string>{"unmap", "seek", "flush"}), stream.calls);
}

TEST(NvJpeg2kExtEncoderOutputTest, FallbackUnmapsBeforeWritingExactProducedLength)
{
    FakeOutputStream stream(64);
    std::vector<unsigned char> bitstream(17);
    for (size_t i = 0; i < bitstream.size(); ++i)
        bitstream[i] = static_cast<unsigned char>(i + 1);
    void* mapped_data = stream.storage.data();

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvjpeg2k::detail::finalize_output(&stream.desc, mapped_data, 64, bitstream.size(), bitstream.data()));

    EXPECT_EQ(nullptr, mapped_data);
    EXPECT_EQ(64u, stream.unmapped_size);
    EXPECT_EQ(bitstream.size(), stream.logical_size);
    EXPECT_TRUE(std::equal(bitstream.begin(), bitstream.end(), stream.storage.begin()));
    EXPECT_EQ((std::vector<std::string>{"unmap", "seek", "write", "flush"}), stream.calls);
}

TEST(NvJpeg2kExtEncoderOutputTest, FallbackSupportsStreamThatCannotMap)
{
    FakeOutputStream stream(64);
    std::vector<unsigned char> bitstream(17, 0x5a);
    void* mapped_data = nullptr;

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvjpeg2k::detail::finalize_output(&stream.desc, mapped_data, 64, bitstream.size(), bitstream.data()));

    EXPECT_EQ(bitstream.size(), stream.logical_size);
    EXPECT_EQ((std::vector<std::string>{"seek", "write", "flush"}), stream.calls);
}

TEST(NvJpeg2kExtEncoderOutputTest, UnmapFailureClearsMappingAndStopsFinalization)
{
    FakeOutputStream stream(64);
    stream.unmap_status = NVIMGCODEC_STATUS_EXECUTION_FAILED;
    void* mapped_data = stream.storage.data();

    EXPECT_EQ(NVIMGCODEC_STATUS_EXECUTION_FAILED,
        nvjpeg2k::detail::finalize_output(&stream.desc, mapped_data, 64, 17, nullptr));
    EXPECT_EQ(nullptr, mapped_data);
    EXPECT_EQ((std::vector<std::string>{"unmap"}), stream.calls);
}

TEST(NvJpeg2kExtEncoderOutputTest, RejectsProducedLengthBeyondMappedCapacityAfterUnmap)
{
    FakeOutputStream stream(64);
    void* mapped_data = stream.storage.data();

    EXPECT_EQ(NVIMGCODEC_STATUS_EXECUTION_FAILED,
        nvjpeg2k::detail::finalize_output(&stream.desc, mapped_data, 64, 65, nullptr));
    EXPECT_EQ(nullptr, mapped_data);
    EXPECT_EQ(64u, stream.unmapped_size);
    EXPECT_EQ((std::vector<std::string>{"unmap"}), stream.calls);
}

TEST(NvJpeg2kExtEncoderOutputTest, RejectsShortFallbackWriteWithoutFlushing)
{
    FakeOutputStream stream(64);
    stream.short_write = true;
    std::vector<unsigned char> bitstream(17, 0x5a);
    void* mapped_data = stream.storage.data();

    EXPECT_EQ(NVIMGCODEC_STATUS_EXECUTION_FAILED,
        nvjpeg2k::detail::finalize_output(&stream.desc, mapped_data, 64, bitstream.size(), bitstream.data()));
    EXPECT_EQ((std::vector<std::string>{"unmap", "seek", "write"}), stream.calls);
}

TEST(NvJpeg2kExtEncoderOutputTest, PropagatesSeekWriteAndFlushFailures)
{
    std::vector<unsigned char> bitstream(17, 0x5a);

    FakeOutputStream seek_failure(64);
    seek_failure.seek_status = NVIMGCODEC_STATUS_INTERNAL_ERROR;
    void* seek_mapping = seek_failure.storage.data();
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR,
        nvjpeg2k::detail::finalize_output(&seek_failure.desc, seek_mapping, 64, 17, nullptr));

    FakeOutputStream write_failure(64);
    write_failure.write_status = NVIMGCODEC_STATUS_INTERNAL_ERROR;
    void* write_mapping = write_failure.storage.data();
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR,
        nvjpeg2k::detail::finalize_output(
            &write_failure.desc, write_mapping, 64, bitstream.size(), bitstream.data()));

    FakeOutputStream flush_failure(64);
    flush_failure.flush_status = NVIMGCODEC_STATUS_INTERNAL_ERROR;
    void* flush_mapping = flush_failure.storage.data();
    EXPECT_EQ(NVIMGCODEC_STATUS_INTERNAL_ERROR,
        nvjpeg2k::detail::finalize_output(&flush_failure.desc, flush_mapping, 64, 17, nullptr));
}

} // namespace
}} // namespace nvimgcodec::test
