// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <vector>

namespace nvimgcodec { namespace test { namespace {

// A CPU batch-only decoder records the actual native submissions. No GPU or
// image decompression is needed to verify batch dispatch and failure reporting.
struct RecordingDecoder
{
    std::vector<int> calls;
    int fail_call = -1;
    int throw_call = -1;
    nvimgcodecProcessingStatus_t failure_status = NVIMGCODEC_PROCESSING_STATUS_FAIL;
    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    nvimgcodecDecoderDesc_t decoder_desc{};
    nvimgcodecExtensionDesc_t extension_desc{};

    RecordingDecoder()
    {
        decoder_desc = {NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC, sizeof(decoder_desc), nullptr, this, "batch_recorder", "bmp",
            NVIMGCODEC_BACKEND_KIND_CPU_ONLY};
        decoder_desc.create = [](void* self, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t*, const char*) {
            *decoder = reinterpret_cast<nvimgcodecDecoder_t>(self);
            return NVIMGCODEC_STATUS_SUCCESS;
        };
        decoder_desc.destroy = [](nvimgcodecDecoder_t) { return NVIMGCODEC_STATUS_SUCCESS; };
        decoder_desc.canDecode = [](nvimgcodecDecoder_t, const nvimgcodecImageDesc_t*, const nvimgcodecCodeStreamDesc_t*,
                                     const nvimgcodecDecodeParams_t*,
                                     int) -> nvimgcodecProcessingStatus_t { return NVIMGCODEC_PROCESSING_STATUS_SUCCESS; };
        decoder_desc.decodeBatch = [](nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t** images, const nvimgcodecCodeStreamDesc_t**,
                                       int count, const nvimgcodecDecodeParams_t*, int) {
            auto& self = *reinterpret_cast<RecordingDecoder*>(decoder);
            bool fail = static_cast<int>(self.calls.size()) == self.fail_call;
            bool should_throw = static_cast<int>(self.calls.size()) == self.throw_call;
            self.calls.push_back(count);
            if (should_throw)
                throw std::runtime_error("Native subbatch failed before reporting image status");
            for (int i = 0; i < count; i++)
                images[i]->imageReady(images[i]->instance, fail ? self.failure_status : NVIMGCODEC_PROCESSING_STATUS_SUCCESS);
            return fail ? NVIMGCODEC_STATUS_EXECUTION_FAILED : NVIMGCODEC_STATUS_SUCCESS;
        };
        extension_desc = {NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC, sizeof(extension_desc), nullptr, this, "batch_recorder_extension",
            NVIMGCODEC_VER, NVIMGCODEC_VER};
        extension_desc.create = [](void* ptr, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework) {
            auto& self = *static_cast<RecordingDecoder*>(ptr);
            self.framework = framework;
            *extension = reinterpret_cast<nvimgcodecExtension_t>(ptr);
            return framework->registerDecoder(framework->instance, &self.decoder_desc, NVIMGCODEC_PRIORITY_HIGH);
        };
        extension_desc.destroy = [](nvimgcodecExtension_t extension) {
            auto& self = *reinterpret_cast<RecordingDecoder*>(extension);
            return self.framework->unregisterDecoder(self.framework->instance, &self.decoder_desc);
        };
    }
};

class BatchDecodeTest : public ::testing::Test
{
  protected:
    RecordingDecoder recorder;
    nvimgcodecInstance_t instance = nullptr;
    nvimgcodecExtension_t extension = nullptr;
    nvimgcodecDecoder_t decoder = nullptr;
    nvimgcodecCodeStream_t stream = nullptr;
    std::array<nvimgcodecImage_t, 5> images{};
    std::array<std::array<unsigned char, 16>, 5> buffers{};
    nvimgcodecDecodeParams_t params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(params), nullptr};

    void SetUp() override
    {
        nvimgcodecInstanceCreateInfo_t info{NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(info), nullptr};
        info.load_builtin_modules = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &info));
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, &recorder.extension_desc));
        nvimgcodecExecutionParams_t exec{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(exec), nullptr};
        exec.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
        exec.max_num_cpu_threads = 1;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec, nullptr));
        static unsigned char bmp[] = {0x42, 0x4d, 30, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 12, 0, 0, 0, 1, 0, 1, 0, 1, 0, 24, 0, 0, 0, 255, 0};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromHostMem(instance, &stream, bmp, sizeof(bmp), nullptr));
        nvimgcodecImageInfo_t output{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(output), nullptr};
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamGetImageInfo(stream, &output));
        output.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
        output.num_planes = 1;
        output.plane_info[0].num_channels = 3;
        output.plane_info[0].row_stride = 16; // Three payload bytes, sixteen decoded storage bytes.
        output.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
        for (size_t i = 0; i < images.size(); i++) {
            output.buffer = buffers[i].data();
            ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &images[i], &output));
        }
    }
    void TearDown() override
    {
        for (auto image : images)
            if (image)
                nvimgcodecImageDestroy(image);
        if (stream)
            nvimgcodecCodeStreamDestroy(stream);
        if (decoder)
            nvimgcodecDecoderDestroy(decoder);
        if (extension)
            nvimgcodecExtensionDestroy(extension);
        if (instance)
            nvimgcodecInstanceDestroy(instance);
    }
    std::vector<nvimgcodecProcessingStatus_t> decode(size_t batch_size = 5)
    {
        std::array<nvimgcodecCodeStream_t, 5> streams{stream, stream, stream, stream, stream};
        nvimgcodecFuture_t future = nullptr;
        EXPECT_EQ(
            NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDecode(decoder, streams.data(), images.data(), batch_size, &params, &future));
        if (!future)
            return {};
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureWaitForAll(future));
        std::vector<nvimgcodecProcessingStatus_t> statuses(batch_size);
        size_t count = 0;
        EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecFutureGetProcessingStatus(future, statuses.data(), &count));
        EXPECT_EQ(batch_size, count);
        nvimgcodecFutureDestroy(future);
        return statuses;
    }
};

TEST_F(BatchDecodeTest, PluginReceivesWholeBatch)
{
    EXPECT_EQ(std::vector<nvimgcodecProcessingStatus_t>(5, NVIMGCODEC_PROCESSING_STATUS_SUCCESS), decode());
    EXPECT_EQ((std::vector<int>{5}), recorder.calls);
}

TEST_F(BatchDecodeTest, ThrowingPluginDoesNotReportSuccess)
{
    recorder.throw_call = 0;
    EXPECT_EQ(std::vector<nvimgcodecProcessingStatus_t>(5, NVIMGCODEC_PROCESSING_STATUS_FAIL), decode());
}

TEST_F(BatchDecodeTest, SpecificFailureSurvivesBatchError)
{
    recorder.fail_call = 0;
    recorder.failure_status = NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED;
    EXPECT_EQ(std::vector<nvimgcodecProcessingStatus_t>(5, recorder.failure_status), decode());
}

TEST_F(BatchDecodeTest, FullLoadBatchDoesNotOverflowAfterSingleton)
{
    RecordingDecoder fallback;
    fallback.decoder_desc.id = "batch_fallback";
    fallback.extension_desc.id = "batch_fallback_extension";
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(stream));
    stream = nullptr;
    nvimgcodecExtension_t fallback_extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &fallback_extension, &fallback.extension_desc));
    nvimgcodecExecutionParams_t exec{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(exec), nullptr};
    exec.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec.max_num_cpu_threads = 1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec, nullptr));
    static unsigned char bmp[] = {
        0x42, 0x4d, 30, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 12, 0, 0, 0, 1, 0, 1, 0, 1, 0, 24, 0, 0, 0, 255, 0};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromHostMem(instance, &stream, bmp, sizeof(bmp), nullptr));
    decode(1);
    decode(5);
    EXPECT_EQ((std::vector<int>{1, 5}), recorder.calls);
    EXPECT_TRUE(fallback.calls.empty());
    nvimgcodecDecoderDestroy(decoder);
    decoder = nullptr;
    recorder.calls.clear();
    fallback.calls.clear();
    nvimgcodecBackend_t backend{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(backend), nullptr, NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
        {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 0.4f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
    exec.num_backends = 1;
    exec.backends = &backend;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec, nullptr));
    decode(5);
    EXPECT_EQ((std::vector<int>{2}), recorder.calls);
    EXPECT_EQ((std::vector<int>{3}), fallback.calls);
    nvimgcodecDecoderDestroy(decoder);
    decoder = nullptr;
    nvimgcodecCodeStreamDestroy(stream);
    stream = nullptr;
    nvimgcodecExtensionDestroy(fallback_extension);
}

}}} // namespace nvimgcodec::test
