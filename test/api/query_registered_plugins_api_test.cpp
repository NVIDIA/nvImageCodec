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

#include <gtest/gtest.h>
#include <nvimgcodec.h>
#include <nvimgcodec_version.h>

#include <array>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace nvimgcodec { namespace test {

namespace {

constexpr const char* kCodecName = "registered_plugins_test_codec";

struct PluginCallState
{
    const char* id;
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_SUCCESS;
    std::vector<std::string>* calls;
};

nvimgcodecStatus_t createDecoder(void* instance, nvimgcodecDecoder_t* decoder, const nvimgcodecExecutionParams_t*, const char*)
{
    *decoder = reinterpret_cast<nvimgcodecDecoder_t>(instance);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t destroyDecoder(nvimgcodecDecoder_t)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecProcessingStatus_t canDecode(
    nvimgcodecDecoder_t decoder, const nvimgcodecImageDesc_t*, const nvimgcodecCodeStreamDesc_t*, const nvimgcodecDecodeParams_t*, int)
{
    auto* state = reinterpret_cast<PluginCallState*>(decoder);
    state->calls->emplace_back(state->id);
    return state->processing_status;
}

nvimgcodecStatus_t decode(
    nvimgcodecDecoder_t, const nvimgcodecImageDesc_t*, const nvimgcodecCodeStreamDesc_t*, const nvimgcodecDecodeParams_t*, int)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t createEncoder(void* instance, nvimgcodecEncoder_t* encoder, const nvimgcodecExecutionParams_t*, const char*)
{
    *encoder = reinterpret_cast<nvimgcodecEncoder_t>(instance);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t destroyEncoder(nvimgcodecEncoder_t)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecProcessingStatus_t canEncode(
    nvimgcodecEncoder_t encoder, const nvimgcodecCodeStreamDesc_t*, const nvimgcodecImageDesc_t*, const nvimgcodecEncodeParams_t*, int)
{
    auto* state = reinterpret_cast<PluginCallState*>(encoder);
    state->calls->emplace_back(state->id);
    return state->processing_status;
}

nvimgcodecStatus_t encode(
    nvimgcodecEncoder_t, const nvimgcodecCodeStreamDesc_t*, const nvimgcodecImageDesc_t*, const nvimgcodecEncodeParams_t*, int)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t canParse(void*, int* result, nvimgcodecCodeStreamDesc_t*)
{
    *result = 1;
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t createParser(void*, nvimgcodecParser_t* parser)
{
    *parser = reinterpret_cast<nvimgcodecParser_t>(1);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t destroyParser(nvimgcodecParser_t)
{
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t getCodeStreamInfo(nvimgcodecParser_t, nvimgcodecCodeStreamInfo_t* code_stream_info, nvimgcodecCodeStreamDesc_t*)
{
    std::strcpy(code_stream_info->codec_name, kCodecName);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t getImageInfo(nvimgcodecParser_t, nvimgcodecImageInfo_t* image_info, nvimgcodecCodeStreamDesc_t*)
{
    std::strcpy(image_info->codec_name, kCodecName);
    return NVIMGCODEC_STATUS_SUCCESS;
}

class RegisteredPluginsTestExtension
{
  public:
    RegisteredPluginsTestExtension(const nvimgcodecFrameworkDesc_t* framework, const nvimgcodecDecoderDesc_t* first_decoder,
        const nvimgcodecDecoderDesc_t* second_decoder, const nvimgcodecEncoderDesc_t* encoder, const nvimgcodecParserDesc_t* parser)
        : framework_(framework)
        , first_decoder_(first_decoder)
        , second_decoder_(second_decoder)
        , encoder_(encoder)
        , parser_(parser)
    {
        status_ = framework_->registerDecoder(framework_->instance, second_decoder_, NVIMGCODEC_PRIORITY_NORMAL);
        if (status_ == NVIMGCODEC_STATUS_SUCCESS) {
            status_ = framework_->registerDecoder(framework_->instance, first_decoder_, NVIMGCODEC_PRIORITY_HIGH);
        }
        if (status_ == NVIMGCODEC_STATUS_SUCCESS) {
            status_ = framework_->registerEncoder(framework_->instance, encoder_, NVIMGCODEC_PRIORITY_NORMAL);
        }
        if (status_ == NVIMGCODEC_STATUS_SUCCESS) {
            status_ = framework_->registerParser(framework_->instance, parser_, NVIMGCODEC_PRIORITY_NORMAL);
        }
    }

    ~RegisteredPluginsTestExtension()
    {
        framework_->unregisterParser(framework_->instance, parser_);
        framework_->unregisterEncoder(framework_->instance, encoder_);
        framework_->unregisterDecoder(framework_->instance, second_decoder_);
        framework_->unregisterDecoder(framework_->instance, first_decoder_);
    }

    nvimgcodecStatus_t status() const { return status_; }

  private:
    const nvimgcodecFrameworkDesc_t* framework_;
    const nvimgcodecDecoderDesc_t* first_decoder_;
    const nvimgcodecDecoderDesc_t* second_decoder_;
    const nvimgcodecEncoderDesc_t* encoder_;
    const nvimgcodecParserDesc_t* parser_;
    nvimgcodecStatus_t status_ = NVIMGCODEC_STATUS_SUCCESS;
};

class RegisteredPluginsTestExtensionFactory
{
  public:
    RegisteredPluginsTestExtensionFactory()
    {
        cpu_decoder_state_.calls = &decoder_calls_;
        gpu_decoder_state_.calls = &decoder_calls_;
        encoder_state_.calls = &encoder_calls_;

        cpu_decoder_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC;
        cpu_decoder_.struct_size = sizeof(cpu_decoder_);
        cpu_decoder_.instance = &cpu_decoder_state_;
        cpu_decoder_.id = "registered_plugins_cpu_decoder";
        cpu_decoder_.codec = kCodecName;
        cpu_decoder_.backend_kind = NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
        cpu_decoder_.create = &createDecoder;
        cpu_decoder_.destroy = &destroyDecoder;
        cpu_decoder_.canDecode = &canDecode;
        cpu_decoder_.decode = &decode;

        gpu_decoder_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODER_DESC;
        gpu_decoder_.struct_size = sizeof(gpu_decoder_);
        gpu_decoder_.instance = &gpu_decoder_state_;
        gpu_decoder_.id = "registered_plugins_gpu_decoder";
        gpu_decoder_.codec = kCodecName;
        gpu_decoder_.backend_kind = NVIMGCODEC_BACKEND_KIND_GPU_ONLY;
        gpu_decoder_.create = &createDecoder;
        gpu_decoder_.destroy = &destroyDecoder;
        gpu_decoder_.canDecode = &canDecode;
        gpu_decoder_.decode = &decode;

        encoder_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_ENCODER_DESC;
        encoder_.struct_size = sizeof(encoder_);
        encoder_.instance = &encoder_state_;
        encoder_.id = "registered_plugins_cpu_encoder";
        encoder_.codec = kCodecName;
        encoder_.backend_kind = NVIMGCODEC_BACKEND_KIND_CPU_ONLY;
        encoder_.create = &createEncoder;
        encoder_.destroy = &destroyEncoder;
        encoder_.canEncode = &canEncode;
        encoder_.encode = &encode;

        parser_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PARSER_DESC;
        parser_.struct_size = sizeof(parser_);
        parser_.id = "registered_plugins_parser";
        parser_.codec = kCodecName;
        parser_.canParse = &canParse;
        parser_.create = &createParser;
        parser_.destroy = &destroyParser;
        parser_.getCodeStreamInfo = &getCodeStreamInfo;
        parser_.getImageInfo = &getImageInfo;

        extension_.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXTENSION_DESC;
        extension_.struct_size = sizeof(extension_);
        extension_.instance = this;
        extension_.id = "registered_plugins_test_extension";
        extension_.version = 1;
        extension_.ext_api_version = NVIMGCODEC_VER;
        extension_.create = &create;
        extension_.destroy = &destroy;
    }

    nvimgcodecExtensionDesc_t* extensionDesc() { return &extension_; }

    const nvimgcodecDecoderDesc_t* cpuDecoderDesc() const { return &cpu_decoder_; }

    const nvimgcodecDecoderDesc_t* gpuDecoderDesc() const { return &gpu_decoder_; }

    const nvimgcodecEncoderDesc_t* encoderDesc() const { return &encoder_; }

    const nvimgcodecParserDesc_t* parserDesc() const { return &parser_; }

    nvimgcodecExtension_t pluginHandle() const { return plugin_handle_; }

    void clearDecoderCalls() { decoder_calls_.clear(); }

    const std::vector<std::string>& decoderCalls() const { return decoder_calls_; }

    void setDecoderStatus(nvimgcodecBackendKind_t backend_kind, nvimgcodecProcessingStatus_t status)
    {
        (backend_kind == NVIMGCODEC_BACKEND_KIND_CPU_ONLY ? cpu_decoder_state_ : gpu_decoder_state_).processing_status = status;
    }

    void clearEncoderCalls() { encoder_calls_.clear(); }

    const std::vector<std::string>& encoderCalls() const { return encoder_calls_; }

  private:
    static nvimgcodecStatus_t create(void* instance, nvimgcodecExtension_t* extension, const nvimgcodecFrameworkDesc_t* framework)
    {
        auto* factory = static_cast<RegisteredPluginsTestExtensionFactory*>(instance);
        auto* implementation = new RegisteredPluginsTestExtension(
            framework, &factory->cpu_decoder_, &factory->gpu_decoder_, &factory->encoder_, &factory->parser_);
        if (implementation->status() != NVIMGCODEC_STATUS_SUCCESS) {
            nvimgcodecStatus_t status = implementation->status();
            delete implementation;
            return status;
        }
        *extension = reinterpret_cast<nvimgcodecExtension_t>(implementation);
        factory->plugin_handle_ = *extension;
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    static nvimgcodecStatus_t destroy(nvimgcodecExtension_t extension)
    {
        delete reinterpret_cast<RegisteredPluginsTestExtension*>(extension);
        return NVIMGCODEC_STATUS_SUCCESS;
    }

    std::vector<std::string> decoder_calls_;
    std::vector<std::string> encoder_calls_;
    PluginCallState cpu_decoder_state_{"registered_plugins_cpu_decoder", NVIMGCODEC_PROCESSING_STATUS_SUCCESS, nullptr};
    PluginCallState gpu_decoder_state_{"registered_plugins_gpu_decoder", NVIMGCODEC_PROCESSING_STATUS_SUCCESS, nullptr};
    PluginCallState encoder_state_{"registered_plugins_cpu_encoder", NVIMGCODEC_PROCESSING_STATUS_SUCCESS, nullptr};
    nvimgcodecDecoderDesc_t cpu_decoder_{};
    nvimgcodecDecoderDesc_t gpu_decoder_{};
    nvimgcodecEncoderDesc_t encoder_{};
    nvimgcodecParserDesc_t parser_{};
    nvimgcodecExtensionDesc_t extension_{};
    nvimgcodecExtension_t plugin_handle_ = nullptr;
};

nvimgcodecInstance_t createEmptyInstance()
{
    nvimgcodecInstanceCreateInfo_t create_info{
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
    nvimgcodecInstance_t instance = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));
    return instance;
}

nvimgcodecCodeStream_t createTestCodeStream(nvimgcodecInstance_t instance)
{
    static const std::array<unsigned char, 1> data{0};
    nvimgcodecCodeStream_t code_stream = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateFromHostMem(instance, &code_stream, data.data(), data.size(), nullptr));
    return code_stream;
}

nvimgcodecImage_t createTestImage(nvimgcodecInstance_t instance)
{
    static std::array<unsigned char, 3> pixels{};
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    image_info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_RGB;
    image_info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
    image_info.num_planes = 1;
    image_info.plane_info[0].width = 1;
    image_info.plane_info[0].height = 1;
    image_info.plane_info[0].row_stride = 3;
    image_info.plane_info[0].num_channels = 3;
    image_info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
    image_info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_HOST;
    image_info.buffer = pixels.data();
    nvimgcodecImage_t image = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageCreate(instance, &image, &image_info));
    return image;
}

unsigned char* resizeTestBuffer(void* context, size_t size)
{
    auto* buffer = static_cast<std::vector<unsigned char>*>(context);
    buffer->resize(size);
    return buffer->data();
}

nvimgcodecCodeStream_t createTestOutputCodeStream(nvimgcodecInstance_t instance, const char* codec_name, std::vector<unsigned char>* buffer)
{
    nvimgcodecImageInfo_t image_info{NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO, sizeof(nvimgcodecImageInfo_t), nullptr};
    std::strcpy(image_info.codec_name, codec_name);
    nvimgcodecCodeStream_t code_stream = nullptr;
    EXPECT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamCreateToHostMem(instance, &code_stream, buffer, &resizeTestBuffer, &image_info));
    return code_stream;
}

} // namespace

TEST(QueryRegisteredPluginsApiTest, exposesFrameworkExtensionsAndRegisteredDescriptions)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);

    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance, &framework));
    ASSERT_NE(nullptr, framework);
    EXPECT_EQ(NVIMGCODEC_STRUCTURE_TYPE_FRAMEWORK_DESC, framework->struct_type);
    EXPECT_NE(nullptr, framework->getCodecNames);
    EXPECT_NE(nullptr, framework->getFileExtensions);
    EXPECT_NE(nullptr, framework->getDecoderDescs);
    EXPECT_NE(nullptr, framework->getEncoderDescs);
    EXPECT_NE(nullptr, framework->getParserDescs);

    int extension_count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, nullptr, &extension_count));
    EXPECT_EQ(0, extension_count);

    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));
    EXPECT_NE(factory.pluginHandle(), extension);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, nullptr, &extension_count));
    ASSERT_EQ(1, extension_count);
    nvimgcodecExtension_t listed_extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, &listed_extension, &extension_count));
    EXPECT_EQ(extension, listed_extension);

    const nvimgcodecExtensionDesc_t* extension_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionGetExtensionDesc(extension, &extension_desc));
    ASSERT_NE(nullptr, extension_desc);
    EXPECT_STREQ(factory.extensionDesc()->id, extension_desc->id);
    EXPECT_EQ(factory.extensionDesc()->create, extension_desc->create);
    EXPECT_EQ(factory.extensionDesc()->destroy, extension_desc->destroy);

    int codec_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getCodecNames(framework->instance, nullptr, &codec_count, nullptr));
    ASSERT_EQ(1, codec_count);
    int too_many_codec_names = 2;
    std::array<char, 2 * NVIMGCODEC_MAX_CODEC_NAME_SIZE> oversized_codec_names{};
    oversized_codec_names[0] = 'x';
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        framework->getCodecNames(framework->instance, oversized_codec_names.data(), &too_many_codec_names, nullptr));
    EXPECT_EQ(2, too_many_codec_names);
    EXPECT_EQ('x', oversized_codec_names[0]);

    std::array<char, NVIMGCODEC_MAX_CODEC_NAME_SIZE> codec_name{};
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, framework->getCodecNames(framework->instance, codec_name.data(), &codec_count, nullptr));
    EXPECT_STREQ(kCodecName, codec_name.data());

    int decoder_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, nullptr, &decoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    ASSERT_EQ(2, decoder_count);
    int truncated_count = 1;
    const nvimgcodecDecoderDesc_t* first_decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, &first_decoder, &truncated_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(1, truncated_count);
    EXPECT_EQ(factory.cpuDecoderDesc(), first_decoder);

    int too_many_decoders = 3;
    std::array<const nvimgcodecDecoderDesc_t*, 3> oversized_decoder_descs{
        reinterpret_cast<const nvimgcodecDecoderDesc_t*>(1), nullptr, nullptr};
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, framework->getDecoderDescs(framework->instance, oversized_decoder_descs.data(),
                                                       &too_many_decoders, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(3, too_many_decoders);
    EXPECT_EQ(reinterpret_cast<const nvimgcodecDecoderDesc_t*>(1), oversized_decoder_descs[0]);

    std::vector<const nvimgcodecDecoderDesc_t*> decoder_descs(decoder_count);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, decoder_descs.data(), &decoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(factory.cpuDecoderDesc(), decoder_descs[0]);
    EXPECT_EQ(factory.gpuDecoderDesc(), decoder_descs[1]);

    decoder_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, nullptr, &decoder_count, nullptr, NVIMGCODEC_BACKEND_KIND_GPU_ONLY));
    ASSERT_EQ(1, decoder_count);
    const nvimgcodecDecoderDesc_t* gpu_decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, &gpu_decoder, &decoder_count, nullptr, NVIMGCODEC_BACKEND_KIND_GPU_ONLY));
    EXPECT_EQ(factory.gpuDecoderDesc(), gpu_decoder);

    int encoder_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getEncoderDescs(framework->instance, nullptr, &encoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_CPU_ONLY));
    ASSERT_EQ(1, encoder_count);
    const nvimgcodecEncoderDesc_t* encoder_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getEncoderDescs(framework->instance, &encoder_desc, &encoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_CPU_ONLY));
    EXPECT_EQ(factory.encoderDesc(), encoder_desc);

    int parser_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, nullptr, &parser_count, kCodecName));
    ASSERT_EQ(1, parser_count);
    const nvimgcodecParserDesc_t* parser_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, &parser_desc, &parser_count, kCodecName));
    EXPECT_EQ(factory.parserDesc(), parser_desc);

    decoder_count = 0;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, nullptr, &decoder_count, "unknown", NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(0, decoder_count);
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        framework->getDecoderDescs(framework->instance, nullptr, &decoder_count, nullptr, static_cast<nvimgcodecBackendKind_t>(99)));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    extension = nullptr;

    extension_count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, nullptr, &extension_count));
    EXPECT_EQ(0, extension_count);
    codec_count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getCodecNames(framework->instance, nullptr, &codec_count, nullptr));
    EXPECT_EQ(0, codec_count);

    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, orderedSelectionOverridesRegisteredDecoderOrder)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);
    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));

    factory.setDecoderStatus(NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
    nvimgcodecDecoder_t decoder = nullptr;
    {
        std::string gpu_id = factory.gpuDecoderDesc()->id;
        std::string cpu_id = factory.cpuDecoderDesc()->id;
        const char* plugin_ids[] = {gpu_id.c_str(), cpu_id.c_str()};
        nvimgcodecPluginSelection_t selection{
            NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION, sizeof(nvimgcodecPluginSelection_t), nullptr, nullptr, 2, plugin_ids};
        nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
        exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
        exec_params.max_num_cpu_threads = 1;
        exec_params.num_selections = 1;
        exec_params.selections = &selection;
        ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    }
    nvimgcodecCodeStream_t code_stream = createTestCodeStream(instance);
    nvimgcodecImage_t image = createTestImage(instance);
    ASSERT_NE(nullptr, code_stream);
    ASSERT_NE(nullptr, image);

    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(decoder, &code_stream, &image, 1, &decode_params, &processing_status, 1));
    EXPECT_EQ(NVIMGCODEC_PROCESSING_STATUS_SUCCESS, processing_status);
    EXPECT_EQ((std::vector<std::string>{factory.gpuDecoderDesc()->id, factory.cpuDecoderDesc()->id}), factory.decoderCalls());

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, absentSelectionPreservesRegisteredDecoderOrder)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);
    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
    exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec_params.max_num_cpu_threads = 1;
    nvimgcodecDecoder_t decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    nvimgcodecCodeStream_t code_stream = createTestCodeStream(instance);
    nvimgcodecImage_t image = createTestImage(instance);

    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(decoder, &code_stream, &image, 1, &decode_params, &processing_status, 1));
    EXPECT_EQ((std::vector<std::string>{factory.cpuDecoderDesc()->id}), factory.decoderCalls());

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, backendFilterTakesPrecedenceOverSelectedDecoderOrder)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);
    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));

    const char* plugin_ids[] = {factory.gpuDecoderDesc()->id, factory.cpuDecoderDesc()->id};
    nvimgcodecPluginSelection_t selection{
        NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION, sizeof(nvimgcodecPluginSelection_t), nullptr, nullptr, 2, plugin_ids};
    nvimgcodecBackend_t backend{NVIMGCODEC_STRUCTURE_TYPE_BACKEND, sizeof(nvimgcodecBackend_t), nullptr, NVIMGCODEC_BACKEND_KIND_CPU_ONLY,
        {NVIMGCODEC_STRUCTURE_TYPE_BACKEND_PARAMS, sizeof(nvimgcodecBackendParams_t), nullptr, 1.0f, NVIMGCODEC_LOAD_HINT_POLICY_FIXED}};
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
    exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec_params.max_num_cpu_threads = 1;
    exec_params.num_backends = 1;
    exec_params.backends = &backend;
    exec_params.num_selections = 1;
    exec_params.selections = &selection;

    nvimgcodecDecoder_t decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    nvimgcodecCodeStream_t code_stream = createTestCodeStream(instance);
    nvimgcodecImage_t image = createTestImage(instance);

    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(decoder, &code_stream, &image, 1, &decode_params, &processing_status, 1));
    EXPECT_EQ((std::vector<std::string>{factory.cpuDecoderDesc()->id}), factory.decoderCalls());

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, codecMappingOmitsUnlistedDecoders)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);
    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));

    std::vector<std::string> other_decoder_calls;
    PluginCallState other_decoder_state{"other_codec_decoder", NVIMGCODEC_PROCESSING_STATUS_SUCCESS, &other_decoder_calls};
    nvimgcodecDecoderDesc_t other_decoder_desc = *factory.cpuDecoderDesc();
    other_decoder_desc.instance = &other_decoder_state;
    other_decoder_desc.id = other_decoder_state.id;
    other_decoder_desc.codec = "other_codec";
    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance, &framework));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->registerDecoder(framework->instance, &other_decoder_desc, NVIMGCODEC_PRIORITY_NORMAL));

    factory.setDecoderStatus(NVIMGCODEC_BACKEND_KIND_GPU_ONLY, NVIMGCODEC_PROCESSING_STATUS_CODESTREAM_UNSUPPORTED);
    const char* plugin_ids[] = {factory.gpuDecoderDesc()->id};
    nvimgcodecPluginSelection_t selection{
        NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION, sizeof(nvimgcodecPluginSelection_t), nullptr, kCodecName, 1, plugin_ids};
    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
    exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec_params.max_num_cpu_threads = 1;
    exec_params.num_selections = 1;
    exec_params.selections = &selection;

    nvimgcodecDecoder_t decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    nvimgcodecCodeStream_t code_stream = createTestCodeStream(instance);
    nvimgcodecImage_t image = createTestImage(instance);

    nvimgcodecDecodeParams_t decode_params{NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS, sizeof(nvimgcodecDecodeParams_t), nullptr};
    nvimgcodecProcessingStatus_t processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCanDecode(decoder, &code_stream, &image, 1, &decode_params, &processing_status, 1));
    EXPECT_EQ((std::vector<std::string>{factory.gpuDecoderDesc()->id}), factory.decoderCalls());

    std::vector<unsigned char> output_buffer;
    nvimgcodecCodeStream_t other_code_stream = createTestOutputCodeStream(instance, other_decoder_desc.codec, &output_buffer);
    processing_status = NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        nvimgcodecDecoderCanDecode(decoder, &other_code_stream, &image, 1, &decode_params, &processing_status, 1));
    EXPECT_TRUE(other_decoder_calls.empty());

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecImageDestroy(image));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(other_code_stream));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->unregisterDecoder(framework->instance, &other_decoder_desc));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, rejectsInvalidPluginSelections)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);
    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
    exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec_params.max_num_cpu_threads = 1;
    exec_params.num_selections = 1;

    const char* unknown_ids[] = {"unknown_decoder"};
    nvimgcodecPluginSelection_t selection{
        NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION, sizeof(nvimgcodecPluginSelection_t), nullptr, nullptr, 1, unknown_ids};
    exec_params.selections = &selection;
    nvimgcodecDecoder_t decoder = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    selection.struct_size = sizeof(nvimgcodecPluginSelection_t) - 1;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    selection.struct_size = sizeof(nvimgcodecPluginSelection_t);
    selection.struct_type = NVIMGCODEC_STRUCTURE_TYPE_UNKNOWN;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));
    selection.struct_type = NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION;

    const char* duplicate_ids[] = {factory.cpuDecoderDesc()->id, factory.cpuDecoderDesc()->id};
    selection.num_plugin_ids = 2;
    selection.plugin_ids = duplicate_ids;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    const char* wrong_codec_ids[] = {factory.cpuDecoderDesc()->id};
    selection.codec_name = "wrong_codec";
    selection.num_plugin_ids = 1;
    selection.plugin_ids = wrong_codec_ids;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    selection.codec_name = nullptr;
    const char* encoder_id[] = {factory.encoderDesc()->id};
    selection.plugin_ids = encoder_id;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    nvimgcodecEncoder_t encoder = nullptr;
    const char* decoder_id[] = {factory.cpuDecoderDesc()->id};
    selection.plugin_ids = decoder_id;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr));

    nvimgcodecPluginSelection_t duplicate_codec_selections[] = {
        {NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION, sizeof(nvimgcodecPluginSelection_t), nullptr, kCodecName, 1, decoder_id},
        {NVIMGCODEC_STRUCTURE_TYPE_PLUGIN_SELECTION, sizeof(nvimgcodecPluginSelection_t), nullptr, kCodecName, 1, decoder_id}};
    exec_params.num_selections = 2;
    exec_params.selections = duplicate_codec_selections;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    duplicate_codec_selections[0].codec_name = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, rejectsExtensionAndInstanceDestroyWhileRegistryClientsAreAlive)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);

    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionCreate(instance, &extension, factory.extensionDesc()));

    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance, &framework));
    auto expect_registry_mutation_blocked = [&] {
        nvimgcodecExtension_t second_extension = nullptr;
        EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE,
            nvimgcodecExtensionCreate(instance, &second_extension, factory.extensionDesc()));
        EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE,
            framework->registerDecoder(framework->instance, factory.cpuDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL));
        EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE,
            framework->registerEncoder(framework->instance, factory.encoderDesc(), NVIMGCODEC_PRIORITY_NORMAL));
        EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE,
            framework->registerParser(framework->instance, factory.parserDesc(), NVIMGCODEC_PRIORITY_NORMAL));
        EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE,
            framework->registerFileExtension(framework->instance, "dynamic_codec", ".dynamic", 0));
        EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE,
            framework->unregisterFileExtension(framework->instance, kCodecName, ".dynamic"));
    };

    nvimgcodecExecutionParams_t exec_params{NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS, sizeof(nvimgcodecExecutionParams_t), nullptr};
    exec_params.device_id = NVIMGCODEC_DEVICE_CPU_ONLY;
    exec_params.max_num_cpu_threads = 1;
    nvimgcodecDecoder_t decoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderCreate(instance, &decoder, &exec_params, nullptr));

    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, nvimgcodecInstanceDestroy(instance));
    expect_registry_mutation_blocked();

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecDecoderDestroy(decoder));

    nvimgcodecEncoder_t encoder = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderCreate(instance, &encoder, &exec_params, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, nvimgcodecInstanceDestroy(instance));
    expect_registry_mutation_blocked();
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecEncoderDestroy(encoder));

    nvimgcodecCodeStream_t code_stream = createTestCodeStream(instance);
    ASSERT_NE(nullptr, code_stream);
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, nvimgcodecInstanceDestroy(instance));
    expect_registry_mutation_blocked();
    nvimgcodecStatus_t concurrent_registration_status = NVIMGCODEC_STATUS_SUCCESS;
    std::thread concurrent_registration([&] {
        concurrent_registration_status =
            framework->registerFileExtension(framework->instance, "concurrent_codec", ".concurrent", 0);
    });
    concurrent_registration.join();
    EXPECT_EQ(NVIMGCODEC_STATUS_RESOURCE_IN_USE, concurrent_registration_status);
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecCodeStreamDestroy(code_stream));

    int dynamic_codec_count = -1;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getCodecNames(framework->instance, nullptr, &dynamic_codec_count, ".dynamic"));
    EXPECT_EQ(0, dynamic_codec_count);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, supportsDirectRegistrationThroughReturnedFramework)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);

    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance, &framework));

    RegisteredPluginsTestExtensionFactory factory;
    nvimgcodecDecoderDesc_t incomplete_desc = *factory.cpuDecoderDesc();
    incomplete_desc.create = nullptr;
    EXPECT_EQ(
        NVIMGCODEC_STATUS_INVALID_PARAMETER, framework->registerDecoder(framework->instance, &incomplete_desc, NVIMGCODEC_PRIORITY_NORMAL));
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, framework->registerDecoder(framework->instance, factory.cpuDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        framework->registerDecoder(framework->instance, factory.cpuDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL));
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, framework->registerDecoder(framework->instance, factory.gpuDecoderDesc(), NVIMGCODEC_PRIORITY_NORMAL));
    ASSERT_EQ(
        NVIMGCODEC_STATUS_SUCCESS, framework->registerEncoder(framework->instance, factory.encoderDesc(), NVIMGCODEC_PRIORITY_NORMAL));
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->registerParser(framework->instance, factory.parserDesc(), NVIMGCODEC_PRIORITY_NORMAL));

    int decoder_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, nullptr, &decoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    ASSERT_EQ(2, decoder_count);
    std::array<const nvimgcodecDecoderDesc_t*, 2> decoder_descs{};
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, decoder_descs.data(), &decoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(factory.cpuDecoderDesc(), decoder_descs[0]);
    EXPECT_EQ(factory.gpuDecoderDesc(), decoder_descs[1]);

    int encoder_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getEncoderDescs(framework->instance, nullptr, &encoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    ASSERT_EQ(1, encoder_count);
    const nvimgcodecEncoderDesc_t* encoder_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getEncoderDescs(framework->instance, &encoder_desc, &encoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(factory.encoderDesc(), encoder_desc);

    int parser_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, nullptr, &parser_count, kCodecName));
    ASSERT_EQ(1, parser_count);
    const nvimgcodecParserDesc_t* parser_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, &parser_desc, &parser_count, kCodecName));
    EXPECT_EQ(factory.parserDesc(), parser_desc);

    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->unregisterParser(framework->instance, factory.parserDesc()));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->unregisterEncoder(framework->instance, factory.encoderDesc()));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->unregisterDecoder(framework->instance, factory.gpuDecoderDesc()));
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->unregisterDecoder(framework->instance, factory.cpuDecoderDesc()));
    decoder_count = -1;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getDecoderDescs(framework->instance, nullptr, &decoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(0, decoder_count);
    encoder_count = -1;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS,
        framework->getEncoderDescs(framework->instance, nullptr, &encoder_count, kCodecName, NVIMGCODEC_BACKEND_KIND_ANY));
    EXPECT_EQ(0, encoder_count);
    parser_count = -1;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, nullptr, &parser_count, kCodecName));
    EXPECT_EQ(0, parser_count);

    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, listsAndDestroysAutomaticallyLoadedBuiltinExtension)
{
    nvimgcodecInstanceCreateInfo_t create_info{
        NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, sizeof(nvimgcodecInstanceCreateInfo_t), nullptr};
    create_info.load_builtin_modules = 1;
    nvimgcodecInstance_t instance = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceCreate(&instance, &create_info));

    int extension_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, nullptr, &extension_count));
    ASSERT_EQ(1, extension_count);
    nvimgcodecExtension_t extension = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, &extension, &extension_count));

    const nvimgcodecExtensionDesc_t* extension_desc = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionGetExtensionDesc(extension, &extension_desc));
    ASSERT_STREQ("nvimgcodec_builtin_parsers", extension_desc->id);

    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance, &framework));
    int parser_count = 0;
    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, nullptr, &parser_count, nullptr));
    EXPECT_GT(parser_count, 0);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecExtensionDestroy(extension));
    extension_count = -1;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetExtensions(instance, nullptr, &extension_count));
    EXPECT_EQ(0, extension_count);
    parser_count = -1;
    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, framework->getParserDescs(framework->instance, nullptr, &parser_count, nullptr));
    EXPECT_EQ(0, parser_count);

    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

TEST(QueryRegisteredPluginsApiTest, validatesPublicQueryArguments)
{
    nvimgcodecInstance_t instance = createEmptyInstance();
    ASSERT_NE(nullptr, instance);

    const nvimgcodecFrameworkDesc_t* framework = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecGetFrameworkDesc(nullptr, &framework));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecGetFrameworkDesc(instance, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecGetExtensions(nullptr, nullptr, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecGetExtensions(instance, nullptr, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecExtensionGetExtensionDesc(nullptr, nullptr));
    nvimgcodecExtension_t extension = reinterpret_cast<nvimgcodecExtension_t>(1);
    int extension_count = 1;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, nvimgcodecGetExtensions(instance, &extension, &extension_count));
    EXPECT_EQ(1, extension_count);
    EXPECT_EQ(reinterpret_cast<nvimgcodecExtension_t>(1), extension);

    ASSERT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecGetFrameworkDesc(instance, &framework));
    int count = -1;
    const nvimgcodecDecoderDesc_t* decoder_desc = nullptr;
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER, framework->getCodecNames(framework->instance, nullptr, nullptr, nullptr));
    EXPECT_EQ(NVIMGCODEC_STATUS_INVALID_PARAMETER,
        framework->getDecoderDescs(framework->instance, &decoder_desc, &count, nullptr, NVIMGCODEC_BACKEND_KIND_ANY));

    EXPECT_EQ(NVIMGCODEC_STATUS_SUCCESS, nvimgcodecInstanceDestroy(instance));
}

}} // namespace nvimgcodec::test
