/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#pragma once

#include <nvimgcodec.h>
#include "iimage_decoder.h"
#include "iimage_decoder_factory.h"
#include "image_generic_codec.h"

namespace nvimgcodec {

class ImageGenericDecoder : public ImageGenericCodec<ImageGenericDecoder, IImageDecoderFactory, IImageDecoder> {
  public:
    explicit ImageGenericDecoder(
            ILogger* logger, ICodecRegistry* codec_registry, const nvimgcodecExecutionParams_t* exec_params, const char* options = nullptr)
                : Base(logger, codec_registry, exec_params, options) {}
    
    ~ImageGenericDecoder() override = default;

    void canDecode(const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images,
            const nvimgcodecDecodeParams_t* params, nvimgcodecProcessingStatus_t* processing_status, int force_format);

    ProcessingResultsPromise::FutureImpl decode(
        const std::vector<ICodeStream*>& code_streams, const std::vector<IImage*>& images, const nvimgcodecDecodeParams_t* params);

    using Base = ImageGenericCodec<ImageGenericDecoder, IImageDecoderFactory, IImageDecoder>;
    using Factory = IImageDecoderFactory;
    using Processor = IImageDecoder;
    using Entry = SampleEntry<typename Base::ProcessorEntry>;

    static const std::string& process_name()
    {
        static std::string name{"decode"};
        return name;
    };

    static size_t getProcessorCount(ICodec* codec) {
        return codec->getDecodersNum();
    }

    static IImageDecoderFactory* getFactory(ICodec* codec, int i) {
        return codec->getDecoderFactory(i);
    }

    static std::string getId(Factory* factory) {
        return factory->getDecoderId();
    }

    static std::unique_ptr<IImageDecoder> createInstance(
        const Factory* factory, const nvimgcodecExecutionParams_t* exec_params, const char* options)
    {
        return factory->createDecoder(exec_params, options);
    }

    static bool hasBatchedAPI(Processor* processor) {
        return processor->hasDecodeBatch();
    }

    static int getMiniBatchSize(Processor* processor) {
        return processor->getMiniBatchSize();
    }

    static int getProcessorsNum(ICodec* codec) {
        return codec->getDecodersNum();
    }

    void sortSamples();
    bool canProcessImpl(Entry& sample, int tid);
    bool processImpl(Entry& sample, int tid);
    bool processBatchImpl(ProcessorEntry& processor);
    bool allocateTempBuffers(Entry& sample);
    void copyToOutputBuffer(const nvimgcodecImageInfo_t& output_info, const nvimgcodecImageInfo_t& info);
    void postSyncCudaThreads(int tid);

    const nvimgcodecDecodeParams_t* curr_params_;

    // to sort
    std::vector<uint8_t> subsampling_score_;
    std::vector<uint64_t> area_;
};

} // namespace nvimgcodec
