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

#include <map>
#include <memory>
#include <string>

#include "iimage_decoder_factory.h"
#include "iimage_encoder_factory.h"
#include "iimage_parser_factory.h"

#include "icodec.h"

namespace nvimgcodec {

class IImageParser;
class IImageEncoder;
class IImageDecoder;
class ILogger;

class Codec : public ICodec
{
  public:
    explicit Codec(ILogger* logger, const char* name);
    const std::string& name() const override;
    std::unique_ptr<IImageParser> createParser(nvimgcodecCodeStreamDesc_t* code_stream) const override;
    int getDecodersNum() const override;
    IImageDecoderFactory* getDecoderFactory(int index) const override;
    int getEncodersNum() const override;
    IImageEncoderFactory* getEncoderFactory(int index) const override;
    void registerEncoderFactory(std::unique_ptr<IImageEncoderFactory> factory, float priority) override;
    void unregisterEncoderFactory(const std::string encoder_id) override;
    void registerDecoderFactory(std::unique_ptr<IImageDecoderFactory> factory, float priority) override;
    void unregisterDecoderFactory(const std::string decoder_id) override;
    void registerParserFactory(std::unique_ptr<IImageParserFactory> factory, float priority) override;
    void unregisterParserFactory(const std::string parser_id) override;

  private:
    ILogger* logger_;
    std::string name_;
    std::multimap<float, std::unique_ptr<IImageParserFactory>> parsers_;
    std::multimap<float, std::unique_ptr<IImageEncoderFactory>> encoders_;
    std::multimap<float, std::unique_ptr<IImageDecoderFactory>> decoders_;
};
} // namespace nvimgcodec
