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

#include "codec_registry.h"
#include "codec.h"
#include "iimage_parser.h"
#include "log.h"

#include <iostream>
#include <stdexcept>

namespace nvimgcodec {

CodecRegistry::CodecRegistry(ILogger* logger)
    : logger_(logger)
{
}

void CodecRegistry::registerCodec(std::unique_ptr<ICodec> codec)
{
    if (by_name_.find(codec->name()) != by_name_.end())
        throw std::invalid_argument("A different codec with the same name already registered.");
    // TODO(janton): Figure out a way to give JPEG a higher priority without this
    //               Perhaps codec should come with a priority too
    if (codec->name() == "jpeg") {
        codec_ptrs_.push_front(codec.get());
    } else {
        codec_ptrs_.push_back(codec.get());
    }
    by_name_.insert(std::make_pair(codec->name(), std::move(codec)));
}

std::unique_ptr<IImageParser> CodecRegistry::getParser(
    nvimgcodecCodeStreamDesc_t* code_stream) const
{
    NVIMGCODEC_LOG_TRACE(logger_, "CodecRegistry::getParser");
    for (auto* codec : codec_ptrs_) {
        std::unique_ptr<IImageParser> parser = codec->createParser(code_stream);
        if (parser) {
            return parser;
        }
    }

    return nullptr;
}

ICodec* CodecRegistry::getCodecByName(const char* name)
{
    auto it = by_name_.find(name);
    if (it != by_name_.end())
        return it->second.get();
    else
        return nullptr;
}

size_t CodecRegistry::getCodecsCount() const
{
    return codec_ptrs_.size();
}

ICodec* CodecRegistry::getCodecByIndex(size_t index)
{
    return codec_ptrs_[index];
}

} // namespace nvimgcodec