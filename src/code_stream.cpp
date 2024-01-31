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

#include "code_stream.h"
#include <cstring>
#include <iostream>
#include <string>
#include "codec.h"
#include "codec_registry.h"
#include "exception.h"
#include "image_parser.h"
namespace nvimgcodec {

CodeStream::CodeStream(ICodecRegistry* codec_registry, std::unique_ptr<IIoStreamFactory> io_stream_factory)
    : codec_registry_(codec_registry)
    , parser_(nullptr)
    , io_stream_factory_(std::move(io_stream_factory))
    , io_stream_(nullptr)
    , io_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_IO_STREAM_DESC, sizeof(nvimgcodecIoStreamDesc_t), nullptr, this, read_static, write_static, putc_static, skip_static,
          seek_static, tell_static, size_static, reserve_static, flush_static, map_static, unmap_static}
    , code_stream_desc_{NVIMGCODEC_STRUCTURE_TYPE_CODE_STREAM_DESC, sizeof(nvimgcodecCodeStreamDesc_t), nullptr, this, &io_stream_desc_, static_get_image_info}
    , image_info_(nullptr)
{
}

CodeStream::~CodeStream()
{
}

void CodeStream::parse()
{
    auto parser = codec_registry_->getParser(&code_stream_desc_);
    if (!parser)
        throw Exception(UNSUPPORTED_FORMAT_STATUS, "The encoded stream did not match any of the available format parsers",
            "CodeStream::parse - Encoded stream parsing");

    parser_ = std::move(parser);
}

void CodeStream::parseFromFile(const std::string& file_name)
{
    io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, false, false);
    parse();
}

void CodeStream::parseFromMem(const unsigned char* data, size_t size)
{
    io_stream_ = io_stream_factory_->createMemIoStream(data, size);
    parse();
}
void CodeStream::setOutputToFile(const char* file_name)
{
    io_stream_ = io_stream_factory_->createFileIoStream(file_name, false, false, true);
}

void CodeStream::setOutputToHostMem(void* ctx, nvimgcodecResizeBufferFunc_t resize_buffer_func)
{
    io_stream_ = io_stream_factory_->createMemIoStream(ctx, resize_buffer_func);
}

nvimgcodecStatus_t CodeStream::getImageInfo(nvimgcodecImageInfo_t* image_info)
{
    assert(image_info);
    if (image_info->struct_next) {
        // If we have some linked structure, we might need to ask the parser again
        assert(parser_);
        return parser_->getImageInfo(&code_stream_desc_, image_info);
    } else if (!image_info_) {
        // If no linked structure, but it's the first time we parse, we ask the parser and store the results
        assert(parser_);
        image_info_ = std::make_unique<nvimgcodecImageInfo_t>();
        image_info_->struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
        image_info_->struct_size = sizeof(nvimgcodecImageInfo_t);
        image_info_->struct_next = image_info->struct_next; // TODO(janton): temp solution but we probably need deep copy
        auto res = parser_->getImageInfo(&code_stream_desc_, image_info_.get());
        if (res != NVIMGCODEC_STATUS_SUCCESS) {
            image_info_.reset();
            return res;
        }
    }
    // Otherwise, we just return the previous info structure
    *image_info = *image_info_.get();
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::setImageInfo(const nvimgcodecImageInfo_t* image_info)
{
    if (!image_info_) {
        image_info_ = std::make_unique<nvimgcodecImageInfo_t>();
    }
    *image_info_.get() = *image_info;
    return NVIMGCODEC_STATUS_SUCCESS;
}

std::string CodeStream::getCodecName() const
{
    return image_info_ ? std::string(image_info_->codec_name) : parser_->getCodecName();
}

ICodec* CodeStream::getCodec() const
{
    return codec_registry_->getCodecByName(getCodecName().c_str());
}

nvimgcodecIoStreamDesc_t* CodeStream::getInputStreamDesc()
{
    return &io_stream_desc_;
}

nvimgcodecCodeStreamDesc_t* CodeStream::getCodeStreamDesc()
{
    return &code_stream_desc_;
}

nvimgcodecStatus_t CodeStream::read(size_t* output_size, void* buf, size_t bytes)
{
    assert(io_stream_);
    *output_size = io_stream_->read(buf, bytes);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::write(size_t* output_size, void* buf, size_t bytes)
{
    assert(io_stream_);
    *output_size = io_stream_->write(buf, bytes);
    return NVIMGCODEC_STATUS_SUCCESS;
}
nvimgcodecStatus_t CodeStream::putc(size_t* output_size, unsigned char ch)
{
    assert(io_stream_);
    *output_size = io_stream_->putc(ch);

    return *output_size == 1 ? NVIMGCODEC_STATUS_SUCCESS : NVIMGCODEC_STATUS_BAD_CODESTREAM;
}

nvimgcodecStatus_t CodeStream::skip(size_t count)
{
    assert(io_stream_);
    io_stream_->skip(count);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::seek(ptrdiff_t offset, int whence)
{
    assert(io_stream_);
    io_stream_->seek(offset, whence);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::tell(ptrdiff_t* offset)
{
    assert(io_stream_);
    *offset = io_stream_->tell();
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::size(size_t* size)
{
    assert(io_stream_);
    *size = io_stream_->size();
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::reserve(size_t bytes)
{
    assert(io_stream_);
    io_stream_->reserve(bytes);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::flush()
{
    assert(io_stream_);
    io_stream_->flush();
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::map(void** addr, size_t offset, size_t size)
{
    assert(io_stream_);
    *addr = io_stream_->map(offset, size);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::unmap(void* addr, size_t size)
{
    assert(io_stream_);
    io_stream_->unmap(addr, size);
    return NVIMGCODEC_STATUS_SUCCESS;
}

nvimgcodecStatus_t CodeStream::read_static(void* instance, size_t* output_size, void* buf, size_t bytes)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->read(output_size, buf, bytes);
}

nvimgcodecStatus_t CodeStream::write_static(void* instance, size_t* output_size, void* buf, size_t bytes)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->write(output_size, buf, bytes);
}

nvimgcodecStatus_t CodeStream::putc_static(void* instance, size_t* output_size, unsigned char ch)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->putc(output_size, ch);
}

nvimgcodecStatus_t CodeStream::skip_static(void* instance, size_t count)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->skip(count);
}

nvimgcodecStatus_t CodeStream::seek_static(void* instance, ptrdiff_t offset, int whence)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->seek(offset, whence);
}

nvimgcodecStatus_t CodeStream::tell_static(void* instance, ptrdiff_t* offset)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->tell(offset);
}

nvimgcodecStatus_t CodeStream::size_static(void* instance, size_t* size)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->size(size);
}

nvimgcodecStatus_t CodeStream::reserve_static(void* instance, size_t bytes)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->reserve(bytes);
}

nvimgcodecStatus_t CodeStream::flush_static(void* instance)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->flush();
}

nvimgcodecStatus_t CodeStream::map_static(void* instance, void** addr, size_t offset, size_t size)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->map(addr, offset, size);
}

nvimgcodecStatus_t CodeStream::unmap_static(void* instance, void* addr, size_t size)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    return handle->unmap(addr, size);
}

nvimgcodecStatus_t CodeStream::static_get_image_info(void* instance, nvimgcodecImageInfo_t* result)
{
    assert(instance);
    CodeStream* handle = reinterpret_cast<CodeStream*>(instance);
    handle->getImageInfo(result);
    return NVIMGCODEC_STATUS_SUCCESS;
}

} // namespace nvimgcodec