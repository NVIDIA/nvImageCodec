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

#include "std_file_io_stream.h"
#include <errno.h>
#include <sys/stat.h>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <memory>
#include <string>

namespace nvimgcodec {

StdFileIoStream::StdFileIoStream(const std::string& path, bool to_write)
    : FileIoStream(path)
{
    fp_ = std::fopen(path.c_str(), to_write ? "wb" : "rb");
    if (fp_ == nullptr)
        throw std::runtime_error("Could not open file " + path + ": " + std::strerror(errno));
}

void StdFileIoStream::close()
{
    if (fp_ != nullptr) {
        std::fclose(fp_);
        fp_ = nullptr;
    }
}

void StdFileIoStream::seek(int64_t pos, int whence)
{
    if (std::fseek(fp_, static_cast<long>(pos), whence))
        throw std::runtime_error(std::string("Seek operation failed: ") + std::strerror(errno));
}

int64_t StdFileIoStream::tell() const
{
    return std::ftell(fp_);
}

std::size_t StdFileIoStream::read(void* buffer, size_t n_bytes)
{
    size_t n_read = std::fread(buffer, 1, n_bytes, fp_);
    return n_read;
}

std::size_t StdFileIoStream::write(void* buffer, size_t n_bytes)
{
    size_t n_written = std::fwrite(buffer, 1, n_bytes, fp_);
    return n_written;
}

std::size_t StdFileIoStream::putc(unsigned char ch)
{
    size_t n_written = std::fputc(ch, fp_);
    return n_written;
}

std::shared_ptr<void> StdFileIoStream::get(size_t /*n_bytes*/)
{
    // this function should return a pointer inside mmaped file
    // it doesn't make sense in case of StdFileIoStream
    return {};
}

std::size_t StdFileIoStream::size() const
{
    struct stat sb;
    if (stat(path_.c_str(), &sb) == -1) {
        throw std::runtime_error("Unable to stat file " + path_ + ": " + std::strerror(errno));
    }
    return sb.st_size;
}
} //namespace nvimgcodec