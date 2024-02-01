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

#include "file_io_stream.h"

namespace nvimgcodec {

class MmapedFileIoStream : public FileIoStream
{
  public:
    explicit MmapedFileIoStream(const std::string& path, bool read_ahead);
    void close() override;
    std::shared_ptr<void> get(size_t n_bytes) override;
    static bool reserveFileMappings(unsigned int num);
    static void freeFileMappings(unsigned int num);
    std::size_t read(void* buffer, size_t n_bytes) override;
    void seek(int64_t pos, int whence = SEEK_SET) override;
    int64_t tell() const override;
    std::size_t size() const override;

    ~MmapedFileIoStream() override { MmapedFileIoStream::close(); }

  private:
    std::shared_ptr<void> p_;
    std::size_t length_;
    std::size_t pos_;
    bool read_ahead_whole_file_;
};
} // namespace nvimgcodec