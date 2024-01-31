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

//#include "mmaped_file_io_stream.h"
#include <cassert>

namespace nvimgcodec {

std::unique_ptr<FileIoStream> FileIoStream::open(
    const std::string& uri, bool read_ahead, bool use_mmap, bool to_write)
{
    std::string processed_uri;

    if (uri.find("file://") == 0) {
        processed_uri = uri.substr(std::string("file://").size());
    } else {
        processed_uri = uri;
    }

    if (use_mmap) {
        assert(!"TODO");
        return std::unique_ptr<FileIoStream>(new StdFileIoStream(processed_uri, to_write));
        // return std::unique_ptr<FileIoStream>(new MmapedFileIoStream(processed_uri,
        // read_ahead));
    } else {
        return std::unique_ptr<FileIoStream>(new StdFileIoStream(processed_uri, to_write));
    }
}

bool FileIoStream::reserveFileMappings(unsigned int num)
{
    return false;
    //MmapedFileIoStream::reserveFileMappings(num);
}

void FileIoStream::freeFileMappings(unsigned int num)
{
    //MmapedFileIoStream::freeFileMappings(num);
}

} // namespace nvimgcodec