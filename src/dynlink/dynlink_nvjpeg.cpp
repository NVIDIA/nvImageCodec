/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cuda.h>
#include <stdio.h>
#include <dlfcn.h>
#include <mutex>
#include <string>
#include <unordered_map>
#include "library_loader.h"

namespace {


#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)

  static const char __NvjpegLibName[] = "libnvjpeg.so";
  #if CUDA_VERSION_MAJOR >= 12
  static const char __NvjpegLibNameCuVer[] = "libnvjpeg.so.12";
  #elif CUDA_VERSION_MAJOR >= 11
  static const char __NvjpegLibNameCuVer[] = "libnvjpeg.so.11";
  #else
  static const char __NvjpegLibNameCuVer[] = "libnvjpeg.so.10";
  #endif

#elif defined(_WIN32) || defined(_WIN64)

  static const char __NvjpegLibName[] = "nvjpeg.dll";

  #if CUDA_VERSION_MAJOR >= 12
  static const char __NvjpegLibNameCuVer[] = "nvjpeg64_12.dll";
  #elif CUDA_VERSION_MAJOR >= 11
  static const char __NvjpegLibNameCuVer[] = "nvjpeg64_11.dll";
  #else
  static const char __NvjpegLibNameCuVer[] = "nvjpeg64_10.dll";
  #endif

#endif


nvimgcodec::ILibraryLoader::LibraryHandle loadNvjpegLibrary()
{
    nvimgcodec::LibraryLoader lib_loader;
    nvimgcodec::ILibraryLoader::LibraryHandle ret = nullptr;
    ret = lib_loader.loadLibrary(__NvjpegLibNameCuVer);
    if (!ret) {
        ret = lib_loader.loadLibrary(__NvjpegLibName);
        if (!ret) {
#if defined(__linux__) || defined(__linux) || defined(linux) || defined(_LINUX)
            fprintf(stderr, "dlopen libnvjpeg.so failed!. Please install CUDA toolkit or nvJPEG python wheel.");
#elif defined(_WIN32) || defined(_WIN64)
            fprintf(stderr, "LoadLibrary nvjpeg.dll failed!. Please install CUDA toolkit or nvJPEG python wheel.");
#endif
        }
    }
    return ret;
}

}  // namespace

void *NvjpegLoadSymbol(const char *name) {
  nvimgcodec::LibraryLoader lib_loader;
  static nvimgcodec::ILibraryLoader::LibraryHandle nvjpegDrvLib = loadNvjpegLibrary();
  // check processing library, core later if symbol not found
  try {
    void *ret = nvjpegDrvLib ? lib_loader.getFuncAddress(nvjpegDrvLib, name) : nullptr;
    return ret;
  } catch (...) {
    return nullptr;
  }
}

bool nvjpegIsSymbolAvailable(const char *name) {
  static std::mutex symbol_mutex;
  static std::unordered_map<std::string, void*> symbol_map;
  std::lock_guard<std::mutex> lock(symbol_mutex);
  auto it = symbol_map.find(name);
  if (it == symbol_map.end()) {
    auto *ptr = NvjpegLoadSymbol(name);
    symbol_map.insert({name, ptr});
    return ptr != nullptr;
  }
  return it->second != nullptr;
}
