#!/bin/bash -xe

# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#zlib
ZLIB_VERSION=1.3.1.2
ZLIB_URL=https://github.com/madler/zlib/archive/refs/tags/v${ZLIB_VERSION}.tar.gz
ZLIB_SHA256=fbf1c8476136693e6c3f1fa26e6d8c4f2c8b5a5c44340c04df349dad02eed09e

# Download and extract zlib if not already present
if [ ! -d external/zlib-${ZLIB_VERSION} ]; then
    pushd external
    wget -O zlib-${ZLIB_VERSION}.tar.gz ${ZLIB_URL}
    # Verify checksum for security
    echo "${ZLIB_SHA256} zlib-${ZLIB_VERSION}.tar.gz" | sha256sum -c || { rm zlib-${ZLIB_VERSION}.tar.gz; exit 1; }
    tar -xf zlib-${ZLIB_VERSION}.tar.gz
    rm zlib-${ZLIB_VERSION}.tar.gz
    popd
fi

pushd external/zlib-${ZLIB_VERSION}

mkdir -p build_dir
pushd build_dir

echo "set(CMAKE_SYSTEM_NAME Linux)" > toolchain.cmake
echo "set(CMAKE_SYSTEM_PROCESSOR ${CMAKE_TARGET_ARCH})" >> toolchain.cmake
echo "set(CMAKE_C_COMPILER ${CC_COMP})" >> toolchain.cmake
echo "set(CMAKE_CXX_COMPILER ${CXX_COMP})" >> toolchain.cmake
# only when cross compiling
if [ "${CC_COMP}" != "gcc" ]; then
    echo "set(CMAKE_FIND_ROOT_PATH ${INSTALL_PREFIX})" >> toolchain.cmake
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)" >> toolchain.cmake
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)" >> toolchain.cmake
    echo "set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)" >> toolchain.cmake
fi
echo "set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")" >> toolchain.cmake
echo "set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")" >> toolchain.cmake
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -DZLIB_BUILD_SHARED=OFF \
      -DZLIB_BUILD_STATIC=ON \
      ..
make -j${NPROC}
make install

popd
rm -rf build_dir

popd

