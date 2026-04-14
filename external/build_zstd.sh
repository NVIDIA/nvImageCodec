#!/bin/bash -xe

# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# zstandard compression library
ZSTD_VERSION=1.5.7
ZSTD_URL=https://github.com/facebook/zstd/archive/refs/tags/v${ZSTD_VERSION}.tar.gz
ZSTD_SHA256=37d7284556b20954e56e1ca85b80226768902e2edabd3b649e9e72c0c9012ee3

# Download and extract zstd if not already present
if [ ! -d external/zstd-${ZSTD_VERSION} ]; then
    pushd external
    wget -O zstd-${ZSTD_VERSION}.tar.gz ${ZSTD_URL}
    # Verify checksum for security
    echo "${ZSTD_SHA256} zstd-${ZSTD_VERSION}.tar.gz" | sha256sum -c || { rm zstd-${ZSTD_VERSION}.tar.gz; exit 1; }
    tar -xf zstd-${ZSTD_VERSION}.tar.gz
    rm zstd-${ZSTD_VERSION}.tar.gz
    popd
fi

pushd external/zstd-${ZSTD_VERSION}

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
      -DZSTD_BUILD_SHARED=OFF \
      -DZSTD_BUILD_STATIC=ON \
      ../build/cmake
make -j${NPROC}
make install

popd
rm -rf build_dir

popd


