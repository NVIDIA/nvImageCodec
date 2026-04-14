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

# libjpeg-turbo
LIBJPEG_TURBO_VERSION=3.1.3
LIBJPEG_TURBO_URL=https://github.com/libjpeg-turbo/libjpeg-turbo/archive/refs/tags/${LIBJPEG_TURBO_VERSION}.tar.gz
LIBJPEG_TURBO_SHA256=3a13a5ba767dc8264bc40b185e41368a80d5d5f945944d1dbaa4b2fb0099f4e5

# Download and extract libjpeg-turbo if not already present
if [ ! -d external/libjpeg-turbo-${LIBJPEG_TURBO_VERSION} ]; then
    pushd external
    wget -O libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz ${LIBJPEG_TURBO_URL}
    # Verify checksum for security
    echo "${LIBJPEG_TURBO_SHA256} libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz" | sha256sum -c || { rm libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz; exit 1; }
    tar -xf libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz
    rm libjpeg-turbo-${LIBJPEG_TURBO_VERSION}.tar.gz
    popd
fi

pushd external/libjpeg-turbo-${LIBJPEG_TURBO_VERSION}/

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
      -DENABLE_SHARED=FALSE -DENABLE_STATIC=TRUE \
      ..
make -j${NPROC}
make install

popd
rm -rf build_dir

popd
