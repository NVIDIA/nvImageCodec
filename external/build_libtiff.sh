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

# libtiff
LIBTIFF_VERSION=4.7.1
LIBTIFF_URL=https://gitlab.com/libtiff/libtiff/-/archive/v${LIBTIFF_VERSION}/libtiff-v${LIBTIFF_VERSION}.tar.gz
LIBTIFF_SHA256=6ab956415ddb6cae497faad18398ff0fd056d17d8bf7b5921cc194c55b0191fc

# Download and extract libtiff if not already present
if [ ! -d external/libtiff-${LIBTIFF_VERSION} ]; then
    pushd external
    wget -O libtiff-${LIBTIFF_VERSION}.tar.gz ${LIBTIFF_URL}
    # Verify checksum for security
    echo "${LIBTIFF_SHA256} libtiff-${LIBTIFF_VERSION}.tar.gz" | sha256sum -c || { rm libtiff-${LIBTIFF_VERSION}.tar.gz; exit 1; }
    tar -xf libtiff-${LIBTIFF_VERSION}.tar.gz
    mv libtiff-v${LIBTIFF_VERSION} libtiff-${LIBTIFF_VERSION}
    rm libtiff-${LIBTIFF_VERSION}.tar.gz
    popd
fi

export CURRDIR=$PWD
pushd external/libtiff-${LIBTIFF_VERSION}
patch -p1 < $CURRDIR/external/patches/0001-Fix-wget-complaing-about-expired-git.savannah.gnu.or.patch

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

# Create a CMake script to define ZSTD::ZSTD and CMath::CMath targets
DEPS_TARGET_CMAKE=${INSTALL_PREFIX}/define_libtiff_deps_targets.cmake
cat > "${DEPS_TARGET_CMAKE}" << EOF
# Define ZSTD::ZSTD target
if(NOT TARGET ZSTD::ZSTD)
  add_library(ZSTD::ZSTD STATIC IMPORTED)
  set_target_properties(ZSTD::ZSTD PROPERTIES
    IMPORTED_LOCATION "${INSTALL_PREFIX}/lib/libzstd.a"
    INTERFACE_INCLUDE_DIRECTORIES "${INSTALL_PREFIX}/include"
    INTERFACE_LINK_LIBRARIES "pthread"
  )
endif()
EOF

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=toolchain.cmake \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
      -DCMAKE_PROJECT_INCLUDE=${DEPS_TARGET_CMAKE} \
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON -DBUILD_SHARED_LIBS=OFF \
      -Dtiff-docs=OFF \
      -Djbig=OFF \
      ..
make -j${NPROC}
make install

popd
rm -rf build_dir

popd
