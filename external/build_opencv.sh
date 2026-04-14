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

# OpenCV
OPENCV_VERSION=4.13.0
OPENCV_URL=https://github.com/opencv/opencv/archive/refs/tags/${OPENCV_VERSION}.tar.gz
OPENCV_SHA256=1d40ca017ea51c533cf9fd5cbde5b5fe7ae248291ddf2af99d4c17cf8e13017d

# Download and extract OpenCV if not already present
if [ ! -d external/opencv-${OPENCV_VERSION} ]; then
    pushd external
    wget -O opencv-${OPENCV_VERSION}.tar.gz ${OPENCV_URL}
    # Verify checksum for security
    echo "${OPENCV_SHA256} opencv-${OPENCV_VERSION}.tar.gz" | sha256sum -c || { rm opencv-${OPENCV_VERSION}.tar.gz; exit 1; }
    tar -xf opencv-${OPENCV_VERSION}.tar.gz
    rm opencv-${OPENCV_VERSION}.tar.gz
    popd
fi

pushd external/opencv-${OPENCV_VERSION}

mkdir -p build_dir
pushd build_dir

# Create a CMake script to define ZSTD::ZSTD and CMath::CMath targets
DEPS_TARGET_CMAKE=${INSTALL_PREFIX}/define_deps_targets.cmake
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

# Define CMath::CMath target (math library)
if(NOT TARGET CMath::CMath)
  find_library(MATH_LIBRARY m)
  if(MATH_LIBRARY)
    add_library(CMath::CMath UNKNOWN IMPORTED)
    set_target_properties(CMath::CMath PROPERTIES
      IMPORTED_LOCATION "\${MATH_LIBRARY}"
    )
  else()
    add_library(CMath::CMath INTERFACE IMPORTED)
  endif()
endif()
EOF

cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DVIBRANTE_PDK:STRING=/ \
      -DCMAKE_TOOLCHAIN_FILE=$PWD/../platforms/${OPENCV_TOOLCHAIN_FILE} \
      -DCMAKE_INSTALL_PREFIX=${INSTALL_PREFIX} \
      -DCMAKE_PREFIX_PATH=${INSTALL_PREFIX} \
      -DCMAKE_PROJECT_INCLUDE=${DEPS_TARGET_CMAKE} \
      -DBUILD_LIST=core,improc,imgcodecs \
      -DBUILD_SHARED_LIBS=OFF \
      -DWITH_EIGEN=OFF \
      -DWITH_CUDA=OFF \
      -DWITH_1394=OFF \
      -DWITH_IPP=OFF \
      -DWITH_OPENCL=OFF \
      -DWITH_GTK=OFF \
      -DBUILD_JPEG=OFF \
      -DWITH_JPEG=ON \
      -DBUILD_TIFF=OFF \
      -DWITH_TIFF=ON \
      -DBUILD_ZLIB=OFF \
      -DWITH_ZLIB=ON \
      -DWITH_QUIRC=OFF \
      -DWITH_ADE=OFF \
      -DBUILD_JASPER=OFF \
      -DBUILD_DOCS=OFF \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DBUILD_PNG=ON \
      -DWITH_WEBP=ON \
      -DBUILD_opencv_cudalegacy=OFF \
      -DBUILD_opencv_stitching=OFF \
      -DWITH_TBB=OFF \
      -DWITH_QUIRC=OFF \
      -DWITH_OPENMP=OFF \
      -DWITH_PTHREADS_PF=OFF \
      -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_java=OFF \
      -DBUILD_opencv_python2=OFF \
      -DBUILD_opencv_python3=OFF \
      -DWITH_PROTOBUF=OFF \
      -DWITH_FFMPEG=OFF \
      -DWITH_GSTREAMER=OFF \
      -DWITH_GSTREAMER_0_10=OFF \
      -DWITH_VTK=OFF \
      -DWITH_OPENEXR=OFF \
      -DINSTALL_C_EXAMPLES=OFF \
      -DINSTALL_TESTS=OFF \
      -DVIBRANTE=TRUE \
      -DWITH_CSTRIPES=OFF \
      VERBOSE=1 ..
make -j${NPROC} install

popd
rm -rf build_dir

popd