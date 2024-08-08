#!/bin/bash -ex
#
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#Determine Architecture
ARCH="$(arch)"
if [ ${ARCH} = "x86_64" ]; then
    ARCH_LONGNAME="x86_64-conda_cos6"
elif [ ${ARCH} = "ppc64le" ]; then
    ARCH_LONGNAME="powerpc64le-conda_cos7"
else
    echo "Error: Unsupported Architecture. Expected: [x86_64|ppc64le] Actual: ${ARCH}"
    exit 1
fi

# Create 'gcc' and 'g++' symlinks so nvcc can find it
ln -s $CC $BUILD_PREFIX/bin/gcc
ln -s $CXX $BUILD_PREFIX/bin/g++

# Force -std=c++17 in CXXFLAGS
export CXXFLAGS=${CXXFLAGS/-std=c++??/-std=c++17}
export PATH=/usr/local/cuda/bin:${PATH}

# Building third party dependencies
cd $SRC_DIR
export INSTALL_PREFIX=$BUILD_PREFIX
export CMAKE_PREFIX_PATH=$BUILD_PREFIX
bash -ex ./external/build_deps.sh

# Create build directory for cmake and enter it
mkdir $SRC_DIR/build
cd $SRC_DIR/build
# Build
cmake -DCUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}            \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}     \
      -DTIMESTAMP=${TIMESTAMP}                            \
      -DGIT_SHA=${GIT_SHA}                                \
      ..
make -j"$(nproc --all)"

# pip install
$PYTHON -m pip install --no-deps --ignore-installed -v $SRC_DIR/build/python

export LD_LIBRARY_PATH="$PREFIX/lib:$LD_LIBRARY_PATH"
NVIMGCODEC_PATH=$($PYTHON -c 'import nvidia.nvimgcodec as nvimgcodec; import os; print(os.path.dirname(nvimgcodec.__file__))')
echo "NVIMGCODEC_PATH is ${NVIMGCODEC_PATH}"
