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

export ROOT_DIR=$(pwd)
export SCRIPT_DIR=$(cd $(dirname $0) && pwd)
export HOST_INSTALL_PREFIX=${HOST_INSTALL_PREFIX:-/usr/local}
export INSTALL_PREFIX=${INSTALL_PREFIX:-$HOST_INSTALL_PREFIX}
export CC_COMP=${CC_COMP:-gcc}
export CXX_COMP=${CXX_COMP:-g++}
export OPENCV_TOOLCHAIN_FILE=${OPENCV_TOOLCHAIN_FILE:-"linux/gnu.toolchain.cmake"}
export CMAKE_TARGET_ARCH=${CMAKE_TARGET_ARCH:-$(uname -m)}
export NPROC=$(grep ^processor /proc/cpuinfo | wc -l)
if [ $NPROC -gt 32 ]; then
    export NPROC=32
fi
echo ${INSTALL_PREFIX}
echo ${CC_COMP}
echo ${CXX_COMP}
echo ${CMAKE_TARGET_ARCH}
echo ${BUILD_ARCH_OPTION}
echo ${HOST_ARCH_OPTION}
echo ${OPENCV_TOOLCHAIN_FILE}

PACKAGE_LIST=(
    "zlib"
    "libjpeg-turbo"
    "zstd"
    "openjpeg"
    "libtiff"
    "opencv"
)

for PACKAGE in "${PACKAGE_LIST[@]}"; do
    ${SCRIPT_DIR}/build_${PACKAGE}.sh
done
