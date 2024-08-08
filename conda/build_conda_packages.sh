#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set -o xtrace
set -e

if [ -z $PYVER ]; then
    echo "PYVER is not set"
    exit 1
fi

export CONDA_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export ROOT_DIR="${CONDA_DIR}/.."
export VERSION=$(grep -Po 'set\(NVIMGCODEC_VERSION "\K[^"]*' ${ROOT_DIR}/CMakeLists.txt)
export BUILD_FLAVOR=${BUILD_FLAVOR:-""}  # nightly, weekly
export BUILD_ID=${BUILD_ID:-$EPOCHSECONDS}
export GIT_SHA=$(git rev-parse HEAD)
export TIMESTAMP=$(date +%Y%m%d)
export VERSION_SUFFIX=$(if [ "${BUILD_FLAVOR}" != "" ]; then \
                          echo .${BUILD_FLAVOR}.${TIMESTAMP}; \
                        fi)
export BUILD_VERSION=${VERSION}${VERSION_SUFFIX}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}
export GIT_LFS_SKIP_SMUDGE=1
export CUDA_VERSION=$(echo $(ls /usr/local/cuda/lib64/libcudart.so*)  | sed 's/.*\.\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1.\2/')
export CONDA_OVERRIDE_CUDA=${CUDA_VERSION}

CONDA_BUILD_OPTIONS="--python=${PYVER} --exclusive-config-file config/conda_build_config.yaml"
CONDA_PREFIX=${CONDA_PREFIX:-$(dirname $CONDA_EXE)/..}

# Adding conda-forge channel for dependencies
conda config --add channels conda-forge
conda config --add channels nvidia
conda config --add channels local

conda build ${CONDA_BUILD_OPTIONS} recipe

# Copying the artifacts from conda prefix
mkdir -p ${ROOT_DIR}/artifacts
cp ${CONDA_PREFIX}/conda-bld/*/nvidia-nvimagecodec*.tar.bz2 ${ROOT_DIR}/artifacts
