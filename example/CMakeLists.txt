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

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    # Build as a stand-alone samples project
    cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
    project(nvimgcodec_samples LANGUAGES CXX CUDA)

    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    set(CMAKE_CXX_EXTENSIONS OFF)

    set(CMAKE_CUDA_STANDARD 14)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_EXTENSIONS OFF)

    option(BUILD_CVCUDA_SAMPLES "Build CVCUDA samples" OFF)

    find_package(CUDAToolkit REQUIRED)

    list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/nvimgcodec_cuda${CUDA_VERSION_MAJOR}")
    find_package(nvimgcodec CONFIG REQUIRED)

    include_directories(${nvimgcodec_INCLUDE_DIR})
    link_directories(${nvimgcodec_LIB_DIR})

    if(NOT DEFINED NVIMGCODEC_INCLUDE_DIR)
        set(NVIMGCODEC_INCLUDE_DIR ${nvimgcodec_INCLUDE_DIR})
    endif()
    include_directories(${NVIMGCODEC_INCLUDE_DIR})
else()
    # Build as a part of nvImageCodec project
    install(DIRECTORY
        ${CMAKE_CURRENT_SOURCE_DIR}/assets
        ${CMAKE_CURRENT_SOURCE_DIR}/nvimtrans
        ${CMAKE_CURRENT_SOURCE_DIR}/nvimproc
        ${CMAKE_CURRENT_SOURCE_DIR}/python
        DESTINATION samples
        COMPONENT samples
    )

    install(FILES
        ${CMAKE_CURRENT_SOURCE_DIR}/README.md
        ${CMAKE_CURRENT_SOURCE_DIR}/NOTICE.md
        ${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt
        DESTINATION samples
        COMPONENT samples
    )
endif()

add_subdirectory(nvimtrans)

if(BUILD_CVCUDA_SAMPLES)
    add_subdirectory(nvimproc)
endif()

