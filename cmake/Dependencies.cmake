# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

find_package(Python COMPONENTS Interpreter)
set(PYTHONINTERP_FOUND ${Python_Interpreter_FOUND})
set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})

##################################################################
# Google C++ testing framework
##################################################################
if (BUILD_TEST)
  set(BUILD_GTEST ON CACHE INTERNAL "Build gtest submodule")
  set(BUILD_GMOCK ON CACHE INTERNAL "Build gmock submodule")
  check_and_add_cmake_submodule(${PROJECT_SOURCE_DIR}/external/googletest EXCLUDE_FROM_ALL)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/googletest/googletest/include)
  include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/googletest/googlemock/include)
  set_target_properties(gtest gmock PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

function(CUDA_find_library out_path lib_name)
    find_library(${out_path} ${lib_name} PATHS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}
                 PATH_SUFFIXES lib lib64)
endfunction()

find_package(CUDAToolkit REQUIRED)

include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
find_path(NVJPEG_INCLUDE
    NAMES nvjpeg.h
    PATHS
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
        ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include
)
include_directories(SYSTEM ${NVJPEG_INCLUDE})

include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/NVTX/c/include)
include_directories(SYSTEM ${PROJECT_SOURCE_DIR}/external/dlpack/include)

# Linking with static nvjpeg2k until there is a python package for it
if (BUILD_NVJPEG2K_EXT)
    if (NOT WITH_DYNAMIC_NVJPEG2K)
        CUDA_find_library(NVJPEG2K_LIBRARY nvjpeg2k_static)
        if (${NVJPEG2K_LIBRARY} STREQUAL "NVJPEG2K_LIBRARY-NOTFOUND")
            message(WARNING "nvjpeg2k not found - disabled")
            set(BUILD_NVJPEG2K_EXT OFF CACHE BOOL INTERNAL)
            set(BUILD_NVJPEG2K_EXT OFF)
        else()
            message(STATUS "Found nvjpeg2k: " ${NVJPEG2K_LIBRARY})
        endif()
    else()
        # Note: We are getting the x86_64 tarball, but we are only interested in the headers.
        include(FetchContent)
        FetchContent_Declare(
            nvjpeg2k_headers
            URL      https://developer.download.nvidia.com/compute/nvjpeg2000/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-0.8.0.38-archive.tar.xz
            URL_HASH SHA512=21acc1bfa7b6967fc2240dac9c9041faa6c10c9fe356f754748b6a6154e92031b0e4d8d1a0a1d1cdfb5c68b929126d548e7ea3d321609d339c2a6668281d2180
        )
        FetchContent_Populate(nvjpeg2k_headers)
        set(nvjpeg2k_SEARCH_PATH "${nvjpeg2k_headers_SOURCE_DIR}/include")
    endif()

    if(NOT DEFINED NVJPEG2K_INCLUDE)
        find_path(NVJPEG2K_INCLUDE
            NAMES nvjpeg2k.h
            PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                  ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include
                  ${nvjpeg2k_SEARCH_PATH}
        )
    endif()
endif()

if (BUILD_NVJPEG2K_EXT)
    message(STATUS "NVJPEG2K_INCLUDE=${NVJPEG2K_INCLUDE}")
    include_directories(SYSTEM ${NVJPEG2K_INCLUDE})
else()
    message(STATUS "nvjpeg2k extension build disabled")
endif()

if (BUILD_NVTIFF_EXT)
    if (WITH_DYNAMIC_NVTIFF)
        message(STATUS "Dynamic nvTIFF extension build")
        # Note: We are getting the x86_64 tarball, but we are only interested in the headers.
        include(FetchContent)
        FetchContent_Declare(
            nvtiff_headers
            URL      https://developer.download.nvidia.com/compute/nvtiff/redist/libnvtiff/linux-x86_64/libnvtiff-linux-x86_64-0.5.0.67_cuda12-archive.tar.xz
            URL_HASH SHA512=62dea5c9b72aa3f4e18b306e3f1e60a00082e67c6cedc95442413b39aa48e41ae4b9371a5ff210a37a6cb3008ed308aeb78cbf2a7b9df55bc5ee7a6da0899832 
        )
        FetchContent_Populate(nvtiff_headers)
        set(nvtiff_SEARCH_PATH "${nvtiff_headers_SOURCE_DIR}/include")

        find_path(NVTIFF_INCLUDE
            NAMES nvtiff.h
            PATHS ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
                ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include
                ${nvtiff_SEARCH_PATH}
        )
    else()
        find_library(NVTIFF_LIB nvtiff_static PATH_SUFFIXES lib lib64)
        if(NOT NVTIFF_LIB)
            message(WARNING, "nvTIFF library not found. Disabling its extensions and tests build.")
            set(BUILD_NVTIFF_EXT OFF CACHE BOOL INTERNAL)
            set(BUILD_NVTIFF_EXT OFF)
        else()
            message(STATUS "Found nvTIFF: ${NVTIFF_LIB}")
            find_path(NVTIFF_INCLUDE NAMES nvtiff.h)
            if(NOT NVTIFF_INCLUDE)
                message(FATAL_ERROR,
                " Found nvTIFF library file but not header file, please check your install"
                " or disable nvTIFF extension build with -DBUILD_NVTIFF_EXT=OFF")
            endif()
        endif()
    endif()
endif()
if (BUILD_NVTIFF_EXT)
    message(STATUS "Using NVTIFF_INCLUDE=${NVTIFF_INCLUDE}")
    include_directories(SYSTEM ${NVTIFF_INCLUDE})
else()
    message(STATUS "nvtiff extension build disabled")
endif()

set(TIFF_LIBRARY_DEPS)

find_package(ZLIB)
if(NOT DEFINED ZLIB_LIBRARY)
    message(STATUS "zlib not found - disabled")
else()
    message(STATUS "Using zlib at ${ZLIB_LIBRARY}")
    list(APPEND TIFF_LIBRARY_DEPS ${ZLIB_LIBRARY})
endif()

find_package(ZSTD)
if(NOT DEFINED ZSTD_LIBRARY)
    message(FATAL_ERROR "zstd not found - disabled")
else()
    message(STATUS "Using zstd at ${ZSTD_LIBRARY}")
    list(APPEND TIFF_LIBRARY_DEPS ${ZSTD_LIBRARY})
endif()

find_package(JPEG 62) # 1.5.3 version
if(NOT DEFINED JPEG_LIBRARY)
    message(STATUS "libjpeg-turbo not found - disabled")
    set(BUILD_LIBJPEG_TURBO_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_LIBJPEG_TURBO_EXT OFF)
else()
    message(STATUS "Using libjpeg-turbo at ${JPEG_LIBRARY}")
    include_directories(SYSTEM ${JPEG_INCLUDE_DIR})
    list(APPEND TIFF_LIBRARY_DEPS ${JPEG_LIBRARY})
endif()

find_package(TIFF)
if(NOT TIFF_FOUND)
    message(STATUS "libtiff not found - disabled")
    set(BUILD_LIBTIFF_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_LIBTIFF_EXT OFF)
else()
    if(NOT TARGET CMath::CMath)
        add_library(CMath::CMath INTERFACE IMPORTED)
    endif()
    message(STATUS "TIFF_INCLUDE_DIR: ${TIFF_INCLUDE_DIR}")
    message(STATUS "TIFF_LIBRARIES: ${TIFF_LIBRARIES}")
    include_directories(SYSTEM ${TIFF_INCLUDE_DIR})
    message(STATUS "libtiff dependencies: ${TIFF_LIBRARY_DEPS}")
endif()

if (NOT DEFINED OpenCV_VERSION AND (BUILD_OPENCV_EXT OR BUILD_TEST))
    if (WIN32)
        set(OpenCV_STATIC ON)
    endif()
    find_package(OpenCV 4.9 QUIET COMPONENTS core imgproc imgcodecs)

    if(NOT OpenCV_FOUND)
        message(STATUS "OpenCV not found - disabled")
        set(BUILD_OPENCV_EXT OFF CACHE BOOL INTERNAL)
        set(BUILD_OPENCV_EXT OFF)

        if (BUILD_TEST)
            message(WARNING "Native tests need OpenCV to run - Disabling tests")
            set(BUILD_TEST OFF CACHE BOOL INTERNAL)
            set(BUILD_TEST OFF)
        endif()
    else()
        message(STATUS "Found OpenCV: ${OpenCV_INCLUDE_DIRS} (found suitable version \"${OpenCV_VERSION}\", minimum required is \"4.9\")")
        message(STATUS "OpenCV libraries: ${OpenCV_LIBRARIES}")
        include_directories(SYSTEM ${OpenCV_INCLUDE_DIRS})
    endif()
endif()

# #################################################################
# Boost preprocessor
# #################################################################
include_directories(${PROJECT_SOURCE_DIR}/external/boost/preprocessor/include)

set(NVIMGCODEC_COMMON_DEPENDENCIES "")
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES rt)
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES pthread)
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES m)
list(APPEND NVIMGCODEC_COMMON_DEPENDENCIES dl)
