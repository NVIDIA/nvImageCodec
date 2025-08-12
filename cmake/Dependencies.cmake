# Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(CTK_SEARCH_PATHS
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${CMAKE_CUDA_COMPILER_TOOLKIT_ROOT}/include
)

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

if (BUILD_NVJPEG2K_EXT)
    if (WITH_DYNAMIC_NVJPEG2K)
        include(FetchContent)
        FetchContent_Declare(
            nvjpeg2k_headers
            URL      https://developer.download.nvidia.com/compute/nvjpeg2000/redist/libnvjpeg_2k/linux-x86_64/libnvjpeg_2k-linux-x86_64-0.9.0.43-archive.tar.xz
            URL_HASH SHA512=22d14a20af67ba414956fd7c4223cf3fd519cec9ccbd0ae27603416ab143eae92457ab0434205fe66617bcc5a54805bfc6183f89205ea2d0068d497321a43783
        )
        FetchContent_Populate(nvjpeg2k_headers)
        set(NVJPEG2K_SEARCH_PATHS "${nvjpeg2k_headers_SOURCE_DIR}/include")
    else()
        set(NVJPEG2K_SEARCH_PATHS ${CTK_SEARCH_PATHS})
        find_library(NVJPEG2K_LIBRARY nvjpeg2k_static PATH_SUFFIXES lib lib64)
        if(NVJPEG2K_LIBRARY)
            message(STATUS "Found nvJPEG2k: ${NVJPEG2K_LIBRARY}")
        else()
            message(WARNING "nvJPEG2k library not found. Disabling nvJPEG2k and tests build.")
            set(BUILD_NVJPEG2K_EXT OFF CACHE BOOL INTERNAL)
            set(BUILD_NVJPEG2K_EXT OFF)
        endif()
    endif()

    find_path(NVJPEG2K_INCLUDE NAMES nvjpeg2k.h HINTS ${NVJPEG2K_SEARCH_PATHS})

    if((NVJPEG2K_LIBRARY OR WITH_DYNAMIC_NVJPEG2K) AND NOT NVJPEG2K_INCLUDE)
        message(FATAL_ERROR
        "nvJPEG2k header file not found, please check your install"
        " or disable nvJPEG2k extension build with -DBUILD_NVJPEG2K_EXT=OFF")
    endif()
endif()

if (BUILD_NVJPEG2K_EXT)
    message(STATUS "Using NVJPEG2K_INCLUDE=${NVJPEG2K_INCLUDE}")
    include_directories(BEFORE SYSTEM ${NVJPEG2K_INCLUDE})
else()
    message(STATUS "nvJPEG2k extension build disabled")
endif()

if (BUILD_NVTIFF_EXT)
    if (WITH_DYNAMIC_NVTIFF)
        include(FetchContent)
        FetchContent_Declare(
            nvtiff_headers
            URL      https://developer.download.nvidia.com/compute/nvtiff/redist/libnvtiff/linux-x86_64/libnvtiff-linux-x86_64-0.5.1.75_cuda12-archive.tar.xz
            URL_HASH SHA512=66332d1cb32d428b8f7fce8ebaf9d44caa01d85f77d880c827ccf15459f3164e6dcfabfb88e4a0c2b0916ef83161c2d9f8990bebb8d61aca938cd9199b514752
        )
        FetchContent_Populate(nvtiff_headers)
        set(NVTIFF_SEARCH_PATHS "${nvtiff_headers_SOURCE_DIR}/include")
    else()
        set(NVTIFF_SEARCH_PATHS ${CTK_SEARCH_PATHS})
        find_library(NVTIFF_LIB nvtiff_static PATH_SUFFIXES lib lib64)
        if(NVTIFF_LIB)
            message(STATUS "Found nvTIFF: ${NVTIFF_LIB}")
        else()
            message(WARNING "nvTIFF library not found. Disabling nvTIFF extension and tests build.")
            set(BUILD_NVTIFF_EXT OFF CACHE BOOL INTERNAL)
            set(BUILD_NVTIFF_EXT OFF)
        endif()
    endif()

    find_path(NVTIFF_INCLUDE NAMES nvtiff.h HINTS ${NVTIFF_SEARCH_PATHS})

    if((NVTIFF_LIB OR WITH_DYNAMIC_NVTIFF) AND NOT NVTIFF_INCLUDE)
        message(FATAL_ERROR
        "nvTIFF header file not found, please check your install"
        " or disable nvTIFF extension build with -DBUILD_NVTIFF_EXT=OFF")
    endif()
endif()

if (BUILD_NVTIFF_EXT)
    message(STATUS "Using NVTIFF_INCLUDE=${NVTIFF_INCLUDE}")
    include_directories(BEFORE SYSTEM ${NVTIFF_INCLUDE})
else()
    message(STATUS "nvTIFF extension build disabled")
endif()

set(TIFF_LIBRARY_DEPS)

find_package(ZLIB)
if(NOT ZLIB_FOUND)
    message(STATUS "zlib not found - disabled")
else()
    message(STATUS "Using zlib at ${ZLIB_LIBRARIES}")
    list(APPEND TIFF_LIBRARY_DEPS ${ZLIB_LIBRARIES})
endif()

find_package(ZSTD)
if(NOT DEFINED ZSTD_LIBRARY)
    message(FATAL_ERROR "zstd not found - disabled")
else()
    message(STATUS "Using zstd at ${ZSTD_LIBRARY}")
    list(APPEND TIFF_LIBRARY_DEPS ${ZSTD_LIBRARY})
endif()

find_package(JPEG 62) # 1.5.3 version
if(NOT JPEG_FOUND)
    message(STATUS "libjpeg-turbo not found - disabled")
    set(BUILD_LIBJPEG_TURBO_EXT OFF CACHE BOOL INTERNAL)
    set(BUILD_LIBJPEG_TURBO_EXT OFF)
else()
    message(STATUS "Using libjpeg-turbo at ${JPEG_LIBRARIES}")
    include_directories(SYSTEM ${JPEG_INCLUDE_DIRS})
    list(APPEND TIFF_LIBRARY_DEPS ${JPEG_LIBRARIES})
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
