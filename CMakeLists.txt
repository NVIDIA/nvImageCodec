# SPDX-FileCopyrightText: Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

cmake_minimum_required(VERSION 3.18 FATAL_ERROR)

include(CMakeDependentOption)

set(NVIMGCODEC_PROJECT_NAME "nvimgcodec")
set(NVIMGCODEC_VERSION "0.4.0")

if (NVIMGCODEC_FLAVOR)
    set(NVIMGCODEC_FLAVOR "${NVIMGCODEC_FLAVOR}")  # could be nightly, weekly, etc if necessary
    set(NVIMGCODEC_FLAVOR_MINUS "-${NVIMGCODEC_FLAVOR}")  # -nightly, -weekly, etc
endif()

# Used when creating special builds
set(PROJECT_VERSION_SUFFIX "")

if (NOT ARCH)
    set(ARCH "x86_64")
endif()

if (NOT NVIMGCODEC_WHL_PLATFORM_NAME)
    if(UNIX)
        set(NVIMGCODEC_WHL_PLATFORM_NAME "manylinux2014_${ARCH}")
    else()
        set(NVIMGCODEC_WHL_PLATFORM_NAME "win_amd64")
    endif()
endif()

if(NOT BUILD_ID)
    set (BUILD_ID 0)
endif()

set(NVIMGCODEC_VERSION_WITH_BUILD "${NVIMGCODEC_VERSION}.${BUILD_ID}")
project(${NVIMGCODEC_PROJECT_NAME} VERSION ${NVIMGCODEC_VERSION} LANGUAGES CXX CUDA)
set(CUDA_VERSION "${CMAKE_CUDA_COMPILER_VERSION}")

# Include custom cmake modules/scripts
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

include(Utils)
parse_cuda_version(${CUDA_VERSION} CUDA_VERSION_MAJOR CUDA_VERSION_MINOR CUDA_VERSION_PATCH CUDA_VERSION_SHORT CUDA_VERSION_SHORT_DIGIT_ONLY)

add_definitions("-DCUDA_VERSION_MAJOR=${CUDA_VERSION_MAJOR}")
add_definitions("-DCUDA_VERSION_MINOR=${CUDA_VERSION_MINOR}")
add_definitions("-DCUDA_VERSION_PATCH=${CUDA_VERSION_PATCH}")

if(UNIX)
    set(NVIMGCODEC_SYSTEM_NAME "${CMAKE_SYSTEM_PROCESSOR}-linux")
else()
    set(NVIMGCODEC_SYSTEM_NAME "${CMAKE_SYSTEM_PROCESSOR}-win")
endif()

set(NVIMGCODEC_BUILD_SUFFIX "cuda${CUDA_VERSION_MAJOR}-${NVIMGCODEC_SYSTEM_NAME}")


# Set a default build type if none was specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Debug' as none was specified.")
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "" "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(WIN32)
  # XXX: cmake appends /MT after /MD that produces D9025 warnings about overriding flags, hence...
  string(REGEX REPLACE "/MD " "" CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE}")
  string(REGEX REPLACE "/MDd " "" CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG}")
endif()

option(BUILD_LIBRARY "Build library" ON)
option(BUILD_TEST "Build tests" ON)
option(BUILD_SAMPLES "Build samples" ON)
option(BUILD_CVCUDA_SAMPLES "Build CVCUDA samples" OFF)
option(BUILD_DOCS "Build documentation" OFF)
option(BUILD_EXTENSIONS "Build extensions modules" ON)
option(BUILD_NVJPEG_EXT "Build nvjpeg extensions module" ON)
option(BUILD_NVJPEG2K_EXT "Build nvjpeg2k extensions module" ON)
option(BUILD_NVTIFF_EXT "Build nvtiff extensions module" ON)
option(SKIP_NVTIFF_WITH_NVCOMP_TESTS "Skip nvTIFF extension tests that require nvCOMP" OFF)
option(BUILD_NVBMP_EXT "Build nvbmp extensions module" ON)
option(BUILD_NVPNM_EXT "Build nvpnm extensions module" ON)
option(BUILD_LIBJPEG_TURBO_EXT "Build libjpeg-turbo extensions module" ON)
option(BUILD_LIBTIFF_EXT "Build libtiff extensions module" ON)
option(BUILD_OPENCV_EXT "Build opencv extensions module" ON)
option(BUILD_PYTHON "Build Python binding" ON)
option(BUILD_WHEEL "Build python wheel package" ON)

cmake_dependent_option(WITH_DYNAMIC_NVJPEG "Dynamically loads nvjpeg at runtime" ON
                      "BUILD_NVJPEG_EXT" OFF)
propagate_option(WITH_DYNAMIC_NVJPEG)
cmake_dependent_option(WITH_DYNAMIC_NVJPEG2K "Dynamically loads nvjpeg2k at runtime" ON
                      "BUILD_NVJPEG2K_EXT" OFF)
propagate_option(WITH_DYNAMIC_NVJPEG2K)
cmake_dependent_option(WITH_DYNAMIC_NVTIFF "Dynamically loads nvtiff at runtime" ON
                      "BUILD_NVTIFF_EXT" OFF)
propagate_option(WITH_DYNAMIC_NVTIFF)

string(TOLOWER ${CMAKE_SYSTEM_NAME} SYS_NAME)

if(WIN32)
    set(CPACK_GENERATOR "ZIP")
    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "C:\\Program Files\\NVIDIA nvImageCodec\\v${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}\\${CUDA_VERSION_MAJOR}" CACHE PATH "where nvImageCodec will be installed" FORCE)
    endif()
else()
    set(CPACK_GENERATOR "TGZ;DEB")

    if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
        set(CMAKE_INSTALL_PREFIX "/opt/nvidia/nvimgcodec_cuda${CUDA_VERSION_MAJOR}" CACHE PATH "where nvImageCodec will be installed" FORCE)
    endif()

    set(CPACK_COMPONENTS_GROUPING IGNORE)

    # Debian options
    set(CPACK_DEBIAN_ENABLE_COMPONENT_DEPENDS true)
    set(CPACK_DEBIAN_PACKAGE_CONTROL_STRICT_PERMISSION ON)
    set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS off)
    set(CPACK_DEBIAN_COMPRESSION_TYPE xz)

    # Create several .debs, one for each component
    set(CPACK_DEB_COMPONENT_INSTALL ON)

    # Archive options
    set(CPACK_ARCHIVE_THREADS 0) # use all cores
    set(CPACK_ARCHIVE_COMPONENT_INSTALL ON)
endif()

set(CPACK_PACKAGE_NAME "${PROJECT_NAME}")
set(CPACK_PACKAGE_VENDOR "NVIDIA CORPORATION")
set(CPACK_PACKAGE_CONTACT "nvImageCodec Support https://github.com/NVIDIA/nvImageCodec")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "GPU Accelerated Image Codecs library")
set(CPACK_PACKAGE_VERSION "${NVIMGCODEC_VERSION_WITH_BUILD}${PROJECT_VERSION_SUFFIX}")
set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION_TWEAK "${BUILD_ID}")
set(CPACK_PACKAGE_VERSION_SUFFIX "${PROJECT_VERSION_SUFFIX}")
#set(PROJECT_VERSION "${PROJECT_VERSION}${PROJECT_VERSION_SUFFIX}")
# we split the file name components with '-', so the version component can't
# have this character, let's replace it by '_'
string(REPLACE "-" "_" tmp ${CPACK_PACKAGE_VERSION})
set(NVIMGCODEC_VERSION_BUILD "${tmp}-${NVIMGCODEC_BUILD_SUFFIX}")
set(CPACK_PACKAGE_FILE_NAME "${PROJECT_NAME}-${NVIMGCODEC_VERSION_BUILD}")
set(CPACK_STRIP_FILES TRUE)
set(CPACK_PROJECT_CONFIG_FILE ${CMAKE_SOURCE_DIR}/cmake/Package.cmake)
set(CPACK_COMPONENTS_ALL lib tests samples)

if(UNIX)
    configure_file(cpack/ld.so.conf.in cpack/ld.so.conf @ONLY)
    install(FILES ${CMAKE_CURRENT_BINARY_DIR}/cpack/ld.so.conf
        DESTINATION "etc/ld.so.conf.d"
        RENAME ${CPACK_PACKAGE_NAME}.conf
        COMPONENT lib)
    configure_file(cpack/debian_lib_postinst.in cpack/lib/postinst @ONLY)
    configure_file(cpack/debian_lib_prerm.in cpack/lib/prerm @ONLY)

    set(CPACK_DEBIAN_LIB_PACKAGE_CONTROL_EXTRA
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/postinst"
        "${CMAKE_CURRENT_BINARY_DIR}/cpack/lib/prerm")

    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libc6 (>= 2.3.1-6), libstdc++6 (>=6.0)")

    # as per debian convention, use the library file name
    set(CPACK_DEBIAN_LIB_PACKAGE_NAME "lib${PROJECT_NAME}")
endif()

include (CPack)

include(cmake/CUDA_utils.cmake)

set(USED_CUDA_COMPILER ${CMAKE_CUDA_COMPILER})
message(STATUS "CUDA .cu files compiler: ${USED_CUDA_COMPILER}")

CUDA_find_supported_arch_values(CUDA_supported_archs ${USED_CUDA_COMPILER} ${CUDA_known_archs})
message(STATUS "CUDA supported archs: ${CUDA_supported_archs}")

set(CUDA_TARGET_ARCHS_SORTED ${CUDA_TARGET_ARCHS})
list(SORT CUDA_TARGET_ARCHS_SORTED)
CUDA_find_supported_arch_values(CUDA_targeted_archs ${USED_CUDA_COMPILER} ${CUDA_TARGET_ARCHS_SORTED})
message(STATUS "CUDA targeted archs: ${CUDA_targeted_archs}")
if (NOT CUDA_targeted_archs)
  message(FATAL_ERROR "None of the provided CUDA architectures ({${CUDA_TARGET_ARCHS}})"
      " is supported by ${USED_CUDA_COMPILER}, use one or more of: ${CUDA_supported_archs}")
endif()

CUDA_get_cmake_cuda_archs(CMAKE_CUDA_ARCHITECTURES ${CUDA_targeted_archs})
message(STATUS "Generated CMAKE_CUDA_ARCHITECTURES: ${CMAKE_CUDA_ARCHITECTURES}")

if(WIN32)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --Wno-deprecated-gpu-targets -Xfatbin -compress-all -t 0")

    include(third_party/generate_product_version)

    generate_product_version(ProductVersionFiles
        NAME "NVIDIA nvImageCodec library"
        VERSION_MAJOR ${CMAKE_PROJECT_VERSION_MAJOR}
        VERSION_MINOR ${CMAKE_PROJECT_VERSION_MINOR}
        VERSION_PATCH ${CMAKE_PROJECT_VERSION_PATCH}
        VERSION_REVISION ${CMAKE_PROJECT_VERSION_TWEAK}
        COMPANY_NAME NVIDIA Corporation)
elseif()
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options \"-fvisibility=hidden -Wno-free-nonheap-object\" --Wno-deprecated-gpu-targets -Xfatbin -compress-all -t 0")

endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)

list(APPEND CMAKE_BUILD_RPATH $ORIGIN)
list(APPEND CMAKE_INSTALL_RPATH $ORIGIN)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_SOURCE_DIR}/test
    )

# Find all dependencies
include(Dependencies)

add_subdirectory(external)

if(BUILD_LIBRARY)
    add_subdirectory(src)
endif()

if(BUILD_SAMPLES)
    add_subdirectory(example)
endif()

if(BUILD_PYTHON)
    add_subdirectory(python)
endif()

if(BUILD_DOCS)
    add_subdirectory(docs)
endif()

if(BUILD_TEST)
    enable_testing()
    add_subdirectory(test)
endif()

set (NVIMGCODEC_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include/)
if(BUILD_EXTENSIONS)
    add_subdirectory(extensions)
endif()

if(UNIX)
add_custom_target(coverity COMMAND ../coverity.sh
  WORKING_DIRECTORY
    ${CMAKE_CURRENT_BINARY_DIR}
  COMMENT
    "Running static analysis for nvImageCodec code with Coverity"
  VERBATIM
  )
endif()
