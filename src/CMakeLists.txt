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

set(NVIMGCODEC_LIBRARY_NAME nvimgcodec)

add_subdirectory(imgproc)

set(NVIMGCODEC_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec.h
    nvimgcodec_capi.cpp
    plugin_framework.cpp
    codec_registry.cpp
    codec.cpp
    image_decoder.cpp
    image_encoder.cpp
    image_parser.cpp
    code_stream.cpp
    file_io_stream.cpp
    std_file_io_stream.cpp
    image.cpp
    image_parser_factory.cpp
    image_encoder_factory.cpp
    image_decoder_factory.cpp
    image_generic_decoder.cpp
    image_generic_encoder.cpp
    nvimgcodec_director.cpp
    processing_results.cpp
    decode_state_batch.cpp
    encode_state_batch.cpp
    thread_pool.cpp
    default_executor.cpp
    builtin_modules.cpp
    parsers/bmp.cpp
    parsers/exif.cpp
    parsers/jpeg.cpp
    parsers/jpeg2k.cpp
    parsers/tiff.cpp
    parsers/png.cpp
    parsers/pnm.cpp
    parsers/webp.cpp
    parsers/parsers_ext_module.cpp
)

if(UNIX)
  list(APPEND NVIMGCODEC_SRCS mmaped_file_io_stream.cpp)
endif()

# Build the library
message(STATUS "props ProductVersionFiles ${ProductVersionFiles}")

add_library(${NVIMGCODEC_LIBRARY_NAME}_obj OBJECT ${NVIMGCODEC_SRCS} ${ProductVersionFiles})
add_library(${NVIMGCODEC_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${NVIMGCODEC_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)
add_library(${NVIMGCODEC_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${NVIMGCODEC_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)

set(LIBTOPACK ${NVIMGCODEC_LIBRARY_NAME} ${NVIMGCODEC_LIBRARY_NAME}_static)

find_package(CUDAToolkit REQUIRED)
add_subdirectory(dynlink)


if (WITH_DYNAMIC_CUDA)
    set(CUDA_DRIVER_LIB dynlink_cuda)
else()
    set(CUDA_DRIVER_LIB CUDA::cuda_driver)
endif()

target_link_libraries(${NVIMGCODEC_LIBRARY_NAME} PRIVATE CUDA::cudart_static ${CUDA_DRIVER_LIB})

if(UNIX)
    # CXX flags
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden")
    target_link_libraries(${NVIMGCODEC_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
    target_link_libraries(${NVIMGCODEC_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
elseif(MSVC)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP")
      message(STATUS "Added parallel build arguments to CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
endif()


# Configure library version
include(ConfigureVersion)
configure_library_version(NVIMGCODEC_LIBRARY_NAME
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec_version.h.in"
    "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec_version.h"
)

set(NVIMGCODEC_PUBLIC_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec.h")
set(NVIMGCODEC_PUBLIC_HEADERS "${NVIMGCODEC_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvimgcodec_version.h")
set(NVIMGCODEC_PUBLIC_HEADERS "${NVIMGCODEC_PUBLIC_HEADERS}\;${CMAKE_CURRENT_SOURCE_DIR}/../include/nvcv_adapter.hpp")

if(UNIX)
    set_target_properties(${NVIMGCODEC_LIBRARY_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PUBLIC_HEADER ${NVIMGCODEC_PUBLIC_HEADERS}
        VERSION ${PROJECT_VERSION}
        SOVERSION ${PROJECT_VERSION_MAJOR})
else()
    set_target_properties(${NVIMGCODEC_LIBRARY_NAME} PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        PUBLIC_HEADER ${NVIMGCODEC_PUBLIC_HEADERS}
        VERSION ${PROJECT_VERSION}
        OUTPUT_NAME ${NVIMGCODEC_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
        ARCHIVE_OUTPUT_NAME ${NVIMGCODEC_LIBRARY_NAME})
endif()

# Installation (https://github.com/forexample/package-example) {

# Layout. This works for all platforms:
#   * <prefix>/cmake/
#   * <prefix>/lib*/
#   * <prefix>/include/
set(config_install_dir "${CMAKE_INSTALL_PREFIX}/cmake")

set(generated_dir "${CMAKE_CURRENT_BINARY_DIR}/generated")

# Configuration
set(version_config "${generated_dir}/${PROJECT_NAME}ConfigVersion.cmake")
set(project_config "${generated_dir}/${PROJECT_NAME}Config.cmake")
set(TARGETS_EXPORT_NAME "${PROJECT_NAME}Targets")
set(namespace "${PROJECT_NAME}::")

# Include module with fuction 'write_basic_package_version_file'
include(CMakePackageConfigHelpers)

# Configure '<PROJECT-NAME>ConfigVersion.cmake'
# Use:
#   * PROJECT_VERSION
write_basic_package_version_file(
    "${version_config}" COMPATIBILITY SameMajorVersion
)

# Configure '<PROJECT-NAME>Config.cmake'
# Use variables:
#   * TARGETS_EXPORT_NAME
#   * PROJECT_NAME
configure_package_config_file(
    "${CMAKE_SOURCE_DIR}/cmake/template/Config.cmake.in"
    "${project_config}"
    INSTALL_DESTINATION "${config_install_dir}"
)

# Targets:
install(
    TARGETS ${LIBTOPACK}
    EXPORT "${TARGETS_EXPORT_NAME}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT lib
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}" COMPONENT lib
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}" COMPONENT lib
)

# License file
install(
    FILES "${CMAKE_SOURCE_DIR}/LICENSE.txt" "${CMAKE_SOURCE_DIR}/Acknowledgements.txt"
    DESTINATION "."
    COMPONENT lib
)

# Config
#   * <prefix>/cmake/nvimgcodec/nvimgcodecConfig.cmake
#   * <prefix>/cmake/nvimgcodec/nvimgcodecConfigVersion.cmake
install(
    FILES "${project_config}" "${version_config}"
    DESTINATION "cmake/nvimgcodec"
    COMPONENT lib
)

# Config
# * <prefix>/cmake/nvimgcodec/nvimgcodecTargets.cmake
install(
    EXPORT "${TARGETS_EXPORT_NAME}"
    NAMESPACE "${namespace}"
    DESTINATION "cmake/nvimgcodec"
    COMPONENT lib
)

# }
