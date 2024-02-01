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

set(NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME nvjpeg_ext)

set(NVIMGCODEC_NVJPEG_EXT_SRC
  cuda_decoder.cpp
  hw_decoder.cpp
  cuda_encoder.cpp
  type_convert.cpp
  nvjpeg_ext.cpp
  nvjpeg_utils.cpp
  exception.cpp
)

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CUDAToolkit_INCLUDE_DIRS}")
set(CMAKE_REQUIRED_LIBRARIES "CUDA::nvjpeg")
check_cxx_source_compiles(
  "#include <nvjpeg.h>
  int main(){
    return NVJPEG_BACKEND_LOSSLESS_JPEG;
  }"
  NVJPEG_LOSSLESS_SUPPORTED)

check_cxx_source_compiles(
  "#include <nvjpeg.h>
  int main(){
    return NVJPEG_FLAGS_REDUCED_MEMORY_DECODE_ZERO_COPY;
  }"
  NVJPEG_REDUCED_MEMORY_ZERO_COPY_SUPPORTED)

include(CheckCXXSymbolExists)
check_cxx_symbol_exists(nvjpegBufferDeviceResize "nvjpeg.h" NVJPEG_BUFFER_RESIZE_API)
if (NVJPEG_BUFFER_RESIZE_API)
  message("nvJPEG buffer resize API available")
  add_definitions(-DNVJPEG_BUFFER_RESIZE_API=1)
endif()

if(NVJPEG_LOSSLESS_SUPPORTED)
  message("nvJPEG lossless supported")
  add_definitions(-DNVJPEG_LOSSLESS_SUPPORTED=1)
else()
  message("nvJPEG lossless NOT supported")
  add_definitions(-DNVJPEG_LOSSLESS_SUPPORTED=0)
endif()

if(NVJPEG_LOSSLESS_SUPPORTED)
  list(APPEND NVIMGCODEC_NVJPEG_EXT_SRC lossless_decoder.cpp)
endif()

if (NVJPEG_REDUCED_MEMORY_ZERO_COPY_SUPPORTED)
  message("nvJPEG reduced memory zero copy supported")
  add_definitions(-DNVJPEG_REDUCED_MEMORY_ZERO_COPY_SUPPORTED=1)
else()
  message("nvJPEG reduced memory zero copy NOT supported")
  add_definitions(-DNVJPEG_REDUCED_MEMORY_ZERO_COPY_SUPPORTED=0)
endif()

add_library(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME} SHARED ${NVIMGCODEC_NVJPEG_EXT_SRC} ext_module.cpp)
add_library(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}_static STATIC ${NVIMGCODEC_NVJPEG_EXT_SRC})

if(NOT WITH_DYNAMIC_NVJPEG)
  if(UNIX)
    target_link_libraries(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}
      PUBLIC CUDA::cudart_static CUDA::nvjpeg_static CUDA::culibos)
  else()
    target_link_libraries(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}
      PUBLIC CUDA::cudart_static CUDA::nvjpeg)
  endif()
else()
  target_link_libraries(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}
    PRIVATE CUDA::cudart_static dynlink_nvjpeg)
  target_link_libraries(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}
    PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_nvjpeg>")
endif()

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden")
  set_target_properties(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME} ${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    NO_SONAME OFF)
else()
  set_target_properties(${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME ${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
    ARCHIVE_OUTPUT_NAME ${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME})
endif()

if(UNIX)
  install(TARGETS ${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME} ${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}_static
    LIBRARY DESTINATION extensions NAMELINK_SKIP COMPONENT lib
    ARCHIVE DESTINATION lib64 COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )

else()
  install(TARGETS ${NVIMGCODEC_NVJPEG_EXT_LIBRARY_NAME}
    RUNTIME DESTINATION extensions COMPONENT lib
    LIBRARY DESTINATION lib64 COMPONENT lib
    ARCHIVE DESTINATION lib64 COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )
endif()
