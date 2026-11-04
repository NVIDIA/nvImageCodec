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

set(EXT_LIBRARY_NAME nvtiff_ext)

set(EXT_SRCS
  cuda_decoder.cpp
  cuda_encoder.cpp
  nvtiff_ext.cpp
  nvtiff_utils.cpp
  metadata_extractor.cpp
)

set(EXT_HEADERS
  cuda_decoder.h
  cuda_encoder.h
  error_handling.h
  log.h
  nvtiff_ext.h
  nvtiff_utils.h
  metadata_extractor.h
)

# Add utils directory for common headers
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/..)

add_library(${EXT_LIBRARY_NAME}_obj OBJECT ${EXT_SRCS} ${EXT_HEADERS})
add_library(${EXT_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> ext_module.cpp $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)

nvimgcodec_apply_version_script(${EXT_LIBRARY_NAME} extension.map)

add_library(${EXT_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> $<TARGET_OBJECTS:nvimgcodec_imgproc_static>)

if (WITH_DYNAMIC_CUDA)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE dynlink_cuda)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_cuda>")
else()
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE CUDA::cuda_driver)
endif()

if (WITH_DYNAMIC_NVTIFF)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE dynlink_nvtiff)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_nvtiff>")
else()
  target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC ${NVTIFF_LIB})
endif()

if(NOT WITH_DYNAMIC_NVJPEG)
  if(UNIX)
    target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC ${CUDA_NVJPEG_LIB} CUDA::culibos)
  else()
    target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC CUDA::nvjpeg)
  endif()
else()
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE dynlink_nvjpeg)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "-Wl,--exclude-libs,$<TARGET_FILE_NAME:dynlink_nvjpeg>")
endif()

target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC ${CUDA_CUDART_LIB})

if (UNIX)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
  target_link_libraries(${EXT_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")

  set_target_properties(${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    NO_SONAME OFF
  )
else()
  set_target_properties(${EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME ${EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
    ARCHIVE_OUTPUT_NAME ${EXT_LIBRARY_NAME}
  )
endif()

if (UNIX)
  install(TARGETS ${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static
    LIBRARY DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}" NAMELINK_SKIP COMPONENT lib
    ARCHIVE DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${NVIMGCODEC_INSTALL_INCLUDEDIR}" COMPONENT lib
  )

else()
  install(TARGETS ${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static
    RUNTIME DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}" COMPONENT lib
    ARCHIVE DESTINATION "${NVIMGCODEC_INSTALL_EXTENSION_LIBDIR}" COMPONENT lib
    PUBLIC_HEADER DESTINATION "${NVIMGCODEC_INSTALL_INCLUDEDIR}" COMPONENT lib
  )
endif()
