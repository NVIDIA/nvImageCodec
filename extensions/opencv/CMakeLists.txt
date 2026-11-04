# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

set(EXT_LIBRARY_NAME opencv_ext)

set(EXT_SRCS
  opencv_ext.cpp
  opencv_utils.cpp
  opencv_encoder.cpp
  opencv_decoder.cpp
  )

set(EXT_HEADERS
  error_handling.h
  log.h
  opencv_decoder.h
  opencv_encoder.h
  opencv_ext.h
  opencv_utils.h
  )

add_library(${EXT_LIBRARY_NAME}_obj OBJECT ${EXT_SRCS} ${EXT_HEADERS})
add_library(${EXT_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> ext_module.cpp)

nvimgcodec_apply_version_script(${EXT_LIBRARY_NAME} extension.map)

add_library(${EXT_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj>)

target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC nvimgcodec)
target_link_libraries(${EXT_LIBRARY_NAME}_static PUBLIC nvimgcodec)
target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC ${OpenCV_LIBRARIES})
target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC ${JPEG_LIBRARY})
if(TIFF_FOUND)
  target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC TIFF::TIFF)
endif()

if(UNIX)
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
  target_link_libraries(${EXT_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")

  set_target_properties(${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    NO_SONAME OFF)
else()
  target_link_directories(${EXT_LIBRARY_NAME} PUBLIC ${CMAKE_LIBRARY_PATH})

  set_target_properties(${EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME ${EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
    ARCHIVE_OUTPUT_NAME ${EXT_LIBRARY_NAME}
    MSVC_RUNTIME_LIBRARY MultiThreadedDLL)
endif()

if(UNIX)
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

  if(NVIMGCODEC_OPENCV_RUNTIME_DLLS)
    install(FILES ${NVIMGCODEC_OPENCV_RUNTIME_DLLS}
      DESTINATION "${NVIMGCODEC_INSTALL_EXTENSIONS_DIR}"
      COMPONENT lib
    )
  endif()
endif()
