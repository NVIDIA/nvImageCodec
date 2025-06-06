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

set(EXT_LIBRARY_NAME libtiff_ext)

set(EXT_SRCS
  libtiff_ext.cpp
  libtiff_decoder.cpp
  )


add_library(${EXT_LIBRARY_NAME}_obj OBJECT ${EXT_SRCS})
add_library(${EXT_LIBRARY_NAME} SHARED $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj> ext_module.cpp)
add_library(${EXT_LIBRARY_NAME}_static STATIC $<TARGET_OBJECTS:${EXT_LIBRARY_NAME}_obj>)

target_link_libraries(${EXT_LIBRARY_NAME} PUBLIC ${TIFF_LIBRARIES} ${TIFF_LIBRARY_DEPS})
target_link_libraries(${EXT_LIBRARY_NAME}_static PUBLIC ${TIFF_LIBRARIES} ${TIFF_LIBRARY_DEPS})

if(UNIX)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fPIC -fvisibility=hidden -Wl,--exclude-libs,ALL")
  target_link_libraries(${EXT_LIBRARY_NAME} PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")
  target_link_libraries(${EXT_LIBRARY_NAME}_static PRIVATE "${NVIMGCODEC_COMMON_DEPENDENCIES}")

  set_target_properties(${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    VERSION ${PROJECT_VERSION}
    NO_SONAME OFF)
else()
  set_target_properties(${EXT_LIBRARY_NAME} PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    VERSION ${PROJECT_VERSION}
    OUTPUT_NAME ${EXT_LIBRARY_NAME}_${PROJECT_VERSION_MAJOR}
    ARCHIVE_OUTPUT_NAME ${EXT_LIBRARY_NAME})
endif()

if(UNIX)
  install(TARGETS ${EXT_LIBRARY_NAME} ${EXT_LIBRARY_NAME}_static
    LIBRARY DESTINATION extensions NAMELINK_SKIP COMPONENT lib
    ARCHIVE COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )
else()
  install(TARGETS ${EXT_LIBRARY_NAME}
    RUNTIME DESTINATION extensions COMPONENT lib
    LIBRARY COMPONENT lib
    ARCHIVE COMPONENT lib
    PUBLIC_HEADER DESTINATION include COMPONENT lib
  )
endif()
