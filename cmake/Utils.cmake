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


# Build a .so library variant for each python version provided in PYTHON_VERSIONS variable
# if it is accesible during the build time. The library sufix is provided and specific for each python version
#
# supported options:
# <TARGET_NAME> - umbrella target name used for this set of libraries; two additional targets are created,
#              which can be used to set PUBLIC and PRIVATE target properties, respectively:
#              TARGET_NAME_public
#              TARGET_NAME_private.
#              Properties for these targets, such as link dependencies, includes or compilation flags,
#              must be set with INTERFACE keyword, but they are propagated as PUBLIC/PRIVATE
#              properties to all version-specific libraries.
# OUTPUT_NAME - library name used for this build
# PREFIX - library prefix, if none is provided, the library will be named ${TARGET_NAME}.python_specific_extension
# OUTPUT_DIR - ouptut directory of the build library
# PUBLIC_LIBS - list of libraries that should be linked in as a public one
# PRIV_LIBS - list of libraries that should be linked in as a private one
# SRC - list of source code files
function(build_per_python_lib)
    set(oneValueArgs TARGET_NAME OUTPUT_NAME OUTPUT_DIR PREFIX)
    set(multiValueArgs PRIV_LIBS PUBLIC_LIBS SRC EXCLUDE_LIBS)

    cmake_parse_arguments(PARSE_ARGV 1 PYTHON_LIB_ARG "${options}" "${oneValueArgs}" "${multiValueArgs}")

    set(PYTHON_LIB_ARG_TARGET_NAME ${ARGV0})
    add_custom_target(${PYTHON_LIB_ARG_TARGET_NAME} ALL)

    # global per target interface library, common for all python variants
    add_library(${PYTHON_LIB_ARG_TARGET_NAME}_public INTERFACE)
    add_library(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE)

    target_sources(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE ${PYTHON_LIB_ARG_SRC})

    target_link_libraries(${PYTHON_LIB_ARG_TARGET_NAME}_public INTERFACE ${PYTHON_LIB_ARG_PUBLIC_LIBS})
    target_link_libraries(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE ${PYTHON_LIB_ARG_PRIV_LIBS})
    target_link_libraries(${PYTHON_LIB_ARG_TARGET_NAME}_private INTERFACE "-Wl,--exclude-libs,${PYTHON_LIB_ARG_EXCLUDE_LIBS}")

    target_include_directories(${PYTHON_LIB_ARG_TARGET_NAME}_private
                               INTERFACE "${PYBIND11_INCLUDE_DIR}"
                               INTERFACE "${pybind11_INCLUDE_DIR}")


    set (PYTHON_VERSIONS "3.8;3.9;3.10;3.11;3.12")
    foreach(PYVER ${PYTHON_VERSIONS})

        set(PYTHON_LIB_TARGET_FOR_PYVER "${PYTHON_LIB_ARG_TARGET_NAME}_${PYVER}")
        # check if listed python versions are accesible
        execute_process(
            COMMAND python${PYVER}-config --help
            RESULT_VARIABLE PYTHON_EXISTS OUTPUT_QUIET)

        if (${PYTHON_EXISTS} EQUAL 0)
            execute_process(
                COMMAND python${PYVER}-config --extension-suffix
                OUTPUT_VARIABLE PYTHON_SUFIX)
            # remove newline and the extension
            string(REPLACE ".so\n" "" PYTHON_SUFIX "${PYTHON_SUFIX}")

            execute_process(
                COMMAND python${PYVER}-config --includes
                OUTPUT_VARIABLE PYTHON_INCLUDES)
            # split and make it a list
            string(REPLACE "-I" "" PYTHON_INCLUDES "${PYTHON_INCLUDES}")
            string(REPLACE "\n" "" PYTHON_INCLUDES "${PYTHON_INCLUDES}")
            separate_arguments(PYTHON_INCLUDES)

            add_library(${PYTHON_LIB_TARGET_FOR_PYVER} SHARED)

            set_target_properties(${PYTHON_LIB_TARGET_FOR_PYVER}
                                    PROPERTIES
                                    LIBRARY_OUTPUT_DIRECTORY ${PYTHON_LIB_ARG_OUTPUT_DIR}
                                    PREFIX "${PYTHON_LIB_ARG_PREFIX}"
                                    OUTPUT_NAME ${PYTHON_LIB_ARG_OUTPUT_NAME}${PYTHON_SUFIX})
            # add includes
            foreach(incl_dir ${PYTHON_INCLUDES})
                target_include_directories(${PYTHON_LIB_TARGET_FOR_PYVER} PRIVATE ${incl_dir})
            endforeach(incl_dir)

            # add interface dummy lib as a dependnecy to easilly propagate options we could set from the above
            target_link_libraries(${PYTHON_LIB_TARGET_FOR_PYVER} PUBLIC ${PYTHON_LIB_ARG_TARGET_NAME}_public)
            target_link_libraries(${PYTHON_LIB_TARGET_FOR_PYVER} PRIVATE ${PYTHON_LIB_ARG_TARGET_NAME}_private)

            add_dependencies(${PYTHON_LIB_ARG_TARGET_NAME} ${PYTHON_LIB_TARGET_FOR_PYVER})
        endif()

    endforeach(PYVER)

endfunction()


function(parse_cuda_version CUDA_VERSION CUDA_VERSION_MAJOR_VAR CUDA_VERSION_MINOR_VAR CUDA_VERSION_PATCH_VAR CUDA_VERSION_SHORT_VAR CUDA_VERSION_SHORT_DIGIT_ONLY_VAR)
  string(REPLACE "." ";" CUDA_VERSION_LIST ${CUDA_VERSION})
  list(GET CUDA_VERSION_LIST 0 ${CUDA_VERSION_MAJOR_VAR})
  list(GET CUDA_VERSION_LIST 1 ${CUDA_VERSION_MINOR_VAR})
  string(REGEX MATCH "^[0-9]*$" ${CUDA_VERSION_MAJOR_VAR}  ${${CUDA_VERSION_MAJOR_VAR}})
  string(REGEX MATCH "^[0-9]*$" ${CUDA_VERSION_MINOR_VAR}  ${${CUDA_VERSION_MINOR_VAR}})

  list(LENGTH CUDA_VERSION_LIST LIST_LENGTH)
  if (${LIST_LENGTH} GREATER 2)
    list(GET CUDA_VERSION_LIST 2 ${CUDA_VERSION_PATCH_VAR})
    string(REGEX MATCH "^[0-9]*$" ${CUDA_VERSION_PATCH_VAR}  ${${CUDA_VERSION_PATCH_VAR}} PARENT_SCOPE)
  endif()

  if ("${${CUDA_VERSION_MAJOR_VAR}}" STREQUAL "" OR "${${CUDA_VERSION_MINOR_VAR}}" STREQUAL "")
    message(FATAL_ERROR "CUDA version is not valid: ${CUDA_VERSION}")
  endif()

  set(${CUDA_VERSION_MAJOR_VAR} "${${CUDA_VERSION_MAJOR_VAR}}" PARENT_SCOPE)
  set(${CUDA_VERSION_MINOR_VAR} "${${CUDA_VERSION_MINOR_VAR}}" PARENT_SCOPE)
  set(${CUDA_VERSION_PATCH_VAR} "${${CUDA_VERSION_PATCH_VAR}}" PARENT_SCOPE)
  message(STATUS "CUDA version: ${CUDA_VERSION}, major: ${${CUDA_VERSION_MAJOR_VAR}}, minor: ${${CUDA_VERSION_MINOR_VAR}}, patch: ${${CUDA_VERSION_PATCH_VAR}}, short: ${${CUDA_VERSION_SHORT_VAR}}, digit-only: ${${CUDA_VERSION_SHORT_DIGIT_ONLY_VAR}}")

  # when building for any version >= 11.0 use CUDA compatibility mode and claim it is a CUDA 110 package
  # TF build image uses cmake 3.5 with not GREATER_EQUAL so split it to GREATER OR EQUAL
  if ((${${CUDA_VERSION_MAJOR_VAR}} GREATER "11" OR ${${CUDA_VERSION_MAJOR_VAR}} EQUAL "11") AND ${${CUDA_VERSION_MINOR_VAR}} GREATER "0")
     set(${CUDA_VERSION_MINOR_VAR} "0")
     set(${CUDA_VERSION_PATCH_VAR} "0")
     set(${CUDA_VERSION_MINOR_VAR} "0" PARENT_SCOPE)
     set(${CUDA_VERSION_PATCH_VAR} "0" PARENT_SCOPE)
  endif()
  set(${CUDA_VERSION_SHORT_VAR} "${${CUDA_VERSION_MAJOR_VAR}}.${${CUDA_VERSION_MINOR_VAR}}" PARENT_SCOPE)
  set(${CUDA_VERSION_SHORT_DIGIT_ONLY_VAR} "${${CUDA_VERSION_MAJOR_VAR}}${${CUDA_VERSION_MINOR_VAR}}" PARENT_SCOPE)

  message(STATUS "Compatible CUDA version: major: ${${CUDA_VERSION_MAJOR_VAR}}, minor: ${${CUDA_VERSION_MINOR_VAR}}, patch: ${${CUDA_VERSION_PATCH_VAR}}, short: ${${CUDA_VERSION_SHORT_VAR}}, digit-only: ${${CUDA_VERSION_SHORT_DIGIT_ONLY_VAR}}")
endfunction()


# add a post-build step to the provided target which copies
# files or directories recursively from SRC to DST
# create phony target first (if not exists with given name yet)
# and add comand attached to it
macro(copy_post_build TARGET_NAME SRC DST)
    if (NOT (TARGET install_${TARGET_NAME}))
        add_custom_target(install_${TARGET_NAME} ALL
             DEPENDS ${TARGET_NAME}
        )
    endif()

    add_custom_command(
    TARGET install_${TARGET_NAME}
    COMMAND mkdir -p "${DST}" && cp -r  "${SRC}" "${DST}")
endmacro(copy_post_build)

# get default compiler include paths, needed by the stub generator
# starting from 3.14.0 CMake will have that inside CMAKE_${LANG}_IMPLICIT_INCLUDE_DIRECTORIES
macro(DETERMINE_GCC_SYSTEM_INCLUDE_DIRS _lang _compiler _flags _result)
    file(WRITE "${CMAKE_BINARY_DIR}/CMakeFiles/dummy" "\n")
    separate_arguments(_buildFlags UNIX_COMMAND "${_flags}")
    execute_process(COMMAND ${_compiler} ${_buildFlags} -v -E -x ${_lang} -dD dummy
                    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/CMakeFiles OUTPUT_QUIET
                    ERROR_VARIABLE _gccOutput)
    file(REMOVE "${CMAKE_BINARY_DIR}/CMakeFiles/dummy")
    if ("${_gccOutput}" MATCHES "> search starts here[^\n]+\n *(.+) *\n *End of (search) list")
        set(${_result} ${CMAKE_MATCH_1})
        string(REPLACE "\n" " " ${_result} "${${_result}}")
        separate_arguments(${_result})
    endif ()
endmacro()

# check_and_add_cmake_submodule
# Checks for presence of a git submodule that includes a CMakeLists.txt
# Usage:
#   check_and_add_cmake_submodule(<submodule_path> ..)
macro(check_and_add_cmake_submodule SUBMODULE_PATH)
  if(NOT EXISTS ${SUBMODULE_PATH}/CMakeLists.txt)
    message(FATAL_ERROR "File ${SUBMODULE_PATH}/CMakeLists.txt not found. "
                        "Did you forget to `git clone --recursive`? Try this:\n"
                        "  cd ${PROJECT_SOURCE_DIR} && \\\n"
                        "  git submodule sync --recursive && \\\n"
                        "  git submodule update --init --recursive && \\\n"
                        "  cd -\n")
  endif()
  add_subdirectory(${SUBMODULE_PATH} ${ARGN})
endmacro(check_and_add_cmake_submodule)

function(propagate_option BUILD_OPTION_NAME)
  string(REPLACE "BUILD_" "" OPTION_NAME ${BUILD_OPTION_NAME})
  set(DEFINE_NAME ${OPTION_NAME}_ENABLED)
  if (${BUILD_OPTION_NAME})
    message(STATUS "${BUILD_OPTION_NAME} -- ON")
    add_definitions(-D${DEFINE_NAME}=1)
  else()
    message(STATUS "${BUILD_OPTION_NAME} -- OFF")
    add_definitions(-D${DEFINE_NAME}=0)
  endif()
endfunction(propagate_option)