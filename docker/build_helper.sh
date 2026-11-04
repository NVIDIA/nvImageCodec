#!/bin/bash

# Stop at any error, show all commands
set -ex

usage="ENV1=VAL1 ENV2=VAL2 [...] $(basename "$0") [-h] -- this nvimgcodec build helper mean to run from the docker environment.
Please don't call it directly.

where:
    -h  show this help text"

while getopts 'h' option; do
  case "$option" in
    h) echo "$usage"
       exit
       ;;
   \?) printf "illegal option: -%s\n" "$OPTARG" >&2
       echo "$usage" >&2
       exit 1
       ;;
  esac
done
shift $((OPTIND - 1))

export ARCH=${ARCH:-x86_64}
export CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-Release}

export BUILD_LIBRARY=${BUILD_LIBRARY:-ON}
export BUILD_TEST=${BUILD_TEST:-ON}
export BUILD_SAMPLES=${BUILD_SAMPLES:-ON}
export BUILD_CVCUDA_SAMPLES=${BUILD_CVCUDA_SAMPLES:-OFF}
export BUILD_DOCS=${BUILD_DOCS:-OFF}
export BUILD_EXTENSIONS=${BUILD_EXTENSIONS:-ON}
export BUILD_NVJPEG_EXT=${BUILD_NVJPEG_EXT:-ON}
export BUILD_NVJPEG2K_EXT=${BUILD_NVJPEG2K_EXT:-ON}
export BUILD_NVBMP_EXT=${BUILD_NVBMP_EXT:-ON}
export BUILD_NVPNM_EXT=${BUILD_NVPNM_EXT:-ON}
export BUILD_LIBJPEG_TURBO_EXT=${BUILD_LIBJPEG_TURBO_EXT:-ON}
export BUILD_LIBTIFF_EXT=${BUILD_LIBTIFF_EXT:-ON}
export BUILD_OPENCV_EXT=${BUILD_OPENCV_EXT:-ON}
export BUILD_NVTIFF_EXT=${BUILD_NVTIFF_EXT:-ON}
export BUILD_PYTHON=${BUILD_PYTHON:-ON}
export BUILD_WHEEL=${BUILD_WHEEL:-ON}
export WITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG:-ON}
export WITH_DYNAMIC_NVJPEG2K=${WITH_DYNAMIC_NVJPEG2K:-ON}
export WITH_DYNAMIC_NVTIFF=${WITH_DYNAMIC_NVTIFF:-ON}

export NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID:-0}
export GIT_SHA=${GIT_SHA}
export NVIMGCODEC_TIMESTAMP=${NVIMGCODEC_TIMESTAMP}
export BUILD_FLAVOR=${BUILD_FLAVOR}
export CUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}
export WHL_PLATFORM_NAME=${WHL_PLATFORM_NAME:-manylinux_2_28_${ARCH}}
export WHL_OUTDIR=${WHL_OUTDIR:-/wheelhouse}
export WHL_COMPRESSION=${WHL_COMPRESSION:-YES}
export PATH=/usr/local/cuda/bin:${PATH}
export EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS}
export BUNDLE_PATH_PREFIX=${BUNDLE_PATH_PREFIX}
export TEST_BUNDLED_LIBS=${TEST_BUNDLED_LIBS:-YES}

# The JOB_SPEC passes INSTALL_LIB_PREFIX_DIR / INSTALL_LIB_ARCHIVE and
# INSTALL_TEST_PREFIX_DIR / INSTALL_TEST_ARCHIVE; accept either naming style.
INSTALL_PREFIX_DIR=${INSTALL_PREFIX_DIR:-${INSTALL_LIB_PREFIX_DIR:-}}
INSTALL_ARCHIVE=${INSTALL_ARCHIVE:-${INSTALL_LIB_ARCHIVE:-}}
INSTALL_TEST_PREFIX_DIR=${INSTALL_TEST_PREFIX_DIR:-}
INSTALL_TEST_ARCHIVE=${INSTALL_TEST_ARCHIVE:-}

export LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH}"

# Pick the interpreter that runs CMake's stub codegen and `setup.py bdist_wheel`.
# HOST_PYTHON_VERSION is set by the builder Dockerfile (ENV HOST_PYTHON_VERSION=...)
# and can be overridden at the call site (e.g. HOST_PYTHON_VERSION=3.13 ./build_helper.sh).
# Falls back to whatever `python` resolves to on PATH for builders that don't bake it in.
if [ -n "${HOST_PYTHON_VERSION}" ]; then
    export Python_EXECUTABLE="/usr/bin/python${HOST_PYTHON_VERSION}"
    if [ ! -x "${Python_EXECUTABLE}" ]; then
        echo "ERROR: Python_EXECUTABLE=${Python_EXECUTABLE} not found (HOST_PYTHON_VERSION=${HOST_PYTHON_VERSION})" >&2
        exit 1
    fi
else
    export Python_EXECUTABLE=$(which python)
fi

export NPROC=$(grep ^processor /proc/cpuinfo | wc -l)
if [ $NPROC -gt 32 ]; then
    export NPROC=32
fi

archive_install_prefix() {
    local archive_path="$1"
    local prefix_path="$2"
    local archive_var="$3"
    local prefix_var="$4"

    if [ -z "${archive_path}" ]; then
        return
    fi
    if [ -z "${prefix_path}" ]; then
        echo "ERROR: ${archive_var} requires ${prefix_var}" >&2
        exit 1
    fi
    tar -czf "${archive_path}" -C "${prefix_path}" .
}

cmake ../                                                              \
      -DBUILD_ID=${NVIDIA_BUILD_ID}                                    \
      -DARCH=${ARCH}                                                   \
      -DCUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}                         \
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}                           \
      -DBUILD_LIBRARY=${BUILD_LIBRARY}                                 \
      -DBUILD_TEST=${BUILD_TEST}                                       \
      -DBUILD_SAMPLES=${BUILD_SAMPLES}                                 \
      -DBUILD_CVCUDA_SAMPLES=${BUILD_CVCUDA_SAMPLES}                   \
      -DBUILD_DOCS=${BUILD_DOCS}                                       \
      -DBUILD_EXTENSIONS=${BUILD_EXTENSIONS}                           \
      -DBUILD_NVJPEG_EXT=${BUILD_NVJPEG_EXT}                           \
      -DBUILD_NVJPEG2K_EXT=${BUILD_NVJPEG2K_EXT}                       \
      -DBUILD_NVBMP_EXT=${BUILD_NVBMP_EXT}                             \
      -DBUILD_NVPNM_EXT=${BUILD_NVPNM_EXT}                             \
      -DBUILD_LIBJPEG_TURBO_EXT=${BUILD_LIBJPEG_TURBO_EXT}             \
      -DBUILD_LIBTIFF_EXT=${BUILD_LIBTIFF_EXT}                         \
      -DBUILD_OPENCV_EXT=${BUILD_OPENCV_EXT}                           \
      -DBUILD_NVTIFF_EXT=${BUILD_NVTIFF_EXT}                           \
      -DSKIP_NVTIFF_WITH_NVCOMP_TESTS=${SKIP_NVTIFF_WITH_NVCOMP_TESTS} \
      -DBUILD_PYTHON=${BUILD_PYTHON}                                   \
      -DBUILD_WHEEL=${BUILD_WHEEL}                                     \
      -DWITH_DYNAMIC_NVJPEG=${WITH_DYNAMIC_NVJPEG}                     \
      -DWITH_DYNAMIC_NVJPEG2K=${WITH_DYNAMIC_NVJPEG2K}                 \
      -DWITH_DYNAMIC_NVTIFF=${WITH_DYNAMIC_NVTIFF}                     \
      -DNVIMGCODEC_FLAVOR=${BUILD_FLAVOR}                       \
      -DNVIMGCODEC_WHL_PLATFORM_NAME=${WHL_PLATFORM_NAME}              \
      -DTIMESTAMP=${NVIMGCODEC_TIMESTAMP} -DGIT_SHA=${GIT_SHA}         \
      -DPython_EXECUTABLE=${Python_EXECUTABLE}                         \
      ${EXTRA_CMAKE_OPTIONS}

make -j${NPROC}

# Guard the linker version scripts: assert every shipped .so exports exactly its
# allowlist and leaks no weak libstdc++ symbols. Run here rather than as a test
# command because CI runs the gtest binary directly from the test tarball and
# never invokes ctest, so the equivalent ctest registration would never fire.
# Linux-only and needs `nm`, both of which hold in this builder image.
# Set SKIP_EXPORTED_SYMBOL_CHECK=1 to bypass.
if [ "${SKIP_EXPORTED_SYMBOL_CHECK:-0}" != "1" ]; then
    if [ ! -x ../tools/check_exported_symbols.sh ]; then
        echo "ERROR: ../tools/check_exported_symbols.sh is missing or not executable" >&2
        exit 1
    fi
    echo "Checking exported symbols..."
    ../tools/check_exported_symbols.sh .
fi

# When INSTALL_PREFIX_DIR is set, run `cmake --install` so the parent CI/CD
# pipeline can pick up a fully-staged install tree (lib/, include/, etc.)
# from a known sibling directory. CI sets this; local invocations leave it
# unset and skip the install step.
if [ -n "${INSTALL_PREFIX_DIR}" ]; then
    cmake --install . --prefix "${INSTALL_PREFIX_DIR}"
fi
if [ -n "${INSTALL_LIB_PREFIX_DIR:-}" ]; then
    cmake --install . --prefix "${INSTALL_LIB_PREFIX_DIR}" --component lib
fi
if [ -n "${INSTALL_TEST_PREFIX_DIR:-}" ]; then
    cmake --install . --prefix "${INSTALL_TEST_PREFIX_DIR}" --component tests
fi

if [ -n "${INSTALL_TEST_PREFIX_DIR}" ]; then
    cmake --install . --prefix "${INSTALL_TEST_PREFIX_DIR}" --component tests
fi

# Guards for the exported CMake package. Run here rather than as a test command
# because the test runner images do not ship cmake, while this builder does.
# The driver stages its own install, so it does not depend on the INSTALL_*
# variables above being set. Set SKIP_CMAKE_PACKAGE_GUARDS=1 to bypass.
if [ "${SKIP_CMAKE_PACKAGE_GUARDS:-0}" != "1" ] && [ -f ../test/cmake_package/run_package_tests.cmake ]; then
    echo "Running CMake package guards..."
    cmake -DNVIMGCODEC_BUILD_DIR=. -P ../test/cmake_package/run_package_tests.cmake
fi

cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
mkdir -p ${WHL_OUTDIR}
cp *.tar.gz *.deb ${WHL_OUTDIR}/

if [ "${BUILD_WHEEL}" = "ON" ]; then
    make wheel
    cp python/*.whl ${WHL_OUTDIR}/
    if [ -n "${INSTALL_PREFIX_DIR}" ]; then
        mkdir -p ${INSTALL_PREFIX_DIR}/test
        cp python/*.whl ${INSTALL_PREFIX_DIR}/test/
    fi
    if [ -n "${INSTALL_TEST_PREFIX_DIR:-}" ]; then
        mkdir -p ${INSTALL_TEST_PREFIX_DIR}/test
        cp python/*.whl ${INSTALL_TEST_PREFIX_DIR}/test/
    fi
    # TODO(janton): custom bundle path prefix(?)
    # TODO(janton): test bundled libs
fi

archive_install_prefix "${INSTALL_ARCHIVE:-}" "${INSTALL_PREFIX_DIR:-}" INSTALL_ARCHIVE INSTALL_PREFIX_DIR
archive_install_prefix "${INSTALL_LIB_ARCHIVE:-}" "${INSTALL_LIB_PREFIX_DIR:-}" INSTALL_LIB_ARCHIVE INSTALL_LIB_PREFIX_DIR
archive_install_prefix "${INSTALL_TEST_ARCHIVE:-}" "${INSTALL_TEST_PREFIX_DIR:-}" INSTALL_TEST_ARCHIVE INSTALL_TEST_PREFIX_DIR
