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
export WHL_PLATFORM_NAME=${WHL_PLATFORM_NAME:-manylinux2014_${ARCH}}
export WHL_OUTDIR=${WHL_OUTDIR:-/wheelhouse}
export WHL_COMPRESSION=${WHL_COMPRESSION:-YES}
export PATH=/usr/local/cuda/bin:${PATH}
export EXTRA_CMAKE_OPTIONS=${EXTRA_CMAKE_OPTIONS}
export BUNDLE_PATH_PREFIX=${BUNDLE_PATH_PREFIX}
export TEST_BUNDLED_LIBS=${TEST_BUNDLED_LIBS:-YES}

# use all available pythons
export LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH}"
export Python_EXECUTABLE=$(which python)
export NPROC=$(grep ^processor /proc/cpuinfo | wc -l)
if [ $NPROC -gt 32 ]; then
    export NPROC=32
fi



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

cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
mkdir -p ${WHL_OUTDIR}
cp *.tar.gz *.deb ${WHL_OUTDIR}/

if [ "${BUILD_WHEEL}" = "ON" ]; then
    make wheel
    cp python/*.whl ${WHL_OUTDIR}/
    # TODO(janton): custom bundle path prefix(?)
    # TODO(janton): test bundled libs
fi
