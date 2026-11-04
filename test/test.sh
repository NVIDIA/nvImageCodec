#!/bin/bash

# This script expects:
#  * nvImageCodec python wheel to be in the same folder

# Usage: ./test.sh <cuda-version-major> <path-to-python> <run-slow-tests>
# Example: ./test.sh 11 ../Python3.9/ false
#

if [ -z "$1" ]; then
    echo "CUDA toolkit version major not specified"
    exit -1
else
    CUDA_VERSION_MAJOR=$1
fi

if [ -z "$2" ]; then
    PATH_TO_PYTHON="python3"
else
    PATH_TO_PYTHON=$2
fi

if [ -z "$3" ]; then
    RUN_SLOW_TESTS=${RUN_SLOW_TESTS:-false}
else
    RUN_SLOW_TESTS=$3
fi

PYTHON_TAG=$("${PATH_TO_PYTHON}" -c 'import sys; print(f"py{sys.version_info.major}{sys.version_info.minor}")')
exit_code=$?
if [ $exit_code -ne 0 ]; then exit $exit_code; fi

VENV_RUN_ID=${BUILD_ID:-$$}
VENV_DIR=".nvimgcodec_test_venv_${PYTHON_TAG}_cu${CUDA_VERSION_MAJOR}_${VENV_RUN_ID}"

if [ ! -x "${VENV_DIR}/bin/python" ]; then
    if [ -d "${VENV_DIR}" ]; then
        echo "Removing incomplete python virtual environment ${VENV_DIR}"
        rm -rf "${VENV_DIR}"
    fi

    echo "Creating python virtual environment ${VENV_DIR} using ${PATH_TO_PYTHON}"
    "${PATH_TO_PYTHON}" -m venv "${VENV_DIR}"
    exit_code=$?
    if [ $exit_code -ne 0 ]; then exit $exit_code; fi
else
    echo "Using existing python virtual environment ${VENV_DIR}"
fi

echo "Activating python virtual environment ${VENV_DIR}"
. "${VENV_DIR}/bin/activate"

# Check the system architecture
ARCH=$(uname -m)

if [ "$ARCH" = "aarch64" ]; then
    echo "Running operation for aarch64 architecture"
    # Preloading the library can help allocate the necessary memory for the TLS block
    # Workaround for issue on aarch64 when importing opencv using python3.10
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libGLdispatch.so.0:$LD_PRELOAD
fi


echo "Installing python requirements"
python -m pip install --upgrade pip setuptools wheel
python -m pip install appdirs
python -m pip install -r requirements_lnx_cu${CUDA_VERSION_MAJOR}.txt

PYTHON_SITE_PACKAGES=$(python -c 'import site; print(site.getsitepackages()[0])')
PYTHON_NVIDIA_LIBS="${PYTHON_SITE_PACKAGES}/nvidia"

if [ -d "${PYTHON_NVIDIA_LIBS}/cu${CUDA_VERSION_MAJOR}/lib" ]; then
    # CUDA 13+ wheels share a versioned package directory, including nvJPEG.
    CUDA_PATH=${PYTHON_NVIDIA_LIBS}/cu${CUDA_VERSION_MAJOR}
    export LD_LIBRARY_PATH="${CUDA_PATH}/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    # CUDA 12 and older wheels use one package directory per component.
    CUDA_PATH=${PYTHON_NVIDIA_LIBS}/cuda_runtime
    export LD_LIBRARY_PATH="${PYTHON_NVIDIA_LIBS}/cuda_runtime/lib/${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${PYTHON_NVIDIA_LIBS}/nvjpeg/lib/:${LD_LIBRARY_PATH}"
fi

export LD_LIBRARY_PATH=${PYTHON_NVIDIA_LIBS}/nvjpeg2k/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PYTHON_NVIDIA_LIBS}/nvtiff/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=${PYTHON_NVIDIA_LIBS}/libnvcomp/lib64/:$LD_LIBRARY_PATH

NVIMGCODEC_ARCHIVE_LIB_DIR=$(pwd)/../lib64/libnvimgcodec/${CUDA_VERSION_MAJOR}
export LD_LIBRARY_PATH=$NVIMGCODEC_ARCHIVE_LIB_DIR:$LD_LIBRARY_PATH

echo "Installing nvImageCodec python wheel(s) from current folder"
for G in ./*.whl; do
    echo "Found and installing: $G"
    python -m pip install -I "$G"
done

# Required GPU lanes must fail when TIFF fixture dependencies are missing instead
# of silently skipping native batching tests (see test_nvtiff_batching._dependency).
export NVIMGCODEC_REQUIRE_TIFF_TESTS=1

echo "Running transcoder (nvimtrans) tests"
pytest -v test_transcode.py
exit_code=$?
if [ $exit_code -ne 0 ]; then exit $exit_code; fi

# The integration tests need packages that requirements_lnx_cu*.txt does not install on
# these versions (CuPy, PyTorch, parts of the DICOM stack).
if [[ "${PYTHON_TAG}" == "py314" || "${PYTHON_TAG}" == "py315" ]]; then
    EXTRA_PYTEST_ARGS="--ignore=python/integration"
else
    EXTRA_PYTEST_ARGS=""
fi

echo "Running python tests (excluding slow tests)"
pytest -v ./python $EXTRA_PYTEST_ARGS
exit_code=$?
if [ $exit_code -ne 0 ]; then exit $exit_code; fi

if [ "$RUN_SLOW_TESTS" = "true" ]; then
    echo "Running slow tests (parallel workers: 6)"
    exit_code=0
    pytest -v ./python -m "slow" -n 6 || exit_code=$?
    if [ $exit_code -eq 5 ]; then
        echo "No slow tests collected or all skipped, continuing."
    elif [ $exit_code -ne 0 ]; then
        exit $exit_code
    fi
fi

echo "Running unit tests"
./nvimgcodec_tests --resources_dir ../resources
exit_code=$?
if [ $exit_code -ne 0 ]; then exit $exit_code; fi

echo "Deactivating python virtual environment ${VENV_DIR}"
deactivate
