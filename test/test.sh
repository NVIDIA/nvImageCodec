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

if [ -d "/opt/.nvimgcodec_test_venv" ]; then
    echo "Python virtual environment /opt/.nvimgcodec_test_venv already exists. Activating."
    . /opt/.nvimgcodec_test_venv/bin/activate
else
    echo "Creating python virtual environment .nvimgcodec_test_venv"
    ${PATH_TO_PYTHON} -m venv .nvimgcodec_test_venv
    exit_code=$?
    if [ $exit_code -ne 0 ]; then exit $exit_code; fi

    echo "Activating python virtual environment .nvimgcodec_test_venv"
    . .nvimgcodec_test_venv/bin/activate
fi

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
pip install appdirs
pip install -r requirements_lnx_cu${CUDA_VERSION_MAJOR}.txt

CUDA_PATH=$(pwd)/.nvimgcodec_test_venv/lib/site-packages/nvidia/cuda_runtime

# echo trick to get correct python version (on linux venv creates python3.minor directory)
LIB_PYTHON_DIR="$(echo "$(pwd)/.nvimgcodec_test_venv/lib/python"*)"
PYTHON_NVIDIA_LIBS="$LIB_PYTHON_DIR/site-packages/nvidia"

export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/cuda_runtime/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvjpeg/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvjpeg2k/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvtiff/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/libnvcomp/lib64/:$LD_LIBRARY_PATH

export NVIMGCODEC_EXTENSIONS_PATH=$(pwd)/../extensions

echo "Installing nvImageCodec python wheel(s) from current folder"
for G in ./*.whl; do
    echo "Found and installing: $G"
    pip install -I $G
done

echo "Running transcoder (nvimtrans) tests"
pytest -v test_transcode.py
exit_code=$?
if [ $exit_code -ne 0 ]; then exit $exit_code; fi

# If PATH_TO_PYTHON contains "python3.14", set EXTRA_PYTEST_ARGS to ignore the integration tests
if [[ "$PATH_TO_PYTHON" == *"python3.14"* ]]; then
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
    pytest -v ./python -m "slow" -n 6
    exit_code=$?
    if [ $exit_code -ne 0 ]; then exit $exit_code; fi
fi

echo "Running unit tests"
./nvimgcodec_tests --resources_dir ../resources
exit_code=$?
if [ $exit_code -ne 0 ]; then exit $exit_code; fi

echo "Deactivating python virtual environment .nvimgcodec_test_venv"
deactivate