#!/bin/bash

# This script expects:
#  * nvImageCodec python wheel to be in the same folder

# Usage: ./test.sh <cuda-version-major > <path-to-python>
# Example: ./test.sh 11 ../Python3.9/

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

echo "Creating python virtual environment .venv"
${PATH_TO_PYTHON} -m venv .venv
if [ $? -ne 0 ]; then exit $?; fi

echo "Activating python virtual environment .venv"
source .venv/bin/activate

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
pip install --ignore-requires-python -r requirements_lnx_cu${CUDA_VERSION_MAJOR}.txt

CUDA_PATH=$(pwd)/.venv/lib/site-packages/nvidia/cuda_runtime

# echo trick to get correct python version (on linux venv creates python3.minor directory)
LIB_PYTHON_DIR="$(echo "$(pwd)/.venv/lib/python"*)"
PYTHON_NVIDIA_LIBS="$LIB_PYTHON_DIR/site-packages/nvidia"

export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/cuda_runtime/lib/:$LD_LIBRARY_PATH

export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvjpeg/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvjpeg2k/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvtiff/lib/:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$PYTHON_NVIDIA_LIBS/nvcomp/:$LD_LIBRARY_PATH

export NVIMGCODEC_EXTENSIONS_PATH=$(pwd)/../extensions

echo "Installing nvImageCodec python wheel(s) from current folder"
for G in ./*.whl; do
    echo "Found and installing: $G"
    pip install -I $G
done

echo "Running transcoder (nvimtrans) tests"
pytest -v test_transcode.py
if [ $? -ne 0 ]; then exit $?; fi

echo "Running python tests"
pytest -v ./python
if [ $? -ne 0 ]; then exit $?; fi

echo "Running unit tests"
./nvimgcodec_tests --resources_dir ../resources
if [ $? -ne 0 ]; then exit $?; fi

echo "Deactivating python virtual environment .venv"
deactivate