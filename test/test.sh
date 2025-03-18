#!/bin/bash

# This script expects:
#  * nvImageCodec python wheel to be in the same folder

# Usage: ./test.sh
# Example: ./test.sh 11 ../Python3.9/

if [ -z "$1" ]; then
    echo "CUDA toolkit version major not specified"
    exit -1
else
    CUDA_VERSION_MAJOR=$1
fi

if [ -z "$2" ]; then
    PATH_TO_PYTHON=""
else
    PATH_TO_PYTHON=$2
fi

echo "Creating python virtual environment .venv"
${PATH_TO_PYTHON} -m venv .venv
if [ $? -ne 0 ]; then exit $?; fi

echo "Activating python virtual environment .venv"
source .venv/bin/activate

CUDA_PATH=$(pwd)/.venv/lib/site-packages/nvidia/cuda_runtime
export PATH=$(pwd)/.venv/lib/${PATH_TO_PYTHON}/site-packages/nvidia/cuda_runtime/bin:$PATH

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

export PATH=$(pwd)/.venv/lib/${PATH_TO_PYTHON}/site-packages/nvidia/nvjpeg/bin:$PATH
export PATH=$(pwd)/.venv/lib/${PATH_TO_PYTHON}/site-packages/nvidia/nvjpeg2k/bin:$PATH
export PATH=$(pwd)/.venv/lib/${PATH_TO_PYTHON}/site-packages/nvidia/nvcomp:$PATH

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