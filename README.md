# nvImageCodec

![Version](https://img.shields.io/badge/Version-v0.3.0--beta-blue)
[![License](https://img.shields.io/badge/License-Apache_2.0-yellogreen.svg)](https://opensource.org/licenses/Apache-2.0)

![Platform](https://img.shields.io/badge/Platform-linux--x86__64_%7C_linux--aarch64_%7C_windows--64_wsl2_%7C_windows--64-blue)

[![Cuda](https://img.shields.io/badge/CUDA-v11.8_%7c_v12.5-%2376B900?logo=nvidia)](https://developer.nvidia.com/cuda-toolkit-archive)
[![GCC](https://img.shields.io/badge/GCC->=v9.0-yellow)](https://gcc.gnu.org/gcc-9/)
[![CMake](https://img.shields.io/badge/CMake->=v3.18-%23008FBA?logo=cmake)](https://cmake.org/)


[![Python](https://img.shields.io/badge/python-v3.8_%7c_v3.9_%7c_v3.10_%7c_v3.11_%7c_v3.12-blue?logo=python)](https://www.python.org/)
![PyPI - Wheel](https://img.shields.io/pypi/wheel/nvidia-nvimgcodec-cu12?pypiBaseUrl=https%3A%2F%2Fpypi.org&label=PyPI&link=https%3A%2F%2Fpypi.org%2Fsearch%2F%3Fq%3Dnvidia-nvimgcodec-cu12)


The nvImageCodec is an open-source library of accelerated codecs with unified interface.
It is designed as a framework for extension modules which delivers codec plugins.

This nvImageCodec release includes the following key features:

- Unified API for decoding and encoding images
- Batch processing, with variable shape and heterogeneous formats images
- Codec prioritization with automatic fallback
- Builtin parsers for image format detection: jpeg, jpeg2000, tiff, bmp, png, pnm, webp 
- Python bindings
- Zero-copy interfaces to CV-CUDA, PyTorch and CuPy 
- End-end accelerated sample applications for common image transcoding

Currently there are following native codec extensions:

- nvjpeg_ext

   - Hardware jpeg decoder
   - CUDA jpeg decoder
   - CUDA lossless jpeg decoder
   - CUDA jpeg encoder

- nvjpeg2k_ext

   - CUDA jpeg 2000 decoder (including High Throughput Jpeg2000)
   - CUDA jpeg 2000 encoder 

- nvbmp_ext (as an example extension module)

   - CPU bmp reader
   - CPU bmp writer

- nvpnm_ext (as an example extension module)

   - CPU pnm (ppm, pbm, pgm) writer

Additionally as a fallback there are following 3rd party codec extensions:

- libturbo-jpeg_ext

   - CPU jpeg decoder

- libtiff_ext 

   - CPU tiff decoder

- opencv_ext

   - CPU jpeg decoder
   - CPU jpeg2k_decoder
   - CPU png decoder
   - CPU bmp decoder
   - CPU pnm decoder
   - CPU tiff decoder
   - CPU webp decoder


## Pre-requisites

This section describes the recommended dependencies to use nvImageCodec.

- Linux distro:
   - x86_64
     - Debian 11, 12
     - Fedora 39
     - RHEL 8, 9
     - OpenSUSE 15
     - SLES 15
     - Ubuntu 20.04, 22.04
     - WSL2 Ubuntu 20.04
   - arm64-sbsa
      - RHEL 8, 9
      - SLES 15
      - Ubuntu 20.04, 22.04
   - aarch64-jetson (CUDA Toolkit >= 12.0)
      - Ubuntu 22.04
- Windows
   - x86_64
     - [Microsoft Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170)   
- NVIDIA driver >= 520.56.06
- CUDA Toolkit > = 11.8
- nvJPEG2000 >= 0.8.0
- Python >= 3.8

## Install nvImageCodec library

You can download and install the appropriate built binary packages from the [nvImageCodec Developer Page](https://developer.nvidia.com/nvimgcodec-downloads) or install nvImageCodec Python from PyPI as it is described below.

### nvImageCodec Python for CUDA 11.x

```
pip install nvidia-nvimgcodec-cu11
```

### nvImageCodec Python for CUDA 12.x

```
pip install nvidia-nvimgcodec-cu12
```

### Optional installation of nvJPEG library

If you do not have CUDA Toolkit installed, or you would like install nvJPEG library independently, you can use the instructions described below.

Install the nvidia-pyindex module

```
pip install nvidia-pyindex
```

Install nvJPEG for CUDA 11.x

```
pip install nvidia-nvjpeg-cu11
```

Install nvJPEG for CUDA 12.x

```
pip install nvidia-nvjpeg-cu12
```

### Optional installation of nvJPEG2000 library

[nvJPEG2000 library](https://developer.nvidia.com/nvjpeg2000-downloads) can be installed in the system, or installed as a Python package. For the latter, follow the instructions below.

Install nvJPEG2000 for CUDA 11.x

```
pip install nvidia-nvjpeg2k-cu11
```

Install nvJPEG2000 for CUDA 12.x

```
pip install nvidia-nvjpeg2k-cu12
```

Install nvJPEG2000 for CUDA 12.x on Tegra platforms

```
pip install nvidia-nvjpeg2k-tegra-cu12
```

Please see also [nvJPEG2000 installation documentation](https://docs.nvidia.com/cuda/nvjpeg2000/userguide.html#installing-nvjpeg2000) for more information


### Documentation

[NVIDIA nvImageCodec Documentation](https://docs.nvidia.com/cuda/nvimagecodec/)

## Build and install from Sources

### Additional pre-requisites
- Linux
  - GCC >= 9.4
  - cmake >= 3.18
  - patchelf >= 0.17.2
- Windows
  - [Microsoft Visual Studio 2022 Build Tools](https://aka.ms/vs/17/release/vs_buildtools.exe)
- Dependencies for extensions. If you would not like to build particular extension you can skip it.
  - nvJPEG2000 >= 0.8.0
  - libjpeg-turbo >= 2.0.0
  - libtiff >= 4.5.0
  - opencv >= 4.10.0
- Python packages: 
  - clang==14.0.1 
  - wheel
  - setuptools
  - sphinx_rtd_theme
  - breathe 
  - future
  - flake8
  - sphinx==4.5.0

Please see also Dockerfiles.

### Build

#### Linux

```
git lfs clone https://github.com/NVIDIA/nvImageCodec.git
cd nvimagecodec
git submodule update --init --recursive --depth 1
mkdir build
cd build
export CUDACXX=nvcc
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```
##### Build CVCUDA samples

To build CV-CUDA samples, additionally CV-CUDA has to be installed and CVCUDA_DIR and NVCV_TYPES_DIR
need to point folders with *-config.cmake files. Apart of that, BUILD_CVCUDA_SAMPLES variable must be set to ON.

#### Windows

Open Developer Command Prompt for VS 2022

```
git lfs clone https://github.com/NVIDIA/nvImageCodec.git
cd nvimagecodec
git submodule update --init --recursive --depth 1
.\externa\build_deps.bat
.\docker\build_helper.bat .\build 12
```

## Build Python wheel

After succesfully built project, execute below commands.

```
cd build
cmake --build . --target wheel
```

## Packaging

From a successfully built project, installers can be generated using cpack:
```
cd build
cpack --config CPackConfig.cmake -DCMAKE_BUILD_TYPE=Release
```
This will generate in build directory *.zip or *tar.xz files


## Installation from locally built packages

#### Tar file installation

```
tar -xvf nvimgcodec-0.3.0.0-cuda12-x86_64-linux-lib.tar.gz -C /opt/nvidia/
```

#### DEB File Installation
```
sudo apt-get install -y ./nvimgcodec-0.3.0.0-cuda12-x86_64-linux-lib.deb
```
#### Python WHL File Installation

```
pip install nvidia_nvimgcodec_cu12-0.3.0-py3-none-manylinux2014_x86_64.whl
```

### Installation from sources

##### Linux
```
cd build
cmake --install . --config Release --prefix /opt/nvidia/nvimgcodec_<major_cuda_ver>
```

After execution there should be:
- all extension modules in /opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/extensions (it is default directory for extension discovery)
- libnvimgcodec.so in /opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/lib64

Add directory with libnvimgcodec.so to LD_LIBRARY_PATH

```
export LD_LIBRARY_PATH=/opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/lib64:$LD_LIBRARY_PATH
```

##### Windows

Open Developer Command Prompt for VS 2022

```
cd build
cmake --install . --config Release --prefix "c:\Program Files\nvimgcodec_cuda<major_cuda_ver>"
```

After execution there should be:

- all extension modules in c:\Program Files\nvimgcodec_cuda<major_cuda_ver>/extensions (it is default directory for extension discovery)
- nvimgcodec_0.dll in c:\Program Files\nvimgcodec_cuda<major_cuda_ver>\bin

Add directory with nvimgcodec_0.dll to PATH

## Testing
Run CTest to execute L0 and L1 tests
```
cd build
cmake --install . --config Release --prefix bin
ctest -C Release
```

Run sample transcoder app tests
```
cd build
cmake --install . --config Release --prefix bin
cd bin/test

LD_LIBRARY_PATH=$PWD/../lib64 pytest -v test_transcode.py

```

Run Python API tests

First install python wheel. You would also need to have installed all Python tests dependencies (see Dockerfiles). 

```
pip install nvidia_nvimgcodec_cu12-0.3.0.x-py3-none-manylinux2014_x86_64.whl
```

Run tests
```
cd tests
pytest -v ./python
```

## CMake package integration

To use nvimagecodec as a dependency in your CMake project, use:
```
list(APPEND CMAKE_PREFIX_PATH "/opt/nvidia/nvimgcodec_cuda<major_cuda_ver>/")  # or the prefix where the package was installed if custom

find_package(nvimgcodec CONFIG REQUIRED)
# Mostly for showing some of the variables defined
message(STATUS "nvimgcodec_FOUND=${nvimgcodec_FOUND}")
message(STATUS "nvimgcodec_INCLUDE_DIR=${nvimgcodec_INCLUDE_DIR}")
message(STATUS "nvimgcodec_LIB_DIR=${nvimgcodec_LIB_DIR}")
message(STATUS "nvimgcodec_BIN_DIR=${nvimgcodec_BIN_DIR}")
message(STATUS "nvimgcodec_LIB=${nvimgcodec_LIB}")
message(STATUS "nvimgcodec_EXTENSIONS_DIR=${nvimgcodec_EXTENSIONS_DIR}")
message(STATUS "nvimgcodec_VERSION=${nvimgcodec_VERSION}")

target_include_directories(<your-target> PUBLIC ${nvimgcodec_INCLUDE_DIR})
target_link_directories(<your-target> PUBLIC ${nvimgcodec_LIB_DIR})
target_link_libraries(<your-target> PUBLIC ${nvimgcodec_LIB})
```

