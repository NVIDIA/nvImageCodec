# nvImageCodec Samples

## Description

These are some sample applications showcasing various nvImageCodec APIs. Sample applications are available in C++ and Python.

## Pre-requisites
- Recommended linux distros:
    - Ubuntu >= 20.04 (tested with 20.04 and 22.04)
    - WSL2 with Ubuntu >= 20.04 (tested with 20.04)
- NVIIDA driver: 
    - Linux: version 520.56.06 or higher
- NVIDIA CUDA Toolkit >= 11.8
- CMake >= 3.18
- gcc >= 9.4
- Python Packages:
    - jupyter >= 1.0.0 
    - matplotlib >= 3.5.2
    - numpy >= 1.23.1
    - cupy >= 11.2.0
    - cucim >= 22.6.0
    - cv-cuda == 0.3.0 Beta
    - opencv-python >= 4.6.0.66
    - torch == 1.13.0
    - torchvision == 0.14.0
    - torchnvjpeg (https://github.com/itsliupeng/torchnvjpeg)


## Build C++ samples

```
mkdir build
cd build
export CUDACXX=nvcc
cmake .. -DCMAKE_BUILD_TYPE=Release
make
```

### Build C++ CVCUDA samples

To build CV-CUDA samples, additionally CV-CUDA has to be installed and CVCUDA_DIR and NVCV_TYPES_DIR
need to point folders with *-config.cmake files. Apart of that, BUILD_CVCUDA_SAMPLES variable must be set to ON.

## Open Python Jupyter notebooks samples

Change directory to nvimgcodec/samples/python and start the notebook server from the command line: 

```
cd samples/python
jupyter notebook
```

This will print some information about the notebook server in your terminal, including the URL of the web application (by default, http://localhost:8888):

```
...
[I 2023-07-28 11:06:53.944 ServerApp] Jupyter Server 2.7.0 is running at:
[I 2023-07-28 11:06:53.944 ServerApp] http://localhost:8888/tree?token=599f185ea12dce78b606ab103ad82510c749ca4c551e0713
[I 2023-07-28 11:06:53.944 ServerApp]     http://127.0.0.1:8888/tree?token=599f185ea12dce78b606ab103ad82510c749ca4c551e0713
[I 2023-07-28 11:06:53.944 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
...

```

Open your web browser to this given URL.

When the notebook opens in your browser, you will see the Notebook Dashboard, which will show a list of the notebooks with samples. 