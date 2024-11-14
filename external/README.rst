nvImageCodec dependencies
=========================
This directory contains links to extra dependencies required to setup a whole development
environment for the nvImageCodec project.

To obtain only the required code for he build, without unnecessary git history, please do:

.. code-block:: bash

  git submodule init
  git submodule update --depth 1 --recursive

Consists of externally hosted subrepositories:

+-----------------+----------------------+---------------------+
| Repository      | Version              | License             |
+=================+======================+=====================+
| nvtx_           | |nvtxver|_           | |nvtxlic|_          |
+-----------------+----------------------+---------------------+
| preprocessor_   | |preprocessorver|_   | |preprocessorlic|_  |
+-----------------+----------------------+---------------------+
| dlpack_         | |dlpackver|_         | |dlpacklic|_        |
+-----------------+----------------------+---------------------+
| googletest_     | |googletestver|_     | |googletestlic|_    |
+-----------------+----------------------+---------------------+
| pybind11_       | |pybind11ver|_       | |pybind11lic|_      |
+-----------------+----------------------+---------------------+
| opencv_         | |opencvver|_         | |opencvlic|_        |
+-----------------+----------------------+---------------------+
| openjpeg_       | |openjpegver|_       | |openjpeglic|_      |
+-----------------+----------------------+---------------------+
| libtiff_        | |libtiffver|_        | |libtifflic|_       |
+-----------------+----------------------+---------------------+
| zstd_           | |zstdver|_           | |zstdlic|_          |
+-----------------+----------------------+---------------------+
| libjpeg-turbo_  | |libjpeg-turbover|_  | |libjpeg-turbolic|_ |
+-----------------+----------------------+---------------------+
| zlib_           | |zlibver|_           | |zliblic|_          |
+-----------------+----------------------+---------------------+

.. |nvtx| replace:: NVTX
.. _nvtx: https://github.com/NVIDIA/NVTX
.. |nvtxver| replace:: 3.1.0
.. _nvtxver: https://github.com/NVIDIA/NVTX/releases/tag/v3.1.0
.. |nvtxlic| replace:: Apache License 2.0
.. _nvtxlic: https://github.com/NVIDIA/NVTX/blob/release-v3/LICENSE.txt

.. |pybind11| replace:: pybind11
.. _pybind11: https://github.com/pybind/pybind11
.. |pybind11ver| replace:: 2.13.6
.. _pybind11ver: https://github.com/pybind/pybind11/releases/tag/v2.13.6
.. |pybind11lic| replace:: BSD 3-Clause License
.. _pybind11lic: https://github.com/pybind/pybind11/blob/master/LICENSE

.. |googletest| replace:: GoogleTest
.. _googletest: https://github.com/google/googletest
.. |googletestver| replace:: 1.14.0
.. _googletestver: https://github.com/google/googletest/releases/tag/v1.14.0
.. |googletestlic| replace:: BSD 3-Clause License
.. _googletestlic: https://github.com/google/googletest/blob/master/LICENSE

.. |dlpack| replace:: DLPack
.. _dlpack: https://github.com/dmlc/dlpack
.. |dlpackver| replace:: 0.8
.. _dlpackver: https://github.com/dmlc/dlpack/releases/tag/v0.8
.. |dlpacklic| replace:: Apache License 2.0
.. _dlpacklic: https://github.com/dmlc/dlpack/blob/main/LICENSE

.. |preprocessor| replace:: Boost Preprocessor
.. _preprocessor: https://github.com/boostorg/preprocessor
.. |preprocessorver| replace:: 1.85.0
.. _preprocessorver: https://github.com/boostorg/preprocessor/releases/tag/boost-1.85.0
.. |preprocessorlic| replace:: Boost Software License 1.0
.. _preprocessorlic: https://github.com/boostorg/boost/blob/master/LICENSE_1_0.txt

.. _opencv: https://github.com/opencv/opencv/
.. |opencvlic| replace:: Apache License 2.0
.. _opencvlic: https://github.com/opencv/opencv/blob/master/LICENSE
.. |opencvver| replace:: 4.10.0
.. _opencvver: https://github.com/opencv/opencv/releases/tag/4.10.0

.. _openjpeg: https://github.com/uclouvain/openjpeg
.. |openjpeglic| replace:: BSD-2 license
.. _openjpeglic: https://github.com/uclouvain/openjpeg/blob/master/LICENSE
.. |openjpegver| replace:: 2.5.2
.. _openjpegver: https://github.com/uclouvain/openjpeg/releases/tag/v2.5.2

.. _libtiff: https://gitlab.com/libtiff/libtiff
.. |libtifflic| replace:: BSD-2 license
.. _libtifflic: https://gitlab.com/libtiff/libtiff/-/blob/master/README.md
.. |libtiffver| replace:: 4.6.0 (+ Build System Patch)
.. _libtiffver: https://gitlab.com/libtiff/libtiff/-/tree/v4.6.0

.. _zstd: https://github.com/facebook/zstd
.. |zstdlic| replace:: BSD-3 license
.. _zstdlic: https://github.com/facebook/zstd/blob/dev/LICENSE
.. |zstdver| replace:: 1.5.6
.. _zstdver: https://github.com/facebook/zstd/releases/tag/v1.5.6

.. _libjpeg-turbo: https://github.com/libjpeg-turbo/libjpeg-turbo/
.. |libjpeg-turbolic| replace:: BSD-3 license, IJG license, zlib license
.. _libjpeg-turbolic: https://github.com/libjpeg-turbo/libjpeg-turbo/blob/master/LICENSE.md
.. |libjpeg-turbover| replace:: 3.0.3
.. _libjpeg-turbover: https://github.com/libjpeg-turbo/libjpeg-turbo/releases/tag/3.0.3

.. _zlib: https://github.com/madler/zlib
.. |zliblic| replace:: zlib License
.. _zliblic: https://github.com/madler/zlib/blob/master/README
.. |zlibver| replace:: 1.3.1
.. _zlibver: https://github.com/madler/zlib/releases/tag/v1.3.1

Installing dependencies locally
===============================

You can install the libraries that need to be installed as a prerequisite by running the repository ``build_deps.sh``
