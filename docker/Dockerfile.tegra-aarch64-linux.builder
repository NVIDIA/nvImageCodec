ARG AARCH64_BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04
FROM ${AARCH64_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ARG CUDA_CROSS_VERSION=12-9
ARG CUDA_CROSS_VERSION_DOT=12.9
ARG CUDA_TARGET_ARCHS="72;87"
# 72 : Volta  - gv11b/Tegra (Jetson AGX Xavier)
# 87 : Ampere - ga10b,ga10c/Tegra (Jetson AGX Orin)

# Host Python that runs the stub codegen and the wheel build. The wheel itself
# is still produced for every CPython ABI listed below; this only pins which
# interpreter drives the build (libclang/setuptools/wheel are installed here).
ARG HOST_PYTHON_VERSION=3.12

ENV CUDA_CROSS_VERSION=${CUDA_CROSS_VERSION}
ENV CUDA_CROSS_VERSION_DOT=${CUDA_CROSS_VERSION_DOT}
ENV CUDA_TARGET_ARCHS=${CUDA_TARGET_ARCHS}
ENV HOST_PYTHON_VERSION=${HOST_PYTHON_VERSION}

RUN apt-get update && \
    apt install -y --no-install-recommends \
        wget \
        software-properties-common \
        ca-certificates \
        curl \
        unzip \
        git \
        rsync \
        libjpeg-dev \
        dh-autoreconf \
        gcc-aarch64-linux-gnu \
        g++-aarch64-linux-gnu \
        pkg-config \
        libtool \
        libtool-bin \
        python3 \
        autogen \
        zip \
    && \
    rm -rf /var/lib/apt/lists/*

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    rm cuda-keyring_1.1-1_all.deb && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/cross-linux-aarch64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    rm cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        cuda-nvcc-${CUDA_CROSS_VERSION} \
        cuda-nvcc-cross-aarch64-${CUDA_CROSS_VERSION} \
        cuda-cudart-cross-aarch64-${CUDA_CROSS_VERSION} \
        cuda-driver-cross-aarch64-${CUDA_CROSS_VERSION} \
        cuda-cccl-cross-aarch64-${CUDA_CROSS_VERSION} \
    && \
    rm -rf /var/lib/apt/lists/*

# This solves a conflict with two packages (cudart and culibos) providing 
# /usr/local/cuda-13.0/targets/aarch64-linux/lib/libculibos.a library.
# The culibos package is a dependency of npp and nvjpeg packages, so we are avoiding
# its installation by manually unpacking the packages and copying the files to the target location.
RUN mkdir /tmp/debs && cd /tmp/debs && \
        apt-get update && apt download \
        libnpp-cross-aarch64-${CUDA_CROSS_VERSION} \
        libnvjpeg-cross-aarch64-${CUDA_CROSS_VERSION} && \
        dpkg-deb -x libnpp*.deb . && \
        dpkg-deb -x libnvjpeg*.deb . && \
        cp -rf usr/* /usr/ && \
        cd / && \
        rm -rf /tmp/debs && \
        rm -rf /var/lib/apt/lists/*

# The six from-source dependencies are built BEFORE the Python enumeration below,
# so adding or dropping an interpreter no longer invalidates the cross-compiled
# zlib/libjpeg-turbo/zstd/openjpeg/libtiff/OpenCV build -- only the CPython
# cross-header loop re-runs. ENV PKG_CONFIG_PATH must stay at the head of this
# block: ENV applies only to later instructions, and leaving it behind would build
# libtiff and OpenCV with different WITH_TIFF/WITH_WEBP detection and no error.
ENV PKG_CONFIG_PATH=/usr/aarch64-linux-gnu/lib/pkgconfig

RUN export CMAKE_VERSION=3.24.3 \
      && curl -fsSL --retry 5 --retry-delay 15 --connect-timeout 30 --speed-limit 1024 --speed-time 60 \
        -o /tmp/cmake-install.sh https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-${CMAKE_VERSION} \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-${CMAKE_VERSION} \
      && rm -f /tmp/cmake-install.sh \
      && rm -f /usr/local/bin/*cmake* \
      && rm -f /usr/local/bin/cpack \
      && rm -f /usr/local/bin/ctest \
      && ln -s /opt/cmake-${CMAKE_VERSION}/bin/* /usr/local/bin

# run in /bin/bash to have more advanced features supported like list
# Only the files external/build_deps.sh actually reads. Copying all of external/
# put ~24 entries (boost, dlpack, googletest, nvjpeg2k, nvtiff, NVTX, pybind11, ...)
# into the cache key of the most expensive layer in the tree -- the six from-source
# builds ending in OpenCV -- so a submodule bump rebuilt all of them for nothing.
# Measured: 29 commits touched external/ in 18 months; 11 touched a real input.
# A seventh package added to build_deps.sh without a matching line here fails loudly
# at build time with a missing-file error, which is the right failure mode.
COPY external/build_deps.sh \
     external/build_zlib.sh \
     external/build_libjpeg-turbo.sh \
     external/build_zstd.sh \
     external/build_openjpeg.sh \
     external/build_libtiff.sh \
     external/build_opencv.sh \
     /opt/src/external/
COPY external/patches/ /opt/src/external/patches/
RUN /bin/bash -c '\
    export CC_COMP=aarch64-linux-gnu-gcc                                                           && \
    export CXX_COMP=aarch64-linux-gnu-g++                                                          && \
    export INSTALL_PREFIX="/usr/aarch64-linux-gnu/"                                                && \
    export HOST_ARCH_OPTION="--host=aarch64-unknown-linux-gnu"                                     && \
    export CMAKE_TARGET_ARCH=aarch64                                                               && \
    export OPENCV_TOOLCHAIN_FILE="linux/aarch64-gnu.toolchain.cmake"                               && \
    pushd /opt/src && external/build_deps.sh && popd'

RUN apt-get update && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    apt-get install -y \
        python3.9 python3.9-dev \
        python3.10 python3.10-dev \
        python3.11 python3.11-dev \
        python3.12 python3.12-dev \
        python3.13 python3.13-dev \
        python3.14 python3.14-dev \
        python3.15 python3.15-dev \
    && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/python3-config /usr/bin/python-config && \
    PYTHON_V=$(python3 -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')") && \
    ln -s /usr/bin/python${PYTHON_V}-config /usr/bin/python3-config

# Bootstrap pip for the host Python (HOST_PYTHON_VERSION) and install the
# packages used by the build: libclang for stub_codegen.py, plus setuptools/wheel
# for `setup.py bdist_wheel`. clang and libclang are installed separately so a
# clang upgrade doesn't downgrade libclang. Only the host Python needs these —
# per-ABI cross headers for the wheels are installed in the next RUN.
RUN curl -fsSL https://bootstrap.pypa.io/get-pip.py | python${HOST_PYTHON_VERSION} - --quiet && \
    python${HOST_PYTHON_VERSION} -m pip install --no-cache-dir clang==14.0 && \
    python${HOST_PYTHON_VERSION} -m pip install --no-cache-dir libclang==14.0.1 setuptools wheel && \
    rm -rf /root/.cache/pip/ && \
    cd /tmp && git clone https://github.com/NixOS/patchelf && cd patchelf && \
    ./bootstrap.sh && ./configure --prefix=/usr/ && make -j install && cd / && rm -rf /tmp/patchelf

# hack - install cross headers in the default python paths, so host python3-config would point to them
RUN export PYVERS="3.9.0 3.10.0 3.11.0 3.12.0 3.13.0 3.14.0 3.15.0" && \
    for PYVER in ${PYVERS}; do \
        cd /tmp && (curl -L https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tgz | tar -xzf -) && \
        rm -rf *.tgz && cd Python*                                                                     && \
        ./configure --disable-ipv6 ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no                         \
            --disable-shared CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++                           \
            --build=x86_64-pc-linux-gnu --host=aarch64-linux-gnu --prefix=/usr/                           \
            --with-build-python=python${PYVER%.*}                                                      && \
        make -j"$(grep ^processor /proc/cpuinfo | wc -l)" inclinstall                                  && \
        cd / && rm -rf /tmp/Python*;                                                                      \
        # exit for loop on first error so that whole build fails.
        # otherwise for loop will return status of last iteration, which may succeed, even if previous iteration failed
        if [ $? -ne 0 ]; then                                                                             \
            echo "failed installing cross headers for ${PYVER}";                                          \
            exit 1;                                                                                       \
        fi                                                                                                \
    done                                                                                               && \
    # hack - patch the host pythonX-config to return --extension-suffix for the target
    find /usr/ -iname x86_64-linux-gnu-python* -exec sed -i "s/\(SO.*\)\(x86_64\)\(.*\)/\1aarch64\3/" {} \;


VOLUME /opt/nvimagecodec
WORKDIR /opt/nvimagecodec
ENV PATH=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}/bin:$PATH
ARG BUILD_DIR=build_aarch64_linux
WORKDIR /opt/nvimagecodec/${BUILD_DIR}

ARG NVIDIA_BUILD_ID=0
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}

ENV ARCH=aarch64-linux
ENV TEST_BUNDLED_LIBS=NO
ENV WHL_PLATFORM_NAME=manylinux_2_28_aarch64
ENV BUNDLE_PATH_PREFIX="/usr/aarch64-linux-gnu"
ENV EXTRA_CMAKE_OPTIONS=" \
-DNVIMGCODEC_FLAVOR=tegra                                                      \
-DCMAKE_TOOLCHAIN_FILE:STRING=../cmake/aarch64-linux-tegra.toolchain.cmake     \
-DCMAKE_COLOR_MAKEFILE=ON                                                      \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}/bin/nvcc       \
-DCUDA_HOST=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}                          \
-DCUDA_TARGET=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}/targets/aarch64-linux  \
-DCMAKE_PREFIX_PATH=/usr/aarch64-linux-gnu/"

CMD mkdir -p /opt/nvimagecodec/${BUILD_DIR} && \
    source /opt/nvimagecodec/docker/build_helper.sh && \
    rm -rf nvidia* && \
    cp -r /wheelhouse /opt/nvimagecodec/
