ARG AARCH64_BASE_IMAGE=nvidia/cuda:12.2.0-devel-ubuntu20.04
FROM ${AARCH64_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive

ARG CUDA_CROSS_VERSION=12-2
ARG CUDA_CROSS_VERSION_DOT=12.2

ENV CUDA_CROSS_VERSION=${CUDA_CROSS_VERSION}
ENV CUDA_CROSS_VERSION_DOT=${CUDA_CROSS_VERSION_DOT}

RUN rm /etc/apt/sources.list.d/cuda.list && \
    apt-key del 7fa2af80 && \
    apt-get update && apt-get install -y --no-install-recommends wget && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-keyring_1.0-1_all.deb && \
    dpkg -i cuda-keyring_1.0-1_all.deb && \
    apt-get update && apt-get install software-properties-common -y --no-install-recommends && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt-get update && apt-get install -y --no-install-recommends \
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
    python3-distutils \
    autogen \
    zip \
    python3.8 python3.8-dev \
    python3.9 python3.9-dev python3.9-distutils \
    python3.10 python3.10-dev python3.10-distutils \
    python3.11 python3.11-dev python3.11-distutils \
    python3.12 python3.12-dev python3.12-distutils && \
    apt-key adv --fetch-key http://repo.download.nvidia.com/jetson/jetson-ota-public.asc && \
    add-apt-repository 'deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/cross-linux-aarch64/ /' && \
    apt-get update && \
    apt-get install -y cuda-cudart-cross-aarch64-${CUDA_CROSS_VERSION} \
                       cuda-driver-cross-aarch64-${CUDA_CROSS_VERSION} \
                       cuda-cccl-cross-aarch64-${CUDA_CROSS_VERSION} \
                       cuda-nvcc-cross-aarch64-${CUDA_CROSS_VERSION} \
                       libnpp-cross-aarch64-${CUDA_CROSS_VERSION} \
                       libnvjpeg-cross-aarch64-${CUDA_CROSS_VERSION} \
    && \
    rm -rf /var/lib/apt/lists/* && \
    PYTHON_VER=$(python3 -c "import sys;print(f'{sys.version_info[0]}{sys.version_info[1]}')") && \
    curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py && \
    # decouple libclang and clang installation so libclang changes are not overriden by clang
    pip install clang==14.0 && pip install libclang==14.0.1 flake8 && \
    rm -rf /root/.cache/pip/ && \
    cd /tmp && git clone https://github.com/NixOS/patchelf && cd patchelf && \
    ./bootstrap.sh && ./configure --prefix=/usr/ && make -j install && cd / && rm -rf /tmp/patchelf && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    ln -s /usr/bin/python3-config /usr/bin/python-config && \
    PYTHON_V=$(python3 -c "import sys;print(f'{sys.version_info[0]}.{sys.version_info[1]}')") && \
    ln -s /usr/bin/python${PYTHON_V}-config /usr/bin/python3-config

ENV PKG_CONFIG_PATH=/usr/aarch64-linux-gnu/lib/pkgconfig

RUN export CMAKE_VERSION=3.24.3 \
      && wget https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-linux-x86_64.sh \
        -q -O /tmp/cmake-install.sh \
      && chmod u+x /tmp/cmake-install.sh \
      && mkdir /opt/cmake-${CMAKE_VERSION} \
      && /tmp/cmake-install.sh --skip-license --prefix=/opt/cmake-${CMAKE_VERSION} \
      && rm -f /tmp/cmake-install.sh \
      && rm -f /usr/local/bin/*cmake* \
      && rm -f /usr/local/bin/cpack \
      && rm -f /usr/local/bin/ctest \
      && ln -s /opt/cmake-${CMAKE_VERSION}/bin/* /usr/local/bin

# run in /bin/bash to have more advanced features supported like list
ADD external /opt/src/external
RUN /bin/bash -c '\
    export CC_COMP=aarch64-linux-gnu-gcc                                                           && \
    export CXX_COMP=aarch64-linux-gnu-g++                                                          && \
    export INSTALL_PREFIX="/usr/aarch64-linux-gnu/"                                                && \
    export HOST_ARCH_OPTION="--host=aarch64-unknown-linux-gnu"                                     && \
    export CMAKE_TARGET_ARCH=aarch64                                                               && \
    export OPENCV_TOOLCHAIN_FILE="linux/aarch64-gnu.toolchain.cmake"                               && \
    pushd /opt/src && external/build_deps.sh && popd'

# hack - install cross headers in the default python paths, so host python3-config would point to them
RUN export PYVERS="3.8.5 3.9.0 3.10.0 3.11.0 3.12.0" && \
    for PYVER in ${PYVERS}; do \
        cd /tmp && (curl -L https://www.python.org/ftp/python/${PYVER}/Python-${PYVER}.tgz | tar -xzf - || exit 1) && \
        rm -rf *.tgz && cd Python*                                                                     && \
        ./configure --disable-ipv6 ac_cv_file__dev_ptmx=no ac_cv_file__dev_ptc=no                         \
            --disable-shared CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++                           \
            --build=x86_64-pc-linux-gnu --host=aarch64-linux-gnu --prefix=/usr/                           \
            --with-build-python=python${PYVER%.*}                                                      && \
        make -j"$(grep ^processor /proc/cpuinfo | wc -l)" inclinstall                                  && \
        cd / && rm -rf /tmp/Python*;                                                                      \
    done                                                                                               && \
    # hack - patch the host pythonX-config to return --extension-suffix for the target
    find /usr/ -iname x86_64-linux-gnu-python* -exec sed -i "s/\(SO.*\)\(x86_64\)\(.*\)/\1aarch64\3/" {} \;

VOLUME /opt/nvimagecodec

WORKDIR /opt/nvimagecodec

ENV PATH=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}/bin:$PATH

ARG BUILD_DIR=build_aarch64_linux

WORKDIR /opt/nvimagecodec/${BUILD_DIR}

# CUDA_TARGET_ARCHS
# 72 : Volta  - gv11b/Tegra (Jetson AGX Xavier)
# 87 : Ampere - ga10b,ga10c/Tegra (Jetson AGX Orin)

ARG NVIDIA_BUILD_ID=0
ENV NVIDIA_BUILD_ID=${NVIDIA_BUILD_ID}

ENV ARCH=aarch64-linux
ENV TEST_BUNDLED_LIBS=NO
ENV WHL_PLATFORM_NAME=manylinux2014_aarch64
ENV BUNDLE_PATH_PREFIX="/usr/aarch64-linux-gnu"
ENV EXTRA_CMAKE_OPTIONS=" \
-DNVIMGCODEC_FLAVOR=tegra                                                      \
-DCMAKE_TOOLCHAIN_FILE:STRING=../cmake/aarch64-linux-tegra.toolchain.cmake     \
-DCMAKE_COLOR_MAKEFILE=ON                                                      \
-DCMAKE_CUDA_COMPILER=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}/bin/nvcc       \
-DCUDA_HOST=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}                          \
-DCUDA_TARGET=/usr/local/cuda-${CUDA_CROSS_VERSION_DOT}/targets/aarch64-linux  \
-DCMAKE_PREFIX_PATH=/usr/aarch64-linux-gnu/                                    \
-DCUDA_TARGET_ARCHS:STRING=72;87"

CMD mkdir -p /opt/nvimagecodec/${BUILD_DIR} && \
    source /opt/nvimagecodec/docker/build_helper.sh && \
    rm -rf nvidia* && \
    cp -r /wheelhouse /opt/nvimagecodec/
