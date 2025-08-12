#!/bin/bash -ex

SCRIPT_DIR=$(dirname $0)
source ${SCRIPT_DIR}/config-docker.sh || source ${SCRIPT_DIR}/default-config-docker.sh

BUILDER_CUDA_VERSION="11.8"
BUILDER_ARCH="x86_64"
BUILDER_IMAGE="${REGISTRY_PREFIX}builder-cuda-${BUILDER_CUDA_VERSION}-${BUILDER_ARCH}:vrelease_0.6_2"
# Note: Use build_dockers.sh to produce the image if needed

docker run --rm -it -v ${PWD}:/opt/src ${BUILDER_IMAGE} /bin/bash -c \
    'WHL_OUTDIR=/opt/src/artifacts && mkdir -p /opt/src/build_docker && cd /opt/src/build_docker && source ../docker/build_helper.sh'
