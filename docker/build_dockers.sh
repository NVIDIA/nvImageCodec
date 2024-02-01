#!/bin/bash -ex

export VERSION=${VERSION:-9}  # Update version when changing anything in the Dockerfiles

SCRIPT_DIR=$(dirname $0)
source ${SCRIPT_DIR}/config-docker.sh || source ${SCRIPT_DIR}/default-config-docker.sh

docker buildx create --name nvimagecodec_builder || echo "nvimagecodec_build already created"
docker buildx use nvimagecodec_builder
docker buildx inspect --bootstrap

####### BASE IMAGES #######

# Manylinux2014 with GCC 9
export MANYLINUX_GCC9="${REGISTRY_PREFIX}manylinux2014_${ARCH}.gcc9"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${MANYLINUX_GCC9} \
    -t ${MANYLINUX_GCC9} -t ${MANYLINUX_GCC9}:v${VERSION} \
    -f docker/Dockerfile.gcc9 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    .

# Manylinux2014 with GCC 10
export MANYLINUX_GCC10="${REGISTRY_PREFIX}manylinux2014_${ARCH}.gcc10"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${MANYLINUX_GCC10} \
    -t ${MANYLINUX_GCC10} -t ${MANYLINUX_GCC10}:v${VERSION} \
    -f docker/Dockerfile.gcc10 \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux2014_${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 11.8.0
export CUDA_118="${REGISTRY_PREFIX}cuda11.8-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${CUDA_118} \
    -t ${CUDA_118} -t ${CUDA_118}:v${VERSION} \
    -f docker/Dockerfile.cuda118.${ARCH}.deps \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 12.3.0
export CUDA_123="${REGISTRY_PREFIX}cuda12.3-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${CUDA_123} \
    -t ${CUDA_123} -t ${CUDA_123}:v${VERSION} \
    -f docker/Dockerfile.cuda123.${ARCH}.deps \
    --platform ${PLATFORM} \
    --push \
    .

# GCC 9 (minimum supported)
export DEPS_GCC9="${REGISTRY_PREFIX}nvimgcodec_deps-${ARCH}-gcc9"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${DEPS_GCC9} \
    -t ${DEPS_GCC9} -t ${DEPS_GCC9}:v${VERSION} \
    -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC9}" \
    --build-arg "ARCH=${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    .

# GCC 10
export DEPS_GCC10="${REGISTRY_PREFIX}nvimgcodec_deps-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${DEPS_GCC10} \
    -t ${DEPS_GCC10} -t ${DEPS_GCC10}:v${VERSION} \
    -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC10}" \
    --build-arg "ARCH=${ARCH}" \
    --platform ${PLATFORM} \
    --push \
    .

####### BUILDER IMAGES #######

# GCC 9 (minimum supported), CUDA 11.8
export BUILDER_GCC9_CUDA_118="${REGISTRY_PREFIX}builder-cuda-11.8-gcc9-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${BUILDER_GCC9_CUDA_118} \
    -t ${BUILDER_GCC9_CUDA_118} -t ${BUILDER_GCC9_CUDA_118}:v${VERSION} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC9}" \
    --build-arg "CUDA_IMAGE=${CUDA_118}" \
    --platform ${PLATFORM} \
    --push \
    .

# GCC 10, CUDA 11.8
export BUILDER_CUDA_118="${REGISTRY_PREFIX}builder-cuda-11.8-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${BUILDER_CUDA_118} \
    -t ${BUILDER_CUDA_118} -t ${BUILDER_CUDA_118}:v${VERSION} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC10}" \
    --build-arg "CUDA_IMAGE=${CUDA_118}" \
    --platform ${PLATFORM} \
    --push \
    .

# GCC 10, CUDA 12.3
export BUILDER_CUDA_123="${REGISTRY_PREFIX}builder-cuda-12.3-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${BUILDER_CUDA_123} \
    -t ${BUILDER_CUDA_123} -t ${BUILDER_CUDA_123}:v${VERSION} \
    -f docker/Dockerfile.cuda.deps \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC10}" \
    --build-arg "CUDA_IMAGE=${CUDA_123}" \
    --platform ${PLATFORM} \
    --push \
    .

####### TEST IMAGES #######

# CUDA 11.3 (minimum supported)
export RUNNER_CUDA_113="${REGISTRY_PREFIX}runner-cuda-11.3-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${RUNNER_CUDA_113} \
    -t ${RUNNER_CUDA_113} -t ${RUNNER_CUDA_113}:v${VERSION} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:11.3.1-runtime-ubuntu20.04" \
    --build-arg "VER_CUDA=11.3.1" \
    --build-arg "VER_UBUNTU=20.04" \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 11.8
export RUNNER_CUDA_118="${REGISTRY_PREFIX}runner-cuda-11.8-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${RUNNER_CUDA_118} \
    -t ${RUNNER_CUDA_118} -t ${RUNNER_CUDA_118}:v${VERSION} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:11.8.0-runtime-ubuntu20.04" \
    --build-arg "VER_CUDA=11.8.0" \
    --build-arg "VER_UBUNTU=20.04" \
    --platform ${PLATFORM} \
    --push \
    .

# CUDA 12.1
export RUNNER_CUDA_121="${REGISTRY_PREFIX}runner-cuda-12.1-${ARCH}"
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${RUNNER_CUDA_121} \
    -t ${RUNNER_CUDA_121} -t ${RUNNER_CUDA_121}:v${VERSION} \
    -f docker/Dockerfile.${ARCH} \
    --build-arg "BASE=nvidia/cuda:12.1.1-runtime-ubuntu20.04" \
    --build-arg "VER_CUDA=12.1.1" \
    --build-arg "VER_UBUNTU=20.04" \
    --platform ${PLATFORM} \
    --push \
    .
