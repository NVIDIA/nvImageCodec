export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
export REGISTRY_PREFIX=${REGISTRY_PREFIX:-""}
export PLATFORM=${PLATFORM:-"linux/amd64"}  # or "linux/arm64"
export ARCH=${ARCH:-"x86_64"}  # or "aarch64"
export PYVER=${PYVER:-"py39"}
export MANYLINUX_IMAGE_TAG="2025.02.12-1"
