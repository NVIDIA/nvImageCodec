export DOCKER_BUILDKIT=${DOCKER_BUILDKIT:-1}
export REGISTRY_PREFIX=${REGISTRY_PREFIX:-""}
export PLATFORM=${PLATFORM:-"linux/amd64"}  # or "linux/arm64"
export ARCH=${ARCH:-"x86_64"}  # or "aarch64"