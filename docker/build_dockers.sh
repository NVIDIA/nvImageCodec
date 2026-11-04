#!/bin/bash -ex

export VERSION=${VERSION:-release_0.10.0}  # Update version when changing anything in the Dockerfiles
export TEGRA_VERSION=${TEGRA_VERSION:-29} # Update version when changing anything in the Dockerfile.tegra-aarch64-linux.builder
export CONDA_TEST_VERSION=${CONDA_TEST_VERSION:-2} # Update version when changing anything in the Dockerfile.conda-test

SCRIPT_DIR=$(dirname $0)
source ${SCRIPT_DIR}/config-docker.sh || source ${SCRIPT_DIR}/default-config-docker.sh
source ${SCRIPT_DIR}/cuda_versions.env
source ${SCRIPT_DIR}/nvidia_libs.env
# Optional. Absent means "official redistributables only", which is the supported
# default -- do not make this a hard source.
if [[ -f ${SCRIPT_DIR}/custom_libs.env ]]; then source ${SCRIPT_DIR}/custom_libs.env; fi

# NVIDIA artifact naming is not the same string as ARCH.
if [[ "${ARCH}" == "aarch64" ]]; then ARTIFACT_ARCH="sbsa"; else ARTIFACT_ARCH="${ARCH}"; fi

# Expand one *_CUSTOM_URL template for this build's arch and CUDA version.
# Empty in, empty out -- and empty out is precisely the signal Dockerfile.runner reads
# as "fetch the official redistributable instead", so an absent or blank
# custom_libs.env needs no other change anywhere to produce a working build.
resolve_custom_url() {
    local template="$1" cuda_tag="$2"
    if [[ -z "${template}" ]]; then return 0; fi
    template="${template//@ARCH@/${ARTIFACT_ARCH}}"
    template="${template//@CUDA_TAG@/${cuda_tag}}"
    echo "${template}"
}

# The whole nvidia_libs.env / custom_libs.env contribution for one runner image, as
# build args, so each runner block below stays a single docker buildx invocation.
nvidia_lib_args() {
    local cuda_major="$1"
    local tag_var="NVTIFF_CUSTOM_CUDA_TAG_${cuda_major}"
    local cuda_tag="${!tag_var:-}"
    echo "--build-arg NVJPEG2K_VERSION=${NVJPEG2K_VERSION}"
    echo "--build-arg NVTIFF_VERSION=${NVTIFF_VERSION}"
    echo "--build-arg NVCOMP_VERSION=${NVCOMP_VERSION}"
    echo "--build-arg NVJPEG2K_CUSTOM_URL=$(resolve_custom_url "${NVJPEG2K_CUSTOM_URL:-}" "${cuda_tag}")"
    echo "--build-arg NVTIFF_CUSTOM_URL=$(resolve_custom_url "${NVTIFF_CUSTOM_URL:-}" "${cuda_tag}")"
    echo "--build-arg NVCOMP_CUSTOM_URL=$(resolve_custom_url "${NVCOMP_CUSTOM_URL:-}" "${cuda_tag}")"
}

# Pin the DALI nightly wheel that goes into the runner images. Left empty it floats, so
# two runner images with the same :vNN tag can differ. Set it to make a runner image
# reproducible, and bump it as a deliberate, reviewable act. The DALI sources and test
# data are not baked in -- bench.sh fetches them at test time against whatever wheel is
# installed -- so this is the only float.
DALI_VERSION=${DALI_VERSION:-}

# Cache export mode.
#
# `type=inline` is min-mode by definition: it can only carry layers that end up in
# the final image, so any stage consumed purely via `COPY --from=` is unexportable
# and re-runs on every builder with cold state. Nothing in these Dockerfiles is in that
# shape today, but it is the trap to remember before adding a build-only stage.
#
# `type=registry,mode=max` keeps every layer of every stage in a separate
# `<image>-buildcache` ref. It is opt-in until two things are settled: whether this
# registry accepts the BuildKit cache manifest (Docker-Distribution-based registries
# commonly reject the exporter's default OCI index media type, hence
# image-manifest=true,oci-mediatypes=true), and who owns retention on the resulting
# repositories -- mode=max across every image is a lot of storage, and storage is
# already the binding constraint on the build host.
#
# Pilot it on one image first:  REGISTRY_CACHE=1 ./docker/build_dockers.sh
cache_args() {
    local ref="$1"
    if [[ "${REGISTRY_CACHE:-0}" == "1" ]]; then
        echo "--cache-to type=registry,ref=${ref}-buildcache,mode=max,image-manifest=true,oci-mediatypes=true"
        echo "--cache-from type=registry,ref=${ref}-buildcache"
    else
        echo "--cache-to type=inline"
    fi
}


# Selectivity. Deliberately a filter around the existing per-image blocks rather than
# a table-driven matrix: tooling elsewhere in the tree matches this script's literal
# text (the netrc block, the TAB-indented --secret lines and `export VERSION=`) with
# unchecked regexes, so restructuring those parts fails silently rather than loudly.
# They are kept byte-identical for that reason.
#
#   --only <substr>   build only images whose ref contains <substr>
#   --cuda <ver>      build only CUDA images for that version, e.g. --cuda 13.4
#   --no-push         build locally (--load) instead of pushing
#   --dry-run         list what would be built, then exit
#
# There is deliberately no "build everything, then push at the end" barrier: the
# builder and runner images consume their parents by registry ref
# (--build-arg FROM_IMAGE_NAME=${DEPS_GCC14}:v${VERSION}), so deferring the pushes
# would break the dependency chain. `set -e` already stops at the first failure.
ONLY_FILTER=""
CUDA_FILTER=""
PUSH_ARG="--push"
DRY_RUN=0
POSITIONAL=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --only)     ONLY_FILTER="$2"; shift 2 ;;
        --cuda)     CUDA_FILTER="$2"; shift 2 ;;
        --no-push)  PUSH_ARG="--load"; shift ;;
        --dry-run)  DRY_RUN=1; shift ;;
        -h|--help)  sed -n '/^#   --only/,/^#   --dry-run/p' "$0"; exit 0 ;;
        *)          POSITIONAL+=("$1"); shift ;;
    esac
done
if [[ ${#POSITIONAL[@]} -gt 0 ]]; then set -- "${POSITIONAL[@]}"; else set --; fi

# Each build block is guarded by this.
should_build() {
    # Match on the image name only. REGISTRY_PREFIX contains "cuda"
    # (.../cuda-hpc-libraries/...), so matching the full ref makes --cuda exclude
    # every image, including the CUDA-agnostic ones.
    local name="${1#${REGISTRY_PREFIX}}"
    if [[ -n "${ONLY_FILTER}" && "${name}" != *"${ONLY_FILTER}"* ]]; then return 1; fi
    if [[ -n "${CUDA_FILTER}" && "${name}" == *cuda* && "${name}" != *"${CUDA_FILTER}"* ]]; then return 1; fi
    if [[ "${DRY_RUN}" == "1" ]]; then echo "would build: ${name}"; return 1; fi
    return 0
}


# --dry-run only lists what would be built (see should_build); it must not touch the
# daemon or a builder, or it stops being a safe planning command.
if [[ "${DRY_RUN}" != "1" ]]; then
    if [[ "${PUSH_ARG}" == "--load" ]]; then
        # nvimagecodec_builder's docker-container driver runs buildkitd in its own
        # container, isolated from the host engine's image store. A parent loaded by
        # --load lands only in the host store, so the next stage's
        # --build-arg FROM_IMAGE_NAME=<ref>:v${VERSION} would resolve against the
        # registry instead -- failing if that tag was never pushed, or silently
        # building on an older pushed parent otherwise. The default builder runs
        # against the host engine directly, so --load output is exactly what the next
        # stage's FROM lookup sees.
        docker buildx use default
    else
        # network=host: in CI the daemon is a docker-in-docker sidecar, and TLS handshakes
        # with github.com stall from behind its bridge while the same connections from
        # the pod's own network succeed. Sharing the daemon's network namespace takes the
        # bridge out of the path of RUN steps. A builder created earlier keeps its old
        # network mode; remove it with `docker buildx rm nvimagecodec_builder`.
        docker buildx create --name nvimagecodec_builder --driver-opt network=host || echo "nvimagecodec_build already created"
        docker buildx use nvimagecodec_builder
        docker buildx inspect --bootstrap
    fi
fi

####### BASE IMAGES #######

# Manylinux_2_28 with GCC 14
export MANYLINUX_GCC14="${REGISTRY_PREFIX}manylinux_2_28_${ARCH}.gcc14"
# No COPY/ADD from the context: pipe the Dockerfile in via stdin (context `-`)
# so nothing is uploaded to the daemon.
if should_build "${MANYLINUX_GCC14}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${MANYLINUX_GCC14} \
    -t ${MANYLINUX_GCC14} -t ${MANYLINUX_GCC14}:v${VERSION} \
    --build-arg "FROM_IMAGE_NAME=quay.io/pypa/manylinux_2_28_${ARCH}:${MANYLINUX_IMAGE_TAG}" \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    - < docker/Dockerfile.gcc14
fi

# CUDA 12.9.0
# Pick this version's descriptor for the current arch (docker/cuda_versions.env).
CUDA_129_RUN_URL_VAR="CUDA_129_RUN_URL_${ARCH}"; CUDA_129_RUN_URL_VALUE="${!CUDA_129_RUN_URL_VAR}"
CUDA_129_RUN_MD5_VAR="CUDA_129_RUN_MD5_${ARCH}"; CUDA_129_RUN_MD5_VALUE="${!CUDA_129_RUN_MD5_VAR}"
export CUDA_129="${REGISTRY_PREFIX}cuda12.9-${ARCH}"
# No COPY/ADD from the context: pipe the Dockerfile in via stdin (context `-`)
# so nothing is uploaded to the daemon.
if should_build "${CUDA_129}"; then
docker buildx build \
    $(cache_args "${CUDA_129}") \
    --cache-from type=registry,ref=${CUDA_129} \
    -t ${CUDA_129} -t ${CUDA_129}:v${VERSION} \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    --build-arg "CUDA_RUN_URL=${CUDA_129_RUN_URL_VALUE}" \
    --build-arg "CUDA_RUN_MD5=${CUDA_129_RUN_MD5_VALUE}" \
    --build-arg "CUDA_RUN_EXTRA_ARGS=${CUDA_129_RUN_EXTRA_ARGS:-}" \
    - < docker/Dockerfile.cuda-toolkit.deps
fi

# CUDA 13.4.1
# Pick this version's descriptor for the current arch (docker/cuda_versions.env).
CUDA_134_RUN_URL_VAR="CUDA_134_RUN_URL_${ARCH}"; CUDA_134_RUN_URL_VALUE="${!CUDA_134_RUN_URL_VAR}"
CUDA_134_RUN_MD5_VAR="CUDA_134_RUN_MD5_${ARCH}"; CUDA_134_RUN_MD5_VALUE="${!CUDA_134_RUN_MD5_VAR}"
export CUDA_134="${REGISTRY_PREFIX}cuda13.4-${ARCH}"
# No COPY/ADD from the context: pipe the Dockerfile in via stdin (context `-`)
# so nothing is uploaded to the daemon.
if should_build "${CUDA_134}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from "type=registry,ref=${CUDA_134}" \
    -t "${CUDA_134}" -t "${CUDA_134}:v${VERSION}" \
    --platform "${PLATFORM}" \
    ${PUSH_ARG} \
    --build-arg "CUDA_RUN_URL=${CUDA_134_RUN_URL_VALUE}" \
    --build-arg "CUDA_RUN_MD5=${CUDA_134_RUN_MD5_VALUE}" \
    --build-arg "CUDA_RUN_EXTRA_ARGS=${CUDA_134_RUN_EXTRA_ARGS:-}" \
    - < docker/Dockerfile.cuda-toolkit.deps
fi

# GCC 14
export DEPS_GCC14="${REGISTRY_PREFIX}nvimgcodec_deps-${ARCH}-manylinux_2_28-gcc14"
if should_build "${DEPS_GCC14}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${DEPS_GCC14} \
    -t ${DEPS_GCC14} -t ${DEPS_GCC14}:v${VERSION} \
    -f docker/Dockerfile.deps \
    --build-arg "FROM_IMAGE_NAME=${MANYLINUX_GCC14}:v${VERSION}" \
    --build-arg "ARCH=${ARCH}" \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    .
fi

####### BUILDER IMAGES #######

# GCC 14, CUDA 12.9
export BUILDER_CUDA_129="${REGISTRY_PREFIX}builder-cuda-12.9-${ARCH}"
# No COPY/ADD from the context: pipe the Dockerfile in via stdin (context `-`)
# so nothing is uploaded to the daemon.
if should_build "${BUILDER_CUDA_129}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${BUILDER_CUDA_129} \
    -t ${BUILDER_CUDA_129} -t ${BUILDER_CUDA_129}:v${VERSION} \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC14}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_129}:v${VERSION}" \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    - < docker/Dockerfile.cuda.deps
fi

# GCC 14, CUDA 13.4
export BUILDER_CUDA_134="${REGISTRY_PREFIX}builder-cuda-13.4-${ARCH}"
# No COPY/ADD from the context: pipe the Dockerfile in via stdin (context `-`)
# so nothing is uploaded to the daemon.
if should_build "${BUILDER_CUDA_134}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from "type=registry,ref=${BUILDER_CUDA_134}" \
    -t "${BUILDER_CUDA_134}" -t "${BUILDER_CUDA_134}:v${VERSION}" \
    --build-arg "FROM_IMAGE_NAME=${DEPS_GCC14}:v${VERSION}" \
    --build-arg "CUDA_IMAGE=${CUDA_134}:v${VERSION}" \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    - < docker/Dockerfile.cuda.deps
fi


# Cross-compiling host=x86_64 target=L4T
if [ "$ARCH" == "x86_64" ]; then
    export BUILDER_CUDA_TEGRA_129="${REGISTRY_PREFIX}builder-cuda-12.9-cross-l4t-aarch64-linux"
    if should_build "${BUILDER_CUDA_TEGRA_129}"; then
    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref=${BUILDER_CUDA_TEGRA_129} \
        -t ${BUILDER_CUDA_TEGRA_129} -t ${BUILDER_CUDA_TEGRA_129}:tegra_v${TEGRA_VERSION} \
        -f docker/Dockerfile.tegra-aarch64-linux.builder \
        --build-arg "CUDA_CROSS_VERSION=12-9" \
        --build-arg "CUDA_CROSS_VERSION_DOT=12.9" \
        --build-arg "CUDA_TARGET_ARCHS=72;87" \
        --platform ${PLATFORM} \
        ${PUSH_ARG} \
        .
    fi
fi

####### TEST IMAGES #######

# Note: we are using devel image because we need CTK to be available for cupy to be able to JIT compile kernels.
# The second --cache-from on the CUDA 12.9 runner below points at the pre-rename
# -py39 repository. It is transitional: without it the first build under the new
# -ubuntu<ver> name imports from a repository that does not exist and rebuilds from
# scratch. Delete both lines once a -ubuntu<ver> image has been pushed. New runner
# blocks (e.g. CUDA 13.4) never had a -py39 tag, so they don't need this line.

# CUDA 12.9
export RUNNER_CUDA_129="${REGISTRY_PREFIX}runner-cuda-12.9-${ARCH}-ubuntu${VER_UBUNTU}"
if should_build "${RUNNER_CUDA_129}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from type=registry,ref=${RUNNER_CUDA_129} \
    --cache-from type=registry,ref=${REGISTRY_PREFIX}runner-cuda-12.9-${ARCH}-py39 \
    -t ${RUNNER_CUDA_129} -t ${RUNNER_CUDA_129}:v${VERSION} \
    -f docker/Dockerfile.runner \
    --build-arg "ARTIFACT_ARCH=${ARTIFACT_ARCH}" \
    --build-arg "DALI_VERSION=${DALI_VERSION}" \
    --build-arg "BASE=nvidia/cuda:12.9.0-devel-ubuntu${VER_UBUNTU}" \
    --build-arg "VER_CUDA=12.9.0" \
    --build-arg "CUDA_MAJOR=12" \
    $(nvidia_lib_args 12) \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    .
fi

# CUDA 13.4
export RUNNER_CUDA_134="${REGISTRY_PREFIX}runner-cuda-13.4-${ARCH}-ubuntu${VER_UBUNTU}"
if should_build "${RUNNER_CUDA_134}"; then
docker buildx build \
    --cache-to type=inline \
    --cache-from "type=registry,ref=${RUNNER_CUDA_134}" \
    -t "${RUNNER_CUDA_134}" -t "${RUNNER_CUDA_134}:v${VERSION}" \
    -f docker/Dockerfile.runner \
    --build-arg "ARTIFACT_ARCH=${ARTIFACT_ARCH}" \
    --build-arg "DALI_VERSION=${DALI_VERSION}" \
    --build-arg "BASE=nvidia/cuda:13.4.1-devel-ubuntu${VER_UBUNTU}" \
    --build-arg "VER_CUDA=13.4.1" \
    --build-arg "CUDA_MAJOR=13" \
    $(nvidia_lib_args 13) \
    --platform ${PLATFORM} \
    ${PUSH_ARG} \
    .
fi

####### CONDA TEST IMAGES #######

if [ "$ARCH" == "x86_64" ]; then
    # Dockerfile.conda-test has no COPY/ADD, so the build context is irrelevant.
    # Pipe the Dockerfile via stdin (context arg `-`) to avoid uploading the
    # entire repo to the Docker daemon on every build.

    # CUDA 12.9
    export CONDA_TEST_CUDA_129="${REGISTRY_PREFIX}conda-test-cuda-12.9-x86_64"
    if should_build "${CONDA_TEST_CUDA_129}"; then
    docker buildx build \
        --cache-to type=inline \
        --cache-from type=registry,ref=${CONDA_TEST_CUDA_129} \
        -t ${CONDA_TEST_CUDA_129} -t ${CONDA_TEST_CUDA_129}:conda_test_v${CONDA_TEST_VERSION} \
        --build-arg "CUDA_VERSION=12.9.0" \
        --platform "linux/amd64" \
        ${PUSH_ARG} \
        - < docker/Dockerfile.conda-test
    fi

    # CUDA 13.4
    export CONDA_TEST_CUDA_134="${REGISTRY_PREFIX}conda-test-cuda-13.4-x86_64"
    if should_build "${CONDA_TEST_CUDA_134}"; then
    docker buildx build \
        --cache-to type=inline \
        --cache-from "type=registry,ref=${CONDA_TEST_CUDA_134}" \
        -t "${CONDA_TEST_CUDA_134}" -t "${CONDA_TEST_CUDA_134}:conda_test_v${CONDA_TEST_VERSION}" \
        --build-arg "CUDA_VERSION=13.4.1" \
        --platform "linux/amd64" \
        ${PUSH_ARG} \
        - < docker/Dockerfile.conda-test
    fi
fi
