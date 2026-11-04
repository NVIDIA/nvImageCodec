#!/bin/bash -ex

nvidia-smi

cuda_major_version=$(python3 -m pip list | grep nvidia-dali | grep -oP 'cuda\K\d{2}')

export DALI_PATH=${DALI_PATH:-/opt/dali}
export DALI_EXTRA_PATH=${DALI_EXTRA_PATH:-/opt/dali_extra}

# The runner image does not ship these; fetching here keeps 2.3 GB of mostly-unused
# test data out of every image and pins what we do fetch to the DALI actually
# installed, which a baked-in clone could not do. Both fetches are skipped when the
# directory is already populated, so a cached volume costs nothing.
#
# Override DALI_GIT_URL / DALI_EXTRA_GIT_URL to pull from an internal mirror where
# github.com is unreachable.
DALI_GIT_URL=${DALI_GIT_URL:-https://github.com/NVIDIA/DALI.git}
DALI_EXTRA_GIT_URL=${DALI_EXTRA_GIT_URL:-https://github.com/NVIDIA/DALI_extra.git}

# Only db/single/jpeg is read below (12 MB). The rest of db/single is 1.7 GB, most of
# it db/single/fits, which nothing in this benchmark opens.
#
# Space-separated so an override can list several paths; read into an array (not left
# as a bare expansion below) so a path never undergoes globbing against the caller's cwd.
DALI_EXTRA_SPARSE_PATHS=${DALI_EXTRA_SPARSE_PATHS:-db/single/jpeg}
read -r -a DALI_EXTRA_SPARSE_PATHS_ARR <<< "${DALI_EXTRA_SPARSE_PATHS}"

# git has no equivalent of curl's --retry, and these clones are the only network
# access in the benchmark. Retry the whole clone rather than leaving a partial one.
retry_clone() {
    local url=$1 dest=$2 attempt
    for attempt in 1 2 3; do
        rm -rf "${dest}"
        if git clone --filter=blob:none --sparse "${url}" "${dest}"; then
            return 0
        fi
        echo "clone of ${url} failed (attempt ${attempt}/3); retrying in 10s" >&2
        sleep 10
    done
    echo "ERROR: could not clone ${url} after 3 attempts" >&2
    return 1
}

# DALI sources: only internal_tools is used, and only for hw_decoder_bench.py.
if [ ! -e "${DALI_PATH}/internal_tools/hw_decoder_bench.py" ]; then
    retry_clone "${DALI_GIT_URL}" "${DALI_PATH}"
    (cd "${DALI_PATH}" && git sparse-checkout set internal_tools)
    # Match the installed wheel when it is a release; nightlies have no matching tag,
    # so those stay on the default branch.
    dali_version=$(python3 -m pip show "nvidia-dali-nightly-cuda${cuda_major_version}0" 2>/dev/null \
                   | sed -n 's/^Version: \(.*\)$/\1/p')
    if [ -n "${dali_version}" ] && ! echo "${dali_version}" | grep -q '\.dev'; then
        (cd "${DALI_PATH}" && git checkout "tags/v${dali_version}") \
            || echo "no tag v${dali_version} in DALI; staying on default branch" >&2
    else
        echo "DALI version '${dali_version}' is a nightly; staying on default branch"
    fi
fi

# DALI_extra: pinned to the revision the checked-out DALI expects, when it says.
if [ ! -d "${DALI_EXTRA_PATH}/db/single/jpeg" ]; then
    retry_clone "${DALI_EXTRA_GIT_URL}" "${DALI_EXTRA_PATH}"
    (cd "${DALI_EXTRA_PATH}" && git sparse-checkout set "${DALI_EXTRA_SPARSE_PATHS_ARR[@]}")
    # `git show HEAD:<path>`, not `cat`: DALI_EXTRA_VERSION lives at the repo root and a
    # sparse checkout does not materialise root files, so the file is not on disk. The
    # old in-Dockerfile pin used `cat` and so silently pinned nothing.
    dali_extra_ref=$(cd "${DALI_PATH}" && git show HEAD:DALI_EXTRA_VERSION 2>/dev/null)
    if [ -n "${dali_extra_ref}" ]; then
        (cd "${DALI_EXTRA_PATH}" && git checkout "${dali_extra_ref}") \
            || echo "could not check out DALI_extra ${dali_extra_ref}; using default branch" >&2
    fi
fi

COMMON_ARGS="--width_hint 6000 --height_hint 6000 -b 408 -d 0 -g gpu -w 100 -t 100000 -i ${DALI_EXTRA_PATH}/db/single/jpeg -p rn50 -j 70 --hw_load 0.12"
HW_DECODER_BENCH="${DALI_PATH}/internal_tools/hw_decoder_bench.py"

python3 ${HW_DECODER_BENCH} ${COMMON_ARGS} 2>&1 | tee /tmp/dali_bench_ref.log
python3 ${HW_DECODER_BENCH} ${COMMON_ARGS} --experimental_decoder 2>&1 | tee /tmp/dali_bench_nvimgcodec.log

THROUGHPUT_DALI=$(grep -oP 'Total Throughput: \K[0-9]+(\.[0-9]+)?(?= frames/sec)' /tmp/dali_bench_ref.log)
THROUGHPUT_NVIMGCODEC=$(grep -oP 'Total Throughput: \K[0-9]+(\.[0-9]+)?(?= frames/sec)' /tmp/dali_bench_nvimgcodec.log)

# Ensure that THROUGHPUT_NVIMGCODEC is no less than 5% smaller than THROUGHPUT_DALI

# TODO(janton): Ideally, we would like to enforce 5% for all CUDA versions,
#               but it is still not met for some cases, so fail only if less than 15% less
if [ "$cuda_major_version" -ge 12 ]; then
    threshold=0.95
else
    threshold=0.85
fi

PERF_RESULT=$(echo "$THROUGHPUT_NVIMGCODEC $THROUGHPUT_DALI" | awk -v threshold="$threshold" '{if ($1 >= $2 * threshold) {print "OK"} else { print "FAIL" }}')
echo "PERF_RESULT=${PERF_RESULT}"

if [ "$PERF_RESULT" == "FAIL" ]; then
    exit 1
else
    exit 0
fi
