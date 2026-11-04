#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Verify that the shipped .so artifacts export exactly the symbols their linker
# version script allows, and nothing else.
#
# Two properties are checked per artifact:
#   1. the set of global text (T) exports equals the version script allowlist;
#   2. no weak symbols -- functions (W) or objects (V) -- reach .dynsym at all.
#
# (2) catches libstdc++ vague-linkage symbols emitted by our own translation
# units. (1) is what keeps the checked-in allowlists honest
# -- adding an NVIMGCODECAPI function without updating cmake/version_scripts/
# would otherwise silently ship a library missing that export.
#
# Usage: tools/check_exported_symbols.sh <build-dir>

set -u -o pipefail

if [[ $# -ne 1 ]]; then
    echo "usage: $0 <build-dir>" >&2
    exit 2
fi

BUILD_DIR="$1"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT_DIR="${REPO_ROOT}/cmake/version_scripts"

if [[ ! -d "$BUILD_DIR" ]]; then
    echo "error: no such build directory: $BUILD_DIR" >&2
    exit 2
fi

# nm has to understand the *target* ELF. The Tegra config cross-compiles aarch64
# on an x86_64 host, where a single-target host nm cannot read the artifacts, so
# prefer an explicit $NM, then the target-triple nm, then plain nm. ARCH is set by
# docker/build_helper.sh (e.g. "aarch64-linux"); it is unset for a local run, which
# is always native.
pick_nm() {
    local candidate
    if [[ -n "${NM:-}" ]]; then
        echo "${NM}"
        return
    fi
    for candidate in "${ARCH%%-*}-linux-gnu-nm" nm; do
        if command -v "${candidate}" >/dev/null 2>&1; then
            echo "${candidate}"
            return
        fi
    done
    echo nm
}
ARCH="${ARCH:-}"
NM_BIN="$(pick_nm)"

# Extract the `global:` allowlist from a version script, one symbol per line.
allowlist_of() {
    sed -n '/global:/,/local:/p' "$1" \
        | sed -e 's/#.*//' -e '/global:/d' -e '/local:/d' \
        | tr -d ' \t;' | grep -v '^$' | sort
}

failures=0
checked=0

check_artifact() {
    local so="$1" script="$2"
    local expected symbols actual weak

    expected="$(allowlist_of "${SCRIPT_DIR}/${script}")"
    # Fail loudly rather than letting an nm error read as "this .so exports nothing",
    # which would surface as a confusing allowlist mismatch.
    if ! symbols="$("${NM_BIN}" -D -g --defined-only "$so")"; then
        echo "FAIL $(basename "$so"): ${NM_BIN} could not read the artifact"
        failures=$((failures + 1))
        checked=$((checked + 1))
        return
    fi
    actual="$(echo "$symbols" | awk '$2 == "T" { print $3 }' | sort)"
    # W is a weak function, V a weak object; both are vague-linkage leaks.
    weak="$(echo "$symbols" | awk '$2 == "W" || $2 == "V" { print $3 }' | sort)"

    checked=$((checked + 1))

    if [[ "$expected" != "$actual" ]]; then
        echo "FAIL $(basename "$so"): exported T symbols differ from ${script}"
        diff <(echo "$expected") <(echo "$actual") \
            | sed -e 's/^</  missing (in allowlist, not exported): /' \
                  -e 's/^>/  unexpected (exported, not in allowlist): /' \
            | grep -E '^  (missing|unexpected)'
        failures=$((failures + 1))
        return
    fi

    if [[ -n "$weak" ]]; then
        echo "FAIL $(basename "$so"): $(echo "$weak" | wc -l) weak symbol(s) leaked into .dynsym"
        echo "$weak" | head -10 | sed 's/^/  /'
        failures=$((failures + 1))
        return
    fi

    echo "ok   $(basename "$so") ($(echo "$expected" | wc -l) exports, 0 weak)"
}

# Resolve the real files behind the SONAME symlinks so each artifact is checked
# once: -type f skips the symlinks and keeps only the fully-versioned real file.
while IFS= read -r so; do
    check_artifact "$so" nvimgcodec.map
done < <(find "$BUILD_DIR" -name 'libnvimgcodec.so.*.*.*' -type f | sort)

while IFS= read -r so; do
    check_artifact "$so" extension.map
done < <(find "$BUILD_DIR" -name 'lib*_ext.so.*.*.*' -type f | sort)

while IFS= read -r so; do
    check_artifact "$so" python_module.map
done < <(find "$BUILD_DIR" -name 'nvimgcodec_impl*.so' -type f | sort)

if [[ $checked -eq 0 ]]; then
    echo "error: found no artifacts to check under ${BUILD_DIR}" >&2
    exit 2
fi

echo
if [[ $failures -ne 0 ]]; then
    echo "${failures}/${checked} artifact(s) FAILED the exported-symbol check"
    exit 1
fi
echo "all ${checked} artifact(s) passed the exported-symbol check"
