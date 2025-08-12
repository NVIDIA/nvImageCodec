#!/bin/bash -ex

nvidia-smi

cuda_major_version=$(python3 -m pip list | grep nvidia-dali | grep -oP 'cuda\K\d{2}')

export DALI_PATH=/opt/dali
export DALI_EXTRA_PATH=/opt/dali_extra

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
