#!/usr/bin/env bash

ALL_BENCHMARKS=(
    # "stencils/adi"
    # "stencils/fdtd-2d"
    "stencils/heat-3d"
    # "stencils/jacobi-1d"
    # "stencils/jacobi-2d"
    # "stencils/seidel-2d"
)

FINETUNE_BIN="/home/rossetti/repos/pesto/test/tools/finetune.py"

LOG_DIR="results/pesto_tpz/$(date +%Y-%m-%d)/"
FAILURE_DIR="failure/pesto_tpz/$(date +%Y-%m-%d)"
ENV_FILE="./omp32_spread.env"

mkdir -p "${LOG_DIR}"
mkdir -p "${FAILURE_DIR}"

SUFFIX="tpz"

CC="gcc-12"
EXTRA_FLAGS="-lm -DEXTRALARGE_DATASET -DPOLYBENCH_TIME"

for benchmark in "${ALL_BENCHMARKS[@]}"; do
    benchmark_name=$(basename "$benchmark")

    orig_src="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.c"
    pesto_src="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.$SUFFIX.c"

    if [ ! -f "$orig_src" ]; then
        echo "Source file $orig_src does not exist. Skipping benchmark $benchmark_name."
        continue
    fi

    log_file="${LOG_DIR}/$(date +%Y-%m-%d-%H%M%S)_${benchmark_name}_pesto.log"

    echo "Benching $orig_src"
    echo "PESTO TPZ"
    cmd=(
        python3
        "${FINETUNE_BIN}"
        "${POLYBENCH_DIR}/utilities/polybench.c"
        "${pesto_src}"
        -I "${POLYBENCH_DIR}/utilities"
        --log-file "${log_file}"
        --compiler-bin "${CC}"
        --output-dump-baseline "${POLYBENCH_DIR}/utilities/polybench.c" "${orig_src}"
        "--compiler-extra-flags=${EXTRA_FLAGS}"
        --env "${ENV_FILE}"
        --save-incorrect-sources "${FAILURE_DIR}"
        --param DIV0 "[2,256,pow2]"
        --param DIV2 "[4,512,pow2]"
        # --param D1_MIN_VOL "dyn[tpz_vol,1,10000000]"
        --param D1_MIN_VOL "dyn[min,1]"
        --timeout 5
        # --param DIV0 "{16}"
        # --param DIV2 "{8}"
        # --param D1_MIN_VOL "{10000000}"
        # --param D1_MIN_VOL "[100000,10000000,50000]"
    )
    echo "Running command:"
    echo "${cmd[@]}"
    "${cmd[@]}"

done
