#!/usr/bin/env bash

ALL_BENCHMARKS=(
    "3d7pt:3"
    "apop:2"
    "game-of-life:3"
    "heat-1d:2"
    "heat-2d:3"
    "heat-3d:4"
)

ROOT_DIR="$(realpath "$(dirname "$0")/../")"

FINETUNE_BIN="${ROOT_DIR}/tools/finetune.py"

if [ -z "$PLUTO_BIN" ]; then
    echo "Please set PLUTO_BIN environment variable to point to the Pluto binary."
    exit 1
fi
if [ ! -x "$PLUTO_BIN" ]; then
    echo "PLUTO_BIN ($PLUTO_BIN) is not a valid executable."
    exit 1
fi
if [ -z "$CC" ]; then
    echo "CC is not set, defaulting to gcc"
    CC="gcc"
fi
if [ -z "$ENV_FILE" ]; then
    echo "ENV_FILE is not set, defaulting to env/omp32.env"
    ENV_FILE="./env/omp32.env"
fi

LOG_DIR="${ROOT_DIR}/results/pesto/$(date +%Y-%m-%d)/"
FAILURE_DIR="${ROOT_DIR}/failure/pesto/$(date +%Y-%m-%d)/"

mkdir -p "${LOG_DIR}"
mkdir -p "${FAILURE_DIR}"

CFLAGS="-march=native -O3 -fopenmp"
EXTRA_FLAGS="-lm -DBENCHMARK_TIME"

PLUTO_FLAGS="--tile --parallel --diamond-tile --nounroll --prevector"
PLUTO_VEC_PRAGMA="#pragma GCC ivdep"

for benchmark in "${ALL_BENCHMARKS[@]}"; do
    IFS=':' read -r benchmark_name depth <<<"$benchmark"

    orig_src="${ROOT_DIR}/pluto_bench/${benchmark_name}/${benchmark_name}.c"
    bench_src="${ROOT_DIR}/pluto_bench/${benchmark_name}/${benchmark_name}.tpz.c"

    log_file="${LOG_DIR}/$(date +%Y-%m-%d-%H%M%S)_${benchmark_name}_pesto.log"

    echo "Benching $orig_src"
    cmd=(
        python3
        "${FINETUNE_BIN}"
        "${ROOT_DIR}/lib/benchmark.c"
        "${bench_src}"
        -I "${ROOT_DIR}/include"
        --log-file "${log_file}"
        --compiler-bin "${CC}"
        --compiler-cflags "${CFLAGS}"
        --output-dump-baseline "${ROOT_DIR}/lib/benchmark.c" "${orig_src}"
        --output-dump-flags="-DBENCHMARK_DUMP_ARRAYS"
        --compiler-extra-flags="${EXTRA_FLAGS}"
        --env "$ENV_FILE"
        --save-incorrect-sources "${FAILURE_DIR}/"
        --timeout 5
        --perf-nrun 40
        --perf-nmedianrun 20
        --param DIV0 "[2,512,pow2]"
        --param DIV1 "[8,256,pow2]"
    )

    for ((i = 2; i < depth; i++)); do
        cmd+=("--param" "DIV$i" "[2,256,pow2]")
    done
    echo "Running command:"
    echo "${cmd[@]}"
    "${cmd[@]}"
done
