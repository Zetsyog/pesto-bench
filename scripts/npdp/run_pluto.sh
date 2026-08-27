#!/usr/bin/env bash

ALL_BENCHMARKS=(
    "counting"
    "counting_im"
    "knuth"
    "mcc"
    "mcm"
    "mea"
    "nussinov"
    "nw"
    "sw"
    "sw3d"
    "triang"
    "zuker"
)

FINETUNE_BIN="./tools/finetune.py"

if [ -z "$NPDP_DIR" ]; then
    echo "Please set NPDP_DIR environment variable to point to the NPDP directory."
    exit 1
fi
if [ ! -d "$NPDP_DIR" ]; then
    echo "NPDP_DIR ($NPDP_DIR) is not a valid directory."
    exit 1
fi
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

LOG_DIR="./results/pluto_all/$(date +%Y-%m-%d)/"

mkdir -p "${LOG_DIR}"

CFLAGS="-march=native -O3 -fopenmp"
PLUTO_FLAGS="--tile --parallel --diamond-tile --nounroll --prevector"
PLUTO_VEC_PRAGMA="#pragma GCC ivdep"

EXTRA_FLAGS="-lm -DN=2000 -DMETRICS_TIME"

for benchmark in "${ALL_BENCHMARKS[@]}"; do
    orig_src="${NPDP_DIR}/${benchmark}/${benchmark}.c"

    for schedule in "static" "dynamic"; do
        log_file="${LOG_DIR}/$(date +%Y-%m-%d-%H%M%S)_${benchmark}_pluto_$schedule.log"

        echo "Benching $orig_src"
        echo "OMP $schedule"
        cmd=(
            python3
            "${FINETUNE_BIN}"
            "${orig_src}"
            -I "${NPDP_DIR}/include"
            --log-file "${log_file}"
            --compiler-bin "${CC}"
            --compiler-cflags "${CFLAGS}"
            --pluto "${PLUTO_BIN}"
            --pluto-flags="${PLUTO_FLAGS}"
            --pluto-custom-vec-pragma="${PLUTO_VEC_PRAGMA}"
            --output-dump-baseline "${orig_src}"
            --compiler-extra-flags="${EXTRA_FLAGS}"
            --env "$ENV_FILE"
            --force-omp-schedule "$schedule"
            --timeout 5
            --param T0 "[2,512,pow2]"
            --param T1 "[2,512,pow2]"
            --param T2 "[2,512,pow2]"
            --perf-nrun 5
            --perf-nmedianrun 3
        )
        echo "Running command:"
        echo "${cmd[@]}"
        "${cmd[@]}"
    done
done
