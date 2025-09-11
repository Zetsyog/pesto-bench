#!/usr/bin/env bash

ALL_BENCHMARKS=(
    "datamining/correlation"
    "datamining/covariance"
    "linear-algebra/blas/gemm"
    "linear-algebra/blas/gemver"
    "linear-algebra/blas/gesummv"
    # "linear-algebra/blas/symm"
    "linear-algebra/blas/syr2k"
    "linear-algebra/blas/syrk"
    "linear-algebra/blas/trmm"
    "linear-algebra/kernels/2mm"
    "linear-algebra/kernels/3mm"
    "linear-algebra/kernels/atax"
    "linear-algebra/kernels/bicg"
    "linear-algebra/kernels/doitgen"
    "linear-algebra/kernels/mvt"
    "linear-algebra/solvers/cholesky"
    "linear-algebra/solvers/gramschmidt"
    "linear-algebra/solvers/lu"
    "stencils/adi"
    "stencils/fdtd-2d"
    "stencils/heat-3d"
    "stencils/jacobi-1d"
    "stencils/jacobi-2d"
    "stencils/seidel-2d"
)

FINETUNE_BIN="./tools/finetune.py"

if [ -z "$POLYBENCH_DIR" ]; then
    echo "Please set POLYBENCH_DIR environment variable to point to the PolyBench directory."
    exit 1
fi
if [ ! -d "$POLYBENCH_DIR" ]; then
    echo "POLYBENCH_DIR ($POLYBENCH_DIR) is not a valid directory."
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

ENV_FILE="./env/omp32.env"

LOG_DIR="./results/pluto_all/$(date +%Y-%m-%d)/"
FAILURE_DIR="./failure/pluto_all/$(date +%Y-%m-%d)/"

mkdir -p "${LOG_DIR}"
mkdir -p "${FAILURE_DIR}"

EXTRA_FLAGS="-lm -DEXTRALARGE_DATASET -DPOLYBENCH_TIME"

PLUTO_FLAGS="--tile --parallel --diamond-tile --nounroll --prevector"
PLUTO_VEC_PRAGMA="#pragma GCC ivdep"

mkdir -p "${FAILURE_DIR}/static/"
mkdir -p "${FAILURE_DIR}/dynamic/"

for benchmark in "${ALL_BENCHMARKS[@]}"; do
    benchmark_name=$(basename "$benchmark")

    orig_src="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.c"

    for schedule in "static" "dynamic"; do
        log_file="${LOG_DIR}/$(date +%Y-%m-%d-%H%M%S)_${benchmark_name}_pluto_$schedule.log"

        echo "Benching $orig_src"
        echo "OMP $schedule"
        cmd=(
            python3
            "${FINETUNE_BIN}"
            "${POLYBENCH_DIR}/utilities/polybench.c"
            "${orig_src}"
            -I "${POLYBENCH_DIR}/utilities"
            --log-file "${log_file}"
            --compiler-bin "${CC}"
            --pluto "${PLUTO_BIN}"
            --pluto-flags="${PLUTO_FLAGS}"
            --pluto-custom-vec-pragma="${PLUTO_VEC_PRAGMA}"
            --output-dump-baseline "${POLYBENCH_DIR}/utilities/polybench.c" "${orig_src}"
            --compiler-extra-flags="${EXTRA_FLAGS}"
            --env "$ENV_FILE"
            --force-omp-schedule "$schedule"
            --save-incorrect-sources "${FAILURE_DIR}/$schedule/"
            --param T0 "[4,512,pow2]"
            --param T1 "[4,512,pow2]"
            --param T2 "[4,512,pow2]"
            --param T3 "[4,256,pow2]"
            --timeout 4
        )
        echo "Running command:"
        echo "${cmd[@]}"
        "${cmd[@]}"
    done
done
