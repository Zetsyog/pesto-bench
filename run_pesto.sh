#!/usr/bin/env bash

ALL_BENCHMARKS=(
    "datamining/correlation"
    "datamining/covariance"
    "linear-algebra/blas/gemm"
    "linear-algebra/blas/gemver"
    "linear-algebra/blas/gesummv"
    "linear-algebra/blas/symm"
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
    # "stencils/adi"
    # "stencils/fdtd-2d"
    # "stencils/heat-3d"
    # "stencils/jacobi-1d"
    # "stencils/jacobi-2d"
    # "stencils/seidel-2d"
)

FINETUNE_BIN="/home/rossetti/repos/pesto/test/tools/finetune.py"

LOG_DIR="results/pesto/$(date +%Y-%m-%d)/"
FAILURE_DIR="failure/pesto/$(date +%Y-%m-%d)"

mkdir -p "${LOG_DIR}"
mkdir -p "${FAILURE_DIR}"

LOG_SUFFIX="pesto"
SRC_SUFFIX="pesto"

CC="gcc-12"
EXTRA_FLAGS="-lm -DEXTRALARGE_DATASET -DPOLYBENCH_TIME"

for benchmark in "${ALL_BENCHMARKS[@]}"; do
    benchmark_name=$(basename "$benchmark")

    orig_src="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.c"
    pesto_src="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.${SRC_SUFFIX}.c"

    if [ ! -f "$orig_src" ]; then
        echo "Source file $orig_src does not exist. Skipping benchmark $benchmark_name."
        continue
    fi

    log_file="${LOG_DIR}/$(date +%Y-%m-%d-%H%M%S)_${benchmark_name}_$LOG_SUFFIX.log"

    echo "Benching $orig_src"
    echo "PESTO STATIC"
    cmd=(
        python3
        "${FINETUNE_BIN}"
        "${POLYBENCH_DIR}/utilities/polybench.c"
        "${pesto_src}"
        -I "${POLYBENCH_DIR}/utilities"
        --log-file "${log_file}"
        --compiler-bin "${CC}"
        --output-dump-baseline "${POLYBENCH_DIR}/utilities/polybench.c" "${orig_src}"
        --compiler-extra-flags="${EXTRA_FLAGS}"
        --env ./omp32_static.env
        --save-incorrect-sources "${FAILURE_DIR}"
        --param DIV0 "[8, 256, pow2]"
        --param DIV1 "[8, 512, pow2]"
        --param DIV2 "[8, 512, pow2]"
    )
    echo "Running command:"
    echo "${cmd[@]}"
    "${cmd[@]}"

done
