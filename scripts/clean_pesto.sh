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
    # "stencils/adi"
    "stencils/fdtd-2d"
    "stencils/heat-3d"
    "stencils/jacobi-1d"
    "stencils/jacobi-2d"
    "stencils/seidel-2d"
)

# if there is at least one argument
if [ $# -lt 1 ]; then
    suffix="pesto"
else
    suffix="$1"
fi

for benchmark in "${ALL_BENCHMARKS[@]}"; do
    benchmark_name=$(basename "$benchmark")
    benchmark_dir=$(dirname "$benchmark")
    pesto_file="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.$suffix.c"

    if [ ! -f "$pesto_file" ]; then
        echo "Source file $pesto_file does not exist. Skipping benchmark $benchmark_name."
        continue
    fi
    echo "Cleaning $pesto_file"
    rm -f "$pesto_file" "${POLYBENCH_DIR}/${benchmark_dir}/${benchmark_name}.$suffix.trahrhe."*.h

done
