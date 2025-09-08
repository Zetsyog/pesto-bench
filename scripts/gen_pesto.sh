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

if [ -z "$PESTO_BIN" ]; then
    echo "PESTO_BIN is not set. Please run fetch_and_build_pesto.sh first."
    exit 1
fi
if [ ! -x "$PESTO_BIN" ]; then
    echo "PESTO_BIN ($PESTO_BIN) is not executable. Please check the path."
    exit 1
fi

# Check if there is at least one argument
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <pesto_json_config> [suffix] [source file]"
    exit 1
fi

if [ "$#" -gt 1 ]; then
    suffix="$2"
fi

gen_single_src=0

if [ "$#" -eq 3 ]; then
    gen_single_src=1
    echo "Generating single source file: $2"
    source_file="$3"
fi

if [ $gen_single_src -eq 1 ] && [ ! -f "$source_file" ]; then
    echo "Source file $source_file does not exist."
    exit 1
fi

config_file="$1"

function gen_pesto_source() {
    local config_file="$1"
    local src_file="$2"
    local output_file

    output_file="$(dirname "$src_file")/$(basename "$src_file" .c).$suffix.c"

    echo "Generating PESTO source for $src_file"
    cmd=(
        "$PESTO_BIN"
        "--config" "$config_file"
        "$src_file"
        "-o" "$output_file"
    )

    echo "Running command: ${cmd[*]}"
    if ! "${cmd[@]}"; then
        echo "PESTO failed for $src_file."
        exit 1
    fi

    if [ ! -f "$output_file" ]; then
        echo "PESTO did not generate output file $output_file."
        return 1
    fi

    echo "PESTO source generated at $output_file"
    return 0
}

if [ $gen_single_src -eq 1 ]; then

    gen_pesto_source "$config_file" "$source_file"
    exit $?
else
    for benchmark in "${ALL_BENCHMARKS[@]}"; do
        benchmark_name=$(basename "$benchmark")
        src_file="${POLYBENCH_DIR}/${benchmark}/${benchmark_name}.c"

        if [ ! -f "$src_file" ]; then
            echo "Source file $src_file does not exist. Skipping."
            continue
        fi

        gen_pesto_source "$config_file" "$src_file"
    done
fi
