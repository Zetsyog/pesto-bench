#!/usr/bin/env bash

set -euo pipefail

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
	# "stencils/fdtd-2d"
	# "stencils/heat-3d"
	# "stencils/jacobi-1d"
	# "stencils/jacobi-2d"
	# "stencils/seidel-2d"
)

declare -A ALG_BEST_DIVS
ALG_BEST_DIVS[correlation]="32"
ALG_BEST_DIVS[covariance]="32"
ALG_BEST_DIVS[gemm]="64"
ALG_BEST_DIVS[gemver]="64"
ALG_BEST_DIVS[gesummv]="64"
ALG_BEST_DIVS[syr2k]="64"
ALG_BEST_DIVS[syrk]="64"
ALG_BEST_DIVS[trmm]="64"
ALG_BEST_DIVS[2mm]="64"
ALG_BEST_DIVS[3mm]="64"
ALG_BEST_DIVS[atax]="64"
ALG_BEST_DIVS[bicg]="64"
ALG_BEST_DIVS[mvt]="64"

declare -A PLUTO_BEST_SIZES

declare -A HYBRID_BEST_SIZES

CC="gcc"
CFLAGS="-O3 -march=native"

if [ "$#" -ne 1 ]; then
	echo "Usage: $0 <output_dir>"
	exit 1
fi

output_dir="$1"
mkdir -p "$output_dir"

for bench in "${ALL_BENCHMARKS[@]}"; do
	echo "==== $bench ===="
	bench_name=$(basename "$bench")
	report_dir="$output_dir/$bench_name"
	mkdir -p "$report_dir"

	for version in "pluto" "pesto"; do
		echo "---- $version ----"
		src=""
		bin="${bench_name}_${version}"
		report="$report_dir/$version.txt"

		if [ "$version" == "pluto" ]; then
			src="${POLYBENCH_DIR}/$bench/$bench_name.pluto.c"
			rm -f "$src"
			$PLUTO_BIN --tile --nounroll --parallel --prevec -o "$src" "${POLYBENCH_DIR}/$bench/$bench_name.c"
			# replace #pragma ivdep
			sed -i 's/#pragma ivdep/#pragma GCC ivdep/' "$src"
			# remove #pragma vector always
			sed -i 's/#pragma vector always//' "$src"
		elif [ "$version" == "pesto" ]; then
			src="${POLYBENCH_DIR}/$bench/$bench_name.pesto.c"
		fi

		if [ ! -f "$src" ]; then
			echo "Source file $src does not exist. Skipping."
			continue
		fi

		cmd="$CC $CFLAGS -fopt-info-vec-all=$report_dir/all.$version.txt -I${POLYBENCH_DIR}/utilities $src ${POLYBENCH_DIR}/utilities/polybench.c -o $bin"
		echo "Compiling with command: $cmd"
		if ! $cmd; then
			echo "Compilation failed for $src. Skipping."
			continue
		fi

		cmd="$CC $CFLAGS -fopt-info-vec-missed=$report_dir/missed.$version.txt -I${POLYBENCH_DIR}/utilities $src ${POLYBENCH_DIR}/utilities/polybench.c -o $bin"
		echo "Compiling with command: $cmd"
		if ! $cmd; then
			echo "Compilation failed for $src. Skipping."
			continue
		fi

		cmd="$CC $CFLAGS -fopt-info-vec-optimized=$report_dir/optimized.$version.txt -I${POLYBENCH_DIR}/utilities $src ${POLYBENCH_DIR}/utilities/polybench.c -o $bin"
		echo "Compiling with command: $cmd"
		if ! $cmd; then
			echo "Compilation failed for $src. Skipping."
			continue
		fi

		rm -f "$bin"

	done
done
