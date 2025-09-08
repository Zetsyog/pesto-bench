#!/usr/bin/env bash

DATASET=(
    "LARGE_DATASET"
    "EXTRALARGE_DATASET"
)

BENCH=(
    "linear-algebra/solvers/lu"
    # "linear-algebra/solvers/cholesky"
)

OUTPUT_DIR="pb_data/"
mkdir -p "$OUTPUT_DIR"

for bench in "${BENCH[@]}"; do
    bench_name=$(basename "$bench")
    src="${POLYBENCH_DIR}/${bench}/${bench_name}.c"

    for dataset in "${DATASET[@]}"; do
        $CC -O3 "$src" "${POLYBENCH_DIR}/utilities/polybench.c" -I "$POLYBENCH_DIR/utilities" -o "__tmp.out" -lm "-D${dataset}" "-DSTORE_IN_ARRAY"
        "./__tmp.out"
        mv "input.dump" "${OUTPUT_DIR}/${bench_name}_${dataset}.pbdata"

        rm -f "__tmp.out"
    done
done
