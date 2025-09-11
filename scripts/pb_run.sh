#!/usr/bin/env bash

if [ -z "$CC" ]; then
    echo "CC not set, please source an env file."
    exit 1
fi
if [ -z "$POLYBENCH_DIR" ]; then
    echo "POLYBENCH_DIR not set, please source an env file."
    exit 1
fi
if [ -z "$CFLAGS" ]; then
    CFLAGS="-march=native -O3 -fopenmp"
fi
if [ -z "$EXTRA_FLAGS" ]; then
    EXTRA_FLAGS="-lm -DEXTRALARGE_DATASET -DPOLYBENCH_TIME"
fi
if [ "$#" -lt 1 ]; then
    echo "Usage: $0 <source_file> [compiler opts...]"
    exit 1
fi

EXTRA_FLAGS="$EXTRA_FLAGS ${*:2}"

src="$1"

$CC $CFLAGS "${POLYBENCH_DIR}/utilities/polybench.c" -I "${POLYBENCH_DIR}/utilities/" -I "$(dirname "$1")" "$src" -o __tmp.out $EXTRA_FLAGS
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi
for i in {1..5}; do
    echo "Run #$i"
    ./__tmp.out
done
rm -f __tmp.out
