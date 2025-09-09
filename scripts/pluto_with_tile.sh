#!/usr/bin/env bash

PLUTO_OPT="--tile --parallel --diamond-tile --nounroll --prevector"
CFLAGS="-march=native -O3 -fopenmp"
EXTRA_FLAGS="${EXTRA_FLAGS} -lm -DEXTRALARGE_DATASET -DPOLYBENCH_TIME"

if [ -z "$PLUTO_BIN" ]; then
    echo "PLUTO_BIN not set, please source an env file."
    exit 1
fi
if [ -z "$CC" ]; then
    echo "CC not set, please source an env file."
    exit 1
fi
if [ -z "$POLYBENCH_DIR" ]; then
    echo "POLYBENCH_DIR not set, please source an env file."
    exit 1
fi

# write all arguments except first one to tile.sizes file
printf "%s\n" "${@:2}" >tile.sizes

echo "PLUTO_OPT: $PLUTO_OPT"
echo "PLUTO_BIN: $PLUTO_BIN"

$PLUTO_BIN $PLUTO_OPT "$1" -o "__tmp.c"
if [ $? -ne 0 ]; then
    echo "Pluto failed"
    exit 1
fi
rm -f *.cloog tile.sizes

# replace #pragma ivdep
sed -i 's/#pragma ivdep/#pragma GCC ivdep/' "$1"
# remove #pragma vector always
sed -i 's/#pragma vector always//' "$1"

$CC $CFLAGS "${POLYBENCH_DIR}/utilities/polybench.c" -I "${POLYBENCH_DIR}/utilities/" -I "$(dirname "$1")" __tmp.c -o __tmp.out $EXTRA_FLAGS
if [ $? -ne 0 ]; then
    echo "Compilation failed"
    exit 1
fi

for i in {1..5}; do
    echo "Run #$i"
    ./__tmp.out
done
rm -f __tmp.c __tmp.out
