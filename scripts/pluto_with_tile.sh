#!/usr/bin/env bash

PLUTO_OPT="--tile --parallel --diamond-tile --nounroll --prevector"

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

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <source_file> <output_file> [<tile_size1> <tile_size2> ...]"
    exit 1
fi

# write all arguments except first one to tile.sizes file
printf "%s\n" "${@:3}" >tile.sizes

echo "PLUTO_OPT: $PLUTO_OPT"
echo "PLUTO_BIN: $PLUTO_BIN"

$PLUTO_BIN $PLUTO_OPT "$1" -o "$2"
if [ $? -ne 0 ]; then
    echo "Pluto failed"
    exit 1
fi
rm -f *.cloog tile.sizes

# replace #pragma ivdep
sed -i 's/#pragma ivdep/#pragma GCC ivdep/' "$1"
# remove #pragma vector always
sed -i 's/#pragma vector always//' "$1"
