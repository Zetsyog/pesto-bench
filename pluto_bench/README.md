# Pluto benchmarks

This directory contains benchmarks from pluto repository, adapted to be used with the finetune.py script. Each benchmark is in its own subdirectory, containing source files and a Makefile for building the baseline version. The benchmarks are mostly from the Polybench suite, but some have been modified or extended.

## Compiling the benchmarks

Prereqs: a C compiler (gcc/clang), `make`, and optional tools used by the benchmarks (`polycc`, `pesto`, `pluto`/`polycc`). You also need to build the `/lib` directory (see the root `README.md` for instructions).

- To build a single benchmark, change into its subdirectory and run a target from the included `Makefile` (these delegate to `../common.mk`). Example:
  - Build the baseline binary:
        ```bash
        cd pluto_bench/3d7pt
        make baseline
        ```

  - Build the Pluto-processed binary (requires `polycc` / Pluto):
        ```bash
        make pluto
        ```

  - Build the Pesto-processed binary (requires `pesto`):
        ```bash
        make pesto
        ```

- Common variables you can override on the `make` command line:

  - `CC` : C compiler (default: `gcc`).
  - `CFLAGS` : compiler flags (default: `-march=native -O3 -fopenmp`).
  - `LDFLAGS` : linker flags used when linking binaries.
  - `POLYCC` : path to the `polycc`/Pluto wrapper used to generate tiled code.
  - `PLUTO_FLAGS` : flags passed to `polycc`/Pluto when producing the `.pluto.c` file.
  - `PESTO` / `PESTO_FLAGS` / `PESTO_CONFIG` : used to build the `pesto` variant.
  - `EXTRA_FLAGS` : appended to compiler command lines; useful to pass `-D` macros.

- Example: build a baseline with timing and dump enabled and a custom compiler:

    ```bash
    make baseline CC=clang CFLAGS="-O3 -march=native -fopenmp" \
        EXTRA_FLAGS="-DBENCHMARK_TIME -DBENCHMARK_DUMP_ARRAYS"
    ```

- The repository root include and lib flags are set in `pluto_bench/common.mk`. If you need to change where headers/libs are found, set `ROOT_DIR` or edit `common.mk` accordingly.

You can also run the combined `check-pluto` target (if available) to build baseline and pluto binaries and compare their dumps:

```bash
make check-pluto
```

## Available macros and their utility

These macros are provided by the benchmark support headers in `include/benchmark/`. Use them by passing `-D<MACRO>` via `EXTRA_FLAGS` or `CFLAGS`.

- `BENCHMARK_TIME` : measure kernel execution time and print it. This is the most basic timing macro.
  - `BENCHMARK_TIME_MFLOPS` : additionally print MFLOPS.
  - `BENCHMARK_TIME_TARGET_FILE` : redirect timing output (default: `stdout`).

- `BENCHMARK_DUMP` : dump program result for verification. Defining this enables both `BENCHMARK_DUMP_ARRAYS` and `BENCHMARK_DUMP_CHKSUM` (full arrays and checksums).
  - `BENCHMARK_DUMP_ARRAYS` : dump full array contents for verification.
  - `BENCHMARK_DUMP_CHKSUM` : print checksums instead of full arrays.
  - `BENCHMARK_DUMP_FILE` : file/stream used for dumps (default: `stderr`).

- `BENCHMARK_USE_RESTRICT` : enables `BENCHMARK_RESTRICT` which expands to `restrict` where supported.

- `BENCHMARK_RSEED` : default random seed (default: 42).

Notes & examples:

- To enable timing and MFLOPS printing: `-DBENCHMARK_TIME -DBENCHMARK_TIME_MFLOPS -DBENCHMARK_NUM_FP_OPS=123456`.
- To enable array dumps for correctness checking: `-DBENCHMARK_DUMP_ARRAYS` (or `-DBENCHMARK_DUMP_CHKSUM`).
- To change the printed dump file: `-DBENCHMARK_DUMP_FILE=\"/tmp/dump.txt\"` (note quoting).

If you need more customization when running the finetune script, the script accepts options to pass compiler flags, pluto path/flags and dump options; see the `finetune.py` usage example above.

## Finetuning

To finetune a model on these benchmarks, you can use the `finetune.py` script. For example, to finetune on the `3d7pt` benchmark, you can run:

```bash
python3 tools/finetune.py pluto_bench/3d7pt/3d7pt.c --log-file tmp/3d7pt.log --env=env/local.env --compiler-bin="gcc" -I ./include/ --compiler-extra-flags="-lm -L./lib -lbenchmark -DBENCHMARK_TIME" --pluto="${PLUTO_BIN}" --pluto-flags="--tile --parallel --diamond-tile --nounroll --prevector" --pluto-custom-vec-pragma="#pragma GCC ivdep" --force-omp-schedule "static" --param T0 "{8}" --param T1 "[2,32,pow2]" --param T2 "{8}" --param T3 "{8}" --timeout 4 --output-dump-flags="-DBENCHMARK_DUMP_ARRAYS" --output-dump-baseline pluto_bench/3d7pt/3d7pt.c
```
