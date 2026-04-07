# Pluto benchmarks

This directory contains benchmarks from pluto repository, adapted to be used with the finetune.py script. Each benchmark is in its own subdirectory, containing source files and a Makefile for building the baseline version. The benchmarks are mostly from the Polybench suite, but some have been modified or extended.

## Finetuning

To finetune a model on these benchmarks, you can use the `finetune.py` script. For example, to finetune on the `3d7pt` benchmark, you can run:

```bash
python3 tools/finetune.py pluto_bench/3d7pt/3d7pt.c --log-file tmp/3d7pt.log --env=env/local.env --compiler-bin="gcc" -I ./include/ --compiler-extra-flags="-lm -L./lib -lbenchmark -DBENCHMARK_TIME" --pluto="${PLUTO_BIN}" --pluto-flags="--tile --parallel --diamond-tile --nounroll --prevector" --pluto-custom-vec-pragma="#pragma GCC ivdep" --force-omp-schedule "static" --param T0 "{8}" --param T1 "[2,32,pow2]" --param T2 "{8}" --param T3 "{8}" --timeout 4 --output-dump-flags="-DBENCHMARK_DUMP_ARRAYS" --output-dump-baseline pluto_bench/3d7pt/3d7pt.c
```
