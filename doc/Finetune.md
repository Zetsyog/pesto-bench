**Finetune**

- **File**: [tools/finetune.py](tools/finetune.py)

This document explains how to use the `finetune.py` helper script to search and tune compilation / tiling parameters for Polybench-style benchmarks. It contains a usage section with examples, a complete list of supported CLI options, the parameter expression syntax, and an advanced section describing the implementation and internal components.

**Usage**

- **Synopsis**: `python tools/finetune.py [OPTIONS] PATH...`
- **Purpose**: compile and run one benchmark consisting of one or several source files while sweeping or finetuning parameters (tile sizes, volumes, etc.) to find best-performing configurations.

Examples:

- Find best tile sizes using exhaustive ranges:

  `python tools/finetune.py --param T0 "[8,256,pow2]" sample/bench.c`

- Run with Pluto transformations and custom flags:

  `python tools/finetune.py --pluto /usr/local/bin/pluto --pluto-flags --tile --pluto-flags --parallel --param T0 "[16,512,16]" bench.c`

- Compare optimized outputs against a baseline implementation (dump arrays with `-DPOLYBENCH_DUMP_ARRAYS`):

  `python tools/finetune.py --output-dump-baseline baseline.c --param T0 "[16,256,16]" bench_variant.c`

- Use a dynamic parameter for minimizing a value (e.g., memory or volume):

  `python tools/finetune.py --param V "dyn[min,1024]" bench.c`


**Options**

- **sources**: One or more source paths to compile/run. Positional arguments.
- **--log-file PATH**: Save the script log output to PATH.
- **-o, --output PATH**: Append an output path (json or csv). Can be specified multiple times.
- **--pluto [PLUTO_CMD]**: Enable transformation via Pluto. Optionally provide a Pluto command; when omitted `--pluto` will enable Pluto and try to resolve the default command.
- **--pluto-flags FLAG...**: Extra flags to pass to Pluto. Can be repeated.
- **--pluto-custom-vec-pragma PRAGMA**: Custom vectorization pragma to inject when using Pluto.
- **--param NAME EXPR**: Declare an experiment parameter to sweep or finetune. See "Parameter expression syntax" below for the supported formats. Can be specified multiple times to supply multiple parameters.
- **--compiler-cflags FLAG...**: Flags passed to the compiler as cflags (e.g. optimization flags). Can be repeated.
- **--compiler-extra-flags FLAG...**: Extra flags passed to the compiler on top of the default cflags.
- **-I, --include-dir PATH...**: Additional include directories for the compiler.
- **--output-dump-baseline PATH...**: Source(s) used to generate baseline outputs for result comparison.
- **--compiler-bin BIN**: Path to compiler binary (e.g. `/usr/bin/gcc`). If omitted the script attempts to use `gcc` from PATH.
- **--env ENV_FILE**: Path to a file with environment variable definitions to load before runs.
- **--save-incorrect-sources [DIR]**: If specified, sources that produce incorrect outputs will be saved to DIR.
- **--force-omp-schedule OMP_SCHEDULE**: Force a specific OpenMP schedule for runs (overrides `OMP_SCHEDULE` in environment).
- **--timeout SECONDS**: Timeout per run in seconds (0 = no timeout).
- **--perf-nrun N**: Number of runs used for performance measurement (default 5).
- **--perf-nmedianrun N**: Number of median runs used when computing trimmed means (default 3).
- **--insert-include FILE...**: Insert header include directives into the source before compilation.


**Parameter expression syntax**

The `--param NAME EXPR` option supports three expression forms. Examples show how to declare parameters to the finetune engine.

- Range (square brackets): ``[MIN,MAX]`` or ``[MIN,MAX,STEP]`` or ``[MIN,MAX,pow2]``
  - Example: `--param T0 "[16,256,16]"` iterates T0 = 16, 32, 48, ..., 256 (step 16)
  - Example: `--param T0 "[8,512,pow2]"` iterates powers-of-two from 8 up to 512 (8,16,32,...)
  - Notes: `MIN` and `MAX` are integers; if `STEP` is `pow2`, the parser sets `pow2=True` and interprets values as powers of two.

- List (curly braces): ``{v1,v2,v3,...}``
  - Example: `--param T0 "{16,32,64}"` tries exactly the listed values.

- Dynamic parameters (prefix `dyn[...]`): tune a parameter using a search strategy instead of exhaustive enumeration.
  - Minimize form: `dyn[min,INIT]` — create a minimization parameter seeded with `INIT`. The engine will try to reduce the value while keeping correct behavior.
    - Example: `--param V "dyn[min,1024]"`
  - Tile-volume form: `dyn[tpz_vol,INIT,MAX]` — dynamic search over a tile-volume-like parameter, seeded with `INIT` and bounded by `MAX` (MAX optional). The implementation uses a heuristic search (golden-ratio-like narrowing and stability checks).
    - Example: `--param VOL "dyn[tpz_vol,32,10000]"`

Only one dynamic parameter is allowed per experiment. Non-dynamic parameters are enumerated first; the dynamic parameter (if any) is tuned via a targeted search.


**Behavior and measurement**

- The script compiles sources into an executable for each parameter configuration (unless a transformation step produces a precompiled binary).
- Performance measurement: the script runs each configuration and records a kernel execution time. For top candidates it re-runs multiple times to compute a trimmed mean and standard deviation (`--perf-nrun`, `--perf-nmedianrun`).
- Safety/correctness: if `--output-dump-baseline` is provided the script will compile and run the baseline executable with `-DPOLYBENCH_DUMP_ARRAYS` and compare outputs (hash-based) to ensure transformed binaries produce correct results. Incorrect outputs can be saved with `--save-incorrect-sources`.
- OpenMP: the script may set `OMP_SCHEDULE` when `--force-omp-schedule` is specified; otherwise it inherits environment values. Use `--env FILE` to load a file of environment variables before runs.


**Advanced: implementation details**

This section is intended for maintainers and contributors who want to understand or extend the finetune engine. See the source for exact implementation: [tools/finetune.py](tools/finetune.py).

- **Core components**:
  - `FTOptions`: central CLI/options parser and validation. It builds the argument parser and converts CLI arguments into typed runtime configuration used by the experiment engine.
  - `FTParameter` family: representations of experiment parameters.
    - `FTParameterRange`: param defined by a numeric range (min/max/step or pow2).
    - `FTParameterList`: param defined by an explicit list of integers.
    - `FTParameterMinimize`: dynamic parameter for minimization search seeded with an initial value.
    - `FTParameterDynVol`: dynamic tile-volume parameter with init and optional max.
  - `FTParamInstance`: small container holding a parameter name and concrete integer value for a single run.

- **Run model**:
  - `FTRunBuilder`: builder pattern that chains transformations (Pluto, OMP schedule insertion, include insertion, compilation) and finally produces an `FTRun`.
  - `FTRunBuilderTransform` and concrete steps:
    - `FTRunBuilderStepPluto`: runs Pluto with configured tile sizes and flags and produces transformed source files.
    - `FTRunBuilderStepCompile`: compiles sources to an executable using the configured compiler and flags.
    - `FTRunBuilderStepOmpSchedule`: injects or replaces OpenMP schedule pragmas.
    - `FTRunBuilderInsertIncludes`: inserts requested `#include` directives near the top of the file.

- **Execution and measurement**:
  - `FTRun.exec()`: executes the prepared binary with a controlled environment, optional timeout, and captures stdout/stderr and exit codes.
  - `FTExperiment.best_runs`: the experiment keeps a fixed-size sorted list (smallest scores) of best runs. The list stores tuples of `(score, param_instances)`.
  - Reruns: after an initial sweep the top candidates are rerun to compute a trimmed mean and standard deviation; results are sorted by trimmed-mean score.

- **Parameter generation & searching**:
  - `_gen_param_instances()`: enumerates all non-dynamic parameters as a Cartesian product. If there is a dynamic parameter it is appended to the end of the parameter list and handled specially.
  - Exhaustive ranges: `FTParameterRange` produces either arithmetic sequences (by `step`) or powers-of-two sequences when `pow2=True`.
  - Dynamic strategies:
    - `FTParameterMinimize`: binary/halving-style strategy that reduces a numeric value until a lower bound (controlled by `options.minimize_min_step`) is reached, ensuring runs remain valid.
    - `FTParameterDynVol` (tpz_vol): performs an adaptive search around observed minima using a golden-ratio-like narrowing and stability checks to avoid noisy artifacts.
  - Only one dynamic parameter is supported per experiment; mixing multiple dynamic parameters raises an error.

- **Correctness checks and output comparison**:
  - When `--output-dump-baseline` is provided the engine builds and runs the baseline executable with `-DPOLYBENCH_DUMP_ARRAYS` and collects its output (via stderr capture) to compute a hash. Each candidate's output is compared against this baseline hash; mismatches are reported and optionally saved.

- **Extensibility notes**:
  - Add new parameter types by subclassing `FTParameter` and extending `_parse_experiment_param` in `FTOptions` to recognize new syntaxes.
  - Add transformations by implementing `FTRunBuilderTransform.apply()` and adding instances to the `FTRunBuilder` chain.
  - Measurement strategies can be adjusted by tuning `FTOptions.perf_nrun` and `perf_nmedianrun`.


**Where to look in the code**

- Primary implementation: [tools/finetune.py](tools/finetune.py)
  - CLI and parsing: `FTOptions` (method `_init_parser` and `parse`).
  - Parameter parsing: `_parse_experiment_param` (search for `def _parse_experiment_param`).
  - Run builder & transforms: classes beginning with `FTRunBuilder` and `FTRunBuilderStep`.
  - Experiment orchestration: `FTExperiment` methods such as `_gen_param_instances`, `_finetune`, `_finetune_minimize`, `_finetune_dyntpz_vol`, `_measure_perf`, and `_compare_output_with_baseline`.
