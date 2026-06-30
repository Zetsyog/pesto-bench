import argparse
import pprint
import pathlib
import re
import re
import datetime
import subprocess

from dataclasses import dataclass
import sys


@dataclass
class PredOptions:
    benchmark_sizes: str
    target_dataset_size: str
    dataset_data: dict
    predictor: str
    thread_nb: int
    polybench_dir: str
    finetune_script: str
    results_dir: str = "results"
    cc: str = "gcc"
    cc_cflags: str = "-march=native -O3 -fopenmp"
    cc_extraflags: str = ""
    pluto: str = "polycc"
    pluto_flags: str = "--parallel --tile"
    pluto_vec_pragam: str = "#pragma GCC ivdep"
    env: str = "omp16.env"
    force_omp_schedule: str = "static"


PESTO_SRC_SUFFIX = "hybrid"

BENCHMARKS = [
    "datamining/correlation",
    "datamining/covariance",
    "linear-algebra/blas/gemm",
    "linear-algebra/blas/gemver",
    "linear-algebra/blas/gesummv",
    "linear-algebra/blas/symm",
    "linear-algebra/blas/syr2k",
    "linear-algebra/blas/syrk",
    "linear-algebra/blas/trmm",
    "linear-algebra/kernels/2mm",
    "linear-algebra/kernels/3mm",
    "linear-algebra/kernels/atax",
    "linear-algebra/kernels/bicg",
    "linear-algebra/kernels/doitgen",
    "linear-algebra/kernels/mvt",
    "linear-algebra/solvers/cholesky",
    "linear-algebra/solvers/durbin",
    "linear-algebra/solvers/gramschmidt",
    "linear-algebra/solvers/lu",
    "linear-algebra/solvers/ludcmp",
    "linear-algebra/solvers/trisolv",
    "medley/deriche",
    "medley/floyd-warshall",
    "medley/nussinov",
    "stencils/adi",
    "stencils/fdtd-2d",
    "stencils/heat-3d",
    "stencils/jacobi-1d",
    "stencils/jacobi-2d",
    "stencils/seidel-2d",
]


def parse_bench_sizes(path: str):
    file = open(path, "r", encoding="utf-8")
    data = {}

    # skip first line
    header = file.readline()
    datasets = header.split()[4:]

    for line in file:
        split = line.split()
        if len(split) == 0:
            continue
        benchmark = split[0]
        nparams = 0
        for i in range(3, len(split)):
            if split[i].isdigit():
                break
            nparams += 1
        data[benchmark] = {
            "category": split[1],
            "datatype": split[2],
            "params": split[3 : 3 + nparams],
            "dataset_sizes": {},
        }

        for i, dataset in enumerate(datasets):
            data[benchmark]["dataset_sizes"][dataset] = tuple(
                map(
                    int,
                    split[
                        3 + nparams + nparams * i : 3 + nparams + nparams * i + nparams
                    ],
                )
            )
    file.close()

    return data


nest_id_pattern = re.compile(r"^\[(\d+)\]")
perf_pattern = re.compile(r"^\s*.+perf~(\d+)G.*$")
tile_sizes_pattern = re.compile(r"^.*tile\.sizes\s*=\s*([\d,b]+).*$")


def parse_predictor_output(output: str, isAlgebraic: bool):
    lines = output.splitlines()

    # print(output)

    current_nest_id = -1

    best_perf_nest_id = 0
    best_perf = 0

    for line in lines:
        match = nest_id_pattern.match(line)
        if match:
            current_nest_id = int(match.group(1))
            continue
        if current_nest_id == -1:
            continue
        match = perf_pattern.match(line)
        if match:
            perf = int(match.group(1))
            if perf > best_perf:
                best_perf = perf
                best_perf_nest_id = current_nest_id
            continue

    best_tile_sizes = []

    current_nest_id = -1
    for line in lines:
        if nest_id_pattern.match(line):
            current_nest_id = int(nest_id_pattern.match(line).group(1))
            continue
        if current_nest_id == -1:
            continue
        if current_nest_id != best_perf_nest_id:
            continue
        if tile_sizes_pattern.match(line):
            matched_sizes = tile_sizes_pattern.match(line).group(1).split(",")
            # remove any 'b' characters from the matched sizes
            matched_sizes = [size.replace("b", "") for size in matched_sizes]
            best_tile_sizes.append(list(map(int, matched_sizes)))

    return best_tile_sizes


def compute_preferred_tile_sizes(options: PredOptions, benchmark: str) -> list[int]:
    cmd = [sys.executable, options.predictor, "--rect", "-t", str(options.thread_nb)]

    basename = benchmark.split("/")[-1]

    benchmark_data = options.dataset_data[basename]
    sizes = benchmark_data["dataset_sizes"][options.target_dataset_size]

    polybench_dir = pathlib.Path(options.polybench_dir).absolute()
    pluto_src = (
        polybench_dir / benchmark_data["category"] / basename / f"{basename}.pluto.c"
    )

    if not pluto_src.exists():
        print(f"Pluto source file {pluto_src} does not exist.")
        return []

    cmd.append(str(pluto_src))

    for i, param in enumerate(benchmark_data["params"]):
        cmd.append(f"{param}={sizes[i]}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        print(f"Error running predictor for benchmark {benchmark}: {err.decode()}")
        return []

    return parse_predictor_output(out.decode(), isAlgebraic=False)


def compute_preferred_dividers(options: PredOptions, benchmark: str) -> list[int]:
    cmd = [sys.executable, options.predictor, "--alg", "-t", str(options.thread_nb)]

    basename = benchmark.split("/")[-1]

    benchmark_data = options.dataset_data[basename]
    sizes = benchmark_data["dataset_sizes"][options.target_dataset_size]

    polybench_dir = pathlib.Path(options.polybench_dir).absolute()
    pluto_src = (
        polybench_dir / benchmark_data["category"] / basename / f"{basename}.pluto.c"
    )

    if not pluto_src.exists():
        print(f"Pluto source file {pluto_src} does not exist.")
        return []

    cmd.append(str(pluto_src))

    for i, param in enumerate(benchmark_data["params"]):
        cmd.append(f"{param}={sizes[i]}")

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out, err = proc.communicate()
    if proc.returncode != 0:
        print(f"Error running predictor for benchmark {benchmark}: {err.decode()}")
        return []

    return parse_predictor_output(out.decode(), isAlgebraic=True)


def run_finetune_pluto(options: PredOptions, benchmark: str, tile_sizes: list):
    print("finetuning pluto with tile sizes:", tile_sizes)
    cmd = [sys.executable, options.finetune_script]

    pluto_src = (
        pathlib.Path(options.polybench_dir).absolute()
        / benchmark
        / f"{benchmark.split('/')[-1]}.pluto.c"
    )
    polybench_utilities = pathlib.Path(options.polybench_dir).absolute() / "utilities"
    cmd += [
        str(pluto_src),
        str(polybench_utilities / "polybench.c"),
        "-I",
        str(polybench_utilities),
    ]

    cmd += ["--compiler-bin", options.cc]
    cmd += ["--compiler-cflags", options.cc_cflags]
    cmd += ["--compiler-extra-flags", options.cc_extraflags]
    cmd += ["--pluto", options.pluto]
    cmd += ["--pluto-flags", options.pluto_flags]
    cmd += ["--pluto-custom-vec-pragma", options.pluto_vec_pragam]
    cmd += ["--force-omp-schedule", options.force_omp_schedule]
    cmd += ["--env", options.env]

    for i in range(len(tile_sizes[0])):
        values = ""
        for j, tile_size_row in enumerate(tile_sizes):
            if j != 0:
                values += ","
            values += str(tile_size_row[i])
        cmd += ["--param", f"T{i}", f"{{={values}=}}"]

    # create dir with current date and time
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    result_dir = pathlib.Path(options.results_dir).absolute() / "pred_bench" / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")

    log_file = result_dir / f"{timestamp}_{benchmark.split('/')[-1]}_pluto.log"

    cmd += ["--log-file", str(log_file)]

    print(" ".join(cmd))

    proc = subprocess.Popen(cmd)
    proc.communicate()
    if proc.returncode != 0:
        print(f"Error running finetune script for benchmark {benchmark}")
        return


def run_finetune_pesto(options: PredOptions, benchmark: str, dividers: list):
    print("finetuning pesto with dividers:", dividers)
    cmd = [sys.executable, options.finetune_script]

    pesto_src = (
        pathlib.Path(options.polybench_dir).absolute()
        / benchmark
        / f"{benchmark.split('/')[-1]}.{PESTO_SRC_SUFFIX}.c"
    )
    polybench_utilities = pathlib.Path(options.polybench_dir).absolute() / "utilities"
    cmd += [
        str(pesto_src),
        str(polybench_utilities / "polybench.c"),
        "-I",
        str(polybench_utilities),
    ]

    cmd += ["--compiler-bin", options.cc]
    cmd += ["--compiler-cflags", options.cc_cflags]
    cmd += ["--compiler-extra-flags", options.cc_extraflags]
    cmd += ["--env", options.env]

    for i in range(len(dividers[0])):
        values = ""
        for j, divider_row in enumerate(dividers):
            if j != 0:
                values += ","
            values += str(divider_row[i])
        cmd += ["--param", f"T{i}", f"{{={values}=}}"]

    # create dir with current date and time
    timestamp = datetime.datetime.now().strftime("%Y%m%d")
    result_dir = pathlib.Path(options.results_dir).absolute() / "pred_bench" / timestamp
    result_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    log_file = result_dir / f"{timestamp}_{benchmark.split('/')[-1]}_pluto.log"

    cmd += ["--log-file", str(log_file)]

    print(" ".join(cmd))

    proc = subprocess.Popen(cmd)
    proc.communicate()
    if proc.returncode != 0:
        print(f"Error running finetune script for benchmark {benchmark}")
        return


def run_benchmark_single(options: PredOptions, benchmark: str):
    basename = benchmark.split("/")[-1]
    if basename not in options.dataset_data:
        print(f"Benchmark {benchmark} not found in dataset sizes.")
        return

    print(f"Running benchmark {benchmark} with dataset {options.target_dataset_size}")

    # preferred_tile_sizes = compute_preferred_tile_sizes(options, benchmark)
    preferred_dividers = compute_preferred_dividers(options, benchmark)

    # print(f"Preferred tile sizes: {preferred_tile_sizes}")
    print(f"Preferred dividers: {preferred_dividers}")

    # run_finetune_pluto(options, benchmark, preferred_tile_sizes)
    run_finetune_pesto(options, benchmark, preferred_dividers)


def run_benchmarks(options: PredOptions):
    for benchmark in BENCHMARKS:
        basename = benchmark.split("/")[-1]
        if basename not in options.dataset_data:
            print(f"Benchmark {benchmark} not found in dataset sizes.")
            continue
        if (
            options.target_dataset_size
            not in options.dataset_data[basename]["dataset_sizes"]
        ):
            print(
                f"Dataset size {options.target_dataset_size} not found for benchmark {benchmark}."
            )
            continue
        print(
            f"Processing benchmark {benchmark} with dataset {options.target_dataset_size}"
        )
        run_benchmark_single(options, benchmark)


def main():
    parser = argparse.ArgumentParser(
        description="Predict best tile sizes and dividers for multiple benchmarks and then finetune the best tile size among the predicted tile sizes for each benchmark."
    )
    parser.add_argument(
        "--benchmark-sizes",
        type=str,
        help="Path to the input dataset size file.",
        required=True,
    )
    parser.add_argument("--dataset-size", type=str, required=True)
    parser.add_argument(
        "--predictor", type=str, help="Path to the predictor model.", required=True
    )
    parser.add_argument(
        "--polybench-dir",
        type=str,
        help="Path to the PolyBench directory.",
        required=True,
    )
    parser.add_argument(
        "--thread-nb", type=int, help="Number of threads to use.", required=True
    )
    parser.add_argument(
        "--finetune-script",
        type=str,
        help="Path to the finetune script.",
        required=True,
    )
    parser.add_argument(
        "--finetune-args",
        type=str,
        help="Additional arguments to pass to the finetune script.",
        default="",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        help="Directory to store the results.",
        default="results",
    )
    parser.add_argument(
        "--cc",
        type=str,
        help="C compiler to use.",
        default="gcc",
    )
    parser.add_argument(
        "--cc-cflags",
        type=str,
        help="C compiler flags to use.",
        default="-march=native -O3 -fopenmp",
    )
    parser.add_argument(
        "--cc-extraflags",
        type=str,
        help="Additional C compiler flags to use.",
        default="",
    )

    parser.add_argument(
        "--pluto",
        type=str,
        help="Path to the Pluto compiler.",
        default="polycc",
    )
    parser.add_argument(
        "--pluto-flags",
        type=str,
        help="Flags to pass to the Pluto compiler.",
        default="--parallel --tile",
    )
    parser.add_argument(
        "--pluto-custom-vec-pragma",
        type=str,
        help="Custom vectorization pragma to use with Pluto.",
        default="#pragma GCC ivdep",
    )
    parser.add_argument(
        "--env",
        type=str,
        help="Environment file to use.",
        default="omp16.env",
    )
    parser.add_argument(
        "--force-omp-schedule",
        type=str,
        help="Force OpenMP schedule to use.",
        default="static",
    )

    args = parser.parse_args()

    data = parse_bench_sizes(args.benchmark_sizes)
    # pprint.pprint(data)

    options = PredOptions(
        benchmark_sizes=args.benchmark_sizes,
        target_dataset_size=args.dataset_size,
        predictor=args.predictor,
        dataset_data=data,
        thread_nb=args.thread_nb,
        polybench_dir=args.polybench_dir,
        finetune_script=args.finetune_script,
        results_dir=args.results_dir,
        cc=args.cc,
        cc_cflags=args.cc_cflags,
        cc_extraflags=args.cc_extraflags,
        pluto=args.pluto,
        pluto_flags=args.pluto_flags,
        pluto_vec_pragam=args.pluto_custom_vec_pragma,
        env=args.env,
        force_omp_schedule=args.force_omp_schedule,
    )

    run_benchmarks(options)


if __name__ == "__main__":
    main()
