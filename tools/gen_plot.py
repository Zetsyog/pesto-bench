#!/usr/bin/env python3

"""Script to parse experiment log files and extract data for plotting."""

import argparse
import pathlib
from typing import TextIO, Any
from typing import TypedDict, List, Dict, Literal, Union
import re
import logging
import sys
import seaborn as sns
import pandas as pd
from pprint import pprint
import json

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stderr)],
)
logger = logging.getLogger(__name__)

DatasetSize = Literal[
    "MINI", "SMALL", "MEDIUM", "LARGE", "EXTRALARGE", "XL2", "XL3", "XL4", "XL5", "XL6"
]


class BenchmarkEntry(TypedDict):
    category: str
    datatype: str
    params: List[str]
    dataset_sizes: Dict[DatasetSize, List[int]]
    flops: Dict[DatasetSize, int]


PolybenchData = Dict[str, BenchmarkEntry]


class PlotOptions:
    """Holds options for plotting."""

    def __init__(self):
        self.input_files: list[pathlib.Path] = []
        self.metric: str = "time"
        self.polybench_data: PolybenchData | None = None
        self.polybench_dataset: str | None = None
        self.ylogscale: bool = False


class PlotData:
    """Holds all parsed experiment data."""

    def __init__(self):
        self.experiments: list[PlotExperiment] = []

    def global_dict(self) -> list:
        """Convert the data to a dictionary format."""
        return [
            {
                "name": exp.name,
                "sources": exp.sources,
                "parameters": exp.parameters,
                "best_scores": [
                    {
                        "param_instances": run.param_instances,
                        "score": run.score,
                        "stddev": run.stddev,
                        "stddev_percent": run.stddev_percent,
                    }
                    for run in exp.best_scores
                ],
                "best_score": exp.best_scores[0] if exp.best_scores else None,
            }
            for exp in self.experiments
        ]

    def global_def(self) -> pd.DataFrame:
        """Convert the data to a pandas DataFrame."""
        data = []
        for exp in self.experiments:
            dexp = {
                "name": exp.name,
                "sources": exp.sources,
                "benchmark_name": exp.benchmark_name,
                "tags": ":".join(exp.tags),
                "parameters_names": exp.parameters,
                "best_score": exp.best_scores[0].score if exp.best_scores else None,
                "Mflops": exp.best_scores[0].flops if exp.best_scores else None,
            }
            for instance in exp.best_scores[0].param_instances:
                dexp[instance[0]] = instance[1]
            data.append(dexp)
        return pd.DataFrame(data)


class PlotExperiment:
    """Represents an experiment with its parameters and best scores."""

    def __init__(self):
        self.name: str = "anonymous"
        self.sources: list[str] = []
        self.parameters: list[str] = []
        self.best_scores: list[ExpRun] = []
        self.env: dict[str, str] = {}
        self.compiler_version: str = "unknown"
        self.compiler_extra_flags: str = ""
        self.original_sources: list[str] = []
        self.tags: list[str] = []
        self.benchmark_name: str = "unknown"
        self.top_n_runs: int = 0
        self.has_error: bool = False
        self.dataset: str = "unknown"

    def __str__(self):
        tags = ":".join(self.tags)
        sources = ", ".join([pathlib.Path(src).name for src in self.sources])
        return (
            f"Experiment(name={self.name}, "
            f"benchmark_name={self.benchmark_name}, "
            f"tags=[{tags}], "
            f"sources=[{sources}], "
            f"parameters={self.parameters}, "
            f"best_score={self.best_scores[0].score if self.best_scores else 'N/A'}, "
            f"compiler_version={self.compiler_version},"
            f"env={self.env}"
            ")"
        )

    def has_same_context(self, other: "PlotExperiment") -> bool:
        """Check if two experiments have the same context"""
        if set(self.parameters) != set(other.parameters):
            return False

        if set(self.sources) != set(other.sources):
            return False

        return True

    def log(self, level: int = logging.INFO):
        """Log the experiment details."""
        logger.log(level, "Experiment: %s", self.name)
        logger.log(level, "Sources: %s", self.sources)
        logger.log(level, "Parameters: %s", self.parameters)
        logger.log(level, "Best Scores:")
        logger.log(
            level,
            "\t%s",
            f"".join(
                [f"{p:<20}" for p in self.parameters + ["score", "stddev", "stddev%"]]
            ),
        )
        for run in self.best_scores:
            logger.log(
                level,
                "\t%s",
                f"".join(
                    [f"{v[1]:<20}" for v in run.param_instances]
                    + [
                        f"{run.score:<20.4f}",
                        f"{run.stddev:<20.4f}",
                        f"{run.stddev_percent:<20.2f}",
                    ]
                ),
            )


class ExpRun:
    """Represents a single run of an experiment with specific parameter settings."""

    def __init__(self):
        self.param_instances: list[tuple[str, str]] = []
        # score is the primary metric
        # it should be the meand value of the metric over multiple runs
        self.score: float = 99999.0  # lower is better
        self.single_score: float = 99999.0  # score of a single run, should be useless
        self.stddev: float = 0.0
        self.stddev_percent: float = 0.0
        self.flops: float = 0  # number of floating point operations, if applicable


class ExpParser:
    """Class to parse input files and extract experiment data."""

    def __init__(self, options: PlotOptions):
        self.options = options

    def _parse_options(self, exp: PlotExperiment, fp: TextIO):
        options_dict: dict[str, Any] = {}
        current_key = None
        in_options = False
        for raw_line in fp:
            if "Parsed options:" in raw_line:
                in_options = True
                continue
            if in_options and "Starting experiment" in raw_line:
                break  # End of options section
            if not in_options:
                continue

            line_match = re.match(
                r"^[\d]+-[\d]+-[\d]+ [\d]+:[\d]+:[\d]+ \[.*\] finetune: [\s](?P<line>.*)$",
                raw_line,
            )

            if not line_match and in_options:
                break
            assert line_match is not None
            clean_line = line_match.group("line")

            # Detect key: value lines
            match = re.match(r"(?P<key>[\S][^:]+):\s*(?P<value>.*)$", clean_line)
            if match:
                key = match.group("key").strip()
                value = match.group("value").strip()
                options_dict[key] = [] if value == "" else value
                current_key = key
                continue

            # Continuation (indented lines)
            if current_key and clean_line.startswith((" ", "\t")):
                val = clean_line.strip()
                if isinstance(options_dict[current_key], list):
                    options_dict[current_key].append(val)
                else:
                    # Convert string -> list on first continuation
                    options_dict[current_key] = (
                        [options_dict[current_key], val]
                        if options_dict[current_key]
                        else [val]
                    )

        # pprint(options_dict)

        # Post-process options_dict to fill in the PlotExperiment fields
        if (
            "Compiler version" in options_dict
            and len(options_dict["Compiler version"]) > 0
        ):
            exp.compiler_version = options_dict["Compiler version"][0]
        if (
            "Compile extra flags" in options_dict
            or "Compiler extra flags" in options_dict
        ):
            flags_key = (
                "Compile extra flags"
                if "Compile extra flags" in options_dict
                else "Compiler extra flags"
            )
            exp.compiler_extra_flags = json.loads(
                options_dict[flags_key].replace("'", '"')
            )

        if "Sources" in options_dict:
            if (
                options_dict["Sources"].count("[") == 1
                and options_dict["Sources"].count("]") == 1
            ):
                sources_str = options_dict["Sources"][1:-1]  # remove brackets
                # remove PosixPath(...) wrappers
                sources_str = re.sub(r"PosixPath\((.*?)\)", r"\1", sources_str)
                exp.sources = [
                    s.strip().strip('"').strip("'") for s in sources_str.split(",")
                ]
        if "Top N runs" in options_dict:
            try:
                exp.top_n_runs = int(options_dict["Top N runs"])
            except ValueError:
                exp.top_n_runs = 0
                logger.warning(
                    "Could not parse Top N runs: %s", options_dict["Top N runs"]
                )
        if "Parameters" in options_dict:
            exp.parameters = [
                p.split(":")[0].strip() for p in options_dict["Parameters"] if ":" in p
            ]
        if "env" in options_dict:
            for env_line in options_dict["env"]:
                if "=" in env_line:
                    k, v = env_line.split("=", 1)
                    exp.env[k.strip()] = v.strip()
        if "Output dump baseline sources" in options_dict:
            if (
                options_dict["Output dump baseline sources"].count("[") == 1
                and options_dict["Output dump baseline sources"].count("]") == 1
            ):
                sources_str = options_dict["Output dump baseline sources"][1:-1]
                sources_str = re.sub(r"PosixPath\((.*?)\)", r"\1", sources_str)
                exp.original_sources = [
                    s.strip().strip('"').strip("'") for s in sources_str.split(",")
                ]
        if "Pluto" in options_dict:
            if options_dict["Pluto"] == "True":
                exp.tags.append("pluto")

    def _parse_best_score(self, exp: PlotExperiment, fp: TextIO):
        for line in fp:
            if "Measuring performance of the best runs" in line:
                break
        try:
            next(fp)  # skip header line
        except StopIteration:
            exp.has_error = True
            return

        for line in fp:
            data = line.split("finetune:")[1].strip().split()
            if len(exp.best_scores) >= exp.top_n_runs:
                break  # reached the end of the best runs
            if len(data) < len(exp.parameters) + 4:
                logger.warning("Not enough data in line: %s", line)
                logger.warning("End parsing best scores.")
                break
            try:
                run = ExpRun()
                run.single_score = float(data[len(exp.parameters) + 0])
                run.score = float(data[len(exp.parameters) + 1])
                run.stddev = float(data[len(exp.parameters) + 2])
                run.stddev_percent = float(data[len(exp.parameters) + 3].strip("%"))
                for i, param in enumerate(exp.parameters):
                    run.param_instances.append((param, data[i]))
                exp.best_scores.append(run)
            except (ValueError, IndexError):
                logger.warning("Could not parse line: %s", line)
                logger.warning("Skipping.")
                continue

    def _guess_name(self, exp: PlotExperiment):
        for src in exp.original_sources:
            if "polybench.c" in src:
                continue

            exp.benchmark_name = pathlib.Path(src).stem
            break

        for src in exp.sources:
            if "polybench.c" in src:
                continue

            exp.name = pathlib.Path(src).stem

    def _guess_tags(self, exp: PlotExperiment):
        if "pesto" in exp.name.lower():
            exp.tags.append("pesto")

        if "OMP_SCHEDULE" in exp.env:
            exp.tags.append(exp.env["OMP_SCHEDULE"].lower())

        if "tpz" in exp.name.lower():
            exp.tags.extend(["pesto", "tpz"])

        if "pesto" in exp.name.lower():
            exp.tags.extend(["pesto"])

    def _compute_flops(self, exp: PlotExperiment) -> None:
        assert self.options.polybench_data is not None
        assert self.options.polybench_dataset is not None

        if exp.benchmark_name not in self.options.polybench_data:
            logger.warning(
                "Benchmark %s not found in polybench data", exp.benchmark_name
            )
            return
        benchmark_entry = self.options.polybench_data[exp.benchmark_name]
        if self.options.polybench_dataset not in benchmark_entry["flops"]:
            logger.warning(
                "Dataset %s not found for benchmark %s",
                self.options.polybench_dataset,
                exp.benchmark_name,
            )
            return
        flops = benchmark_entry["flops"][self.options.polybench_dataset]

        for run in exp.best_scores:
            run.flops = (flops / run.score) / 10**6  # Mflops per second

    def _guess_dataset(self, exp: PlotExperiment):
        for flag in exp.compiler_extra_flags:
            if flag.startswith("-D") and flag.endswith("_DATASET"):
                # remove the -D prefix
                exp.dataset = flag[2:-8]

    def parse(self, path: pathlib.Path) -> PlotExperiment:
        """Parse a single input file."""
        exp = PlotExperiment()
        with path.open("r", encoding="utf-8") as file:
            self._parse_options(exp, file)
            self._parse_best_score(exp, file)

        self._guess_name(exp)
        self._guess_tags(exp)
        self._guess_dataset(exp)

        if self.options.metric == "flops":
            self._compute_flops(exp)
        logger.info("Parsed experiment: %s", exp)
        exp.best_scores.sort(key=lambda x: x.score)

        return exp


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Process input files.")
    parser.add_argument(
        "input_file", nargs="+", help="Input file(s)", type=pathlib.Path
    )

    parser.add_argument(
        "--pb-data",
        nargs=1,
        help="Path to polybench dataset JSON database",
        type=pathlib.Path,
        dest="polybench_db",
        default=None,
    )

    parser.add_argument(
        "--pb-dataset",
        nargs=1,
        help="Name of the polybench dataset to use",
        type=str,
        dest="polybench_dataset",
    )

    parser.add_argument(
        "--flops-scale",
        action="store_true",
        help="Scale the scores by the number of FLOPS in the benchmark",
        dest="flops_scale",
        default=False,
    )

    parser.add_argument(
        "--logscale",
        action="store_true",
        help="Use logarithmic scale for the y-axis",
        dest="ylogscale",
        default=False,
    )

    return parser.parse_args()


def options_from_args(args: argparse.Namespace) -> PlotOptions:
    """Convert parsed arguments to PlotOptions."""
    options = PlotOptions()
    options.input_files = args.input_file

    if args.polybench_db and len(args.polybench_db) == 1:
        options.polybench_data = json.loads(
            args.polybench_db[0].read_text(encoding="utf-8")
        )
    if args.polybench_dataset and len(args.polybench_dataset) == 1:
        options.polybench_dataset = args.polybench_dataset[0]
    if args.flops_scale:
        options.metric = "flops"
    if args.ylogscale:
        options.ylogscale = True
    return options


def main():
    """Main function to execute the script logic."""

    logger.info("Starting the plot generation script.")

    options = options_from_args(parse_args())

    data = PlotData()

    data_parser = ExpParser(options)
    for input_file in options.input_files:
        logger.info("parsing %s", input_file)
        exp = data_parser.parse(pathlib.Path(input_file))
        if not exp.has_error:
            data.experiments.append(exp)

    logger.info("Finished processing all input files.")

    df = data.global_def()
    print(df)

    sns.set_theme()
    ydata = "best_score"
    if options.metric == "flops":
        ydata = "Mflops"
    sns.barplot(data=df, x="benchmark_name", y=ydata, hue="tags")

    import matplotlib.pyplot as plt

    plt.xticks(rotation=45)
    if options.ylogscale:
        plt.yscale("log")
    plt.minorticks_on()
    plt.grid(which="minor", linestyle=":", linewidth="0.5", color="gray", axis="y")
    plt.grid(which="major", linestyle="-", linewidth="0.75", color="black", axis="y")
    plt.show()


if __name__ == "__main__":
    main()
