import sys
import argparse
import bisect
import subprocess
import logging
import shutil
import itertools
import statistics
import hashlib
import os

from pathlib import Path
from collections.abc import Iterator

from typing import Optional, Generic, TypeVar, NamedTuple
import threading

logger = logging.getLogger("finetune")

T = TypeVar("T")


class FixedSizeSortedList(Generic[T]):
    """A fixed-size sorted list that maintains the smallest elements."""

    def __init__(self, max_size: int):
        self.max_size = max_size
        self.data: list[T] = []

    def insert(self, value: T):
        """Insert a value into the sorted list, maintaining the size limit."""
        # Insert the value in sorted order
        bisect.insort(self.data, value, key=lambda x: x[0])
        # Remove the largest value if the size limit is exceeded
        if len(self.data) > self.max_size:
            self.data.pop()  # Remove the last (largest) element

    def __repr__(self):
        return repr(self.data)


class FTParameter:
    """Base class for a benchmarking parameter."""

    def __init__(self, name: str):
        self.name = name


class FTParameterRange(FTParameter):
    """Parameter that has a range of values to search for."""

    def __init__(
        self,
        name: str,
        min_value: int,
        max_value: int,
        pow2: bool = True,
        step: int = 10,
    ):
        super().__init__(name)
        self.min_value = min_value
        self.max_value = max_value
        self.pow2 = pow2
        self.step = step


class FTParameterList(FTParameter):
    """Parameter that has a list of discrete values to search for."""

    def __init__(self, name: str, values: list[int]):
        super().__init__(name)
        self.values = values


class FTParameterMinimize(FTParameter):
    """Parameter that the experiment will try to minimize."""

    def __init__(self, name: str, min_value: int):
        super().__init__(name)
        self.init_value = min_value
        self.step = self.init_value // 10
        if self.step < 1:
            self.step = 1


class FTParamInstance:
    """Instance of a parameter with a specific value.
    This is used to represent a specific configuration of parameters for a run.
    For example, it can represent an explicit parameter value for a FTParameterRange
    """

    def __init__(self, name: str, value: int):
        self.name = name
        self.value = value

    def __repr__(self):
        return f"{self.name}={self.value}"


class FTOptions:
    """Global options for the finetune script."""

    def __init__(self):
        self.parser = self._init_parser()

        self.sources: list[Path] = []
        self.log_output: Optional[Path] = None
        self.outputs: list[Path] = []
        self.pluto_enabled: bool = False
        self.pluto_bin: Optional[Path] = None
        self.pluto_flags: list[str] = ["--tile", "--nounrolljam", "--parallel"]
        self.pluto_custom_vec_pragma: Optional[str] = None
        self.parameters: list[FTParameter] = []
        self.compiler_binary: Path = Path(shutil.which("gcc") or "gcc")
        self.compiler_cflags: list[str] = [
            "-march=native",
            "-O3",
            "-fopenmp",
        ]
        self.compiler_extra_flags: list[str] = []
        self.top_n_runs: int = 20
        self.output_dump_baseline_sources: list[Path] = []
        self.env: dict[str, str] = {}
        self.save_incorrect_sources: Optional[Path] = None
        self.force_omp_schedule: Optional[str] = None
        self.minimize_min_step: int = 500
        self.timeout: int = 0

    def _init_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description="Finetune parameters for a benchmark to find best performance."
        )
        parser.add_argument(
            "sources",
            default=None,
            nargs="*",
            type=Path,
            metavar="PATH",
            help="Path to the source file to be compiled and run.",
        )

        parser.add_argument(
            "--log-file",
            action="store",
            default=None,
            type=Path,
            required=False,
            metavar="LOG_FILE",
            dest="log_output",
            help="Path to the log file where the output will be written.",
        )

        parser.add_argument(
            "-o",
            "--output",
            action="append",
            nargs=1,
            default=[],
            type=Path,
            metavar="PATH",
            dest="outputs",
            help="Path to the output files where the results will be written."
            "Can be specified multiple times. Files can be json or csv.",
        )

        parser.add_argument(
            "--pluto",
            nargs="?",
            const=True,
            default=None,
            type=str,
            metavar="PLUTO_CMD",
            dest="pluto",
            help="Finetuning is done using Pluto.",
        )

        parser.add_argument(
            "--pluto-flags",
            action="extend",
            nargs="+",
            default=[],
            type=str,
            metavar="FLAG",
            dest="pluto_flags",
            help="Additional flags to pass to the Pluto compiler. "
            "Can be specified multiple times.",
        )

        parser.add_argument(
            "--pluto-custom-vec-pragma",
            default=None,
            type=str,
            dest="pluto_custom_vec_pragma",
            metavar="PRAGMA",
            help="Custom vectorization pragma to use with Pluto. "
            "If not specified, the default pragma will be used. ",
        )

        parser.add_argument(
            "--param",
            action="append",
            nargs=2,
            default=[],
            type=str,
            metavar=("NAME", "EXPR"),
            dest="experiment_params",
            help="Experiment parameters that must be finetuned. ",
        )

        parser.add_argument(
            "--param-minimize",
            action="append",
            nargs=2,
            default=[],
            type=str,
            metavar=("NAME", "INIT_VALUE"),
            dest="param_minimize",
            help="Parameters to minimize during the search. ",
        )

        parser.add_argument(
            "--compiler-extra-flags",
            action="extend",
            nargs="+",
            default=[],
            type=str,
            metavar="FLAG",
            dest="compiler_extra_flags",
            help="Extra flags to pass to the compiler when compiling the source file. ",
        )

        parser.add_argument(
            "-I",
            "--include-dir",
            action="extend",
            nargs="+",
            default=[],
            type=Path,
            metavar="PATH",
            dest="include_dirs",
            help="Additional include directories for the compiler. "
            "Can be specified multiple times.",
        )

        parser.add_argument(
            "--output-dump-baseline",
            action="extend",
            nargs="+",
            default=[],
            type=Path,
            metavar="PATH",
            dest="output_dump_baseline",
            help="Path to the the source(s) file(s) that will be used as "
            "a baseline for output comparison. ",
        )

        parser.add_argument(
            "--compiler-bin",
            type=str,
            default=None,
            metavar="BIN",
            dest="compiler_binary",
            help="Path to the compiler binary to use for compilation. "
            "If not specified, the default compiler [gcc] will be used.",
        )

        parser.add_argument(
            "--env",
            type=Path,
            default=None,
            metavar="ENV_FILE",
            dest="env",
            help="Path to a file containing environment variables to load "
            "before running the benchmark. ",
        )

        parser.add_argument(
            "--save-incorrect-sources",
            nargs="?",
            const=True,
            default=None,
            type=Path,
            metavar="DIR",
            dest="save_incorrect_sources",
            help="If specified, the sources that produce incorrect results "
            "will be saved in the given directory. ",
        )

        parser.add_argument(
            "--force-omp-schedule",
            default=False,
            type=str,
            metavar="OMP_SCHEDULE",
            dest="force_omp_schedule",
            help="Force the OpenMP schedule to be used for the benchmark. "
            "This will override the OMP_SCHEDULE environment variable. ",
        )

        parser.add_argument(
            "--timeout",
            default=0,
            type=int,
            metavar="SECONDS",
            dest="timeout",
            help="Timeout in seconds for each run. 0 means no timeout. ",
        )

        return parser

    def _parse_experiment_param(self, name: str, expr: str) -> FTParameter | None:
        # for now we only support range parameters

        # check if string contains exactly one [ and one ]

        if expr.startswith("[") and expr.endswith("]"):
            if not (expr.count("[") == 1 and expr.count("]") == 1):
                return None

            range_part = expr[1:-1].strip()
            # check if the range part contains a comma
            if "," not in range_part:
                return None
            range_parts = range_part.split(",")
            if len(range_parts) < 2 or len(range_parts) > 3:
                return None

            # strip whitespaces
            range_parts = [part.strip() for part in range_parts]
            pow2 = False

            try:
                min_value = int(range_parts[0])
                max_value = int(range_parts[1])
                if min_value > max_value:
                    raise ValueError("min_value must be less than max_value")
                step = 10
                if len(range_parts) == 3:
                    if range_parts[2] == "pow2":
                        step = -1
                        pow2 = True
                    else:
                        step = int(range_parts[2])
                        if step <= 0:
                            raise ValueError("step must be greater than 0")
            except ValueError as e:
                logger.error("Invalid parameter range: %s", e)
                return None

            return FTParameterRange(name, min_value, max_value, pow2, step)
        if expr.startswith("{") and expr.endswith("}"):
            if not (expr.count("{") == 1 and expr.count("}") == 1):
                return None

            list_part = expr[1:-1].strip()
            list_parts = list_part.split(",")
            if len(list_parts) < 1:
                return None

            # strip whitespaces and convert to int
            try:
                values = [int(part.strip()) for part in list_parts]
            except ValueError as e:
                logger.error("Invalid parameter list: %s", e)
                return None

            return FTParameterList(name, values)
        return None

    def parse(self) -> None:
        """Parse the command line arguments and initialize the options."""

        ns = self.parser.parse_args()

        self.sources = ns.sources
        self.log_output = ns.log_output
        self.outputs = ns.outputs

        if not ns.sources or len(self.sources) == 0:
            logger.error(
                "No source files specified. Please provide at least one source."
            )
            sys.exit(1)

        for param in ns.experiment_params:
            name, expr = param

            if any(p.name == name for p in self.parameters):
                logger.error("Parameter '%s' is already defined.", name)
                sys.exit(1)

            parsed_param = self._parse_experiment_param(name, expr)
            if not parsed_param:
                logger.error(
                    "Invalid parameter expression for parameter %s: '%s'", name, expr
                )
                sys.exit(1)
            self.parameters.append(parsed_param)

        if len(ns.param_minimize) > 1:
            logger.error(
                "Only one parameter can be minimized at a time. "
                "Multiple minimizing parameters are not supported yet."
            )
            sys.exit(1)
        for param in ns.param_minimize:
            name, min_value = param
            try:
                min_value = int(min_value)
            except ValueError as e:
                logger.error("Invalid parameter minimize: %s", e)
                sys.exit(1)

            # check if a parameter with the same name already exists
            if any(p.name == name for p in self.parameters):
                logger.error("Parameter '%s' is already defined.", name)
                sys.exit(1)

            self.parameters.append(FTParameterMinimize(name, min_value))

        if ns.include_dirs:
            for include_dir in ns.include_dirs:
                if not include_dir.exists():
                    logger.error("Include directory '%s' does not exist.", include_dir)
                    sys.exit(1)
                self.compiler_extra_flags.extend(["-I", f"{include_dir.absolute()}"])

        if ns.compiler_extra_flags:
            for flag in ns.compiler_extra_flags:
                self.compiler_extra_flags.extend(flag.split())

        self.output_dump_baseline_sources.extend(ns.output_dump_baseline)
        # check all output dump baseline sources exist
        for src in self.output_dump_baseline_sources:
            if not src.exists():
                logger.error(
                    "Output dump baseline '%s' does not exist.",
                    src,
                )
                sys.exit(1)

        if ns.compiler_binary:
            compiler_path = shutil.which(str(ns.compiler_binary))
            if not compiler_path:
                logger.error(
                    "Compiler binary '%s' not found. Please specify a valid compiler binary.",
                    ns.compiler_binary,
                )
                sys.exit(1)
            self.compiler_binary = Path(compiler_path)

        if not self.compiler_binary:
            compiler = shutil.which("gcc")
            if not compiler:
                logger.error(
                    "No compiler binary specified and gcc cannot be found."
                    "Please specify a valid compiler binary."
                )
                sys.exit(1)
            self.compiler_binary = Path(compiler)

        # check if the given option is a valid bash command
        try:
            result = subprocess.run(
                [self.compiler_binary, "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            )
            if result.returncode != 0:
                logger.error(
                    "Compiler '%s' is not valid: %s",
                    self.compiler_binary,
                    result.stderr,
                )
                sys.exit(1)
            logger.info(
                "Using compiler '%s': %s", self.compiler_binary, result.stdout.strip()
            )
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            logger.error(
                "Compiler '%s' is not valid or not found: %s", self.compiler_binary, e
            )
            sys.exit(1)

        # pluto stuff
        self.pluto_enabled = ns.pluto is not None
        if ns.pluto:
            pluto_path = (
                shutil.which("polycc")
                if ns.pluto is True
                else shutil.which(str(ns.pluto))
            )
            if not pluto_path:
                logger.error(
                    "Pluto binary '%s' not found. Please install Pluto or specify a valid path.",
                    ns.pluto,
                )
                sys.exit(1)
            self.pluto_bin = Path(pluto_path)

        if ns.pluto_flags:
            self.pluto_flags = []
            for flag in ns.pluto_flags:
                self.pluto_flags.extend(flag.split())

        if ns.pluto_custom_vec_pragma:
            self.pluto_custom_vec_pragma = ns.pluto_custom_vec_pragma

        if self.pluto_enabled:
            # check if pluto is available
            try:
                result = subprocess.run(
                    [str(self.pluto_bin), "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )
                # result code of polycc --version is 1 so no check for it
                # we could check the output but for now we just check that it runs
            except FileNotFoundError:
                logger.error(
                    "Pluto binary '%s' is not valid. "
                    "Please provide a valid path to the Pluto compiler.",
                    self.pluto_bin,
                )
                sys.exit(1)
            except subprocess.CalledProcessError as e:
                logger.error("Pluto execution '%s' failed: %s", self.pluto_bin, e)
                sys.exit(1)
            logger.info(
                "Using Pluto binary '%s': %s", self.pluto_bin, result.stdout.strip()
            )

        if ns.env:
            if not ns.env.exists():
                logger.error("Environment file '%s' does not exist.", ns.env)
                sys.exit(1)
            with open(ns.env, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    key, _, value = line.partition("=")
                    if not key or not value:
                        logger.error("Invalid environment variable line: '%s'", line)
                        continue
                    self.env[key] = value.strip('"').strip("'")

        if ns.save_incorrect_sources:
            if ns.save_incorrect_sources is True:
                self.save_incorrect_sources = Path("incorrect_sources")
            else:
                self.save_incorrect_sources = Path(ns.save_incorrect_sources)

            if not self.save_incorrect_sources.exists():
                try:
                    self.save_incorrect_sources.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    logger.error(
                        "Failed to create directory '%s' for saving incorrect sources: %s",
                        self.save_incorrect_sources,
                        e,
                    )
                    sys.exit(1)

        if ns.force_omp_schedule:
            if not ns.force_omp_schedule:
                logger.error(
                    "Force OpenMP schedule is specified but no value is provided."
                )
                sys.exit(1)
            self.force_omp_schedule = ns.force_omp_schedule
            # set the OMP_SCHEDULE environment variable to the given value
            self.env["OMP_SCHEDULE"] = str(self.force_omp_schedule)

        self.timeout = ns.timeout

    def dump(self, level: int = logging.INFO) -> None:
        """Dump the parsed options to the logger."""
        logger.log(level, "Parsed options:")
        logger.log(level, "\tSources: %s", self.sources)
        logger.log(level, "\tOutputs: %s", self.outputs)
        logger.log(level, "\tLog output: %s", self.log_output)
        logger.log(level, "\tSave incorrect sources: %s", self.save_incorrect_sources)
        logger.log(level, "\tForce OpenMP schedule: %s", self.force_omp_schedule)

        logger.log(level, "\tCompiler binary: %s", self.compiler_binary)
        logger.log(level, "\tCompiler version:")
        try:
            result = subprocess.run(
                [str(self.compiler_binary.absolute()), "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                for line in result.stdout.splitlines():
                    logger.log(level, "\t\t%s", line.strip())
            else:
                for line in result.stderr.splitlines():
                    logger.error("\t\t%s", line.strip())
        except FileNotFoundError:
            logger.error("Compiler binary '%s' not found.", self.compiler_binary)
        except subprocess.CalledProcessError as e:
            logger.error("Failed to get compiler version: %s", e)
        logger.log(level, "\tCompiler cflags: %s", self.compiler_cflags)
        logger.log(level, "\tCompiler extra flags: %s", self.compiler_extra_flags)
        logger.log(level, "\tParameters:")
        for param in self.parameters:
            if isinstance(param, FTParameterRange):
                logger.log(
                    level,
                    "\t\t%s:\t%s[%d, %d]%s",
                    param.name,
                    "pow2 " if param.pow2 else "",
                    param.min_value,
                    param.max_value,
                    f" step {param.step}" if not param.pow2 else "",
                )
            elif isinstance(param, FTParameterMinimize):
                logger.log(level, "\t\t%s:\t<%d", param.name, param.init_value)
            else:
                logger.log(level, "\t\t%s", param.name)

        logger.log(level, "\tPluto: %s", self.pluto_enabled)
        logger.log(level, "\tPluto binary: %s", self.pluto_bin)
        if self.pluto_enabled:
            assert self.pluto_bin is not None
            logger.log(level, "\tPluto version:")
            try:
                result = subprocess.run(
                    [str(self.pluto_bin.absolute()), "--version"],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                )

                for line in result.stdout.splitlines():
                    logger.log(level, "\t\t%s", line.strip())

            except FileNotFoundError:
                logger.error("Pluto binary '%s' not found.", self.pluto_bin)
            except subprocess.CalledProcessError as e:
                logger.error("Failed to get Pluto version: %s", e)

        logger.log(level, "\tPluto flags: %s", self.pluto_flags)
        logger.log(
            level,
            "\tPluto custom vectorization pragma: %s",
            self.pluto_custom_vec_pragma,
        )
        logger.log(level, "\tTop N runs: %d", self.top_n_runs)
        logger.log(
            level,
            "\tOutput dump baseline sources: %s",
            self.output_dump_baseline_sources,
        )
        logger.log(level, "\tenv:")
        for key, value in self.env.items():
            logger.log(level, "\t\t%s=%s", key, value)


class FTRun:
    """FTRun represents a single run of the benchmark with a given set of parameters.
    It can have a compile step if the benchmark needs to be compiled with the given parameters.
    Other processing steps can be added. Thoses steps are executed for each run. This means that
    those steps should depend on the parameters.
    Else they should be moved to the FTExperiment class.
    """

    def __init__(self, params: tuple[FTParamInstance, ...], command: str):
        self.params: tuple[FTParamInstance, ...] = params

        self.command: str = command

        self.exit_code: Optional[int] = None
        self.kernel_execution_time: Optional[float] = None

        self.stderr_fd: Optional[int] = None

        self.cleanup_files: list[Path] = []
        self.error_message: Optional[str] = None

    def exec(
        self, env: Optional[dict[str, str]] = None, timeout: int | None = None
    ) -> None:
        """Execute the run
        An FTRun is a single run of the benchmark with a given set of parameters
        """
        self.exit_code = None
        self.kernel_execution_time = None

        result = subprocess.Popen(
            [self.command],
            stdout=subprocess.PIPE,
            stderr=self.stderr_fd if self.stderr_fd else subprocess.PIPE,
            text=True,
            env=env,
            restore_signals=False,
        )
        try:
            stdout, _ = result.communicate(timeout=timeout)

            self.exit_code = result.returncode

            # parse the output to extract kernel execution time
            for line in stdout.splitlines():
                try:
                    self.kernel_execution_time = float(line)
                    break
                except ValueError:
                    self.kernel_execution_time = None

        except subprocess.TimeoutExpired:
            self.exit_code = -1
            self.error_message = "Timeout expired"
        except subprocess.SubprocessError as e:
            logger.error("Failed to run binary: %s", e)

    def cleanup(self) -> None:
        """Cleanup the run
        This will remove any file generated by the run, such as output files.
        """
        for file in self.cleanup_files:
            if file.exists():
                try:
                    file.unlink()
                except OSError as e:
                    logger.warning("Failed to remove file '%s': %s", file, e)

    def has_error(self) -> bool:
        """Check if an error occurred during the run."""
        return (
            self.exit_code is not None and self.exit_code != 0
        ) or self.kernel_execution_time is None


class FTTransformSourceBundle(NamedTuple):
    """Bundle of sources and generated files for transformations."""

    sources: list[Path]
    generated_files: list[Path]
    cleanup_files: list[Path]


class FTRunBuilderTransform:
    """Base class for FTRunBuilder steps.
    This is used to define the steps in the builder pattern for FTRun.
    """

    def __init__(self):
        pass

    def apply(
        self, input_bundle: FTTransformSourceBundle
    ) -> FTTransformSourceBundle | None:
        """Apply a transformation on the sources."""
        raise NotImplementedError("Subclasses must implement this method.")


class FTRunBuilderStepCompile(FTRunBuilderTransform):
    """Step for compiling the sources in the FTRunBuilder."""

    def __init__(self, compiler_bin: Path, cflags: list[str], extra_flags: list[str]):
        super().__init__()
        self.compiler_bin = compiler_bin
        self.cflags = cflags
        self.extra_flags = extra_flags

    def _build_compile_cmd(self, sources: list[Path], output: Path) -> list[str]:
        """Build the compile command for the sources."""
        cmd = [str(self.compiler_bin.absolute())]
        cmd.extend(self.cflags)
        cmd.extend([str(s.absolute()) for s in sources])
        cmd.append("-o")
        cmd.append(str(output.absolute()))
        cmd.extend(self.extra_flags)
        return cmd

    def apply(
        self, input_bundle: FTTransformSourceBundle
    ) -> FTTransformSourceBundle | None:
        """Compile the sources with the given compiler and flags."""

        sources = input_bundle.sources
        output_bundle = FTTransformSourceBundle([], [], [])

        output = sources[-1].parent / (sources[-1].stem + ".bin")
        cmd = self._build_compile_cmd(sources, output)
        result = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            restore_signals=False,
        )
        try:
            stdout, stderr = result.communicate()
            if result.returncode != 0:
                logger.error("Compilation failed:\n%s\n%s", stdout, stderr)
                logger.error("Compile command: %s", " ".join(cmd))
                return None
        except subprocess.SubprocessError as e:
            logger.error("Failed to compile binary: %s", e)
            logger.error("Compile command: %s", " ".join(cmd))
            return None
        if not output.exists():
            logger.error("Compilation failed: Binary does not exist.")
            logger.error("Compile command: %s", " ".join(cmd))
            return None

        logger.debug("Compilation succeeded:\n%s", stdout)

        output_bundle.sources.append(output)
        output_bundle.generated_files.append(output)
        output_bundle.cleanup_files.append(output)

        return output_bundle


class FTRunBuilderStepPluto(FTRunBuilderTransform):
    """Step for applying Pluto transformations on the sources in the FTRunBuilder."""

    def __init__(
        self,
        pluto_bin: Path,
        pluto_flags: list[str],
        tile_sizes: Optional[list[int]] = None,
        custom_vec_pragma: Optional[str] = None,
    ):
        super().__init__()
        self._pluto_bin = pluto_bin
        self._pluto_flags = pluto_flags
        self._pluto_tile_sizes: list[int] = tile_sizes if tile_sizes else []
        self._custom_vec_pragma: Optional[str] = custom_vec_pragma

    def apply(self, input_bundle: FTTransformSourceBundle) -> FTTransformSourceBundle:
        output_bundle = FTTransformSourceBundle([], [], [])

        sources = input_bundle.sources
        for src in sources:
            if not src.is_file():
                output_bundle.sources.append(src)
                continue

            # Check if the file contains at least one #pragma scop directive
            has_scop = False
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    if "#pragma scop" in line:
                        has_scop = True
                        break
            if not has_scop:
                # no scop found, no transformation is done
                # keep the source file as is
                output_bundle.sources.append(src)
                continue

            output_file = src.with_suffix(".pluto.c")
            tile_size_file = Path("tile.sizes")

            if tile_size_file.exists():
                tile_size_file.unlink()

            # Apply pluto on this source file
            pluto_cmd: list[str] = [
                str(self._pluto_bin),
                str(src.absolute()),
                "-o",
                str(output_file.absolute()),
            ]

            pluto_cmd.extend(self._pluto_flags)

            if len(self._pluto_tile_sizes) > 0:
                with open(tile_size_file, "w", encoding="utf-8") as f:
                    for size in self._pluto_tile_sizes:
                        f.write(f"{size}\n")

            # call pluto
            res = subprocess.run(
                pluto_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
            )

            if tile_size_file.exists():
                tile_size_file.unlink()

            if res.returncode != 0:
                logger.error(
                    "Failed to call pluto '%s': %s",
                    " ".join(pluto_cmd),
                    res.stdout.strip() + "\n" + res.stderr.strip(),
                )
                continue

            if not output_file.exists():
                logger.error(
                    "Pluto did not generate output file '%s' for source '%s'.",
                    output_file,
                    src,
                )
                continue

            if self._custom_vec_pragma:
                # find the #pragma ivdep and #pragma vector always lines in generated file
                # and replace them with the custom pragma
                with open(output_file, "r+", encoding="utf-8") as f:
                    content = f.read()
                    content = content.replace("#pragma ivdep", "")
                    content = content.replace(
                        "#pragma vector always", self._custom_vec_pragma
                    )
                    f.seek(0)
                    f.write(content)
                    f.truncate()

            output_bundle.generated_files.append(output_file)
            output_bundle.sources.append(output_file)
            output_bundle.cleanup_files.append(output_file)
            output_bundle.cleanup_files.append(output_file.with_suffix(".pluto.cloog"))

            logger.debug(
                "Generated pluto source file '%s' from '%s'.",
                output_file.absolute(),
                src,
            )
        return output_bundle


class FTRunBuilderStepOmpSchedule(FTRunBuilderTransform):
    """Step for applying OpenMP schedule on the sources in the FTRunBuilder."""

    def __init__(self, omp_schedule: str):
        super().__init__()
        self._omp_schedule = omp_schedule

    def apply(
        self, input_bundle: FTTransformSourceBundle
    ) -> FTTransformSourceBundle | None:
        """Apply OpenMP schedule to the sources."""
        output_bundle = FTTransformSourceBundle([], [], [])

        for src in input_bundle.sources:
            if not src.is_file():
                output_bundle.sources.append(src)
                continue

            # Check if the file contains at least one #pragma omp directive
            has_omp = False
            with open(src, "r", encoding="utf-8") as f:
                for line in f:
                    if "#pragma omp" in line:
                        has_omp = True
                        break
            if not has_omp:
                # no OpenMP found, no transformation is done
                # keep the source file as is
                output_bundle.sources.append(src)
                continue

            # update all omp clauses in the source file
            output_file = src.with_suffix(".omp.c")
            with open(src, "r", encoding="utf-8") as fin, open(
                output_file, "w", encoding="utf-8"
            ) as fout:
                for line in fin:
                    if "#pragma omp" in line and "schedule(" not in line:
                        # Insert schedule clause before any closing parenthesis in the pragma
                        idx = line.find(")")
                        if idx != -1:
                            # Insert schedule clause before the closing parenthesis
                            new_line = (
                                line[: idx + 1]
                                + f" schedule({self._omp_schedule})"
                                + line[idx + 1 :]
                            )
                        else:
                            # If no parenthesis, just append schedule clause at the end
                            new_line = (
                                line.rstrip() + f" schedule({self._omp_schedule})\n"
                            )
                        fout.write(new_line)
                    else:
                        fout.write(line)
            output_bundle.generated_files.append(output_file)
            output_bundle.sources.append(output_file)
            output_bundle.cleanup_files.append(output_file)

        return output_bundle


class FTRunBuilder:
    """Builder for FTRun instances.
    This is used to build FTRun instances with a fluent interface.
    """

    def __init__(self):
        self._sources: list[Path] = []

        self._transforms: list[FTRunBuilderTransform] = []

        self._generated_files: list[Path] = []

        self._params: tuple[FTParamInstance, ...] = ()

    def with_params(self, params: tuple[FTParamInstance, ...]) -> "FTRunBuilder":
        """Set the parameters for the run."""
        self._params = params
        return self

    def from_sources(self, sources: list[Path]) -> "FTRunBuilder":
        """Set the sources for the run."""
        if not sources:
            logger.error("No sources provided for the run.")
            raise ValueError("Sources cannot be empty.")
        self._sources = [src.absolute() for src in sources]
        return self

    def pluto(
        self, pluto_bin: Path, pluto_flags: list[str], custom_vec_pragma: Optional[str]
    ) -> "FTRunBuilder":
        """Apply pluto scheduler on the sources."""
        # Retrieve tile sizes from the parameters
        tiling_parameters = [
            p for p in self._params if p.name.startswith("T") and p.name[1:].isdigit()
        ]
        tile_sizes = [32] * len(tiling_parameters)
        # extract all parameters that have a name like "T[0-9]"
        for param in tiling_parameters:
            tile_sizes[int(param.name[1:])] = param.value

        self._transforms.append(
            FTRunBuilderStepPluto(
                pluto_bin=pluto_bin,
                pluto_flags=pluto_flags,
                tile_sizes=tile_sizes,
                custom_vec_pragma=custom_vec_pragma,
            )
        )

        return self

    def compile(
        self,
        compiler_bin: Path,
        cflags: Optional[list[str]],
        extra_flags: Optional[list[str]],
        set_parameters_macro: bool = True,
    ) -> "FTRunBuilder":
        """Indicate that the binary should be compiled from the given sources"""
        _extra_flags = extra_flags if extra_flags else []
        if set_parameters_macro:
            # Add a macro to the extra flags to set the parameters
            _extra_flags.extend(
                [f"-D{param.name}={param.value}" for param in self._params]
            )

        self._transforms.append(
            FTRunBuilderStepCompile(
                compiler_bin=compiler_bin,
                cflags=cflags if cflags else [],
                extra_flags=_extra_flags,
            )
        )
        return self

    def add_transform(self, transform: FTRunBuilderTransform) -> "FTRunBuilder":
        """Add a custom transformation step to the builder."""
        self._transforms.append(transform)
        return self

    def build(self) -> FTRun:
        """Build the FTRun instance."""
        sources = self._sources
        cleanup_files: list[Path] = []

        if not sources or len(sources) == 0:
            logger.error("No sources provided for the run.")
            raise ValueError("Sources cannot be empty.")

        for transform in self._transforms:
            output_bundle = transform.apply(
                FTTransformSourceBundle(
                    sources=sources,
                    generated_files=self._generated_files,
                    cleanup_files=cleanup_files,
                )
            )
            if output_bundle is None:
                logger.error(
                    "Transformation step '%s' failed.", transform.__class__.__name__
                )
                continue
            sources = output_bundle.sources
            cleanup_files.extend(output_bundle.cleanup_files)
            self._generated_files.extend(output_bundle.generated_files)

        if len(sources) != 1:
            logger.error(
                "FTRunBuilder requires exactly one output file after transformations. "
                "Got %d sources: %s",
                len(sources),
                sources,
            )

        ftrun = FTRun(
            params=self._params,
            command=str(sources[0].absolute()),
        )
        ftrun.cleanup_files.extend(cleanup_files)

        return ftrun


class FTExperiment:
    """FTExperiment represents a single experiment that will finetune the benchmark parameters
    and run the benchmark with the given parameters.
    It will also perform output comparison with the baseline to ensure correctness of results.
    """

    def __init__(self):
        self.should_exit: bool = False

        self.best_runs: FixedSizeSortedList[
            tuple[float, tuple[FTParamInstance, ...]]
        ] = FixedSizeSortedList(max_size=20)

        self._run_number: int = 5
        self._median_runs: int = 3
        self.trimmed_mean_results: list[tuple[float, tuple[FTParamInstance, ...]]] = []

    def _gen_param_instances(
        self, param_list: list[FTParameter]
    ) -> Iterator[tuple[FTParamInstance, ...]]:
        """
        Generate all possible parameter instances for the given parameters.
        This will generate a Cartesian product of the parameters.
        """

        # Generate instances for range parameters
        def powers_of_two_in_range(min_value: int, max_value: int):
            """Yield all powers of two in [min_value, max_value] (inclusive)."""
            v = 1
            while v <= max_value:
                if v >= min_value:
                    yield v
                v <<= 1

        param_product_list: list[list[FTParamInstance]] = []

        nonminimize_params = [
            p for p in param_list if not isinstance(p, FTParameterMinimize)
        ]
        for param in nonminimize_params:
            if isinstance(param, FTParameterRange):
                if param.pow2:
                    values = powers_of_two_in_range(param.min_value, param.max_value)
                else:
                    values = range(param.min_value, param.max_value + 1, param.step)
                param_product_list.append(
                    [FTParamInstance(param.name, v) for v in values]
                )
            if isinstance(param, FTParameterList):
                param_product_list.append(
                    [FTParamInstance(param.name, v) for v in param.values]
                )

        minimize_params = [p for p in param_list if isinstance(p, FTParameterMinimize)]
        if len(minimize_params) > 1:
            logger.error(
                "Only one parameter can be minimized at a time. "
                "Multiple minimizing parameters are not supported yet."
            )
            sys.exit(1)
        if len(minimize_params) == 1:
            param_product_list.append(
                [
                    FTParamInstance(
                        minimize_params[0].name, minimize_params[0].init_value
                    )
                ]
            )

        return itertools.product(*param_product_list)

    def _ftrun_from_options(
        self, options: FTOptions, params: tuple[FTParamInstance, ...]
    ) -> FTRun:
        """Create an FTRun instance from the given options."""
        builder = FTRunBuilder().from_sources(options.sources).with_params(params)

        if options.pluto_enabled:
            assert options.pluto_bin is not None
            builder.pluto(
                pluto_bin=options.pluto_bin,
                pluto_flags=options.pluto_flags,
                custom_vec_pragma=options.pluto_custom_vec_pragma,
            )
        if options.force_omp_schedule:
            builder.add_transform(
                FTRunBuilderStepOmpSchedule(options.force_omp_schedule)
            )

        builder.compile(
            compiler_bin=options.compiler_binary,
            cflags=options.compiler_cflags,
            extra_flags=options.compiler_extra_flags,
            set_parameters_macro=True,
        )

        return builder.build()

    def _finetune_minimize(
        self, options: FTOptions, init_param_instance: tuple[FTParamInstance, ...]
    ) -> None:
        if not isinstance(options.parameters[-1], FTParameterMinimize):
            logger.error(
                "Minimize parameter must be the last parameter in the list. Got %s",
                init_param_instance[-1],
            )
            return

        init_value: int = int(options.parameters[-1].init_value)

        value: int = init_value
        step: int = value // 2

        while step > options.minimize_min_step:
            param_instance = init_param_instance[:-1] + (
                FTParamInstance(options.parameters[-1].name, value),
            )

            print(f"{param_instance}")

            ftrun = self._ftrun_from_options(options, param_instance)
            ftrun.exec(env=options.env)
            ftrun.cleanup()

            score_str = ""

            # keep track of the best runs
            if not ftrun.has_error() and ftrun.kernel_execution_time is not None:
                self.best_runs.insert((ftrun.kernel_execution_time, ftrun.params))

            if ftrun.has_error():
                value = value + step
            else:
                value = value - step
                step = step // 2

            # round value to the nearest hundred
            if value >= 100:
                value = (value + 50) // 100 * 100
            # round step to the nearest hundred
            if step >= 100:
                step = (step + 50) // 100 * 100

            # Log the result
            if ftrun.has_error():
                # If the run has an error, we log it as N/A
                score_str += "N/A"
                if ftrun.exit_code is not None:
                    # If there is an exit code, we log it
                    score_str += f" (exit code: {ftrun.exit_code})"
            else:  # valid run with execution time
                # add execution time to log
                score_str += f"{ftrun.kernel_execution_time:.6f}"

            logger.info(
                "\t%s%s",
                " ".join(f"{str(p.value)[:15]:<15}" for p in ftrun.params),
                score_str,
            )

    def _finetune(self, options: FTOptions) -> None:
        """
        Finetune the benchmark with the given options.
        Perform an exhaustive search over the given parameters (essentially tiles sizes)
        """

        param_instance_tuple_list = self._gen_param_instances(options.parameters)

        has_minimize = any(
            isinstance(param, FTParameterMinimize) for param in options.parameters
        )

        param_name_list = [
            p for p in options.parameters if not isinstance(p, FTParameterMinimize)
        ]
        param_name_list.extend(
            [p for p in options.parameters if isinstance(p, FTParameterMinimize)]
        )

        logger.info(
            "\t%s%s",
            " ".join(f"{param.name[:15]:<15}" for param in param_name_list),
            "score",
        )
        for param_instance_tuple in param_instance_tuple_list:
            if self.should_exit:
                logger.warning("Experiment interrupted. Exiting...")
                return

            if has_minimize:
                self._finetune_minimize(options, param_instance_tuple)
                continue

            ftrun = self._ftrun_from_options(options, param_instance_tuple)
            ftrun.exec(
                env=options.env,
                timeout=options.timeout if options.timeout > 0 else None,
            )
            ftrun.cleanup()

            score_str = ""

            # keep track of the best runs
            if not ftrun.has_error() and ftrun.kernel_execution_time is not None:
                self.best_runs.insert((ftrun.kernel_execution_time, ftrun.params))

            # Log the result
            if ftrun.has_error():
                # If the run has an error, we log it as N/A
                score_str += "N/A"
                if ftrun.error_message:
                    score_str += f" ({ftrun.error_message})"
                elif ftrun.exit_code is not None:
                    # If there is an exit code, we log it
                    score_str += f" (exit code: {ftrun.exit_code})"
            else:  # valid run with execution time
                # add execution time to log
                score_str += f"{ftrun.kernel_execution_time:.6f}"

            logger.info(
                "\t%s%s",
                " ".join(f"{str(p.value)[:15]:<15}" for p in ftrun.params),
                score_str,
            )

    def _measure_perf(self, options: FTOptions) -> None:
        """Re-run the best runs multiple times to get a better average performance."""
        logger.info(
            "\t%s%s%s%s%s",
            " ".join(f"{p.name[:15]:<15}" for p in options.parameters),
            f"{'score':<15}",
            f"{'Trimmed Mean':<15}",
            f"{'STD Dev':<15}",
            f"{'STD Dev %':<15}",
        )
        self.trimmed_mean_results.clear()

        for score, param_instances in self.best_runs.data:
            if self.should_exit:
                logger.warning("Experiment interrupted. Exiting...")
                return

            score_list: list[tuple[float, tuple[FTParamInstance, ...]]] = []

            ftrun = self._ftrun_from_options(options, param_instances)

            # run the best runs multiple times to get a better average
            for _ in range(self._run_number):
                ftrun.exec(env=options.env)

                if ftrun.exit_code != 0:
                    logger.error("Run failed with exit code: %d", ftrun.exit_code)
                    break

                if ftrun.kernel_execution_time is None:
                    logger.error("No execution time found in output.")
                    break

                score_list.append((ftrun.kernel_execution_time, ftrun.params))

            ftrun.cleanup()

            if len(score_list) < self._run_number:
                logger.error(
                    "Not enough runs completed. Expected %d, got %d.",
                    self._run_number,
                    len(score_list),
                )
                logger.error("An error may have occurred during the runs. Skipping.")
                continue

            # sort the scores and compute the mean of the median runs
            score_list.sort(key=lambda x: x[0])

            # compute the standard deviation of the scores
            scores = [score[0] for score in score_list]
            stddev = statistics.stdev(scores) if len(scores) > 1 else 0.0

            median_scores = score_list[
                len(score_list) // 2
                - self._median_runs // 2 : len(score_list) // 2
                + self._median_runs // 2
            ]
            mean_score = sum(score[0] for score in median_scores) / len(median_scores)
            self.trimmed_mean_results.append((mean_score, param_instances))

            logger.info(
                "\t%s%s%s%s%s",
                " ".join(f"{p.value:<15}" for p in param_instances),
                f"{score:<15.6f}",
                f"{mean_score:<15.6f}",
                f"{stddev:<15.6f}",
                f"{(stddev / mean_score * 100):<15.2f}",
            )

        # Sort the trimmed mean results by score
        self.trimmed_mean_results.sort(key=lambda x: x[0])

    def _compare_output_with_baseline(self, options: FTOptions) -> None:
        """Compare the output of the runs with the baseline output."""
        if not options.output_dump_baseline_sources:
            logger.error("No baseline output specified for comparison.")
            return

        if len(self.trimmed_mean_results) == 0:
            logger.error("No trimmed mean results available for comparison.")
            return

        # First, we need to run the baseline output to generate the expected output
        baseline_run = (
            FTRunBuilder()
            .from_sources(options.output_dump_baseline_sources)
            .compile(
                options.compiler_binary,
                cflags=options.compiler_cflags,
                extra_flags=options.compiler_extra_flags + ["-DPOLYBENCH_DUMP_ARRAYS"],
            )
            .build()
        )

        # Redirect stderr to a file to capture the output
        stderr_file = open("__stderr.dump", "w", encoding="utf-8")
        baseline_run.stderr_fd = stderr_file.fileno()

        baseline_run.exec()
        # Check if program ran successfully
        if baseline_run.exit_code != 0:
            logger.error(
                "Baseline run failed with exit code: %d", baseline_run.exit_code
            )
            return

        stderr_file.close()

        # Compute the hash of the baseline output
        baseline_hash = None
        with open("__stderr.dump", "rb") as f:
            baseline_hash = hashlib.sha256(f.read()).hexdigest()

        if not baseline_hash:
            logger.error("Failed to compute hash of the baseline output.")
            return

        if os.path.exists("__stderr.dump"):
            os.remove("__stderr.dump")
        baseline_run.cleanup()

        logger.info("Baseline output hash: %s", baseline_hash)

        # Compare the output of the best runs with the baseline output
        # We stop comparing after the first match
        # We want to find the run that has the best performance and matches the baseline output
        for _, param_instances in self.trimmed_mean_results:
            if self.should_exit:
                logger.warning("Experiment interrupted. Exiting...")
                return
            builder = (
                FTRunBuilder()
                .from_sources(options.sources)
                .with_params(param_instances)
            )
            if options.pluto_enabled:
                assert options.pluto_bin is not None
                builder.pluto(
                    pluto_bin=options.pluto_bin,
                    pluto_flags=options.pluto_flags,
                    custom_vec_pragma=options.pluto_custom_vec_pragma,
                )
            builder = builder.compile(
                compiler_bin=options.compiler_binary,
                cflags=options.compiler_cflags,
                extra_flags=options.compiler_extra_flags + ["-DPOLYBENCH_DUMP_ARRAYS"],
            )
            ftrun = builder.build()

            # Redirect stderr to a file to capture the output
            stderr_file = open("__stderr.dump", "w", encoding="utf-8")
            ftrun.stderr_fd = stderr_file.fileno()

            ftrun.exec(env=options.env)

            if ftrun.exit_code != 0:
                logger.error("Run failed with exit code: %d", ftrun.exit_code)
                continue

            stderr_file.close()

            # Check if the output file exists
            if not os.path.exists("__stderr.dump"):
                logger.error(
                    "Run with parameters {%s} did not produce any output.",
                    ", ".join(f"{p.name}={p.value}" for p in param_instances),
                )
                continue

            # Compute the hash of the output
            run_hash = None
            with open("__stderr.dump", "rb") as f:
                run_hash = hashlib.sha256(f.read()).hexdigest()

            if not run_hash:
                logger.error("Failed to compute hash of the optimized run output.")
                continue

            os.remove("__stderr.dump")

            if ftrun.has_error():
                logger.error(
                    "Run with parameters {%s} failed with exit code: %d",
                    ", ".join(f"{p.name}={p.value}" for p in param_instances),
                    ftrun.exit_code,
                )
                logger.error("Weird, this should not happen...")
                ftrun.cleanup()
                continue

            if options.save_incorrect_sources and run_hash != baseline_hash:
                for file in ftrun.cleanup_files:
                    if file.exists():
                        try:
                            # Save the incorrect source file
                            param_instances_str = "_".join(
                                f"{p.name}={p.value}" for p in param_instances
                            )
                            name = f"{file.stem}_({param_instances_str}){file.suffix}"
                            dest = options.save_incorrect_sources / name
                            shutil.copy(file, dest)
                        except OSError as e:
                            logger.error(
                                "Failed to save incorrect source file '%s': %s",
                                file,
                                e,
                            )

            ftrun.cleanup()

            logger.info(
                "Run {%s} output hash: %s",
                ", ".join(f"{p.name}={p.value}" for p in param_instances),
                run_hash,
            )

            if run_hash == baseline_hash:
                logger.info(
                    "Run with parameters {%s} is correct.",
                    ", ".join(f"{p.name}={p.value}" for p in param_instances),
                )
                break

    def start(self, options: FTOptions) -> None:
        """Start the given experiment
        An experiment will first try to finetune the benchmark parameters
        and then run the benchmark with the given parameters.
        Then it will perform output comparison with the baseline to ensure correctness of results.
        This is the main entry point for starting an experiment.

        Args:
            options (FTOptions): _description_
        """

        # do finetuning
        logger.info("Starting finetuning...")
        self._finetune(options)

        if self.should_exit:
            logger.warning("Experiment interrupted. Exiting...")
            return

        # then re-run the best runs but multiple times to get a better average
        # allows to get a better estimate of the performance
        logger.info("Measuring performance of the best runs...")
        self._measure_perf(options)

        if len(self.trimmed_mean_results) > 0:
            best_score, best_params = self.trimmed_mean_results[0]
            logger.info("Best run (trimmed mean):")
            logger.info(
                "\t%s%s",
                " ".join(f"{p.name[:15]:<15}" for p in options.parameters),
                f"{'Trimmed Mean':<15}",
            )
            logger.info(
                "\t%s%s",
                " ".join(f"{str(p.value)[:15]:<15}" for p in best_params),
                f"{best_score:<15.6f}",
            )

        if self.should_exit:
            logger.warning("Experiment interrupted. Exiting...")
            return

        if options.output_dump_baseline_sources:
            logger.info("Comparing output with baseline...")
            self._compare_output_with_baseline(options)


def main():
    """Main entry point for the finetune script."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stderr)],
    )
    options = FTOptions()
    options.parse()
    if options.log_output:
        handler = logging.FileHandler(
            filename=options.log_output, mode="a", encoding="utf-8"
        )
        handler.setLevel(logging.INFO)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logging.getLogger("").addHandler(handler)

    logger.info("Starting finetune script...")

    options.dump()

    experiment = FTExperiment()

    logger.info("Starting experiment...")
    experiment_thread = threading.Thread(target=experiment.start, args=(options,))
    experiment_thread.start()
    try:
        experiment_thread.join()
    except KeyboardInterrupt:
        logger.warning("Experiment interrupted by user. Stopping...")
        experiment.should_exit = True
        experiment_thread.join()

    logger.info("finetune done.")


if __name__ == "__main__":
    main()
