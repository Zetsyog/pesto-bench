#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import TextIO
import subprocess
import os


def _write_likwid_init(file: TextIO, nregions: int):
    file.write(
        """
LIKWID_MARKER_INIT;
#pragma omp parallel
{
    LIKWID_MARKER_THREADINIT;
"""
    )
    for i in range(nregions):
        file.write(f'    LIKWID_MARKER_REGISTER("nest_{i}");\n')
    file.write("}")


def _write_likwid_macros(file: TextIO):
    file.write(
        """
#ifdef LIKWID_PERFMON
#include <likwid-marker.h>
#else
#define LIKWID_MARKER_INIT
#define LIKWID_MARKER_THREADINIT
#define LIKWID_MARKER_SWITCH
#define LIKWID_MARKER_REGISTER(regionTag)
#define LIKWID_MARKER_START(regionTag)
#define LIKWID_MARKER_STOP(regionTag)
#define LIKWID_MARKER_CLOSE
#define LIKWID_MARKER_GET(regionTag, nevents, events, time, count)
#include <omp.h>
#endif
               """
    )


def _write_likwid_begin_region(file: TextIO, region_counter: int):
    file.write(
        f"""
#pragma omp parallel
{{
    LIKWID_MARKER_START("nest_{region_counter}");
    #pragma omp for nowait"""
    )


def _write_likwid_end_region(file: TextIO, region_counter: int):
    file.write(
        f"""
    LIKWID_MARKER_STOP("nest_{region_counter}");
}} // end parallel region
LIKWID_MARKER_SWITCH;
"""
    )


def transform_file(input_path: Path, output_path: Path):
    fd = output_path.open("w")

    region_counter = 0
    _write_likwid_macros(fd)
    braces = 0
    with input_path.open("r") as input_fd:
        state = "idle"
        for line in input_fd:
            if state == "idle":
                if line.strip().startswith("#pragma omp parallel for"):
                    state = "copy_omp_clauses"
                    _write_likwid_begin_region(fd, region_counter)
                    pragmas = line.strip().split("#pragma omp parallel for")[1]
                    if pragmas:
                        fd.write(pragmas + "\n")
                else:
                    fd.write(line)
                if "polybench_start_instruments" in line:
                    _write_likwid_init(fd, region_counter)
                if "polybench_stop_instruments" in line:
                    fd.write("LIKWID_MARKER_CLOSE;\n")
            elif state == "copy_omp_clauses":
                if line.strip().startswith("for"):
                    state = "in_for_loop"
                if "{" in line:
                    braces += line.count("{")
                fd.write(line)
            elif state == "in_for_loop":
                if line.strip() == "":
                    continue
                if "{" in line:
                    braces += line.count("{")
                if "}" in line:
                    braces -= line.count("}")
                fd.write(line)
                if braces == 0:
                    state = "idle"
                    _write_likwid_end_region(fd, region_counter)
                    region_counter += 1
    fd.close()


def compile_source(
    source_file: Path,
    output_file: Path,
    cflags: list[str] = [],
    extra_flags: list[str] = [],
):
    cc = os.getenv("CC", "gcc")
    polybench_dir = os.getenv("POLYBENCH_DIR", "../..")
    compile_cmd = (
        [
            cc,
            str(source_file),
            str(Path(polybench_dir) / "utilities" / "polybench.c"),
        ]
        + cflags
        + [
            "-I",
            str(Path(polybench_dir) / "utilities"),
            "-o",
            str(output_file),
            "-llikwid",
        ]
        + extra_flags
    )
    print("Compiling with command:")
    print(" ".join(compile_cmd))
    subprocess.run(compile_cmd, check=True)
    if not output_file.exists():
        raise RuntimeError(f"Compilation failed, {output_file} not found!")


def main():
    parser = argparse.ArgumentParser(
        description="Transform input C file to be monitored by LIKWID."
    )
    parser.add_argument("input_file", help="Path to the input C file")
    parser.add_argument("-o", "--output", help="Path to the output file", default=None)
    parser.add_argument("--cc", help="Compile source file", action="store_true")
    parser.add_argument(
        "--cflags", nargs="*", help="Additional compiler flags", default=[], type=str
    )
    parser.add_argument(
        "--extra-flags",
        nargs="*",
        help="Additional flags for the compiler (e.g., -DEXTRALARGE_DATASET)",
        type=str,
        default=[],
    )

    args = parser.parse_args()

    input_file = Path(args.input_file)
    if not input_file.exists() or not input_file.is_file():
        print(f"Error: The file {input_file} does not exist or is not a file.")
        return

    output_src_file = Path(input_file.stem + "_likwid.c")
    if args.output and not args.cc:
        output_src_file = Path(args.output)

    transform_file(input_file, output_src_file)
    print(f"Transformed file written to {output_src_file}")
    if args.cc:
        output_binary = Path(input_file.stem + "_likwid")
        if args.output:
            output_binary = Path(args.output)
        cflags: list[str] = ["-march=native", "-O3", "-fopenmp"]
        if args.cflags:
            cflags = []
            for flag in args.cflags:
                cflags += flag.split()

        extra_flags: list[str] = ["-I", str(input_file.parent), "-DLIKWID_PERFMON"]
        for flag in args.extra_flags:
            extra_flags += flag.split()

        compile_source(
            output_src_file, output_binary, cflags=cflags, extra_flags=extra_flags
        )


if __name__ == "__main__":
    main()
