import os
from pathlib import Path
import subprocess

ALL_BENCHMARKS=[
    {"name": "counting", "config": "configuration_files/NPDP/counting.json"},
    {"name": "counting_im", "config": "configuration_files/NPDP/counting_im.json"},
    {"name": "knuth", "config": "configuration_files/NPDP/knuth.json"},
    {"name": "mcc", "config": "configuration_files/NPDP/mcc.json"},
    {"name": "mcm", "config": "configuration_files/NPDP/mcm.json"},
    # {"name": "mea", "config": "configuration_files/NPDP/mea.json"},
    {"name": "nussinov", "config": "configuration_files/NPDP/nussinov.json"},
    {"name": "nw", "config": "configuration_files/NPDP/nw.json"},
    {"name": "sw", "config": "configuration_files/NPDP/sw.json"},
    # {"name": "sw3d", "config": "configuration_files/NPDP/sw3d.json"},
    {"name": "triang", "config": "configuration_files/NPDP/triang.json"},
    # {"name": "zuker", "config": "configuration_files/NPDP/zuker.json"}
]

class Benchmark:
    def __init__(self, name: str, src: Path, config: str, output_src: Path):
        self.name = name
        self.src = src
        self.config = config
        self.output_src = output_src


def check_binary_exists(binary: str) -> bool:
    """Check if a binary exists in the system PATH."""
    from shutil import which
    return which(binary) is not None

def generate_pesto_source(pesto_bin: str,benchmark: Benchmark) -> None:
    """Generate PESTO source code for a given benchmark."""
    print(f"Generating PESTO files for benchmark: {benchmark.name} using config: {benchmark.config}")
    cmd = [pesto_bin, "--config", benchmark.config, benchmark.src.as_posix(), "-o", benchmark.output_src.as_posix()]
    print(f"Running command: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print(result.stderr)

def main() -> None:
    PESTO_BIN = os.getenv("PESTO", "pesto")
    if not check_binary_exists(PESTO_BIN):
        print(f"Error: {PESTO_BIN} is not found in the system PATH.")
        exit(1)
    
    for benchmark in ALL_BENCHMARKS:
        src_path = (Path("NPDP_bench") / benchmark["name"] / f"{benchmark['name']}.c").absolute()
        if not src_path.exists():
            print(f"Warning: Source file {src_path} does not exist. Skipping benchmark {benchmark['name']}.")
            continue
        output_path = (src_path.parent / Path(src_path.stem).with_suffix(".pesto.c")).absolute()
        
        benchmark = Benchmark(
            name=benchmark["name"], 
            src=src_path,
            config=benchmark["config"], 
            output_src=output_path
        )
        generate_pesto_source(PESTO_BIN, benchmark)


if __name__ == "__main__":
    main()