"""Run bounded, serial one-factor-at-a-time graph benchmarks on a hosted runner."""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time


def cases():
    baseline = dict(groups=4, times=4, replicates=1, interaction_depth=1)
    result = [baseline]
    for field, values in (
        ("groups", (16, 64)),
        ("times", (8, 16)),
        ("replicates", (4,)),
        ("interaction_depth", (2, 3)),
    ):
        result.extend({**baseline, field: value} for value in values)
    return [{**case, "sparse": sparse} for case in result for sparse in (False, True)]


def run_case(case, destination, memory_bytes, seconds):
    import psutil

    command = [
        sys.executable,
        str(Path(__file__).with_name("benchmark.py")),
        "--compile",
        "--predict",
        "--output",
        str(destination / "benchmark.json"),
    ]
    for field, value in case.items():
        if field == "sparse":
            if value:
                command.append("--sparse")
        else:
            command.extend(["--" + field.replace("_", "-"), str(value)])
    destination.mkdir(parents=True, exist_ok=False)
    start, peak, failure = time.monotonic(), 0, None
    with (destination / "run.log").open("w") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        root = psutil.Process(process.pid)
        while process.poll() is None:
            try:
                processes = [root, *root.children(recursive=True)]
            except psutil.NoSuchProcess:
                processes = []
            memory = 0
            for child in processes:
                try:
                    memory += child.memory_info().rss
                except psutil.NoSuchProcess:
                    pass
            peak = max(peak, memory)
            if memory > memory_bytes or time.monotonic() - start > seconds:
                failure = "memory_limit" if memory > memory_bytes else "time_limit"
                for child in reversed(processes):
                    try:
                        child.kill()
                    except psutil.NoSuchProcess:
                        pass
                break
            time.sleep(0.1)
        code = process.wait()
    result = dict(
        case=case,
        wall_seconds=time.monotonic() - start,
        sampled_process_tree_peak_rss=peak,
        exit_code=code,
        failure=failure,
    )
    (destination / "resources.json").write_text(json.dumps(result, indent=2))
    if code or failure:
        raise RuntimeError(f"Benchmark failed: {result}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--memory-gib", type=float, default=8)
    parser.add_argument("--case-seconds", type=int, default=300)
    args = parser.parse_args()
    if args.memory_gib <= 0 or args.case_seconds < 1 or args.output.exists():
        parser.error("Use positive limits and a separate output directory")
    args.output.mkdir(parents=True)
    for index, case in enumerate(cases()):
        run_case(
            case, args.output / f"case-{index:02d}", args.memory_gib * 2**30, args.case_seconds
        )


if __name__ == "__main__":
    main()
