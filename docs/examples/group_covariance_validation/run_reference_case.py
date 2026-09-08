"""Run one matched reference pair serially, retaining every completed stage."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from validation_identity import write_identity

DIRECTORY = Path(__file__).resolve().parent


def run(command, output):
    """Keep logs on failure; never retry a statistical failure automatically."""
    with (output / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(command) + "\n")
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["ar1", "known", "us-slopes", "us-visits"])
    parser.add_argument("family", choices=["bernoulli", "binomial"])
    parser.add_argument("output", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--seed", type=int, default=20260908)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Output exists; preserve it and choose another run directory")
    args.output.mkdir(parents=True)
    for variable in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "STAN_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ[variable] = "1"
    fixture = args.output / "fixture"
    run(
        [
            "Rscript",
            str(DIRECTORY / "prepare_reference.R"),
            args.family,
            str(fixture),
            str(args.seed),
        ],
        args.output,
    )
    write_identity(fixture / "data.csv", args.mode)
    if args.smoke:
        phases = [("prior", 1, 10, 10)]
    else:
        phases = [("prior", 1, 100, 100), ("posterior", 4, 1000, 1000)]
    for phase, chains, warmup, draws in phases:
        bambi_output = args.output / f"bambi-{phase}"
        reference_output = args.output / f"reference-{phase}"
        command = [
            sys.executable,
            str(DIRECTORY / "bambi_reference.py"),
            args.mode,
            str(fixture / "data.csv"),
            str(bambi_output),
            phase,
            "--chains",
            str(chains),
            "--warmup",
            str(warmup),
            "--draws",
            str(draws),
            "--seed",
            str(args.seed + 1),
        ]
        if args.mode == "known":
            command += ["--rho", "0.6", "-0.35", "0.2", "0.45"]
        run(command, args.output)
        settings = [phase, str(chains), str(warmup), str(draws), str(args.seed + 2)]
        if args.mode == "ar1":
            run(
                [
                    "Rscript",
                    str(DIRECTORY / "stan_reference.R"),
                    str(DIRECTORY / "four_block_ar1.stan"),
                    str(fixture / "data.json"),
                    str(reference_output),
                    *settings,
                ],
                args.output,
            )
        else:
            run(
                [
                    "Rscript",
                    str(DIRECTORY / "reference.R"),
                    args.mode,
                    str(fixture / "input.rds"),
                    str(reference_output),
                    *settings,
                ],
                args.output,
            )
        if phase == "posterior":
            reference_summary = reference_output / "summary.json"
            run(
                [
                    "Rscript",
                    str(DIRECTORY / "export_summary.R"),
                    "stan" if args.mode == "ar1" else "brms",
                    args.mode,
                    str(reference_output / "fit.rds"),
                    str(fixture / "data.csv"),
                    str(reference_summary),
                ],
                args.output,
            )
            run(
                [
                    sys.executable,
                    str(DIRECTORY / "compare_summaries.py"),
                    str(bambi_output / "summary.json"),
                    str(reference_summary),
                    "--output",
                    str(args.output / "comparison.json"),
                ],
                args.output,
            )
            run(
                [
                    sys.executable,
                    str(DIRECTORY / "posterior_checks.py"),
                    str(bambi_output),
                    str(args.output / "bambi-prior"),
                    str(args.output / "posterior-checks"),
                ],
                args.output,
            )
    (args.output / "status.json").write_text(
        json.dumps(
            {
                "status": "smoke_only" if args.smoke else "reference_comparison_passed",
                "scope": "Joint prediction, SBC, sensitivity and regression gates are separate",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
