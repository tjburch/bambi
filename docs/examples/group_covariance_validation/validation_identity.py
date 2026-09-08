"""Record the exact source, inputs and statistical contract for a reference run."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[3]
GROUPS = [
    ["subject"],
    ["subject", "condition"],
    ["subject", "context"],
    ["subject", "condition", "context"],
]


def source_identity():
    """Hash source bytes too: a commit alone cannot identify a dirty checkout."""
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    digest = hashlib.sha256()
    paths = sorted(
        list((ROOT / "bambi").rglob("*.py"))
        + list(Path(__file__).parent.glob("*.py"))
        + list(Path(__file__).parent.glob("*.R"))
        + list(Path(__file__).parent.glob("*.stan"))
        + [ROOT / "pyproject.toml"]
    )
    for path in paths:
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return {"source_commit": commit, "source_sha256": digest.hexdigest()}


def create_identity(data_path, mode, design=None, priors=None):
    """Return an engine-neutral contract; exporters must verify their own design."""
    data_path = Path(data_path)
    with data_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("Reference data must contain observations")
    if design is None:
        if mode not in {"ar1", "known", "us-slopes", "us-visits"}:
            raise ValueError("An explicit design and priors are required for this mode")
        levels = sorted({float(row["year"]) for row in rows})
        groups = GROUPS if mode in {"ar1", "known"} else [["subject"]]
        design = {
            "fixed_columns": ["one", "x1", "x2"],
            "likelihood": "binomial_logit",
            "grouping_columns": groups,
            "time_levels": levels,
            "coefficient_columns": ["one", "x1"] if mode == "us-slopes" else levels,
            "row_groups": [[[row[name] for name in group] for group in groups] for row in rows],
            "fixed_matrix": [[1.0, float(row["x1"]), float(row["x2"])] for row in rows],
            "row_times": [float(row["year"]) for row in rows],
            "trials": [int(row["trials"]) for row in rows],
            "response": [int(row["y"]) for row in rows],
        }
    if priors is None:
        priors = {
            "beta": {"distribution": "Normal", "mu": 0, "sigma": 1.5},
            "sd": {"distribution": "HalfNormal", "sigma": 2.5},
            "scale_meaning": "marginal",
        }
        if mode == "ar1":
            priors["rho"] = {
                "distribution": "TruncatedNormal",
                "mu": 0,
                "sigma": 0.5,
                "lower": -1,
                "upper": 1,
            }
        elif mode.startswith("us-"):
            priors["correlation"] = {"distribution": "LKJ", "eta": 2}
        elif mode == "known":
            priors["fixed_rho"] = [0.6, -0.35, 0.2, 0.45]
        else:
            raise ValueError("Explicit priors are required for this mode")
    return {
        "schema_version": 1,
        **source_identity(),
        "mode": mode,
        "data_md5": hashlib.md5(data_path.read_bytes()).hexdigest(),
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "design": design,
        "priors": priors,
    }


def write_identity(data_path, mode, design=None, priors=None):
    identity = create_identity(data_path, mode, design, priors)
    path = Path(data_path).parent / f"identity-{mode}.json"
    with path.open("x") as stream:
        json.dump(identity, stream, indent=2, allow_nan=False)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data", type=Path)
    parser.add_argument("mode", choices=["ar1", "known", "us-slopes", "us-visits"])
    args = parser.parse_args()
    print(write_identity(args.data, args.mode))


if __name__ == "__main__":
    main()
