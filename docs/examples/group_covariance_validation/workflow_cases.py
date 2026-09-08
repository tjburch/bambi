"""Select a bounded manual CI batch without starting any numerical work."""

import argparse
import json
from pathlib import Path


def select_cases(suite, start=0, count=10):
    if isinstance(start, bool) or isinstance(count, bool) or start < 0 or not 1 <= count <= 10:
        raise ValueError("Use a nonnegative start and a batch size from 1 to 10")
    families = ("bernoulli", "binomial")
    modes = ("ar1", "known", "us-slopes", "us-visits")
    if suite in {"references", "compile-smoke"}:
        cases = [
            dict(mode=mode, family=family, id=f"{mode}-{family}")
            for mode in modes
            for family in families
        ]
    elif suite == "single-block":
        cases = [
            dict(mode=mode, family=family, id=f"{mode}-{family}")
            for mode in ("ar1", "ou", "cs", "toep", "us")
            for family in families
        ]
    elif suite == "samplers":
        cases = [
            dict(mode=mode, family="binomial", id=mode)
            for mode in ("pymc", "nutpie", "numpyro", "blackjax")
        ]
    elif suite == "benchmarks":
        cases = [dict(mode="benchmarks", family="binomial", id="benchmarks")]
    elif suite == "heldout":
        cases = [
            dict(mode=mode, family=family, id=f"{mode}-{family}")
            for mode in ("subject", "future", "three-way")
            for family in families
        ]
    elif suite in {"sbc-pilot", "sbc-full"}:
        replicates = 5 if suite == "sbc-pilot" else 50
        cases = [
            dict(mode=mode, family=family, id=f"{mode}-{family}-{replicate:03d}")
            for mode in ("ar1", "ou", "cs", "toep", "us", "four-block-ar1")
            for family in families
            for replicate in range(replicates)
        ]
    else:
        raise ValueError("Unknown validation suite")
    if start >= len(cases):
        raise ValueError("Batch start is outside the campaign")
    return {"include": cases[start : start + count]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("suite")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    encoded = json.dumps(select_cases(args.suite, args.start, args.count))
    print(encoded)
    if args.github_output:
        with args.github_output.open("a") as stream:
            stream.write(f"matrix={encoded}\n")


if __name__ == "__main__":
    main()
