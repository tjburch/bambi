"""Merge downloaded SBC artifacts without overwriting or dropping case evidence."""

import argparse
import json
from pathlib import Path
import shutil


def merge_campaigns(sources, destination):
    """Check all identities and duplicates before copying any campaign files."""
    if destination.exists():
        raise ValueError("Destination exists; select a separate campaign directory")
    manifests = [
        path
        for source in sources
        for path in source.rglob("manifest.json")
        if (path.parent / "cases").is_dir()
    ]
    if not manifests:
        raise ValueError("No downloaded campaign manifests found")
    identity = json.loads(manifests[0].read_text())
    allowed = {case["id"] for case in identity["cases"]}
    cases = {}
    for path in manifests:
        if json.loads(path.read_text()) != identity:
            raise ValueError("Cannot merge different campaign specifications or environments")
        for case in (path.parent / "cases").iterdir():
            if not case.is_dir() or case.name not in allowed or case.name in cases:
                raise ValueError(f"Unknown or duplicate campaign case: {case.name}")
            if any(item.is_symlink() for item in case.rglob("*")):
                raise ValueError("Campaign artifacts must not contain symbolic links")
            cases[case.name] = case
    destination.mkdir(parents=True)
    shutil.copy2(manifests[0], destination / "manifest.json")
    for name, source in cases.items():
        shutil.copytree(source, destination / "cases" / name)
    return len(cases)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("sources", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        f"Preserved {merge_campaigns(args.sources, args.destination)} cases; run sbc.py check next"
    )


if __name__ == "__main__":
    main()
