#!/usr/bin/env python3
"""Fetch pinned dependency repositories inside DreamProver, preserving existing trees."""

import argparse
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help="Dependencies to fetch: tbps and/or kimina (default: both)")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "dependencies.lock.json").read_text())
    names = args.names or ["tbps", "kimina"]
    for name in names:
        if name not in manifest:
            parser.error(f"Unknown dependency {name!r}; choose tbps or kimina")
        entry = manifest[name]
        destination = ROOT / entry["path"]
        if destination.exists():
            if not (destination / ".git").exists():
                raise SystemExit(f"{destination} already exists without .git; move it before setup")
            revision = subprocess.check_output(["git", "-C", str(destination), "rev-parse", "HEAD"], text=True).strip()
            if revision != entry["revision"]:
                raise SystemExit(f"{destination}: expected {entry['revision']}, found {revision}; existing checkout preserved")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "clone", "--no-checkout", entry["url"], str(destination)], check=True)
            subprocess.run(["git", "-C", str(destination), "checkout", "--detach", entry["revision"]], check=True)
        print(f"{name}: {entry['revision']} at {destination}")


if __name__ == "__main__":
    main()
