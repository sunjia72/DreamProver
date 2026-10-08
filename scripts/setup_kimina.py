#!/usr/bin/env python3
"""Prepare the vendored Kimina server and a matching Lean REPL.

The default builds the Lean 4.15 REPL. --mathlib also fetches Mathlib and its
cache inside the Kimina checkout for imported proofs.
Elan/lake must already be on PATH; no global default toolchain is changed.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(command, *, cwd=ROOT):
    subprocess.run(command, cwd=cwd, check=True)


def checkout(destination, url, version):
    if not destination.exists():
        run(
            ["git", "clone", "--depth", "1", "--branch", version, url, str(destination)]
        )
    else:
        toolchain = (destination / "lean-toolchain").read_text().strip()
        if toolchain != f"leanprover/lean4:{version}":
            raise SystemExit(
                f"{destination} uses {toolchain}; requested {version}. Existing checkout preserved."
            )
    return subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean-version", default="v4.15.0")
    parser.add_argument("--mathlib", action="store_true")
    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=ROOT / "artifacts/kimina-v4.15",
        help="Directory for setup state; use separate directories for concurrent toolchains",
    )
    parser.add_argument(
        "--skip-python",
        action="store_true",
        help="Skip Python dependency installation and Prisma generation",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"v[0-9]+\.[0-9]+\.[0-9]+(?:-rc[0-9]+)?", args.lean_version):
        parser.error("--lean-version must be a release such as v4.15.0")
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    run([sys.executable, str(ROOT / "scripts/setup_dependencies.py"), "kimina"])
    server = ROOT / "vendor/kimina-lean-server"
    repl = server / (
        "repl" if args.lean_version == "v4.33.0" else f"repl-{args.lean_version}"
    )
    repl_revision = checkout(
        repl, "https://github.com/leanprover-community/repl.git", args.lean_version
    )
    run(["lake", "build"], cwd=repl)
    if args.mathlib:
        project = server / f"mathlib4-{args.lean_version}"
        mathlib_revision = checkout(
            project,
            "https://github.com/leanprover-community/mathlib4.git",
            args.lean_version,
        )
        run(["lake", "exe", "cache", "get"], cwd=project)
    else:
        project = repl
        mathlib_revision = None
    if not args.skip_python:
        run([sys.executable, "-m", "pip", "install", "-e", str(server) + "[server]"])
        run(
            [
                sys.executable,
                "-m",
                "prisma",
                "generate",
                "--schema",
                str(server / "prisma/schema.prisma"),
            ],
            cwd=server,
        )
    metadata = {
        "lean_version": args.lean_version,
        "repl_revision": repl_revision,
        "mathlib_revision": mathlib_revision,
        "project_dir": str(project),
        "repl_path": str(repl / ".lake/build/bin/repl"),
    }
    directory = args.metadata_dir.expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "setup.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
