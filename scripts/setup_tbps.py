#!/usr/bin/env python3
"""Build the TBPS Lean expression parser."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, check=True, text=True, **kwargs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tbps-root", type=Path, default=REPO_ROOT / "vendor/tbps")
    parser.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts/tbps")
    parser.add_argument("--lean", default=os.environ.get("TBPS_LEAN", "lean"))
    args = parser.parse_args(argv)
    artifacts = args.artifacts.expanduser().resolve()
    artifacts.mkdir(parents=True, exist_ok=True)
    tbps_root = args.tbps_root.expanduser().resolve()
    source = tbps_root / "tbps-be/Lean_tool/Mathlib_Construction.lean"
    if not source.is_file():
        parser.error(f"Missing TBPS checkout: {source}")
    executable = shutil.which(args.lean)
    if executable is None:
        parser.error(f"Lean executable not found: {args.lean}")
    version = run([executable, "--version"], capture_output=True).stdout.strip()
    parser_dir = artifacts / "parser"
    parser_dir.mkdir(exist_ok=True)
    shutil.copyfile(source, parser_dir / source.name)
    run(
        [executable, "-o", "Mathlib_Construction.olean", "Mathlib_Construction.lean"],
        cwd=parser_dir,
    )
    metadata_path = artifacts / "setup.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    metadata.update(
        lean=executable,
        lean_version=version,
        tbps_root=str(tbps_root),
    )
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Built TBPS parser with {version}")
    print(
        f"export TBPS_LEAN={shlex.quote(executable)} TBPS_PARSER_DIR={shlex.quote(str(parser_dir))}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(f"TBPS setup failed: {exc}")
