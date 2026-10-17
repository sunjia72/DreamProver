"""Run the vendored TBPS backend in the foreground with configurable dependencies."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from dreamprover.paths import repository_root


REPO_ROOT = repository_root()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tbps-root",
        type=Path,
        default=os.environ.get("TBPS_DIR", REPO_ROOT / "vendor/tbps"),
    )
    parser.add_argument("--host", default=os.environ.get("TBPS_HOST", "127.0.0.1"))
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("TBPS_PORT", "8000"))
    )
    parser.add_argument(
        "--lean",
        default=os.environ.get("TBPS_LEAN"),
        help="Lean executable matching the parser .olean (defaults to setup metadata)",
    )
    parser.add_argument(
        "--parser-dir",
        type=Path,
        default=os.environ.get("TBPS_PARSER_DIR", REPO_ROOT / "artifacts/tbps/parser"),
    )
    parser.add_argument(
        "--lean-path",
        default=os.environ.get("TBPS_LEAN_PATH", ""),
        help="Additional Lean module paths, including Mathlib dependencies if needed",
    )
    parser.add_argument(
        "--lean-import",
        action="append",
        default=[],
        help="Additional module to import (repeatable, e.g. Mathlib)",
    )
    parser.add_argument(
        "--url-file", type=Path, help="Optionally write the advertised URL here"
    )
    args = parser.parse_args(argv)
    root = args.tbps_root.expanduser().resolve()
    backend = root / "tbps-be" if (root / "tbps-be").is_dir() else root
    if not (backend / "base_server.py").is_file():
        parser.error(f"Not a TBPS backend: {backend}")
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    parser_dir = args.parser_dir.expanduser().resolve()
    if not (parser_dir / "Mathlib_Construction.olean").is_file():
        parser.error(
            f"Parser module missing in {parser_dir}; run scripts/setup_tbps.py"
        )
    metadata_path = parser_dir.parent / "setup.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    lean = args.lean or metadata.get("lean", "lean")
    env = os.environ.copy()
    env.update(
        DREAMPROVER_REPO=str(REPO_ROOT),
        TBPS_BACKEND_DIR=str(backend),
        TBPS_PARSER_DIR=str(parser_dir),
        TBPS_LEAN=lean,
        TBPS_LEAN_PATH=args.lean_path,
    )
    if args.lean_import:
        env["TBPS_LEAN_IMPORTS"] = ",".join(args.lean_import)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(REPO_ROOT / "src"), env.get("PYTHONPATH")])
    )
    url = f"http://{'127.0.0.1' if args.host in {'0.0.0.0', '::'} else args.host}:{args.port}"
    if args.url_file:
        args.url_file.parent.mkdir(parents=True, exist_ok=True)
        args.url_file.write_text(url + "\n", encoding="utf-8")
    print(f"Starting TBPS at {url}; set TBPS_URL={url} for clients", flush=True)
    # Each upstream query already forks workers. One web worker avoids multiplying
    # that pool by the machine's CPU count.
    command = [
        sys.executable,
        "-m",
        "uvicorn",
        "dreamprover.lean.tbps_app:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--workers",
        "1",
    ]
    os.chdir(backend)
    # Replace this wrapper so terminating it also terminates the foreground
    # server, including when a scheduler sends SIGTERM to just the launcher.
    os.execve(sys.executable, command, env)
    return 0  # os.execve only returns by raising an error.


if __name__ == "__main__":
    raise SystemExit(main())
