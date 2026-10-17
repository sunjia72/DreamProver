#!/usr/bin/env python3
"""Start the vendored Kimina server using its recorded setup paths."""

import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=10001)
    parser.add_argument(
        "--setup-dir",
        type=Path,
        default=ROOT / "artifacts/kimina-v4.15",
        help="Directory containing the selected toolchain's setup.json",
    )
    parser.add_argument(
        "--workers", type=int, default=2, help="Maximum Lean REPL processes"
    )
    parser.add_argument(
        "--max-memory", default="8G", help="Virtual memory limit per REPL, e.g. 8G"
    )
    parser.add_argument(
        "--lean-threads", type=int, default=2, help="Lean runtime threads per REPL"
    )
    parser.add_argument(
        "--max-repl-uses",
        type=int,
        default=8,
        help="Recycle cached Lean processes after this many uses",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.port <= 65535
        or args.workers < 1
        or args.lean_threads < 1
        or args.max_repl_uses < 1
    ):
        parser.error(
            "--port must be 1..65535; --workers, --lean-threads and --max-repl-uses must be positive"
        )
    if not re.fullmatch(r"[1-9][0-9]*[MG]", args.max_memory):
        parser.error("--max-memory must be a positive limit such as 8G or 8192M")
    metadata_path = args.setup_dir.expanduser().resolve() / "setup.json"
    if not metadata_path.exists():
        parser.error("Run python scripts/setup_kimina.py first")
    metadata = json.loads(metadata_path.read_text())
    env = os.environ.copy()
    env.update(
        DREAMPROVER_REPO=str(ROOT),
        LEAN_SERVER_HOST="127.0.0.1",
        LEAN_SERVER_PORT=str(args.port),
        LEAN_SERVER_MAX_REPLS=str(args.workers),
        LEAN_SERVER_MAX_REPL_MEM=args.max_memory,
        LEAN_SERVER_MAX_REPL_USES=str(args.max_repl_uses),
        LEAN_NUM_THREADS=str(args.lean_threads),
        LEAN_SERVER_LEAN_VERSION=metadata["lean_version"],
        LEAN_SERVER_REPL_PATH=metadata["repl_path"],
        LEAN_SERVER_PROJECT_DIR=metadata["project_dir"],
    )
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")])
    )
    # Popen would orphan the server on Ctrl-C; replace this wrapper instead.
    os.chdir(ROOT / "vendor/kimina-lean-server")
    os.execve(
        sys.executable,
        [sys.executable, "-m", "dreamprover.lean.kimina_server"],
        env,
    )


if __name__ == "__main__":
    main()
