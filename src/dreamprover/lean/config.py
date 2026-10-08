"""Default paths for the built Lean project and REPL."""
from pathlib import Path
import os

from dreamprover.paths import repository_root

REPO_ROOT = repository_root()
LEAN_PROJECT_ROOT = Path(os.environ.get(
    "LEAN_PROJECT_ROOT", REPO_ROOT / "vendor/kimina-lean-server/mathlib4-v4.15.0"
)).expanduser().resolve()
REPL_EXECUTABLE = os.environ.get("REPL_EXECUTABLE", os.environ.get(
    "REPL_DIR", str(REPO_ROOT / "vendor/kimina-lean-server/repl-v4.15.0/.lake/build/bin/repl")
))
