"""Validate completed learning checkpoints before reusing their libraries."""
from __future__ import annotations

import json
from pathlib import Path

from dreamprover.runtime.budget import fingerprint


def load_cycle_library(directory: str | Path) -> dict | None:
    """Return a completed cycle's library, rejecting missing or changed data.

    A library written before complete.json is an unfinished checkpoint and is
    not reused. Once that marker exists, its recorded hash must match the
    complete library. Validation reads files without changing the checkpoint.
    """
    directory = Path(directory)
    marker = directory / "complete.json"
    if not marker.exists():
        return None
    library_path = directory / "library" / "full_library.json"
    try:
        complete = json.loads(marker.read_text(encoding="utf-8"))
        library = json.loads(library_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read completed library checkpoint: {directory}") from exc
    if not isinstance(complete, dict) or not isinstance(library, dict):
        raise ValueError(f"Invalid completed library checkpoint: {directory}")
    expected = complete.get("library_sha256")
    actual = fingerprint(library)
    if not isinstance(expected, str) or expected != actual:
        raise ValueError(
            f"Completed library checkpoint hash mismatch: {directory} "
            f"(recorded {expected!r}, actual {actual})"
        )
    return library
