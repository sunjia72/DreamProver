"""Locate the checkout's configuration, data, and local dependencies.

Runtime assets live outside the installed Python package. An explicit checkout
override also keeps their location stable when a server changes into a vendored
dependency before importing DreamProver.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


def _checkout_ancestor(path: Path) -> Path | None:
    for candidate in (path, *path.parents):
        if (
            (candidate / "pyproject.toml").is_file()
            and (candidate / "configs").is_dir()
            and (candidate / "dependencies.lock.json").is_file()
        ):
            return candidate
    return None


def repository_root(config_path: str | Path | None = None) -> Path:
    """Resolve the workspace without requiring a checkout at import time.

    DREAMPROVER_REPO is authoritative. Otherwise prefer a checkout containing
    an explicit configuration, then the source package, then the current
    directory. An installed wheel used outside a checkout falls back to the
    current directory; commands that need assets report missing files normally.
    """
    override = os.environ.get("DREAMPROVER_REPO")
    if override:
        return Path(override).expanduser().resolve()

    candidates = []
    if config_path is not None:
        candidates.append(Path(config_path).expanduser().resolve())
    candidates.extend((Path(__file__).resolve().parent, Path.cwd()))
    for candidate in candidates:
        root = _checkout_ancestor(candidate)
        if root is not None:
            return root
    return Path.cwd()


def config_directory(config_path: str | Path | None = None) -> Path:
    """Return an explicit YAML's directory or the checkout's ``configs``."""
    if config_path is not None:
        path = Path(config_path).expanduser().resolve()
        return path.parent if path.suffix.lower() in {".yaml", ".yml"} else path
    override = os.environ.get("DREAMPROVER_CONFIG_DIR")
    return (
        Path(override).expanduser().resolve()
        if override
        else repository_root() / "configs"
    )


@contextmanager
def working_directory(path: str | Path) -> Iterator[None]:
    """Run a synchronous command in its workspace and restore the caller's cwd."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
