"""Query a Tree-Based Premise Selection service through its HTTP API."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
import yaml

from dreamprover.paths import config_directory


DEFAULT_CONFIG = config_directory() / "experiment/dreamprover.yaml"


def load_tbps_url(config_path: str | Path | None = None, url: str | None = None) -> str:
    """Resolve an explicit override, TBPS_URL, or the repository's experiment YAML."""
    value = url or os.environ.get("TBPS_URL")
    if not value:
        path = Path(config_path) if config_path else DEFAULT_CONFIG
        with path.open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream) or {}
        if not isinstance(config, dict):
            raise ValueError(f"Expected a YAML mapping in {path}")
        value = config.get("tbps_url")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Set --url, TBPS_URL, or tbps_url in the experiment YAML")
    value = value.strip().rstrip("/")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid HTTP service URL: {value!r}")
    return value


def query_health(
    url: str | None = None,
    *,
    config_path: str | Path | None = None,
    timeout: float = 10,
) -> dict[str, Any]:
    response = requests.get(
        load_tbps_url(config_path, url) + "/health", timeout=timeout
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("The TBPS health response must be a JSON object")
    return payload


def query_tbps_server(
    expr: str,
    url: str | None = None,
    *,
    config_path: str | Path | None = None,
    k: int = 20,
    timeout: float = 120,
) -> dict[str, Any]:
    """Return upstream JSON; HTTP errors and application failures are errors."""
    if not expr.strip():
        raise ValueError("The Lean expression cannot be empty")
    if not 1 <= k <= 100:
        raise ValueError("k must be between 1 and 100")
    response = requests.post(
        load_tbps_url(config_path, url) + "/find-similar-theorems",
        json={"expression": expr, "k": k},
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict) or payload.get("success") is not True:
        raise ValueError(f"TBPS search failed: {payload}")
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Service URL (overrides TBPS_URL and YAML)")
    parser.add_argument(
        "--config", type=Path, help="Experiment YAML containing tbps_url"
    )
    parser.add_argument("--expr", default="forall (a b : Nat), a + b = b + a")
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--health", action="store_true", help="Query only /health")
    args = parser.parse_args(argv)
    try:
        if args.timeout <= 0:
            raise ValueError("timeout must be positive")
        payload = (
            query_health(args.url, config_path=args.config, timeout=args.timeout)
            if args.health
            else query_tbps_server(
                args.expr,
                args.url,
                config_path=args.config,
                k=args.k,
                timeout=args.timeout,
            )
        )
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return int(args.health and payload.get("status") != "healthy")
    except (OSError, ValueError, requests.RequestException, yaml.YAMLError) as exc:
        parser.exit(1, f"TBPS request failed: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
