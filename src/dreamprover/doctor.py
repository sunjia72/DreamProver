"""Inspect configuration without invoking a model or downloading embeddings."""

import argparse
import importlib.metadata
import json
import os
import shutil
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from dreamprover.paths import config_directory, repository_root, working_directory


def inspect_dataset(path):
    """Validate the input records early, with useful line-level diagnostics."""
    records = []
    names = set()
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc.msg}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected an object")
            for key in ("header", "formal_statement"):
                if not isinstance(record.get(key), str):
                    raise ValueError(f"{path}:{line_number}: {key} must be a string")
            if not record["formal_statement"].strip():
                raise ValueError(f"{path}:{line_number}: empty formal_statement")
            name = str(record.get("name", record.get("id", len(records))))
            if name in names:
                raise ValueError(f"{path}:{line_number}: duplicate name {name!r}")
            names.add(name)
            records.append(record)
    if not records:
        raise ValueError(f"{path}: dataset is empty")
    return len(records)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Hydra YAML file (default: REPOSITORY/configs/run.yaml)")
    parser.add_argument("--services", action="store_true", help="Probe Kimina and model catalog URLs")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. experiment=dreamprover data=neq_test")
    args = parser.parse_args(argv)
    config_path = (args.config.expanduser().resolve() if args.config is not None
                   else (config_directory() / "run.yaml").resolve())
    if not config_path.is_file():
        parser.error(f"Configuration file does not exist: {config_path}")
    with working_directory(repository_root(config_path)):
        with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
            cfg = compose(config_name=config_path.stem, overrides=args.overrides)
        return _inspect_configuration(cfg, services=args.services)


def _inspect_configuration(cfg, *, services=False):
    report = {"checks": [], "ok": True}

    def check(name, function):
        try:
            value = function()
            report["checks"].append({"name": name, "ok": True, "detail": value})
        except Exception as exc:
            report["ok"] = False
            report["checks"].append({"name": name, "ok": False, "detail": str(exc)})

    for package in ("hydra-core", "openai", "httpx", "kimina-client"):
        check(package, lambda package=package: importlib.metadata.version(package))
    check("dataset", lambda: {"path": cfg.data.file_path, "records": inspect_dataset(cfg.data.file_path)})
    check("proof_budget", lambda: OmegaConf.to_container(cfg.experiment.get("proof_attempt_config", {}), resolve=True))
    report["lean"] = shutil.which("lean")
    report["api_key_present"] = bool(os.environ.get("OPENAI_API_KEY"))
    report["retrieval_enabled"] = cfg.experiment.get("enable_retrieval", False)
    library = cfg.experiment.get("lemma_library_path")
    if library:
        from dreamprover.lean.library import LemmaLibrary
        check("lemma_library", lambda: {"path": str(library), "characters": len(LemmaLibrary.from_file(library).source)})
    if services:
        import httpx

        def probe(url, headers=None):
            with httpx.Client(timeout=15) as client:
                response = client.get(url, headers=headers)
                response.raise_for_status()
                return {"status": response.status_code}

        check("kimina", lambda: probe(cfg.experiment.verifier_base_url.rstrip("/") + "/health"))
        for label, section in (("prover", "prover_llm"), ("reasoner", "informal_llm")):
            if section not in cfg.experiment:
                continue
            llm = cfg.experiment[section]
            if llm.get("provider", "openai") != "openai":
                continue
            key = llm.get("api_key") or os.environ.get("OPENAI_API_KEY")
            headers = {"Authorization": f"Bearer {key}"} if key else None
            check(label, lambda llm=llm, headers=headers: probe(llm.base_url.rstrip("/") + "/models", headers))
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
