#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import argparse
import json
import logging
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from dreamprover.paths import config_directory, repository_root, working_directory


logger = logging.getLogger(__name__)


def run_experiment(cfg):
    name = cfg.experiment.name
    if name == "dreamprover":
        from dreamprover.prover.experiment import run_async_hilbert
        results = run_async_hilbert(cfg)
    else:
        raise ValueError(f"Unknown experiment {name!r}")
    if not isinstance(results, dict):
        raise TypeError(f"Experiment {name!r} returned {type(results).__name__}, expected dict")
    output = Path(cfg.results_dir) / name / "result.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    logger.info("Results saved to %s", output)
    return results


def main(argv=None):
    """Compose an external Hydra config and run its proving experiment."""
    parser = argparse.ArgumentParser(description=__doc__ or "Run recursive Lean theorem proving")
    parser.add_argument("--config", type=Path, help="Hydra YAML file (default: REPOSITORY/configs/run.yaml)")
    parser.add_argument("overrides", nargs="*", help="Hydra overrides, e.g. data.file_path=YOUR.jsonl")
    args = parser.parse_args(argv)
    config_path = (args.config.expanduser().resolve() if args.config is not None
                   else (config_directory() / "run.yaml").resolve())
    if not config_path.is_file():
        parser.error(f"Configuration file does not exist: {config_path}")
    with working_directory(repository_root(config_path)):
        with initialize_config_dir(config_dir=str(config_path.parent), version_base=None):
            cfg: DictConfig = compose(config_name=config_path.stem, overrides=args.overrides)
        run_experiment(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
