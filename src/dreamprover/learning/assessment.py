"""Matched retries of unsolved training targets after one completed cycle.

This diagnostic compares fresh direct proof attempts with an empty or frozen
learned library. Training targets cannot establish held-out transfer.
"""
from __future__ import annotations

import argparse
import json
import logging
import threading
from dataclasses import replace
from pathlib import Path

from dreamprover.paths import config_directory, repository_root, working_directory
from dreamprover.runtime.budget import BudgetExhausted, RunStopped, atomic_json, fingerprint
from dreamprover.runtime.io import safe_problem_filename

logger = logging.getLogger(__name__)
ARMS = ("baseline", "learned_library")


def _read(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def run_assessment(pipeline, *, cycle: int = 1, max_prover_calls: int = 28,
                   max_reasoner_calls: int = 28) -> dict:
    """Compare complete retry pairs under the original shared run budget."""
    if not isinstance(cycle, int) or isinstance(cycle, bool) or not 1 <= cycle <= pipeline.config.cycles:
        raise ValueError("cycle must identify a configured training cycle")
    for name, value in (("max_prover_calls", max_prover_calls), ("max_reasoner_calls", max_reasoner_calls)):
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    library = pipeline.load_cycle_library(cycle)
    cycle_dir = pipeline.root / "cycles" / f"cycle_{cycle:02d}"
    wake = {}
    for problem in pipeline.train:
        result = _read(cycle_dir / "wake" / safe_problem_filename(problem["id"]) / "result.json")
        if (result is None or result.get("id") != problem["id"]
                or not isinstance(result.get("success"), bool)
                or result.get("stop_reason") not in {"completed", "per_problem_call_limit"}):
            raise ValueError("Complete every wake target before assessing its learned library")
        wake[problem["id"]] = result
    targets = [problem for problem in pipeline.train if not wake[problem["id"]]["success"]]
    original_proved = sum(result["success"] for result in wake.values())
    proof_config = replace(pipeline.config.proof_config(False), subgoal_decomp_attempts=0,
                           max_prover_llm_calls=max_prover_calls,
                           max_reasoner_llm_calls=max_reasoner_calls)
    directory = pipeline.root / "assessment" / f"cycle_{cycle:02d}"
    contract = dict(schema_version=1, cycle=cycle, library_sha256=fingerprint(library),
        training_problem_ids=[problem["id"] for problem in pipeline.train],
        target_ids=[problem["id"] for problem in targets], original_wake_completed=len(wake),
        original_wake_proved=original_proved,
        original_wake_sha256={identifier: fingerprint(result) for identifier, result in wake.items()},
        proof_config=proof_config.to_dict(), max_depth=0,
        max_completion_tokens=pipeline.config.max_completion_tokens, model=pipeline.config.model,
        base_url=pipeline.config.base_url,
        arm_policy="fresh matched direct-only retries, empty versus frozen learned library, no library updates",
        interpretation="training-target diagnostic, not held-out transfer")
    previous = _read(directory / "manifest.json")
    if previous is not None and previous != contract:
        raise ValueError("Assessment targets, wake results, frozen library, or matched settings changed on resume")
    atomic_json(directory / "manifest.json", contract)

    def status(state, **details):
        atomic_json(directory / "status.json", dict(phase="training_assessment", status=state,
                    cycle=cycle, budget=pipeline.ledger.summary(), **details))

    status("running")
    interrupted = None
    state = "assessed"
    failure = None
    try:
        jobs = []
        for index, problem in enumerate(targets):
            arms = ARMS if index % 2 == 0 else tuple(reversed(ARMS))
            for arm in arms:
                jobs.append((problem, library if arm == "learned_library" else {},
                             directory / arm / safe_problem_filename(problem["id"])))
        pipeline.run_jobs(jobs, training=False, proof_config=proof_config, max_depth=0)
    except BudgetExhausted as exc:
        state, interrupted = "budget_exhausted", str(exc)
    except RunStopped as exc:
        state, interrupted = "stopped", str(exc)
    except Exception as exc:
        state, interrupted, failure = "failed", str(exc), exc

    rows = [dict(id=problem["id"], **{
        arm: _read(directory / arm / safe_problem_filename(problem["id"]) / "result.json")
        for arm in ARMS}) for problem in targets]
    paired = [row for row in rows if all(row[arm] is not None for arm in ARMS)]
    successes = {arm: sum(bool(row[arm]["success"]) for row in paired) for arm in ARMS}
    outcomes = dict(both=0, baseline_only=0, learned_library_only=0, neither=0)
    for row in paired:
        baseline, learned = (bool(row[arm]["success"]) for arm in ARMS)
        key = ("both" if baseline and learned else "baseline_only" if baseline
               else "learned_library_only" if learned else "neither")
        outcomes[key] += 1
    summary = dict(schema_version=1, cycle=cycle, status=state, reason=interrupted,
        interpretation=contract["interpretation"], total_training_problems=len(pipeline.train),
        original_wake_proved=original_proved, previously_unsolved=len(targets),
        paired_completed=len(paired),
        completed_by_arm={arm: sum(row[arm] is not None for row in rows) for arm in ARMS},
        additional_solved_on_complete_pairs=successes, paired_outcomes=outcomes,
        training_solved_with_matched_retries={arm: original_proved + successes[arm] for arm in ARMS},
        newly_solved_ids={arm: [row["id"] for row in paired if row[arm]["success"]] for arm in ARMS},
        learned_library_only_ids=[row["id"] for row in paired
                                 if row["learned_library"]["success"] and not row["baseline"]["success"]],
        library_size=len(library), library_sha256=contract["library_sha256"],
        successful_library_mentions=sum(row["learned_library"]["success"]
            and bool(row["learned_library"].get("library_lemmas_used")) for row in paired),
        library_mention_definition="lexical mentions in successful generated proofs, not audited proof dependencies",
        usage_by_arm={arm: pipeline.ledger.summary(f"assessment/cycle_{cycle:02d}/{arm}/") for arm in ARMS},
        budget=pipeline.ledger.summary(), results=rows)
    atomic_json(directory / "summary.json", summary)
    status(state, reason=interrupted, paired_completed=len(paired))
    if failure is not None:
        raise failure
    return summary


def main(argv=None):
    import yaml
    from dreamprover.pipeline import Pipeline, PipelineConfig, cli_stop_signals, run_lock

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_directory() / "pipeline/inequalities.yaml")
    parser.add_argument("--cycle", type=int, default=1)
    parser.add_argument("--max-cost-usd", type=float)
    parser.add_argument("--problem-concurrency", type=int)
    parser.add_argument("--max-prover-calls", type=int, default=28)
    parser.add_argument("--max-reasoner-calls", type=int, default=28)
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")
    values = yaml.safe_load(config_path.read_text()) or {}
    for name in ("max_cost_usd", "problem_concurrency"):
        if getattr(args, name) is not None:
            values[name] = getattr(args, name)
    config = PipelineConfig.from_mapping(values)
    if config.max_cost_usd is None:
        parser.error("Set --max-cost-usd before a paid assessment. The cap covers the original run and both retry arms.")
    stop_event = threading.Event()
    with working_directory(repository_root(config_path)):
        with cli_stop_signals(stop_event) as received:
            try:
                with run_lock(config.output_dir):
                    pipeline = Pipeline(config, phase="training_assessment", stop_event=stop_event)
                    summary = run_assessment(pipeline, cycle=args.cycle,
                        max_prover_calls=args.max_prover_calls, max_reasoner_calls=args.max_reasoner_calls)
            except BaseException as exc:
                if received:
                    if isinstance(exc, Exception):
                        logger.exception("Assessment failed while draining after signal %s", received[0])
                    raise SystemExit(128 + received[0])
                raise
    print(json.dumps({key: value for key, value in summary.items() if key != "results"}, indent=2))
    if received:
        raise SystemExit(128 + received[0])
    return 2 if summary["status"] == "budget_exhausted" else 0


if __name__ == "__main__":
    raise SystemExit(main())
