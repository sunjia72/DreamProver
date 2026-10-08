"""Write a Markdown summary from recorded checkpoints, without model or Lean calls."""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from dreamprover.runtime.monitor import _cycle_progress, ledger_summary
from dreamprover.runtime.io import safe_problem_filename


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def reservation_completion_summary(ledger: dict, configured_default) -> dict:
    """Recover requested ceilings without assigning usage to unresolved calls.

    reserve() records R = input + output and C = input_rate * input +
    output_rate * output. Unequal recorded rates recover output uniquely.
    Missing/invalid fields remain unknown; the request fingerprint alone does
    not expose its token limit. This function never changes ledger records.
    """
    def nonnegative_number(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0

    def positive_integer(value):
        return isinstance(value, int) and not isinstance(value, bool) and value > 0

    prices = ledger.get("prices", {})
    rates = [prices.get(name) for name in ("input_per_million", "cached_input_per_million", "output_per_million")]
    input_rate = max(rates[:2]) if all(nonnegative_number(rate) for rate in rates) else None
    output_rate = rates[2]
    rates_available = input_rate is not None and input_rate != output_rate
    default = configured_default if positive_integer(configured_default) else None
    histogram = Counter()
    unknown_ceilings = 0
    outputs = []
    requests = ledger.get("requests", [])
    for request in requests:
        ceiling = None
        total = request.get("reserved_tokens")
        cost = request.get("reserved_cost_usd")
        if rates_available and positive_integer(total) and nonnegative_number(cost):
            try:
                inferred = (cost * 1_000_000 - input_rate * total) / (output_rate - input_rate)
                if math.isfinite(inferred):
                    nearest = round(inferred)
                    # reserve() adds at least 4096 input-framing tokens. Avoid
                    # interpreting malformed totals or a fractional ceiling.
                    if positive_integer(nearest) and total - nearest >= 4096 and abs(inferred - nearest) <= 1e-6:
                        ceiling = nearest
            except (OverflowError, ZeroDivisionError):
                pass
        if ceiling is None:
            unknown_ceilings += 1
        else:
            histogram[ceiling] += 1
        output = request.get("usage", {}).get("output_tokens") if request.get("state") == "complete" else None
        if isinstance(output, int) and not isinstance(output, bool) and output >= 0:
            outputs.append(output)
    return dict(request_reservations=len(requests),
                reserved_requested_ceiling_histogram=dict(sorted(histogram.items())),
                unknown_ceiling_reservations=unknown_ceilings,
                known_reservations_above_default=sum(count for ceiling, count in histogram.items() if ceiling > default) if default is not None else None,
                maximum_settled_output_tokens=max(outputs) if outputs else None,
                known_settled_outputs_above_default=sum(output > default for output in outputs) if default is not None else None,
                output_usage_unknown_reservations=len(requests) - len(outputs))


def collect_report(directory: str | Path) -> dict:
    """Read the frozen plan and require coverage of its exact cycles and IDs."""
    from dreamprover.pipeline import evaluation_metrics

    root = Path(directory).resolve()
    manifest = _read(root / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"No run manifest found: {root}")
    config = manifest["identity"]["config"]
    status = _read(root / "status.json", {})
    training_ids = manifest["training_ids"]
    evaluation_ids = manifest["evaluation_ids"]
    records = {record["id"]: record for record in manifest["identity"].get("evaluation", [])}
    cycles = []
    for cycle in range(1, config["cycles"] + 1):
        path = root / "cycles" / f"cycle_{cycle:02d}"
        progress = _cycle_progress(path, len(training_ids))
        if not path.is_dir():
            progress["stage"] = "not_started"
        if not (path / "library" / "full_library.json").is_file():
            progress["library_size"] = None
        update = _read(path / "update_report.json", {})
        progress["recorded_stage_counts"] = update.get("stage_counts", {})
        if "verified_candidates" not in progress["recorded_stage_counts"]:
            proved = _read(path / "verified_candidates.json")
            if proved is not None:
                progress["recorded_stage_counts"]["verified_candidates"] = len(proved)
            elif (path / "candidate_proofs").is_dir():
                progress["recorded_stage_counts"]["verified_candidates"] = sum(
                    len(_read(checkpoint, {}).get("proofs", {}))
                    for checkpoint in (path / "candidate_proofs").glob("*.json"))
        progress["complete"] = (path / "complete.json").is_file()
        cycles.append(progress)
    rows = []
    for identifier in evaluation_ids:
        row = dict(id=identifier, benchmark=records.get(identifier, {}).get("benchmark", "unspecified"))
        for arm in ("baseline", "learned_library"):
            result = _read(root / "evaluation" / arm / safe_problem_filename(identifier) / "result.json")
            if result is not None:
                if not isinstance(result.get("success"), bool) or result.get("id", identifier) != identifier:
                    raise ValueError(f"Invalid recorded evaluation result for {arm}/{identifier}")
            row[arm] = result
        rows.append(row)
    paired = [row for row in rows if all(row[arm] is not None for arm in ("baseline", "learned_library"))]
    library = _read(root / "final_library.json")
    if library is not None and not isinstance(library, dict):
        raise ValueError("The final library checkpoint must be a dictionary")
    verified = sum(record.get("verification_status") == "verified" and bool(record.get("proof"))
                   for record in (library or {}).values())
    diagnostic = _read(root / "final_library_diagnostic.json")
    training_complete = all(row["complete"] for row in cycles)
    evaluation_complete = len(paired) == len(evaluation_ids)
    complete = (training_complete and evaluation_complete and library is not None
                and verified == len(library) and status.get("status") == "evaluated"
                and not (diagnostic and diagnostic.get("proved") is False))
    ledger = _read(root / "budget.json", {})
    history = _read(root / "execution_history.json", [])
    return dict(run_dir=str(root), generated_at=datetime.now(timezone.utc).isoformat(),
                model=manifest.get("model", config.get("model")), config=config, status=status,
                runtime=manifest["identity"].get("runtime", {}),
                complete=complete, budget_interrupted=status.get("status") == "budget_exhausted",
                training_complete=training_complete, evaluation_complete=evaluation_complete,
                cycles=cycles, planned_training=len(training_ids), planned_evaluation=len(evaluation_ids),
                paired_completed=len(paired),
                completed_by_arm={arm: sum(row[arm] is not None for row in rows)
                                  for arm in ("baseline", "learned_library")},
                successes={arm: sum(row[arm]["success"] for row in paired)
                           for arm in ("baseline", "learned_library")},
                metrics=evaluation_metrics(rows), library_size=len(library) if library is not None else None,
                verified_library_size=verified if library is not None else None, library_diagnostic=diagnostic,
                budget=ledger_summary(ledger), prices=ledger.get("prices", {}),
                completion_ceilings=reservation_completion_summary(ledger, config.get("max_completion_tokens")),
                last_execution=history[-1] if history else {})


def _cell(value) -> str:
    return "—" if value is None else str(value).replace("|", "\\|").replace("\n", " ")


def _rate(successes: int, denominator: int) -> str:
    return f"{successes}/{denominator} ({successes / denominator:.1%})" if denominator else "0/0 (not available)"


def _number(value) -> str:
    return "not available" if value is None else f"{value:.2f}"


def render_report(report: dict) -> str:
    config, budget, metrics = report["config"], report["budget"], report["metrics"]
    state = "Complete" if report["complete"] else "Incomplete — budget interrupted" if report["budget_interrupted"] else "Incomplete"
    lines = ["# DreamProver experiment report", "", f"**{state}.** Recorded status: {_cell(report['status'].get('status', 'unknown'))}.",
             f"Generated: {report['generated_at']}. Run: `{report['run_dir']}`.", "",
             f"Model: **{_cell(report['model'])}**. Planned training: {report['planned_training']} targets × {config['cycles']} cycles; "
             f"held-out evaluation: {report['planned_evaluation']} target pairs.",
             f"Completed cycle markers: {sum(row['complete'] for row in report['cycles'])}/{config['cycles']}. "
             f"Evaluation coverage: {report['paired_completed']}/{report['planned_evaluation']} pairs "
             f"(baseline {report['completed_by_arm']['baseline']}, learned library {report['completed_by_arm']['learned_library']}).",
             "Accuracy denominators below include completed pairs only; pending targets are not counted as failures."]
    if report["status"].get("reason"):
        lines.append(f"Recorded stop reason: {_cell(report['status']['reason'])}.")
    lines += ["", "## Recorded configuration", "",
              f"Wake/evaluation depth: {config.get('wake_depth', 'unknown')}/{config.get('inference_depth', 'unknown')}; "
              f"direct starts per role: {config.get('direct_attempts', 'unknown')}; corrections: {config.get('corrections', 'unknown')}; "
              f"sketch generations: {config.get('wake_sketch_attempts', 'unknown')}/{config.get('inference_sketch_attempts', 'unknown')}; "
              f"configured default completion ceiling: {config.get('max_completion_tokens', 'unknown')} tokens.",
              "Extracted subgoals may receive prover starts followed by reasoner shallow-proof starts before recursion. "
              "The configured direct count applies to each role. Both roles use the recorded model.",
              f"Library capacity: {config.get('capacity', 'unknown')}; structural relevance/duplicate thresholds: "
              f"{config.get('relevance_threshold', 'unknown')}/{config.get('duplicate_threshold', 'unknown')}; seed: {config.get('seed', 'unknown')}.",
              f"Embedding model: `{_cell(config.get('embedding_model'))}`; revision: `{_cell(config.get('embedding_revision'))}`.",
              f"Lean project: `{_cell(config.get('lean_project'))}`; verification timeout: {config.get('verification_timeout', 'unknown')} seconds. "
              f"Last recorded worker concurrency: {report['last_execution'].get('problem_concurrency', config.get('problem_concurrency', 'unknown'))}.",
              f"Recorded Lean toolchain: `{_cell(report['runtime'].get('lean_toolchain'))}`.",
              "", "## Cycle checkpoints", "",
              "| Cycle | Stage | Wake proved/completed/planned | Experiences | Annotated | Proposed | Retained for proving | Verified candidates | Library |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in report["cycles"]:
        counts = row["recorded_stage_counts"]
        values = [row["cycle"], row["stage"], f"{row['wake_proved']}/{row['wake_completed']}/{report['planned_training']}",
                  counts.get("learnable_experiences", row["extracted_experiences"]), counts.get("annotated", row["annotated"]),
                  counts.get("proposed", row["proposed"]), counts.get("retained_for_proving", row["retained_for_proving"]),
                  counts.get("verified_candidates"), counts.get("final_library", row["library_size"])]
        lines.append("| " + " | ".join(map(_cell, values)) + " |")
    if report["library_size"] is None:
        lines += ["", "Final library: unavailable; training has not produced its final checkpoint."]
    else:
        lines += ["", f"Recorded verified final library: {report['verified_library_size']}/{report['library_size']} lemmas"
                  + (" (empty)." if not report["library_size"] else ".")]
        diagnostic = report["library_diagnostic"]
        if diagnostic is not None:
            lines.append(f"Recorded combined-library diagnostic: {_cell(diagnostic.get('proved'))}; {_cell(diagnostic.get('reason') or diagnostic.get('error'))}.")
    lines += ["", "## Paired held-out evaluation", "",
              f"Baseline: {_rate(report['successes']['baseline'], report['paired_completed'])}; "
              f"learned library: {_rate(report['successes']['learned_library'], report['paired_completed'])}.",
              "Paired outcomes: " + ", ".join(f"{name.replace('_', ' ')} {count}" for name, count in metrics["paired_outcomes"].items()) + ".", "",
              "| Benchmark | Completed/planned pairs | Baseline | Learned library | Both | Baseline only | Library only | Neither |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, row in metrics["by_benchmark"].items():
        outcomes = row["paired_outcomes"]
        values = [name, f"{row['paired_completed']}/{row['total_problems']}",
                  _rate(row["paired_successes"]["baseline"], row["paired_completed"]),
                  _rate(row["paired_successes"]["learned_library"], row["paired_completed"]),
                  *[outcomes[key] for key in ("both", "baseline_only", "learned_library_only", "neither")]]
        lines.append("| " + " | ".join(map(_cell, values)) + " |")
    lines += ["", "Shared-success proof lengths compare the same targets solved by both arms. "
              + metrics["proof_length_definition"] + ".", "",
              "| Arm | Shared successes | Mean/median nonempty lines | Mean/median characters | Successful problems reusing library | Distinct reused lemmas |",
              "| --- | --- | --- | --- | --- | --- |"]
    for arm in ("baseline", "learned_library"):
        lengths = metrics["proof_lengths"]["shared_successes"][arm]
        reuse = metrics["successful_library_reuse"][arm]
        values = [arm, lengths["proofs"], "/".join(_number(lengths["nonempty_lines"][key]) for key in ("mean", "median")),
                  "/".join(_number(lengths["characters"][key]) for key in ("mean", "median")), reuse["problems"], len(reuse["distinct_lemmas"])]
        lines.append("| " + " | ".join(map(_cell, values)) + " |")
    reused = metrics["successful_library_reuse"]["learned_library"]["distinct_lemmas"]
    lines += ["", "Successfully reused learned lemmas: " + (", ".join(f"`{_cell(name)}`" for name in reused) if reused else "none recorded") + ".",
              "", "## Recorded API accounting", "",
              f"Requests: {budget['api_calls']} recorded, {budget['completed_calls']} settled, "
              f"{budget['in_flight_or_unsettled_calls'] + budget['uncertain_calls']} unsettled (pending or uncertain). "
              f"Settled usage: {budget['input_tokens']:,} input tokens (including {budget['cached_input_tokens']:,} cached), "
              f"{budget['output_tokens']:,} output tokens, {budget['total_tokens']:,} total. Output usage includes reasoning tokens.",
              f"Recorded settled cost: **${budget['actual_cost_usd']:.6f}**; outstanding conservative reservations: "
              f"${budget['reserved_cost_usd']:.6f}; total charged budget ceiling: ${budget['charged_cost_usd']:.6f}."]
    prices = report["prices"]
    lines.append("Recorded USD rates per million tokens: " + ", ".join(
        f"{label} " + (f"${prices[key]:g}" if key in prices else "not available") for label, key in
        (("input", "input_per_million"), ("cached input", "cached_input_per_million"), ("output", "output_per_million"))) + ".")
    lines.append("Recorded shared limits: " + ", ".join(f"{key}={value}" for key, value in budget["limits"].items()) + ".")
    ceilings = report["completion_ceilings"]
    histogram = ceilings["reserved_requested_ceiling_histogram"]
    lines += ["Reserved/requested completion ceilings recovered from reservation totals: "
              + (", ".join(f"{limit} tokens: {count}" for limit, count in histogram.items()) if histogram else "none recoverable")
              + f". Unknown ceilings: {ceilings['unknown_ceiling_reservations']}/{ceilings['request_reservations']} reservations. "
              + f"Known reservations above the configured default: {_cell(ceilings['known_reservations_above_default'])}.",
              f"Maximum settled output: {_cell(ceilings['maximum_settled_output_tokens'])} tokens; "
              f"known settled outputs above the configured default: {_cell(ceilings['known_settled_outputs_above_default'])}. "
              f"Output usage unknown: {ceilings['output_usage_unknown_reservations']}/{ceilings['request_reservations']} reservations.",
              "Ceiling recovery requires valid recorded reservation fields and unequal rates. "
              "An unresolved reservation does not establish request dispatch or output usage."]
    lines += ["", "This report reads recorded checkpoints only. It makes no model or Lean calls and does not independently reverify proofs. "
              "Completeness requires all configured cycle markers, every planned evaluation pair, a recorded verified final library, and evaluated status."]
    return "\n".join(lines) + "\n"


def write_report(directory: str | Path, output: str | Path | None = None) -> Path:
    root = Path(directory).resolve()
    target = Path(output).expanduser().resolve() if output is not None else root / "report.md"
    if target.suffix.lower() not in {".md", ".markdown"}:
        raise ValueError("Use a .md or .markdown report path; checkpoint files are never overwritten")
    content = render_report(collect_report(root))
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=target.parent,
                                         prefix=f".{target.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return target


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="Markdown output (default: RUN/report.md)")
    args = parser.parse_args(argv)
    print(write_report(args.run_dir, args.output))


if __name__ == "__main__":
    main()
