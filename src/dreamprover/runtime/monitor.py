"""Read live run checkpoints and budget accounting without changing the run.

`python -m dreamprover.runtime.monitor --run-dir runs/inequalities/gpt-6-luna` reports the
ledger rather than the older budget snapshot in status.json. No model, verifier
or embedding calls are made, and monitoring does not acquire the writer lock.
unfinished_problem_ids lists started directories without a terminal result;
these can include interrupted work and do not identify proven live workers.
"""
from __future__ import annotations

import argparse
import json
import time
import math
import os
import socket
from collections import Counter
from pathlib import Path


def _read(path, default=None):
    return json.loads(path.read_text()) if path.is_file() else default


def _writer(directory):
    """Read local lock ownership; a recorded owner on another host is unknown."""
    path = directory / ".run.lock"
    if not path.is_file():
        return dict(running=False, pid=None)
    pid = path.read_text().strip()
    owner = dict(pid=int(pid) if pid.isdigit() else None)
    history = _read(directory / "execution_history.json", [])
    latest = history[-1] if history else {}
    if (owner["pid"] is not None and latest.get("pid") == owner["pid"]
            and isinstance(latest.get("hostname"), str) and latest["hostname"]):
        owner["owner_hostname"] = latest["hostname"]
        if latest.get("slurm_job_id"):
            owner["owner_slurm_job_id"] = latest["slurm_job_id"]
        # /proc/locks describes this node only. Absence from that table cannot
        # establish that a writer recorded on a different node has stopped.
        if latest["hostname"] != socket.gethostname():
            return dict(running=None, **owner)
    info = path.stat()
    expected = (os.major(info.st_dev), os.minor(info.st_dev), info.st_ino)
    try:
        locks = Path("/proc/locks").read_text().splitlines()
    except OSError:
        return dict(running=None, **owner)
    # Inspect Linux's lock table without briefly taking a lock that could make
    # a simultaneous nonblocking writer start fail.
    for line in locks:
        for field in line.split():
            values = field.split(":")
            if len(values) == 3:
                try:
                    identity = (int(values[0], 16), int(values[1], 16), int(values[2]))
                except ValueError:
                    continue
                if identity == expected:
                    return dict(running=True, **owner)
    return dict(running=False, **owner)


def ledger_summary(ledger):
    requests = ledger.get("requests", [])
    actual = dict(input_tokens=0, output_tokens=0, cached_input_tokens=0, total_tokens=0)
    actual_cost = reserved_cost = 0.0
    pending = []
    for index, request in enumerate(requests):
        if request["state"] == "complete":
            for name in actual:
                actual[name] += request["usage"].get(name, 0)
            actual_cost += request["cost_usd"]
        else:
            reserved_cost += request["reserved_cost_usd"]
            pending.append(dict(request_index=index, scope=request["scope"], state=request["state"],
                                error_type=request.get("error_type")))
    return dict(actual, api_calls=len(requests), completed_calls=sum(r["state"] == "complete" for r in requests),
                in_flight_or_unsettled_calls=sum(r["state"] == "reserved" for r in requests),
                uncertain_calls=sum(r["state"] == "uncertain" for r in requests),
                actual_cost_usd=actual_cost, reserved_cost_usd=reserved_cost,
                charged_cost_usd=actual_cost + reserved_cost, limits=ledger.get("limits", {}),
                pending_requests=pending)


def _cycle_progress(directory, planned):
    """Count saved work; unfinished directories are distinct from live workers."""
    results = [_read(path) for path in sorted((directory / "wake").glob("*/result.json"))]
    results = [record for record in results if record is not None]
    unfinished = [path.name for path in sorted((directory / "wake").glob("*"))
                  if path.is_dir() and not (path / "result.json").exists()]
    experiences = _read(directory / "experiences.json")
    annotations = _read(directory / "annotations.json", {})
    proposals = _read(directory / "proposals.json")
    filtered = _read(directory / "filtered_proposals.json")
    candidates_done = len(list((directory / "candidate_proofs").glob("*.json")))
    if (directory / "complete.json").exists():
        stage = "complete"
    elif len(results) < planned:
        stage = "wake"
    elif experiences is None:
        stage = "extract_and_verify_experiences"
    elif len(annotations) < len(experiences):
        stage = "annotate_experiences"
    elif proposals is None:
        stage = "cluster_and_abstract" if (directory / "embeddings.npy").exists() else "embed_experiences"
    elif filtered is None:
        stage = "structural_filter_and_deduplicate"
    elif candidates_done < len(filtered.get("candidates", [])):
        stage = "prove_candidates"
    else:
        stage = "verify_and_update_library"
    return dict(cycle=directory.name, stage=stage, wake_planned=planned, wake_completed=len(results),
                wake_proved=sum(record.get("success", False) for record in results), unfinished_problem_ids=unfinished,
                stop_reasons=dict(Counter(record.get("stop_reason", "unknown") for record in results)),
                verified_direct_exports=len(list((directory / "wake").glob("*/experiences/*.lean"))),
                extracted_experiences=len(experiences) if experiences is not None else None,
                annotated=len(annotations), proposed=len(proposals.get("candidates", [])) if proposals else None,
                retained_for_proving=len(filtered.get("candidates", [])) if filtered else None,
                candidate_proofs_completed=candidates_done,
                library_size=len(_read(directory / "library" / "full_library.json", {})))


def inspect_run(directory, *, include_response_diagnostics=False) -> dict:
    directory = Path(directory)
    manifest = _read(directory / "manifest.json")
    if manifest is None:
        raise FileNotFoundError(f"No run manifest found: {directory}")
    status = _read(directory / "status.json", {})
    ledger = _read(directory / "budget.json", {})
    planned = len(manifest.get("training_ids", []))
    cycles = [_cycle_progress(path, planned) for path in sorted((directory / "cycles").glob("cycle_*"))
              if path.is_dir()]
    evaluation_ids = manifest.get("evaluation_ids", [])
    evaluation_completed = {arm: {path.parent.name for path in (directory / "evaluation" / arm).glob("*/result.json")}
                            for arm in ("baseline", "learned_library")}
    budget_file = directory / "budget.json"
    report = dict(run_dir=str(directory.resolve()), phase=status.get("phase"), status=status.get("status"),
        writer=_writer(directory), model=manifest.get("model"), live_budget=ledger_summary(ledger),
        seconds_since_ledger_update=max(0, time.time() - budget_file.stat().st_mtime) if budget_file.exists() else None,
        training=dict(planned_cycles=manifest["identity"]["config"]["cycles"], problems_per_cycle=planned,
                      completed_cycles=sum(row["stage"] == "complete" for row in cycles),
                      total_completed_problems=sum(row["wake_completed"] for row in cycles), cycles=cycles),
        evaluation=dict(planned=len(evaluation_ids), completed_by_arm={arm: len(values) for arm, values in evaluation_completed.items()},
                        paired_completed=len(evaluation_completed["baseline"] & evaluation_completed["learned_library"])),
        last_execution=_read(directory / "execution_history.json", [None])[-1])
    if include_response_diagnostics:
        responses = [_read(path) for path in (directory / "responses").glob("*.json")]
        diagnostics = [record["diagnostics"] for record in responses if "diagnostics" in record]
        reasoning = [record["reasoning_tokens"] for record in diagnostics if record.get("reasoning_tokens") is not None]
        report["response_diagnostics"] = dict(cached_responses=len(responses),
            empty_visible_responses=sum(not record.get("text") for record in responses),
            responses_with_metadata=len(diagnostics),
            finish_reasons=dict(Counter(reason for record in diagnostics for reason in record.get("finish_reasons", []) if reason)),
            reasoning_usage_known_responses=len(reasoning), reasoning_tokens=sum(reasoning) if reasoning else None,
            visible_text_characters=sum(len(record.get("text") or "") for record in responses))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--watch", type=float, help="Refresh interval in seconds; Ctrl-C stops monitoring")
    parser.add_argument("--response-diagnostics", action="store_true", help="Also scan cached responses for empty/truncated output metadata")
    args = parser.parse_args(argv)
    if args.watch is not None and (not math.isfinite(args.watch) or args.watch <= 0):
        parser.error("--watch must be positive and finite")
    try:
        while True:
            print(json.dumps(inspect_run(args.run_dir, include_response_diagnostics=args.response_diagnostics), indent=2), flush=True)
            if args.watch is None:
                return
            time.sleep(args.watch)
    except KeyboardInterrupt:
        return


if __name__ == "__main__":
    main()
