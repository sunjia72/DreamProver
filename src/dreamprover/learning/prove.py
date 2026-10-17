"""Prove sleep-stage candidate statements with finite LLM/Lean attempts.

Candidate statements are preserved and only proofs accepted by actual Lean
without sorry or introduced axioms enter the output. Apply the paper's
structural relevance filtering before this step when reproducing its method.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dreamprover.learning.extract import library_lean_source
from dreamprover.lean.source import extract_declarations, referenced_names
from dreamprover.learning.update import _statement_key
from dreamprover.clients.completion import query_llm, run_informal_llm
from dreamprover.lean.compiler import check_lean_source

PROOF_PROMPT = """You are a Lean 4 expert. Complete this theorem with a correct proof:
```lean
{statement} := by
```
The available preamble is:
```lean
{header}
```
These previously proved lemmas may be used:
{library}
Return one ```lean``` block containing exactly the original theorem statement
and its proof. Do not change its statement or name. Do not add imports, open
commands, definitions, axioms, or other theorems. Do not use sorry or admit.
{feedback}
"""


def prove_candidates(candidates: list[dict], complete, *, header: str, library: dict | None = None, project_root=None, attempts: int = 4, corrections: int = 6, timeout: float = 120) -> tuple[dict, list[dict]]:
    if attempts < 1 or corrections < 0:
        raise ValueError("attempts must be positive and corrections nonnegative")
    accepted = {name: dict(record) for name, record in (library or {}).items()}
    if accepted:
        source = library_lean_source(accepted)
        if {record.get("header", "").strip() for record in accepted.values()} != {header.strip()}:
            raise ValueError("Candidates and library must use the same preamble")
        checked = check_lean_source(source, project_root=project_root, timeout=timeout)
        if not checked["proved"]:
            raise ValueError(
                f"Existing library verification failed ({checked.get('outcome', 'mathematical_invalid')}):\n"
                + checked.get("diagnostic", checked["stdout"] + checked["stderr"])
            )
    reports = []
    for candidate in candidates:
        name = candidate["name"]
        statement = candidate["statement"].strip()
        if name in accepted:
            raise ValueError(f"Candidate name is already in the library: {name}")
        feedback = ""
        last_error = ""
        solved = False
        calls = 0
        verification_outcomes = []
        last_outcome = "mathematical_invalid"
        last_resource_kind = None
        for attempt in range(attempts):
            feedback = ""
            for correction in range(corrections + 1):
                library_statements = "\n".join(record["statement"] for record in accepted.values())
                prompt = PROOF_PROMPT.format(statement=statement, header=header, library=library_statements, feedback=feedback)
                response = complete(prompt)
                calls += 1
                blocks = re.findall(r"```(?:lean4?|Lean4?)\s*\n(.*?)```", response, re.DOTALL)
                last_outcome = "mathematical_invalid"
                last_resource_kind = None
                try:
                    if len(blocks) != 1:
                        raise ValueError("Expected one Lean proof block")
                    preamble, declarations = extract_declarations(blocks[0])
                    if preamble.strip() or len(declarations) != 1:
                        raise ValueError("Proof block must contain exactly one theorem and no added preamble")
                    declaration = declarations[0]
                    if declaration.name != name or _statement_key(declaration.statement) != _statement_key(statement):
                        raise ValueError("Model changed the original candidate statement")
                    supporting_source = library_lean_source(accepted) if accepted else header
                    result = check_lean_source(supporting_source + "\n\n" + declaration.source, project_root=project_root, timeout=timeout)
                    last_outcome = result.get("outcome", "valid" if result["proved"] else "mathematical_invalid")
                    last_resource_kind = result.get("resource_kind")
                    verification_outcomes.append({
                        "attempt": attempt + 1, "correction": correction,
                        "outcome": last_outcome, "resource_kind": last_resource_kind,
                        "diagnostic": result.get("diagnostic", ""), "timeout_seconds": timeout,
                    })
                    if not result["proved"]:
                        if last_outcome == "resource_exhausted":
                            raise ValueError(
                                f"Lean verification resource allowance exhausted ({last_resource_kind or 'resource'}). "
                                f"The timeout remains {timeout} seconds. Use a proof that fits the same allowance.\n"
                                + result.get("diagnostic", result["stdout"] + result["stderr"])
                            )
                        raise ValueError("Lean did not verify this as a proof without sorry/axioms:\n" + result["stdout"] + result["stderr"])
                except ValueError as exc:
                    last_error = str(exc)
                    feedback = f"The previous attempt failed. Correct its proof while preserving the statement.\nPrevious response:\n{response}\nFeedback:\n{last_error}"
                    continue
                accepted[name] = {
                    "statement": statement,
                    "header": header,
                    "proof": declaration.proof,
                    "children": referenced_names(statement + "\n" + declaration.proof, set(accepted)),
                    "parents": [],
                    "verification_status": "verified",
                    "generated_from": candidate.get("generated_from", []),
                    "usage_count": 0,
                    "last_used_cycle": -1,
                }
                solved = True
                break
            if solved:
                break
        reports.append({
            "name": name, "proved": solved, "model_calls": calls,
            "last_error": "" if solved else last_error,
            "outcome": "valid" if solved else last_outcome,
            "resource_kind": None if solved else last_resource_kind,
            "verification_outcomes": verification_outcomes,
        })
    proofs = {name: record for name, record in accepted.items() if not library or name not in library}
    return proofs, reports


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="dreamprover.learning.sleep candidate JSON")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preamble-file", type=Path, required=True)
    parser.add_argument("--library", type=Path, help="Existing verified full_library.json")
    parser.add_argument("--lean-project", type=Path)
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--attempts", type=int, default=4)
    parser.add_argument("--corrections", type=int, default=6)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--max-tokens", type=int, default=4096)
    args = parser.parse_args()
    if bool(args.base_url) != bool(args.model):
        parser.error("--base-url and --model must be supplied together")
    candidates = json.loads(args.input.read_text(encoding="utf-8"))["candidates"]
    header = args.preamble_file.read_text(encoding="utf-8")
    library = json.loads(args.library.read_text(encoding="utf-8")) if args.library else {}
    def complete(prompt):
        if args.base_url:
            return query_llm(prompt, api_url=args.base_url, model_name=args.model, max_tokens=args.max_tokens)["text"]
        return run_informal_llm(prompt, max_tokens=args.max_tokens)
    proved, reports = prove_candidates(candidates, complete, header=header, library=library, project_root=args.lean_project, attempts=args.attempts, corrections=args.corrections, timeout=args.timeout)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "verified_candidates.json").write_text(json.dumps(proved, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    combined = dict(library, **proved)
    if combined:
        (args.output_dir / "verified_candidates.lean").write_text(library_lean_source(combined), encoding="utf-8")
    (args.output_dir / "proof_report.json").write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Lean verified {len(proved)}/{len(candidates)} candidate proofs ({sum(record['model_calls'] for record in reports)} model calls)")
    if candidates and not proved:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
