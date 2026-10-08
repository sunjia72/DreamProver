"""Verify proved candidate exports and update a bounded lemma library.

Admission uses actual Lean proofs. A configured StructuralComparator performs
normalized upstream TBPS tree-edit duplicate pruning on full typed trees,
preserving operator and constant identities. Without a structural comparator,
duplicates are matched by normalized statements.
Capacity pruning ranks frequency, then recency, and preserves dependencies.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dreamprover.learning.extract import build_library, dependency_order, library_lean_source, write_library
from dreamprover.lean.source import NAME_PATTERN, mask_comments_and_strings, referenced_names
from dreamprover.lean.compiler import check_lean_source


def _statement_key(statement: str) -> str:
    without_name = re.sub(rf"^(?:theorem|lemma)\s+{NAME_PATTERN}", "theorem", statement.strip(), count=1)
    # Keep string literal whitespace: changing it changes the proposition.
    return " ".join(re.findall(r'"(?:\\.|[^"\\])*"|[^\s"]+', without_name))


def update_library(library: dict, candidates: dict, *, capacity: int = 99, cycle: int = 0, usage_sources: list[str] | None = None, project_root=None, timeout: float = 120, structural_comparator=None, duplicate_threshold: float = 0.95) -> tuple[dict, dict]:
    if not 0 <= duplicate_threshold <= 1:
        raise ValueError("duplicate_threshold must be between 0 and 1")
    if capacity < 1 or cycle < 0:
        raise ValueError("Capacity must be positive and cycle nonnegative")
    combined = {name: dict(record) for name, record in library.items()}
    for name, record in combined.items():
        if record.get("verification_status") != "verified":
            raise ValueError(f"Existing library lemma {name!r} is not verified")
    usages = usage_sources or []
    for name, record in combined.items():
        count = sum(name in referenced_names(source, {name}) for source in usages)
        record["usage_count"] = record.get("usage_count", 0) + count
        if count:
            record["last_used_cycle"] = cycle
    duplicates = []
    duplicate_support = {}
    admitted = []
    keys = {_statement_key(record["statement"]) for record in combined.values()}
    tree_references = {}
    duplicate_details = []
    if structural_comparator:
        structural_comparator.preload([(record["statement"], record.get("header", ""))
                                      for record in list(combined.values()) + list(candidates.values())])
        for name, record in combined.items():
            tree_references[name] = structural_comparator.parse_statement(record["statement"], record.get("header", ""))
    for name, record in candidates.items():
        if record.get("verification_status") != "verified":
            raise ValueError(f"Candidate {name!r} is not verified; extract candidates with --verify")
        key = _statement_key(record["statement"])
        if name in combined and _statement_key(combined[name]["statement"]) != key:
            raise ValueError(f"Candidate name conflicts with existing lemma: {name}")
        parsed = None
        duplicate_of = None
        similarity = None
        if structural_comparator:
            parsed = structural_comparator.parse_statement(record["statement"], record.get("header", ""))
            scores = {reference: structural_comparator.duplicate_similarity(parsed, tree) for reference, tree in tree_references.items()}
            if scores:
                nearest = max(scores, key=scores.get)
                if scores[nearest] >= duplicate_threshold:
                    duplicate_of, similarity = nearest, scores[nearest]
        elif key in keys:
            duplicate_of = next((reference for reference, existing in combined.items() if _statement_key(existing["statement"]) == key), None)
            similarity = 1.0
        if duplicate_of is not None:
            duplicates.append(name)
            duplicate_details.append({"name": name, "duplicate_of": duplicate_of, "similarity": similarity})
            duplicate_support[name] = dict(record, usage_count=0, last_used_cycle=cycle)
            continue
        combined[name] = dict(record, usage_count=0, last_used_cycle=cycle)
        keys.add(key)
        if structural_comparator:
            tree_references[name] = parsed
        admitted.append(name)
    # Deduplicated declarations can still be required by another candidate's
    # proof. Retain those support proofs; dropping their names would make the
    # exported library invalid. Dependencies are expanded transitively.
    needed = set()
    known_names = set(combined) | set(duplicate_support)
    pending = list(combined.values())
    while pending:
        record = pending.pop()
        support_names = referenced_names(record["statement"] + "\n" + record["proof"], known_names)
        for support_name in support_names:
            if support_name in duplicate_support and support_name not in needed and support_name not in combined:
                needed.add(support_name)
                pending.append(duplicate_support[support_name])
    for name in duplicate_support:
        if name in needed:
            combined[name] = duplicate_support[name]
    # Determine dependencies again because newly added proofs can use the
    # existing library. Consumers are diagnostic metadata, not proof support.
    names = set(combined)
    for name, record in combined.items():
        record["children"] = [child for child in referenced_names(record["statement"] + "\n" + record["proof"], names) if child != name]
        record["parents"] = []
    for name, record in combined.items():
        for child in record["children"]:
            combined[child]["parents"].append(name)
    dependency_order(combined)
    ranked = sorted(combined, key=lambda name: (-combined[name].get("usage_count", 0), -combined[name].get("last_used_cycle", -1), name))
    retained = set()
    def closure(name):
        result = {name}
        for child in combined[name]["children"]:
            result.update(closure(child))
        return result
    for name in ranked:
        proof_support = closure(name)
        if len(retained | proof_support) <= capacity:
            retained.update(proof_support)
    result = {name: combined[name] for name in dependency_order(combined) if name in retained}
    for record in result.values():
        record["parents"] = [parent for parent in record["parents"] if parent in retained]
    # Verify the final combined source, including the original library: labels
    # from input files alone are insufficient to establish library soundness.
    if result:
        check = check_lean_source(library_lean_source(result), project_root=project_root, timeout=timeout)
        if not check["proved"]:
            raise ValueError(f"Updated library does not verify without sorry:\n{check['stdout']}{check['stderr']}")
    dedupe_metadata = dict(structural_comparator.metadata(), duplicate_threshold=duplicate_threshold) if structural_comparator else {"implementation": "exact_normalized_statement"}
    return result, {"cycle": cycle, "capacity": capacity, "deduplication": "upstream_normalized_tree_edit" if structural_comparator else "exact_normalized_statement", "structural_deduplication": dedupe_metadata, "duplicate_details": duplicate_details, "retention": "frequency_then_recency_with_dependency_closure", "admitted": [name for name in admitted if name in result], "duplicates": duplicates, "retained_duplicate_support": sorted(needed & retained), "forgotten": sorted(set(combined) - retained)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, help="Existing verified full_library.json; absent means an empty library")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--candidate-proof-dir", type=Path, help="Exported proved Lean candidates, already structurally filtered")
    inputs.add_argument("--candidate-library", type=Path, help="verified_candidates.json from dreamprover.learning.prove")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path)
    parser.add_argument("--capacity", type=int, default=99)
    parser.add_argument("--cycle", type=int, default=0)
    parser.add_argument("--usage-proof-dir", type=Path, help="Wake proofs for counting each lemma's use per proof file")
    parser.add_argument("--dedupe", choices=["tree", "exact"], default="tree")
    parser.add_argument("--parser-dir", type=Path)
    parser.add_argument("--tbps-root", type=Path)
    parser.add_argument("--structural-cache", type=Path)
    parser.add_argument("--duplicate-threshold", type=float, default=0.95)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    library = json.loads(args.library.read_text(encoding="utf-8")) if args.library else {}
    candidates = json.loads(args.candidate_library.read_text(encoding="utf-8")) if args.candidate_library else build_library(args.candidate_proof_dir, verify=True, project_root=args.lean_project, timeout=args.timeout)
    sources = [path.read_text(encoding="utf-8") for path in sorted(args.usage_proof_dir.rglob("*.lean"))] if args.usage_proof_dir else []
    comparator = None
    if args.dedupe == "tree":
        from dreamprover.lean.structural import StructuralComparator
        from dreamprover.lean.config import LEAN_PROJECT_ROOT
        comparator = StructuralComparator(args.lean_project or LEAN_PROJECT_ROOT, args.parser_dir, args.tbps_root, timeout=args.timeout, cache_dir=args.structural_cache)
    result, report = update_library(library, candidates, capacity=args.capacity, cycle=args.cycle, usage_sources=sources, project_root=args.lean_project, timeout=args.timeout, structural_comparator=comparator, duplicate_threshold=args.duplicate_threshold)
    write_library(result, args.output_dir, export_lean=True)
    (args.output_dir / "update_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Stored {len(result)} verified lemmas; admitted {len(report['admitted'])}, duplicate {len(report['duplicates'])}, forgotten {len(report['forgotten'])}")


if __name__ == "__main__":
    main()
