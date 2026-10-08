"""Filter sleep proposals using Lean/TBPS expression-tree comparisons.

Outputs record configurable thresholds and pinned upstream algorithms. Parser
failures reject candidates. Relevance uses TBPS's CSE abstraction, while
comparison for duplicates retains full typed trees to distinguish operators.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from dreamprover.lean.structural import (
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_RELEVANCE_THRESHOLD,
    StructuralComparator,
    StructuralParseError,
)


def _threshold(value: float, name: str) -> None:
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")


def filter_candidates(proposals: dict, source_library: dict, comparator: StructuralComparator, *, relevance_threshold: float = DEFAULT_RELEVANCE_THRESHOLD) -> dict:
    """Keep a candidate only when max cluster relevance exceeds the threshold."""
    _threshold(relevance_threshold, "relevance_threshold")
    result = copy.deepcopy(proposals)
    kept = []
    rejected = list(result.get("rejected_candidates", []))
    parsed_sources = {}
    to_parse = [(record["statement"], record.get("header", "")) for record in source_library.values()]
    for candidate in proposals.get("candidates", []):
        members = proposals.get("clusters", {}).get(str(candidate.get("cluster")), candidate.get("generated_from", []))
        headers = {source_library[name].get("header", "").strip() for name in members if name in source_library}
        if len(headers) == 1 and "statement" in candidate:
            to_parse.append((candidate["statement"], next(iter(headers))))
    comparator.preload(to_parse)
    for candidate in proposals.get("candidates", []):
        try:
            if candidate.get("verification_status", "unproved") != "unproved":
                raise ValueError("Proposal must be explicitly unproved before structural filtering")
            members = proposals.get("clusters", {}).get(str(candidate.get("cluster")), candidate.get("generated_from"))
            if not members:
                raise ValueError("Candidate has no source cluster")
            if candidate.get("generated_from") and set(candidate["generated_from"]) != set(members):
                raise ValueError("Candidate provenance does not match its source cluster")
            missing = set(members) - set(source_library)
            if missing:
                raise ValueError(f"Missing source theorems: {sorted(missing)}")
            headers = {source_library[name].get("header", "").strip() for name in members}
            if len(headers) != 1:
                raise StructuralParseError("Source cluster uses incompatible Lean preambles")
            header = next(iter(headers))
            proposed_tree = comparator.parse_statement(candidate["statement"], header)
            scores = {}
            for name in members:
                if name not in parsed_sources:
                    record = source_library[name]
                    parsed_sources[name] = comparator.parse_statement(record["statement"], record.get("header", ""))
                scores[name] = comparator.relevance(proposed_tree, parsed_sources[name])
            nearest = max(scores, key=scores.get)
            score = scores[nearest]
            filtered = dict(candidate, header=header + "\n", structural_relevance={"score": score, "nearest_source": nearest, "source_scores": scores, "node_count": proposed_tree.node_count})
            # §3.2 says the maximum must exceed the threshold (strict >).
            if score > relevance_threshold:
                kept.append(filtered)
            else:
                rejected.append(dict(filtered, rejection_reason="below_structural_relevance_threshold"))
        except (ValueError, KeyError) as exc:
            rejected.append(dict(candidate, rejection_reason="structural_parse_or_provenance_error", structural_error=str(exc)))
    result["candidates"] = kept
    result["rejected_candidates"] = rejected
    result["structural_filter"] = dict(comparator.metadata(), relevance_threshold=relevance_threshold, comparison="maximum_cluster_score_strictly_greater_than_threshold", input_candidates=len(proposals.get("candidates", [])), retained_candidates=len(kept), rejected_candidates=len(rejected))
    return result


def deduplicate_candidates(proposals: dict, existing_library: dict, comparator: StructuralComparator, *, duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD) -> dict:
    """Remove high-tree-similarity proposals before spending proof attempts."""
    _threshold(duplicate_threshold, "duplicate_threshold")
    result = copy.deepcopy(proposals)
    comparator.preload([(record["statement"], record.get("header", "")) for record in existing_library.values()] +
                       [(candidate["statement"], candidate.get("header", "")) for candidate in proposals.get("candidates", [])])
    references = []
    for name, record in existing_library.items():
        # Failing to parse a supposed existing reference stops deduplication;
        # ignoring it would make the configured duplicate policy misleading.
        references.append((name, comparator.parse_statement(record["statement"], record.get("header", "")), "existing_library"))
    kept = []
    duplicates = list(result.get("duplicate_candidates", []))
    errors = list(result.get("rejected_candidates", []))
    for candidate in proposals.get("candidates", []):
        try:
            parsed = comparator.parse_statement(candidate["statement"], candidate.get("header", ""))
        except (ValueError, KeyError) as exc:
            errors.append(dict(candidate, rejection_reason="structural_duplicate_parse_error", structural_error=str(exc)))
            continue
        scores = [(name, comparator.duplicate_similarity(parsed, tree), kind) for name, tree, kind in references]
        duplicate = max(scores, key=lambda item: item[1]) if scores else None
        if duplicate and duplicate[1] >= duplicate_threshold:
            duplicates.append(dict(candidate, duplicate_of=duplicate[0], duplicate_similarity=duplicate[1], duplicate_reference=duplicate[2]))
        else:
            kept.append(candidate)
            references.append((candidate["name"], parsed, "earlier_candidate"))
    result["candidates"] = kept
    result["duplicate_candidates"] = duplicates
    result["rejected_candidates"] = errors
    result["structural_deduplication"] = dict(comparator.metadata(), duplicate_threshold=duplicate_threshold, comparison="normalized_tree_edit_similarity_greater_or_equal", retained_candidates=len(kept), duplicate_candidates=len(duplicates))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="dreamprover.learning.sleep proposal JSON")
    parser.add_argument("--sources", type=Path, required=True, help="Annotated wake theorem library JSON")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--library", type=Path, help="Previous verified library for duplicate pruning")
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--parser-dir", type=Path)
    parser.add_argument("--tbps-root", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--relevance-threshold", type=float, default=DEFAULT_RELEVANCE_THRESHOLD)
    parser.add_argument("--duplicate-threshold", type=float, default=DEFAULT_DUPLICATE_THRESHOLD)
    parser.add_argument("--skip-deduplication", action="store_true")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    proposals = json.loads(args.input.read_text(encoding="utf-8"))
    sources = json.loads(args.sources.read_text(encoding="utf-8"))
    library = json.loads(args.library.read_text(encoding="utf-8")) if args.library else {}
    comparator = StructuralComparator(args.lean_project, args.parser_dir, args.tbps_root, timeout=args.timeout, cache_dir=args.cache_dir)
    result = filter_candidates(proposals, sources, comparator, relevance_threshold=args.relevance_threshold)
    if not args.skip_deduplication:
        result = deduplicate_candidates(result, library, comparator, duplicate_threshold=args.duplicate_threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Retained {len(result['candidates'])}/{len(proposals['candidates'])} unproved candidates; rejected {len(result['rejected_candidates'])}, duplicate {len(result.get('duplicate_candidates', []))}")


if __name__ == "__main__":
    main()
