"""Cluster annotated wake theorems and propose candidate lemmas.

Proposals pass through structural relevance filtering, tree-edit deduplication,
and Lean proof verification before entering the library.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dreamprover.lean.source import theorem_name, split_statement_proof
from dreamprover.clients.completion import query_llm, run_informal_llm

ABSTRACTION_PROMPT = """You are a mathematician and an expert in Lean. These theorems were proved:
{theorem_section}
Propose general, reusable theorems in the {domain} domain that can be used to
prove these theorems. Align types and the number of variables with the inputs.
Use a separate ```lean``` block for each theorem. Do not include imports, open
commands, definitions or proofs. Output only theorem statements, without :=.
Do not duplicate theorem names. Make each proposed statement mathematically
and syntactically correct.
"""


def cluster_embeddings(embeddings, *, clusters: int | None = None, max_clusters: int = 12, seed: int = 0) -> tuple[list[int], dict]:
    """K-Means on normalized embeddings with an explicit reproducible elbow.

    Select the point of maximum distance below the endpoint chord of normalized
    inertia. Record the selection rule and random seed in the output metadata.
    """
    import numpy as np
    from sklearn.cluster import KMeans

    vectors = np.asarray(embeddings, dtype=float)
    if vectors.ndim != 2 or not len(vectors) or not vectors.shape[1]:
        raise ValueError("Embeddings must be a nonempty matrix")
    if not np.isfinite(vectors).all():
        raise ValueError("Embeddings must be finite")
    norms = np.linalg.norm(vectors, axis=1)
    if (norms == 0).any():
        raise ValueError("Zero embeddings have undefined cosine similarity")
    vectors = vectors / norms[:, None]
    unique_count = len(np.unique(vectors, axis=0))
    limit = min(max_clusters, len(vectors), unique_count)
    if max_clusters < 1 or (clusters is not None and not 1 <= clusters <= limit):
        raise ValueError(f"Cluster count must be between 1 and {limit}")
    if clusters is None and limit > 2:
        models = [KMeans(n_clusters=k, n_init=10, random_state=seed).fit(vectors) for k in range(1, limit + 1)]
        inertias = [float(model.inertia_) for model in models]
        x = np.linspace(0, 1, limit)
        y = (np.asarray(inertias) - inertias[-1]) / (inertias[0] - inertias[-1])
        selected = int(np.argmax((1 - x) - y))
        model = models[selected]
        rule = "maximum_distance_to_endpoint_chord"
    else:
        # Two points do not determine an elbow. Choose the single cluster,
        # unless a caller explicitly supplies k.
        clusters = clusters or 1
        model = KMeans(n_clusters=clusters, n_init=10, random_state=seed).fit(vectors)
        inertias = [float(model.inertia_)]
        rule = "explicit" if clusters != 1 or limit > 2 else "small_sample_single_cluster"
    # Canonical IDs make output invariant to sklearn's label permutation.
    label_map = {}
    labels = []
    for label in model.labels_:
        label_map.setdefault(int(label), len(label_map))
        labels.append(label_map[int(label)])
    return labels, {"clusters": len(label_map), "seed": seed, "normalized": True, "elbow_rule": rule, "inertias": inertias}


def parse_candidates(response: str) -> list[str]:
    blocks = re.findall(r"```(?:lean4?|Lean4?)\s*\n(.*?)```", response, re.DOTALL)
    statements = []
    for block in blocks:
        statement = block.strip()
        theorem_name(statement)
        if re.search(r"\n\s*(?:theorem|lemma|import|open|def)\b", statement):
            raise ValueError("Each candidate block must contain exactly one theorem statement")
        try:
            split_statement_proof(statement)
        except ValueError:
            pass
        else:
            raise ValueError("Candidate proposals must contain statements without proofs")
        statements.append(statement)
    if not statements:
        raise ValueError("No candidate Lean statements found")
    names = [theorem_name(statement) for statement in statements]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate names in candidate proposals")
    return statements


def propose_from_clusters(library: dict, embeddings, complete, *, domain: str, clusters: int | None = None, max_clusters: int = 12, seed: int = 0, name_prefix: str | None = None) -> dict:
    if not library:
        raise ValueError("Cannot abstract an empty library")
    names = list(library)
    if len(embeddings) != len(names):
        raise ValueError("One embedding is required per theorem in input order")
    for name in names:
        if not library[name].get("description"):
            raise ValueError(f"Missing description for {name}; run dreamprover.learning.annotate first")
    labels, metadata = cluster_embeddings(embeddings, clusters=clusters, max_clusters=max_clusters, seed=seed)
    candidates = []
    seen_names = set()
    for cluster in range(metadata["clusters"]):
        member_names = [name for name, label in zip(names, labels) if label == cluster]
        section = "\n\n".join(f"```lean\n{library[name]['statement']}\n```\n{library[name]['description']}" for name in member_names)
        response = complete(ABSTRACTION_PROMPT.format(theorem_section=section, domain=domain))
        for candidate_index, statement in enumerate(parse_candidates(response)):
            if name_prefix is not None:
                from dreamprover.lean.structural import _rename_declaration
                replacement = f"{name_prefix}{cluster}_{candidate_index}"
                if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", replacement):
                    raise ValueError("name_prefix must produce a valid unquoted Lean identifier")
                statement = _rename_declaration(statement, replacement)
            name = theorem_name(statement)
            if name in seen_names:
                raise ValueError(f"Duplicate candidate name across clusters: {name}")
            seen_names.add(name)
            candidates.append({"name": name, "statement": statement, "cluster": cluster, "generated_from": member_names, "verification_status": "unproved"})
    return {"domain": domain, "clustering": metadata, "clusters": {str(c): [name for name, label in zip(names, labels) if label == c] for c in range(metadata["clusters"])}, "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--domain", required=True)
    parser.add_argument("--embeddings", type=Path, help="JSON matrix in input library iteration order (offline/reproducible)")
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--proposals", type=Path, help="JSON list of recorded model responses, one per cluster; avoids API calls")
    parser.add_argument("--base-url")
    parser.add_argument("--model")
    parser.add_argument("--clusters", type=int)
    parser.add_argument("--max-clusters", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--name-prefix", help="Deterministic candidate names, e.g. dream_c1_")
    args = parser.parse_args()
    library = json.loads(args.input.read_text(encoding="utf-8"))
    if args.embeddings:
        embeddings = json.loads(args.embeddings.read_text(encoding="utf-8"))
    else:
        from sentence_transformers import SentenceTransformer
        embeddings = SentenceTransformer(args.embedding_model).encode([record["description"] for record in library.values()])
    responses = iter(json.loads(args.proposals.read_text(encoding="utf-8"))) if args.proposals else None
    if bool(args.base_url) != bool(args.model):
        parser.error("--base-url and --model must be supplied together")
    def complete(prompt):
        if responses is not None:
            try:
                return next(responses)
            except StopIteration as exc:
                raise ValueError("Recorded proposals contain fewer responses than clusters") from exc
        if args.base_url:
            return query_llm(prompt, api_url=args.base_url, model_name=args.model)["text"]
        return run_informal_llm(prompt)
    result = propose_from_clusters(library, embeddings, complete, domain=args.domain, clusters=args.clusters, max_clusters=args.max_clusters, seed=args.seed, name_prefix=args.name_prefix)
    result["clustering"]["embedding_source"] = str(args.embeddings) if args.embeddings else args.embedding_model
    result["proposal_source"] = str(args.proposals) if args.proposals else args.model or "configured_llm"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Proposed {len(result['candidates'])} unproved candidates from {result['clustering']['clusters']} clusters")


if __name__ == "__main__":
    main()
