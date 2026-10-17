"""Load explicit lemma libraries without inventing axioms or placeholder proofs."""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from dreamprover.lean.source import split_statement_proof, mask_comments_and_strings

from dreamprover.lean.helpers import extract_theorem_signature


def assert_complete_source(source: str) -> None:
    """Conservatively reject proof holes and newly declared axioms in generated code.

    Lean compilation remains the authority for types and elaboration. This guard
    also catches holes when a server omits or suppresses its warning messages.
    """
    code = mask_comments_and_strings(source)
    if re.search(r"\b(sorry|admit|sorryAx|axiom)\b", code):
        raise ValueError("Lean source contains an unproved declaration (sorry/admit/axiom)")


def combine_lean_sources(*sources: str) -> str:
    """Place module imports before context commands and declarations."""
    imports = []
    bodies = []
    for source in sources:
        body = []
        for line in source.splitlines():
            if re.match(r"^\s*import\s+", line):
                if line.strip() not in imports:
                    imports.append(line.strip())
            else:
                body.append(line)
        bodies.append("\n".join(body).strip())
    return "\n".join(imports + [body for body in bodies if body]) + "\n"


@dataclass(frozen=True)
class LemmaLibrary:
    source: str = ""
    context: str = ""

    def with_header(self, header: str) -> str:
        return combine_lean_sources(header, self.source)

    @classmethod
    def from_file(cls, path):
        path = Path(path)
        text = path.read_text(encoding="utf-8")
        if path.suffix == ".lean":
            assert_complete_source(text)
            return cls(source=text, context=text)
        records = json.loads(text)
        if isinstance(records, list):
            records = {str(record.get("name", i)): record for i, record in enumerate(records)}
        if not isinstance(records, dict):
            raise ValueError("Lemma library must be a Lean file or JSON dictionary/list")

        ordered = []
        visiting, visited = set(), set()

        def visit(name):
            if name in visited:
                return
            if name in visiting:
                raise ValueError(f"Cyclic lemma dependency at {name}")
            record = records[name]
            if not isinstance(record, dict):
                raise ValueError(f"Library record {name} must be an object")
            status = record.get("verification_status")
            if status is not None and status not in {"verified", "source_only"}:
                raise ValueError(f"Library lemma {name} has unproved status {status!r}")
            visiting.add(name)
            for child in record.get("children", []):
                if child in records:
                    visit(child)
            visiting.remove(name)
            visited.add(name)
            ordered.append((name, record))

        for name in records:
            visit(name)
        source_parts, context_parts = [], []
        for name, record in ordered:
            statement = record.get("statement", "").strip()
            proof = record.get("proof", "")
            if not isinstance(proof, str) or not statement or not proof.strip():
                raise ValueError(f"Library lemma {name} requires a statement and actual proof")
            signature = extract_theorem_signature(statement)
            try:
                split_statement_proof(statement)
                has_proof_assignment = True
            except ValueError:
                has_proof_assignment = False
            if not signature or has_proof_assignment:
                raise ValueError(f"Library lemma {name} statement must be a theorem declaration without :=")
            # Original exported records contain only indented tactic lines;
            # new exports contain the entire RHS (by ... or a proof term).
            rhs = proof if proof[:1].isspace() and not proof.lstrip().startswith("by") else proof.strip()
            if proof[:1].isspace() and not proof.lstrip().startswith("by"):
                rhs = "by\n" + proof.rstrip()
            declaration = statement + " := " + rhs + "\n"
            assert_complete_source(declaration)
            source_parts.extend([record.get("header", ""), declaration])
            context_parts.append(signature + ("\n" + record["description"] if record.get("description") else ""))
        source = combine_lean_sources(*source_parts)
        assert_complete_source(source)
        return cls(source=source, context="\n\n".join(context_parts))
