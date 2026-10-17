"""Extract a portable lemma library from exported wake-stage Lean proofs.

Run `python -m dreamprover.learning.extract --help`. Importing this module performs no
filesystem writes. Extraction alone never marks a declaration as verified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dreamprover.lean.source import extract_declarations, referenced_names, split_statement_proof, theorem_name
from dreamprover.lean.compiler import check_lean_source

# Backward-compatible helper names, with a correct statement/proof split.
extract_theorem_name = theorem_name
extract_statement_proof = split_statement_proof


def extract_theorems_header(lean_code_list):
    header, declarations = extract_declarations("".join(lean_code_list))
    return [decl.source + "\n" for decl in declarations], header


def build_library(proof_dir: str | Path, *, verify: bool = False, project_root=None, timeout: float = 120) -> dict[str, dict[str, Any]]:
    directory = Path(proof_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Proof directory not found: {directory}")
    library: dict[str, dict[str, Any]] = {}
    for path in sorted(directory.rglob("*.lean")):
        source = path.read_text(encoding="utf-8")
        header, declarations = extract_declarations(source)
        if verify:
            check = check_lean_source(source, project_root=project_root, timeout=timeout)
            if not check["proved"]:
                raise ValueError(f"Proof file did not pass verification without sorry: {path}\n{check['stdout']}{check['stderr']}")
        names = {decl.name for decl in declarations}
        for declaration in declarations:
            if declaration.name in library:
                raise ValueError(f"Duplicate theorem name {declaration.name!r} in {path}; use unique exported names")
            library[declaration.name] = {
                "statement": declaration.statement,
                "header": header,
                "proof": declaration.proof,
                "children": [name for name in referenced_names(declaration.statement + "\n" + declaration.proof, names) if name != declaration.name],
                "parents": [],
                "source_file": str(path.resolve()),
                "verification_status": "verified" if verify else "source_only",
                "usage_count": 0,
                "last_used_cycle": -1,
            }
    # Track dependencies across files too; Lean verification still determines
    # whether those references are available under the file's own imports.
    names = set(library)
    for name, record in library.items():
        record["children"] = [child for child in referenced_names(record["statement"] + "\n" + record["proof"], names) if child != name]
        for child in record["children"]:
            library[child]["parents"].append(name)
    return library


def dependency_order(library: dict[str, dict[str, Any]]) -> list[str]:
    ordered = []
    visiting = set()
    visited = set()

    def visit(name):
        if name in visiting:
            raise ValueError(f"Cyclic library dependency involving {name}")
        if name in visited:
            return
        visiting.add(name)
        for child in library[name].get("children", []):
            if child not in library:
                raise ValueError(f"Missing library dependency {child!r} required by {name!r}")
            visit(child)
        visiting.remove(name)
        visited.add(name)
        ordered.append(name)

    for name in library:
        visit(name)
    return ordered


def library_lean_source(library: dict[str, dict[str, Any]], *, only_reused: bool = False) -> str:
    headers = {record.get("header", "").strip() for record in library.values()}
    if len(headers) > 1:
        raise ValueError("Cannot combine declarations with different preambles; export separate libraries")
    header = next(iter(headers), "")
    selected = {name for name, record in library.items() if not only_reused or record.get("parents")}
    # A reused lemma may itself depend on other extracted declarations.
    pending = list(selected)
    while pending:
        for child in library[pending.pop()].get("children", []):
            if child not in selected:
                selected.add(child)
                pending.append(child)
    declarations = []
    for name in dependency_order(library):
        if name not in selected:
            continue
        record = library[name]
        if record.get("verification_status") != "verified":
            raise ValueError(f"{name!r} is not verified; build with --verify before exporting Lean")
        if not record.get("proof", "").strip():
            raise ValueError(f"No proof stored for {name!r}")
        declarations.append(f"{record['statement']} := {record['proof']}\n")
    return header + "\n\n" + "\n".join(declarations)


def write_library(library: dict[str, dict[str, Any]], output_dir: str | Path, *, export_lean: bool = False, only_reused: bool = False) -> None:
    # Validate requested export before creating output files.
    source = library_lean_source(library, only_reused=only_reused) if export_lean else None
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "full_library.json").write_text(json.dumps(library, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    graph = {name: {"children": record["children"]} for name, record in library.items()}
    (directory / "simple_library.json").write_text(json.dumps(graph, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if source is not None:
        (directory / "lemmas.lean").write_text(source, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proof-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--verify", action="store_true", help="Check exported proof files with Lean, rejecting sorry/admit")
    parser.add_argument("--lean-project", type=Path, help="Built Lake project (or set LEAN_PROJECT_ROOT)")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--export-lean", action="store_true", help="Export actual proofs; requires --verify")
    parser.add_argument("--only-reused", action="store_true", help="Export only referenced lemmas and their dependencies")
    args = parser.parse_args()
    library = build_library(args.proof_dir, verify=args.verify, project_root=args.lean_project, timeout=args.timeout)
    write_library(library, args.output_dir, export_lean=args.export_lean, only_reused=args.only_reused)
    print(f"Extracted {len(library)} declarations into {args.output_dir}")


if __name__ == "__main__":
    main()
