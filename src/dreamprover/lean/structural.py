"""Lean-elaborated TBPS expression trees for sleep-stage comparisons.

No premise database is needed: Lean elaborates the declaration type, the pinned
TBPS serializer converts it to YourExpr, and the upstream CSE/tree algorithms
compare each pair. Scores are structural heuristics, never proof validation.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dreamprover.lean.source import mask_comments_and_strings, split_statement_proof, theorem_name
from dreamprover.paths import repository_root

REPO_ROOT = repository_root()
DEFAULT_RELEVANCE_THRESHOLD = 0.5
DEFAULT_DUPLICATE_THRESHOLD = 0.95

EXPORT_COMMAND = r'''
open Lean Elab Command
syntax "#dream_export_type " ident str : command
elab_rules : command
  | `(#dream_export_type $decl:ident $out:str) => do
    let info ← getConstInfo decl.getId
    let encoded ← Lean.Elab.Command.runTermElabM (fun _ =>
      exprToYourExprIter info.type 1000)
    let result := Lean.Json.mkObj [("your_expr", toJson encoded)]
    IO.FS.writeFile out.getString (toString result)
'''


class StructuralParseError(ValueError):
    """The statement cannot be reliably converted to a supported Lean tree."""


@dataclass(frozen=True)
class ParsedExpression:
    expression: Any
    tree: Any
    node_count: int
    json_expression: dict
    duplicate_expression: Any = None
    duplicate_node_count: int | None = None


def _canonical_binders(expression: dict) -> dict:
    """Remove alpha-renaming noise before upstream CSE names bound variables.

    The serializer uses de Bruijn indices for occurrences. Renaming binder
    declarations therefore preserves binding and mathematical meaning.
    """
    counter = 0

    def visit(value):
        nonlocal counter
        if not isinstance(value, dict) or len(value) != 1:
            raise StructuralParseError("Malformed TBPS expression node")
        kind, data = next(iter(value.items()))
        if kind in {"fvar", "mvar"}:
            raise StructuralParseError(f"Unresolved {kind} in a closed theorem type")
        if kind in {"sort", "const", "bvar", "lit"}:
            return value
        if kind == "app":
            return {kind: {"fn": visit(data["fn"]), "arg": visit(data["arg"])}}
        if kind in {"lam", "forallE"}:
            name = f"dream_bound_{counter}"
            counter += 1
            return {kind: dict(data, binderName=name, binderType=visit(data["binderType"]), body=visit(data["body"]))}
        if kind == "letE":
            # Upstream CSE's de-Bruijn conversion interprets the let value in
            # the extended body scope. Avoid relying on that unsupported case.
            raise StructuralParseError("letE types are unsupported by the upstream CSE binding conversion")
        if kind == "mdata":
            return {kind: dict(data, expr=visit(data["expr"]))}
        if kind == "proj":
            return {kind: dict(data, struct=visit(data["struct"]))}
        raise StructuralParseError(f"Unknown TBPS expression constructor: {kind}")

    return visit(expression)


def _rename_declaration(statement: str, replacement: str) -> str:
    pattern = r'^(\s*(?:theorem|lemma)\s+)(«[^»]+»|[^\s:({\[⦃]+)'
    def replace(match):
        universe_dot = "." if match.group(2).endswith(".") and statement[match.end():].startswith("{") else ""
        return match.group(1) + replacement + universe_dot
    return re.sub(pattern, replace, statement, count=1)


class StructuralComparator:
    """Local Lean/TBPS parser with reusable on-disk expression caching."""

    def __init__(self, project_root: str | Path, parser_dir: str | Path | None = None, tbps_root: str | Path | None = None, *, timeout: float = 120, cache_dir: str | Path | None = None, batch_size: int = 32):
        self.project_root = Path(project_root).expanduser().resolve()
        if not self.project_root.is_dir():
            raise FileNotFoundError(f"Lean project does not exist: {self.project_root}")
        toolchain_file = self.project_root / "lean-toolchain"
        self.toolchain = toolchain_file.read_text().strip() if toolchain_file.is_file() else "unknown"
        if parser_dir is None:
            version = self.toolchain.rsplit(":", 1)[-1]
            suffix = "tbps-v4.15" if version == "v4.15.0" else "tbps"
            parser_dir = REPO_ROOT / "artifacts" / suffix / "parser"
        self.parser_dir = Path(parser_dir).expanduser().resolve()
        module = self.parser_dir / "Mathlib_Construction.olean"
        if not module.is_file():
            raise FileNotFoundError(f"Matching TBPS parser missing: {module}. Build it with scripts/setup_tbps.py and the project's Lean version.")
        self.tbps_root = Path(tbps_root or REPO_ROOT / "vendor" / "tbps").expanduser().resolve()
        backend = self.tbps_root / "tbps-be"
        if not (backend / "search_app" / "myexpr.py").is_file():
            raise FileNotFoundError(f"TBPS backend missing: {backend}")
        # Only import pure expression/tree routines; PostgreSQL is not required.
        backend_string = str(backend)
        if backend_string not in sys.path:
            sys.path.insert(0, backend_string)
        self.myexpr = importlib.import_module("search_app.myexpr")
        self.cse = importlib.import_module("search_app.cse")
        self.compute = importlib.import_module("search_app.compute.zss_compute")
        imported_root = Path(self.myexpr.__file__).resolve().parents[1]
        if imported_root != backend:
            raise RuntimeError(f"Another TBPS backend is already imported: {imported_root}")
        self.timeout = timeout
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else None
        self._memory_cache: dict[str, ParsedExpression] = {}
        self.parser_hash = hashlib.sha256(module.read_bytes()).hexdigest()
        revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=self.tbps_root, capture_output=True, text=True, check=False)
        self.revision = revision.stdout.strip() if revision.returncode == 0 else "unknown"

    def metadata(self) -> dict:
        return {
            "implementation": "upstream_tbps_lean_type_trees",
            "tbps_revision": self.revision,
            "lean_toolchain": self.toolchain,
            "parser_sha256": self.parser_hash,
            "simplification": "alpha_canonical_binders_then_upstream_cse_then_simplify_forall_expr_iter",
            "relevance_algorithm": "can_t1_collapse_match_t2_soft",
            "collapse_direction": "wake_goal_to_candidate_premise",
            "duplicate_algorithm": "zss_edit_distance_full_typed_tree_normalized_by_max_nodes",
            "duplicate_representation": "full_alpha_canonical_binder_converted_typed_tree_without_cse",
            "edit_costs": "upstream_zss_edit_distance_unit_cost",
            "batch_preload_size": self.batch_size,
        }

    def _decode(self, encoded: dict) -> ParsedExpression:
        normalized = _canonical_binders(encoded)
        try:
            original = self.myexpr.deserialize_expr(normalized)
            # Retrieval intentionally abstracts repeated subexpressions. That
            # abstraction can erase operator identity, making a positive sum
            # and positive product identical. Duplicate pruning instead keeps
            # the complete typed expression, including all constant labels.
            duplicate_expression = self.cse.deBruijn_to_bindername(original)
            duplicate_nodes = self.compute.count_nodes(self.compute.your_expr_to_treenode(duplicate_expression))
            expression = self.cse.cse(original)
            expression = self.myexpr.simplify_forall_expr_iter(expression)
            tree = self.compute.your_expr_to_treenode(expression)
            count = self.compute.count_nodes(tree)
        except (ValueError, KeyError, IndexError, RecursionError) as exc:
            raise StructuralParseError(f"Upstream TBPS cannot simplify this expression: {exc}") from exc
        if count < 1:
            raise StructuralParseError("Empty expression tree")
        return ParsedExpression(expression, tree, count, encoded, duplicate_expression, duplicate_nodes)

    def _statement_key(self, statement: str, header: str) -> tuple[str, str]:
        name = theorem_name(statement)
        try:
            split_statement_proof(statement)
        except ValueError:
            pass
        else:
            raise StructuralParseError("Structural comparisons require statements without outer proof assignments")
        masked = mask_comments_and_strings(statement)
        commands = r"(?:theorem|lemma|import|open|namespace|section|end|axiom|constant|def|abbrev|opaque|instance|structure|class|inductive|set_option|attribute|syntax|elab|macro|notation|variable|noncomputable|private|protected)"
        additional = re.search(rf"\n\s*{commands}\b", masked)
        if additional:
            raise StructuralParseError("Candidate contains an additional Lean declaration")
        canonical_statement = _rename_declaration(statement, "dream_structural_type")
        key = hashlib.sha256(json.dumps([self.toolchain, self.parser_hash, self.revision, header.strip(), canonical_statement], ensure_ascii=False).encode()).hexdigest()
        return name, key

    def _cached(self, key: str) -> ParsedExpression | None:
        if key in self._memory_cache:
            return self._memory_cache[key]
        cache_path = self.cache_dir / f"{key}.json" if self.cache_dir else None
        if cache_path and cache_path.is_file():
            try:
                parsed = self._decode(json.loads(cache_path.read_text())["your_expr"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise StructuralParseError(f"Malformed structural cache: {cache_path}") from exc
            self._memory_cache[key] = parsed
            return parsed
        return None

    def _store(self, key: str, encoded: dict, parsed: ParsedExpression) -> None:
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = self.cache_dir / f"{key}.json"
            with tempfile.NamedTemporaryFile(mode="w", dir=self.cache_dir, prefix=key + "-", suffix=".tmp", encoding="utf-8", delete=False) as cache_file:
                cache_file.write(json.dumps(encoded, ensure_ascii=False) + "\n")
                temporary_path = Path(cache_file.name)
            temporary_path.replace(cache_path)
        self._memory_cache[key] = parsed

    def _compile_types(self, lean_file: Path) -> subprocess.CompletedProcess:
        env = os.environ.copy()
        env["LEAN_PATH"] = os.pathsep.join(filter(None, [str(self.parser_dir), env.get("LEAN_PATH")]))
        try:
            return subprocess.run(["lake", "env", "lean", str(lean_file)], cwd=self.project_root, env=env, capture_output=True, text=True, timeout=self.timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise StructuralParseError(f"Lean structural parser failed: {exc}") from exc

    def preload(self, statements: list[tuple[str, str]]) -> None:
        """Cache valid types in batches without relaxing individual validation.

        Only an entirely successful Lean compilation supplies cached trees. If
        any declaration makes a batch fail, ordinary per-statement parsing
        subsequently supplies its precise diagnostic. Unsupported TBPS trees
        are likewise left uncached and rejected by the individual parser.
        """
        groups = {}
        for statement, header in statements:
            try:
                _, key = self._statement_key(statement, header)
                if self._cached(key) is not None:
                    continue
            except (ValueError, KeyError):
                continue
            groups.setdefault(header.strip(), {})[key] = statement
        for header, by_key in groups.items():
            items = list(by_key.items())
            for start in range(0, len(items), self.batch_size):
                with tempfile.TemporaryDirectory(prefix="dream-structure-batch-", dir=self.project_root) as temporary:
                    directory = Path(temporary)
                    source = "import Mathlib_Construction\n" + header + "\n" + EXPORT_COMMAND + "\n"
                    exports = []
                    for index, (key, statement) in enumerate(items[start:start + self.batch_size]):
                        name = "dream_structural_" + key[:24]
                        output = directory / f"{index}.json"
                        source += _rename_declaration(statement, name) + " := by sorry\n"
                        source += f"#dream_export_type {name} {json.dumps(str(output))}\n"
                        exports.append((key, output))
                    lean_file = directory / "Parse.lean"
                    lean_file.write_text(source, encoding="utf-8")
                    try:
                        result = self._compile_types(lean_file)
                    except StructuralParseError:
                        continue
                    if result.returncode != 0:
                        continue
                    for key, output in exports:
                        try:
                            encoded = json.loads(output.read_text())
                            parsed = self._decode(encoded["your_expr"])
                        except (OSError, ValueError, KeyError):
                            continue
                        self._store(key, encoded, parsed)

    def parse_statement(self, statement: str, header: str = "") -> ParsedExpression:
        name, key = self._statement_key(statement, header)
        cached = self._cached(key)
        if cached is not None:
            return cached
        with tempfile.TemporaryDirectory(prefix="dream-structure-", dir=self.project_root) as temporary:
            directory = Path(temporary)
            output = directory / "type.json"
            # Absolute output filename is a JSON-escaped Lean string literal.
            temporary_name = "dream_structural_" + key[:24]
            renamed = _rename_declaration(statement, temporary_name)
            source = "import Mathlib_Construction\n" + header + "\n" + EXPORT_COMMAND + "\n" + renamed + " := by sorry\n" + f"#dream_export_type {temporary_name} {json.dumps(str(output))}\n"
            lean_file = directory / "Parse.lean"
            lean_file.write_text(source, encoding="utf-8")
            result = self._compile_types(lean_file)
            if result.returncode != 0 or not output.is_file():
                raise StructuralParseError(f"Lean could not elaborate {name}:\n{result.stdout}{result.stderr}")
            try:
                encoded = json.loads(output.read_text())
                parsed = self._decode(encoded["your_expr"])
            except (json.JSONDecodeError, KeyError) as exc:
                raise StructuralParseError("Lean structural parser returned malformed JSON") from exc
        self._store(key, encoded, parsed)
        return parsed

    def relevance(self, candidate: ParsedExpression, wake_theorem: ParsedExpression) -> float:
        # The expanded goal can collapse to the premise's simpler expression.
        score = float(self.compute.can_t1_collapse_match_t2_soft(wake_theorem.tree, candidate.tree))
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid upstream structural relevance score: {score}")
        return score

    def duplicate_similarity(self, first: ParsedExpression, second: ParsedExpression) -> float:
        first_expression = first.expression if first.duplicate_expression is None else first.duplicate_expression
        second_expression = second.expression if second.duplicate_expression is None else second.duplicate_expression
        distance = float(self.compute.zss_edit_distance(first_expression, second_expression))
        if math.isinf(distance):
            return 0.0  # upstream rejects pairs with a node-size ratio >1.5
        if not math.isfinite(distance) or distance < 0:
            raise ValueError(f"Invalid upstream tree-edit distance: {distance}")
        return max(0.0, 1.0 - distance / max(first.duplicate_node_count or first.node_count, second.duplicate_node_count or second.node_count))
