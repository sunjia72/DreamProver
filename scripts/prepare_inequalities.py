#!/usr/bin/env python3
"""Prepare reproducible inequality training and evaluation datasets."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import urllib.request
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AIPS_REVISION = "b9c476cb1c847e5141f7fb0caf9d232b3ea5b721"
AIPS_SHA256 = "d420d14aa0515122de15405ab7c285cf0b138260f8e16b9eafe2d40eafcbd443"
AIPS_URL = f"https://huggingface.co/datasets/llllvvuu/AIPS_inequalities/resolve/{AIPS_REVISION}/inequalities.parquet"
HEADER = "import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 0\n\nopen BigOperators Real Nat Topology Rat\n\n"
BENCHMARKS = {"567NEQ": "neq_test_567.jsonl", "ChenNEQ": "neq_test_chen.jsonl", "MO-INT": "neq_test_MO.jsonl"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(node: ast.AST) -> Fraction | None:
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return Fraction(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = rational(node.operand)
        return -value if value is not None and isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        left, right = rational(node.left), rational(node.right)
        if left is not None and right:
            return left / right
    return None


def lean_expression(node: ast.AST) -> str:
    """Translate a whitelist of SymPy's printed arithmetic, without eval()."""
    if isinstance(node, ast.Name) and node.id in {"a", "b", "c"}:
        return node.id
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return f"({node.value} : ℝ)"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = lean_expression(node.operand)
        return f"(-{value})" if isinstance(node.op, ast.USub) else value
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "sqrt" and len(node.args) == 1 and not node.keywords:
        return f"(Real.sqrt {lean_expression(node.args[0])})"
    if isinstance(node, ast.BinOp):
        left = lean_expression(node.left)
        if isinstance(node.op, ast.Pow):
            power = rational(node.right)
            if power is None:
                raise ValueError("AIPS exponent is not an exact rational")
            if power.denominator == 1 and power >= 0:
                return f"({left} ^ {power.numerator})"
            if power.denominator == 1:
                return f"({left} ^ ({power.numerator} : ℤ))"
            return f"({left} ^ ({power.numerator} / {power.denominator} : ℝ))"
        symbols = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
        symbol = symbols.get(type(node.op))
        if symbol:
            return f"({left} {symbol} {lean_expression(node.right)})"
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        symbols = {ast.LtE: "≤", ast.GtE: "≥", ast.Lt: "<", ast.Gt: ">", ast.Eq: "="}
        symbol = symbols.get(type(node.ops[0]))
        if symbol:
            return f"{lean_expression(node.left)} {symbol} {lean_expression(node.comparators[0])}"
    raise ValueError(f"Unsupported AIPS syntax: {ast.dump(node, include_attributes=False)}")


def convert_aips(expression: str, source_index: int) -> dict:
    tree = ast.parse(expression, mode="eval").body
    if not isinstance(tree, ast.Compare):
        raise ValueError("Expected an inequality")
    name = f"aips_{source_index:06d}"
    goal = lean_expression(tree)
    return {
        "name": name, "split": "train", "benchmark": "AIPS", "header": HEADER,
        "formal_statement": f"theorem {name} (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : {goal} := by\n",
        "informal_prefix": f"For positive real a, b, c, prove: {expression}",
        "source_index": source_index, "source_expression": expression,
        "source_revision": AIPS_REVISION,
    }


def normalized_statement(statement: str) -> str:
    statement = re.sub(r"^\s*(?:theorem|lemma)\s+(?:«[^»]+»|\S+)", "theorem", statement)
    return re.sub(r"\s+", "", statement).removesuffix(":=by")


def load_evaluation(source_dir: Path) -> tuple[list[dict], dict]:
    records, provenance = [], {}
    for benchmark, filename in BENCHMARKS.items():
        path = source_dir / filename
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for row in rows:
            original = row["formal_statement"]
            # Lean identifiers cannot start with a digit. Change only spelling.
            fixed = re.sub(r"^(theorem|lemma)\s+(\d[^\s:]+)", r"\1 «\2»", original)
            # Without this annotation Lean infers Nat.div in the exponent.
            fixed = re.sub(r"\^\s*\(\s*([+-]?\d+)\s*/\s*(\d+)\s*\)", r"^ (\1 / \2 : ℝ)", fixed)
            fixed = re.sub(r"\b(Real\.sqrt|sqrt)\(", r"\1 (", fixed)
            record = {**row, "split": "test", "benchmark": benchmark}
            record["formal_statement"] = fixed
            if fixed != original:
                record["original_formal_statement"] = original
            records.append(record)
        repaired = [record["name"] for record in records if record["benchmark"] == benchmark and "original_formal_statement" in record]
        exponent_repairs = [row["name"] for row in rows if re.search(r"\^\s*\(\s*([+-]?\d+)\s*/\s*(\d+)\s*\)", row["formal_statement"])]
        spacing_repairs = [row["name"] for row in rows if re.search(r"\b(?:Real\.sqrt|sqrt)\(", row["formal_statement"])]
        provenance[benchmark] = {"records": len(rows), "file": filename, "sha256": sha256(path),
                                 "repaired_declarations": repaired, "real_exponent_repairs": exponent_repairs,
                                 "sqrt_spacing_repairs": spacing_repairs}
    names = [record["name"] for record in records]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate evaluation IDs")
    return records, provenance


def prepare(raw_file: Path, source_dir: Path, output_dir: Path, count: int, seed: int) -> dict:
    import pyarrow.parquet as pq
    if sha256(raw_file) != AIPS_SHA256:
        raise ValueError("AIPS checksum does not match the pinned dataset")
    expressions = pq.read_table(raw_file).column(0).to_pylist()
    evaluation, provenance = load_evaluation(source_dir)
    held_out = {normalized_statement(row["formal_statement"]) for row in evaluation}
    indices = list(range(len(expressions)))
    random.Random(seed).shuffle(indices)
    selected, excluded, seen = [], [], set()
    for index in indices:
        try:
            record = convert_aips(expressions[index], index)
        except (SyntaxError, ValueError) as exc:
            excluded.append({"source_index": index, "reason": str(exc)})
            continue
        key = normalized_statement(record["formal_statement"])
        if key in held_out or key in seen:
            excluded.append({"source_index": index, "reason": "duplicate training or held-out statement"})
            continue
        seen.add(key)
        selected.append(record)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Could select only {len(selected)} training records")
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in [("train", selected), ("eval", evaluation)]:
        (output_dir / f"{name}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    for benchmark in BENCHMARKS:
        rows = [row for row in evaluation if row["benchmark"] == benchmark]
        (output_dir / f"{benchmark.lower().replace('-', '_')}.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
    manifest = {
        "training": {"source": AIPS_URL, "authors_source": "https://sites.google.com/view/aips2/home/synthetic-training-dataset",
                     "revision": AIPS_REVISION, "raw_sha256": AIPS_SHA256, "raw_records": len(expressions),
                     "records": count, "seed": seed, "source_indices": [row["source_index"] for row in selected],
                     "selection": "seeded shuffle; first convertible unique records", "excluded": excluded,
                     "license": "cc-by-nc-sa-2.0"},
        "evaluation": provenance,
        "conversion": "positive real a,b,c; exact typed rational powers; sqrt=Real.sqrt; numeric theorem names quoted; untyped evaluation fractional exponents explicitly normalized to real powers",
        "leakage_check": "disjoint IDs and normalized statements",
        "files": {name: sha256(output_dir / name) for name in ["train.jsonl", "eval.jsonl"]},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    (output_dir / "preamble.lean").write_text(HEADER)
    return manifest


def check_syntax(output_dir: Path, project: Path) -> dict:
    """Elaborate dataset statements with Lean."""
    import os
    import subprocess
    rows = [json.loads(line) for name in ["train", "eval"]
            for line in (output_dir / f"{name}.jsonl").read_text().splitlines() if line.strip()]
    artifacts = ROOT / "artifacts/datasets/inequalities"
    artifacts.mkdir(parents=True, exist_ok=True)
    source = artifacts / "syntax.lean"
    source.write_text(rows[0]["header"] + "\n".join(row["formal_statement"] + "  sorry\n" for row in rows))
    env = {**os.environ, "LEAN_NUM_THREADS": "2"}
    result = subprocess.run(["lake", "env", "lean", str(source)], cwd=project.resolve(),
                            env=env, text=True, capture_output=True, timeout=300)
    (artifacts / "syntax.log").write_text(result.stdout + result.stderr)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    status = {"records": len(rows), "exit_code": result.returncode,
              "toolchain": (project / "lean-toolchain").read_text().strip(), "mathlib_revision": revision,
              "validation_type": "statement_elaboration"}
    (artifacts / "validation.json").write_text(json.dumps(status, ensure_ascii=False, indent=2) + "\n")
    if result.returncode:
        raise ValueError(f"Dataset syntax validation failed. See {artifacts / 'syntax.log'}")
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-file", type=Path, default=ROOT / "artifacts/datasets/aips/inequalities.parquet")
    parser.add_argument("--source-dir", type=Path, default=ROOT / "data/inequalities/source")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/inequalities")
    parser.add_argument("--check-syntax", action="store_true")
    parser.add_argument("--lean-project", type=Path, default=ROOT / "vendor/kimina-lean-server/mathlib4-v4.15.0")
    args = parser.parse_args()
    if args.count < 1:
        parser.error("--count must be positive")
    if not args.raw_file.exists():
        args.raw_file.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.raw_file.with_suffix(".part")
        with urllib.request.urlopen(AIPS_URL, timeout=120) as response, temporary.open("wb") as handle:
            import shutil
            shutil.copyfileobj(response, handle)
        if sha256(temporary) != AIPS_SHA256:
            temporary.unlink()
            raise ValueError("Downloaded AIPS checksum does not match")
        temporary.replace(args.raw_file)
    manifest = prepare(args.raw_file, args.source_dir, args.output_dir, args.count, args.seed)
    if args.check_syntax:
        check_syntax(args.output_dir, args.lean_project)
    print(json.dumps({"train": manifest["training"]["records"], "eval": sum(item["records"] for item in manifest["evaluation"].values()), "output": str(args.output_dir)}, indent=2))


if __name__ == "__main__":
    main()
