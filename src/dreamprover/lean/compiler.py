"""Bounded Lean compiler and optional JSON REPL helpers."""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from dreamprover.lean.config import LEAN_PROJECT_ROOT, REPL_EXECUTABLE


# These exact declarations are in official Lean 4.15/4.33 Init sources. The
# compiler-backed names preserve Lean 4.15 native_decide/reduction proofs.
# Newer native_decide generates separate axioms; their names alone establish
# no trustworthy origin, so they are deliberately not admitted here.
TRUSTED_AXIOMS = frozenset({
    "propext", "Classical.choice", "Quot.sound",
    "Lean.ofReduceBool", "Lean.ofReduceNat", "Lean.trustCompiler",
})


def untrusted_axiom_dependencies(dependencies: dict[str, list[str]]) -> dict[str, list[str]]:
    """Return every export dependency outside the exact foundation allowance."""
    return {name: untrusted for name, axioms in dependencies.items()
            if (untrusted := sorted(set(axioms) - TRUSTED_AXIOMS))}


def axiom_dependency_error(dependencies: dict[str, list[str]]) -> str | None:
    """Explain failed proof admission while preserving complete Lean audits."""
    untrusted = untrusted_axiom_dependencies(dependencies)
    if not untrusted:
        return None
    details = "; ".join(f"{name}: {', '.join(axioms)}" for name, axioms in untrusted.items())
    diagnostic = "Lean exports depend on untrusted axioms: " + details
    if any(axiom.rsplit(".", 1)[-1] == "sorryAx" for axioms in untrusted.values() for axiom in axioms):
        diagnostic += ". Imported declarations depending on sorryAx are unproved."
    if any("._native." in axiom and ".ax_" in axiom for axioms in untrusted.values() for axiom in axioms):
        diagnostic += ". Unsupported generated native axioms: Lean 4.33 native_decide is not supported by this audit."
    return diagnostic


def _project_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root or os.environ.get("LEAN_PROJECT_ROOT", LEAN_PROJECT_ROOT)).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Lean project not found: {root}. Set LEAN_PROJECT_ROOT to a built Lake project.")
    return root


def _infrastructure_failure(diagnostic: str):
    # Import locally: proof-verification helpers also use compiler axiom audits.
    from dreamprover.lean.proof import VerificationInfrastructureError, VerificationOutcome

    raise VerificationInfrastructureError(
        diagnostic, (VerificationOutcome("infrastructure_failure", diagnostic),)
    )


def _resource_kind(diagnostic: str) -> str:
    if re.search(r"std::bad_alloc|out of memory|memory allocation failed", diagnostic, re.IGNORECASE):
        return "memory"
    if "heartbeat" in diagnostic.lower():
        return "heartbeats"
    if "recursion depth" in diagnostic.lower():
        return "recursion"
    if "timed out" in diagnostic.lower() or "timeout" in diagnostic.lower():
        return "time"
    return "resource"


def compile_lean_file(
    lean_file_path: str | Path,
    *,
    project_root: str | Path | None = None,
    timeout: float = 120,
) -> dict[str, Any]:
    """Compile source with distinct proof, resource, and infrastructure verdicts."""
    path = Path(lean_file_path).resolve()
    if not path.is_file():
        _infrastructure_failure(f"Lean file not found: {path}")
    try:
        result = subprocess.run(
            ["lake", "env", "lean", str(path)],
            cwd=_project_root(project_root),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        from dreamprover.lean.proof import bounded_verification_diagnostic

        def output_text(value):
            return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value or ""
        diagnostic = bounded_verification_diagnostic(
            output_text(exc.stdout) + output_text(exc.stderr) + f"\nLean compiler timed out after {timeout} seconds"
        )
        return {
            "returncode": 124,
            "stdout": output_text(exc.stdout),
            "stderr": output_text(exc.stderr) + f"\nerror: Lean compiler timed out after {timeout} seconds\n",
            "has_errors": True,
            "timed_out": True,
            "outcome": "resource_exhausted",
            "diagnostic": diagnostic,
            "resource_kind": "time",
        }
    except OSError as exc:
        _infrastructure_failure(f"Cannot execute Lean compiler: {exc}")
    from dreamprover.lean.proof import bounded_verification_diagnostic, is_verification_resource_failure

    diagnostic = bounded_verification_diagnostic(result.stdout + result.stderr)
    outcome = "valid"
    resource_kind = None
    if result.returncode != 0:
        if is_verification_resource_failure(diagnostic):
            outcome = "resource_exhausted"
            resource_kind = _resource_kind(diagnostic)
        elif (result.returncode < 0 or result.returncode >= 128 or re.search(
                r"segmentation fault|illegal instruction|bus error|core dumped|"
                r"(?:toolchain[^\n]*(?:not installed|not found|unavailable))|"
                r"no default toolchain|failed to (?:download|execute)|"
                r"could not execute|error while loading shared libraries",
                diagnostic, re.IGNORECASE)):
            _infrastructure_failure(f"Lean compiler unavailable or crashed (exit {result.returncode}):\n{diagnostic}")
        elif not diagnostic.strip():
            _infrastructure_failure(f"Lean compiler exited with status {result.returncode} without diagnostics")
        else:
            outcome = "mathematical_invalid"
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "has_errors": result.returncode != 0,
        "timed_out": False,
        "outcome": outcome,
        "diagnostic": diagnostic,
        "resource_kind": resource_kind,
    }


def run_lean_file(lean_file_path: str | Path, *, project_root=None, timeout: float = 120) -> str:
    """Compile a file and return its diagnostic text.

    Use compile_lean_file/check_lean_source when checking the compiler exit
    status alongside its output.
    """
    result = compile_lean_file(lean_file_path, project_root=project_root, timeout=timeout)
    output = result["stdout"] + result["stderr"]
    if result["has_errors"] and not output.strip():
        output = f"error: Lean exited with status {result['returncode']}\n"
    return output


def _with_source(lean_source: str, runner, *, project_root=None, timeout=120):
    try:
        root = _project_root(project_root)
        with tempfile.TemporaryDirectory(prefix="dreamprover-", dir=root) as directory:
            path = Path(directory) / "source.lean"
            path.write_text(lean_source, encoding="utf-8")
            return runner(path, project_root=root, timeout=timeout)
    except OSError as exc:
        _infrastructure_failure(f"Cannot prepare Lean compiler input: {exc}")


def append_axiom_audits(lean_source: str) -> tuple[str, list[str]]:
    """Append Lean's recursive axiom audit for each top-level theorem export.

    Imported sorry-backed declarations do not emit a warning when reused.
    `#print axioms` exposes those transitive sorryAx dependencies instead.
    Namespace exports require Lean-aware name resolution and are unsupported
    by the existing lexical library extractor as well as this helper.
    """
    from dreamprover.lean.source import NAME_PATTERN, mask_comments_and_strings, theorem_name

    masked = mask_comments_and_strings(lean_source)
    if re.search(r"^\s*namespace\b", masked, re.MULTILINE):
        raise ValueError("Axiom audits require top-level theorem exports; scoped declarations need Lean-aware extraction")
    attributes = r"(?:@\[[^]]*\]\s*)*"
    if re.search(rf"^\s*{attributes}private\s+(?:theorem|lemma)\b", masked, re.MULTILINE):
        raise ValueError("Axiom audits require public theorem exports; private declarations have module-dependent names")
    declarations = re.compile(rf"^\s*{attributes}(?:protected\s+|noncomputable\s+)*(?P<declaration>(?:theorem|lemma)\s+{NAME_PATTERN})", re.MULTILINE)
    names = list(dict.fromkeys(theorem_name(lean_source[match.start('declaration'):]) for match in declarations.finditer(masked)))
    audits = "\n".join(f"#print axioms {name}" for name in names)
    return lean_source + ("\n\n" + audits + "\n" if audits else ""), names


def axioms_from_diagnostics(diagnostics: str) -> dict[str, list[str]]:
    """Read standard Lean `#print axioms` information messages."""
    result = {}
    # Lean warnings can contain quoted words before an audit on a later line.
    # Only dependency lists may span lines, never the displayed theorem name.
    for match in re.finditer(r"'([^\r\n]+?)' depends on axioms:\s*\[([^]]*)\]", diagnostics, re.DOTALL):
        result[match.group(1)] = [name.strip() for name in match.group(2).split(",") if name.strip()]
    for match in re.finditer(r"'([^\n]+)' does not depend on any axioms", diagnostics):
        result[match.group(1)] = []
    return result


def axiom_audits_complete(names: list[str], dependencies: dict[str, list[str]]) -> bool:
    """Require an audit for every name, allowing Lean's quote display changes."""
    def normalize(name):
        return name.replace("«", "").replace("»", "")
    return {normalize(name) for name in names} <= {normalize(name) for name in dependencies}


def check_lean_source(lean_source: str, *, project_root=None, timeout: float = 120) -> dict[str, Any]:
    """A syntax check may accept sorry; a proved lemma may not.

    Warnings about sorry are emitted by Lean, and the conservative source
    check also catches sorry/admit/sorryAx when warnings are disabled.
    """
    audited_source, audited_names = append_axiom_audits(lean_source)
    result = _with_source(audited_source, compile_lean_file, project_root=project_root, timeout=timeout)
    from dreamprover.lean.source import mask_comments_and_strings

    masked = mask_comments_and_strings(lean_source)
    has_sorry = bool(re.search(r"\b(?:sorry|admit|sorryAx)\b", masked))
    diagnostics = result["stdout"] + result["stderr"]
    result["axiom_dependencies"] = axioms_from_diagnostics(diagnostics)
    imported_sorry = any(axiom.rsplit(".", 1)[-1] == "sorryAx"
                         for dependencies in result["axiom_dependencies"].values() for axiom in dependencies)
    result["has_sorry"] = has_sorry or "uses 'sorry'" in diagnostics.lower() or imported_sorry
    result["audited_declarations"] = audited_names
    result["axiom_audit_complete"] = axiom_audits_complete(audited_names, result["axiom_dependencies"])
    # A newly introduced axiom is an assumption, not a proved candidate.
    result["has_untrusted_declarations"] = bool(re.search(r"^\s*(?:(?:private|protected)\s+)?(?:axiom|constant|unsafe)\b", masked, re.MULTILINE))
    result["untrusted_axiom_dependencies"] = untrusted_axiom_dependencies(result["axiom_dependencies"])
    result["axiom_audit_error"] = axiom_dependency_error(result["axiom_dependencies"])
    result["proved"] = (not result["has_errors"] and not result["has_sorry"]
                        and not result["has_untrusted_declarations"] and result["axiom_audit_complete"]
                        and not result["untrusted_axiom_dependencies"])
    if not result["has_errors"] and not result["has_sorry"] and not result["axiom_audit_complete"]:
        _infrastructure_failure("Lean compilation returned no required axiom audit for: " + ", ".join(audited_names))
    if not result["proved"] and result["outcome"] == "valid":
        result["outcome"] = "mathematical_invalid"
        result["diagnostic"] = result["axiom_audit_error"] or (
            "Lean proof uses sorry" if result["has_sorry"] else "Lean source introduces untrusted declarations"
        )
    return result


def run_lean_code(lean_source: str, *, project_root=None, timeout: float = 120) -> str:
    return _with_source(lean_source, run_lean_file, project_root=project_root, timeout=timeout)


def run_lean_file_repl(lean_file_path: str | Path, *, project_root=None, timeout: float = 120) -> dict[str, Any]:
    path = Path(lean_file_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Lean file not found: {path}")
    executable = os.environ.get("REPL_EXECUTABLE", os.environ.get("REPL_DIR", REPL_EXECUTABLE))
    result = subprocess.run(
        ["lake", "env", executable],
        cwd=_project_root(project_root),
        input=json.dumps({"path": str(path), "allTactics": True}) + "\n\n",
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Lean REPL exited with status {result.returncode}:\n{result.stderr}")
    try:
        return json.loads(result.stdout.strip())
    except json.JSONDecodeError as exc:
        raise RuntimeError("Lean REPL did not return valid JSON; check REPL_EXECUTABLE") from exc


def run_lean_code_repl(lean_source: str, *, project_root=None, timeout: float = 120) -> dict[str, Any]:
    return _with_source(lean_source, run_lean_file_repl, project_root=project_root, timeout=timeout)


def get_extract_goal_results(lean_code: str) -> list[str]:
    return [msg.get("data", "") for msg in run_lean_code_repl(lean_code).get("messages", []) if "extracted_" in msg.get("data", "")]


def get_tactic_at_index(repl_json: dict[str, Any], idx: int) -> dict[str, Any]:
    return repl_json["tactics"][idx]


def extract_all_used_constants(repl_json: dict[str, Any]) -> set[str]:
    return {constant for tactic in repl_json.get("tactics", []) for constant in tactic.get("usedConstants", [])}


def extract_tactics_for_have(repl_json: dict[str, Any]) -> list[str]:
    return [t["goals"] for t in repl_json.get("tactics", []) if t.get("tactic", "").lstrip().startswith("have")]
