#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import json
import re
from dataclasses import dataclass, asdict


MAX_VERIFICATION_DIAGNOSTIC_CHARS = 16384


def bounded_verification_diagnostic(value) -> str:
    """Retain useful server diagnostics without unbounded checkpoint growth."""
    text = str(value)
    if len(text) <= MAX_VERIFICATION_DIAGNOSTIC_CHARS:
        return text
    marker = "\n[verification diagnostic truncated]\n"
    head = (MAX_VERIFICATION_DIAGNOSTIC_CHARS - len(marker)) // 2
    tail = MAX_VERIFICATION_DIAGNOSTIC_CHARS - len(marker) - head
    return text[:head] + marker + text[-tail:]


@dataclass(frozen=True)
class VerificationOutcome:
    """A Lean verdict distinguishes failed proofs from an unavailable verifier."""
    outcome: str
    diagnostic: str = ""
    proof_valid: bool = False
    retry_count: int = 0

    def to_dict(self):
        return asdict(self)


class VerificationInfrastructureError(RuntimeError):
    """Verification did not produce a trustworthy verdict and must be resumed."""

    def __init__(self, diagnostic, outcomes=()):
        self.diagnostic = bounded_verification_diagnostic(diagnostic)
        self.outcomes = tuple(outcomes)
        super().__init__(self.diagnostic)


def is_verification_resource_failure(diagnostic: str) -> bool:
    """Recognize explicit Lean allowances, not arbitrary HTTP timeout failures."""
    return bool(re.search(
        r"lean repl command timed out|(?<![\w:])std::bad_alloc(?!\w)|"
        r"maximum (?:number of )?heartbeats|maximum recursion depth (?:has been )?reached|"
        r"deterministic timeout|\b(?:out of memory|memory allocation failed)\b",
        diagnostic, re.IGNORECASE))


def _failure_outcome(diagnostic):
    kind = "resource_exhausted" if is_verification_resource_failure(str(diagnostic)) else "infrastructure_failure"
    return VerificationOutcome(kind, bounded_verification_diagnostic(diagnostic))


def classify_client_result(result):
    """Classify a complete Kimina result without treating protocol errors as Lean errors."""
    error = getattr(result, "error", None)
    if error:
        return _failure_outcome(error)
    payload = getattr(result, "response", None)
    if not isinstance(payload, dict):
        return VerificationOutcome("infrastructure_failure", bounded_verification_diagnostic(f"Verifier returned no structured Lean response: {payload!r}"))
    if payload.get("error") or payload.get("message"):
        return _failure_outcome(payload.get("error") or payload.get("message"))
    messages = payload.get("messages", [])
    if messages is None:
        messages = []
    if not isinstance(messages, list) or any(
            not isinstance(message, dict)
            or message.get("severity") not in ("error", "warning", "information", "info")
            or not isinstance(message.get("data"), str)
            for message in messages):
        return VerificationOutcome("infrastructure_failure", bounded_verification_diagnostic(f"Malformed Lean diagnostic messages: {payload!r}"))
    for message in messages:
        for key in ("pos", "endPos"):
            position = message.get(key)
            if position is not None and (
                    not isinstance(position, dict)
                    or not isinstance(position.get("line"), int)
                    or ("column" in position and not isinstance(position["column"], int))):
                return VerificationOutcome("infrastructure_failure", bounded_verification_diagnostic(f"Malformed Lean diagnostic positions: {message!r}"))
    errors = [message for message in messages if message["severity"] == "error"]
    if errors:
        diagnostic = bounded_verification_diagnostic(json.dumps(errors, ensure_ascii=False))
        kind = "resource_exhausted" if any(
            is_verification_resource_failure(message["data"]) for message in errors
        ) else "mathematical_invalid"
        return VerificationOutcome(kind, diagnostic)
    # An empty or partial payload cannot certify that the submitted command finished.
    # Actual successful Lean REPL commands always return an environment identifier.
    env = payload.get("env")
    if isinstance(env, bool) or not isinstance(env, int) or env < 0:
        return VerificationOutcome("infrastructure_failure", bounded_verification_diagnostic(f"Incomplete Lean response: missing valid environment identifier: {payload!r}"))
    sorries = payload.get("sorries", [])
    if sorries is None:
        sorries = []
    if not isinstance(sorries, list):
        return VerificationOutcome("infrastructure_failure", bounded_verification_diagnostic(f"Malformed Lean sorry diagnostics: {payload!r}"))
    has_sorry = bool(sorries) or any(
        message["severity"] == "warning" and "sorry" in message["data"].lower()
        for message in messages
    )
    if has_sorry:
        return VerificationOutcome("mathematical_invalid", "Lean declaration uses 'sorry'")
    return VerificationOutcome("valid", proof_valid=True)


def read_client_response(response):
    """Read genuine Lean verdicts and expose resource exhaustion explicitly."""
    results = getattr(response, "results", None)
    if not isinstance(results, list):
        raise VerificationInfrastructureError("Verifier returned malformed result collection")
    outcomes = [classify_client_result(result) for result in results]
    if any(outcome.outcome == "infrastructure_failure" for outcome in outcomes):
        raise VerificationInfrastructureError(
            "\n".join(outcome.diagnostic for outcome in outcomes if outcome.outcome == "infrastructure_failure"),
            outcomes,
        )
    return [{
        "is_correct_with_sorry": outcome.proof_valid or outcome.diagnostic == "Lean declaration uses 'sorry'",
        "is_correct_no_sorry": outcome.proof_valid,
        "outcome": outcome.outcome,
        "diagnostic": outcome.diagnostic,
    } for outcome in outcomes]

def split_header_body(proof: str) -> tuple[str, str]:
    """
    Splits `proof` into:
    - header: the consecutive `import ...` lines at the beginning of the proof.
    We remove all "import Mathlib." lines and add "import Mathlib" if necessary.
    - body: rest of the proof

    Args:
        proof (str): The proof code to split

    Returns:
        tuple[str, str]: The header and body of the proof
    """
    proof = proof.strip()
    lines = proof.splitlines()
    header_lines = []
    proof_idx = 0

    mathlib_found = False
    for i, line in enumerate(lines):
        line = line.strip()
        if line.startswith("import"):
            if line.startswith("import Mathlib."):
                mathlib_found = True
            else:
                header_lines.append(line)
            proof_idx = i + 1
        else:
            break
    if mathlib_found and 'import Mathlib' not in header_lines:
        header_lines.insert(0, 'import Mathlib')
    header = "\n".join(header_lines).strip()
    body = "\n".join(lines[proof_idx:]).strip()

    return header, body


def can_reuse_lean_environment(source: str) -> bool:
    """Reuse an import environment only when the source has an import header.

    An empty-header Kimina REPL can store the first checked declaration in its
    base environment. Rechecking it then produces an already-declared error.
    Fresh environments isolate bare declarations and avoid that server issue.
    """
    return bool(split_header_body(source)[0])


def response_has_server_failure(response) -> bool:
    """Only infrastructure failures may receive a fresh-environment retry."""
    results = getattr(response, "results", None)
    return not isinstance(results, list) or any(
        classify_client_result(result).outcome == "infrastructure_failure" for result in results
    )


def order_client_response(response, expected_ids):
    """Validate snippet identities and restore the SDK's input order."""
    results = getattr(response, "results", None)
    if not isinstance(results, list):
        raise VerificationInfrastructureError("Verifier returned malformed result collection")
    by_id = {getattr(result, 'id', None): result for result in results}
    if (len(results) != len(expected_ids) or len(by_id) != len(expected_ids)
            or set(by_id) != set(expected_ids)):
        raise VerificationInfrastructureError(
            f"Verifier response IDs or count do not match submitted proofs. "
            f"Expected {list(expected_ids)!r}, received {[getattr(result, 'id', None) for result in results]!r}"
        )
    response.results = [by_id[value] for value in expected_ids]
    return response


def prepare_lean_verification(source, is_sorry_ok=False):
    """Add recursive dependency audits for strictly proved top-level exports."""
    if not source.strip():
        return source, [], "Empty Lean source"
    if is_sorry_ok:
        return source, [], None
    from dreamprover.lean.library import assert_complete_source
    from dreamprover.lean.compiler import append_axiom_audits
    try:
        assert_complete_source(source)
        audited, names = append_axiom_audits(source)
        return audited, names, None
    except ValueError as exc:
        return source, [], str(exc)


def proof_axiom_audit_error(result, expected_names):
    """Require complete audits using only the shared trusted foundation axioms."""
    if not expected_names:
        return None
    from dreamprover.lean.compiler import axiom_dependency_error, axioms_from_diagnostics
    payload = result.response
    messages = payload.get("messages", []) if isinstance(payload, dict) else []
    text = "\n".join(str(message.get("data", "")) for message in messages if isinstance(message, dict))
    normalize = lambda value: value.replace("«", "").replace("»", "")
    audits = {normalize(name): dependencies for name, dependencies in axioms_from_diagnostics(text).items()}
    missing = [name for name in expected_names if normalize(name) not in audits]
    if missing:
        return "Missing Lean axiom audits for: " + ", ".join(missing)
    return axiom_dependency_error({name: audits[normalize(name)] for name in expected_names})
