#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import re


def read_client_response(response):
    parsed_answers = []

    for r in response.results:
        curr_response = r.response
        if (not isinstance(curr_response, dict) or curr_response.get('error') or curr_response.get('message')
                or getattr(r, 'error', None)
                or not any(key in curr_response for key in ('messages', 'env', 'sorries', 'tactics', 'infotree'))):
            parsed_answers.append({
                "is_correct_with_sorry": False,
                "is_correct_no_sorry": False
            })
            continue
        messages = curr_response.get("messages") or []
        if not isinstance(messages, list):
            parsed_answers.append({"is_correct_with_sorry": False, "is_correct_no_sorry": False})
            continue
        severities = [m.get("severity") for m in messages]
        datas = [m.get("data", "") for m in messages]

        if "error" in severities:
            parsed_answers.append({
                "is_correct_with_sorry": False,
                "is_correct_no_sorry": False
            })
            continue

        has_sorry = bool(curr_response.get('sorries')) or any(
            sev == "warning" and "sorry" in str(data).lower()
            for sev, data in zip(severities, datas)
        )

        parsed_answers.append({
            "is_correct_with_sorry": True,
            "is_correct_no_sorry": not has_sorry
        })

    return parsed_answers

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
    """Recover infrastructure failures without replaying resource-exhausted proofs.

    Kimina destroys a REPL after a configured command timeout. The unchanged
    proof has already used its full verification allowance. Likewise, a known
    std::bad_alloc from the proof cannot gain memory by replaying under the same
    limit. The next corrected proof obtains a new REPL automatically.
    """
    for result in response.results:
        error = getattr(result, 'error', None)
        if error and str(error).strip().lower().startswith("lean repl command timed out"):
            continue
        if error and re.search(r"(?<![\w:])std::bad_alloc(?!\w)", str(error)):
            continue
        if error:
            return True
        payload = result.response
        if isinstance(payload, dict) and payload.get('message') == 'Unknown environment.':
            return True
    return False


def order_client_response(response, expected_ids):
    """Kimina's sync SDK merges batches by completion order; restore input order."""
    by_id = {getattr(result, 'id', None): result for result in response.results}
    if (len(response.results) != len(expected_ids) or len(by_id) != len(expected_ids)
            or set(by_id) != set(expected_ids)):
        raise RuntimeError("Verifier response IDs do not match submitted proof IDs")
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
