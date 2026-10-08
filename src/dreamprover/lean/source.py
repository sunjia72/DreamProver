"""Conservative lexical helpers for exported, top-level Lean declarations.

This is not a Lean parser. Lean itself remains the authority for checking
extracted declarations. Namespace/section commands and intervening definitions
are rejected by the library builder instead of silently changing their scope.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


def mask_comments_and_strings(source: str) -> str:
    """Preserve positions/newlines while hiding nested comments and strings."""
    chars = list(source)
    index = 0
    depth = 0
    string = False
    line_comment = False
    while index < len(source):
        pair = source[index:index + 2]
        char = source[index]
        if line_comment:
            if char == "\n":
                line_comment = False
            else:
                chars[index] = " "
        elif depth:
            if pair == "/-":
                chars[index:index + 2] = "  "
                depth += 1
                index += 1
            elif pair == "-/":
                chars[index:index + 2] = "  "
                depth -= 1
                index += 1
            elif char != "\n":
                chars[index] = " "
        elif string:
            if char == "\\" and index + 1 < len(source):
                chars[index] = " "
                index += 1
                chars[index] = " "
            elif char == '"':
                chars[index] = " "
                string = False
            elif char != "\n":
                chars[index] = " "
        elif pair == "--":
            chars[index:index + 2] = "  "
            line_comment = True
            index += 1
        elif pair == "/-":
            chars[index:index + 2] = "  "
            depth = 1
            index += 1
        elif char == '"':
            chars[index] = " "
            string = True
        index += 1
    if depth or string:
        raise ValueError("Unterminated Lean comment or string")
    return "".join(chars)


NAME_PATTERN = r"(?:«[^»]+»|[^\s:({\[⦃]+)"
DECLARATION = re.compile(rf"^(?:theorem|lemma)\s+({NAME_PATTERN})", re.MULTILINE)
COMMAND = re.compile(r"^(?:import|namespace|section|end|def|abbrev|axiom|opaque|instance|structure|class|inductive|open|variable|set_option|attribute|syntax|elab|macro|notation|noncomputable|private|protected)\b", re.MULTILINE)


def split_statement_proof(declaration: str) -> tuple[str, str]:
    """Find the declaration assignment, ignoring defaults inside binders."""
    masked = mask_comments_and_strings(declaration)
    depth = 0
    let_assignments = 0
    for index, char in enumerate(masked):
        if char in "([{⦃":
            depth += 1
        elif char in ")] }⦄".replace(" ", ""):
            depth -= 1
        elif depth == 0 and masked[index:index + 3] == "let" and (index == 0 or not (masked[index - 1].isalnum() or masked[index - 1] in "_«")) and (index + 3 == len(masked) or not (masked[index + 3].isalnum() or masked[index + 3] == "_")):
            let_assignments += 1
        elif masked[index:index + 2] == ":=" and depth == 0:
            if let_assignments:
                let_assignments -= 1
            else:
                return declaration[:index].rstrip(), declaration[index + 2:].strip()
    raise ValueError("Declaration has no top-level := proof assignment")


def theorem_name(statement: str) -> str:
    masked = mask_comments_and_strings(statement).strip()
    match = DECLARATION.match(masked)
    if not match:
        raise ValueError("Expected a top-level theorem or lemma declaration")
    # Universe parameters are part of the declaration syntax, not its name.
    name = match.group(1)
    if name.endswith(".") and masked[match.end():].startswith("{"):
        return name[:-1]
    return name


@dataclass(frozen=True)
class Declaration:
    name: str
    statement: str
    proof: str
    source: str


def extract_declarations(source: str) -> tuple[str, list[Declaration]]:
    """Extract simple top-level theorem/lemma exports with a shared preamble.

    Files with scope changes or other commands between declarations require
    Lean-aware extraction and are rejected here. This prevents a standalone
    extracted statement from acquiring a different mathematical meaning.
    """
    masked = mask_comments_and_strings(source)
    starts = list(DECLARATION.finditer(masked))
    if not starts:
        return source, []
    header = source[:starts[0].start()]
    if re.search(r"^\s*(?:namespace|section)\b", mask_comments_and_strings(header), re.MULTILINE):
        raise ValueError("Scoped theorem exports need Lean-aware extraction; export top-level declarations first")
    result = []
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(source)
        block = source[match.start():stop].strip()
        command = COMMAND.search(mask_comments_and_strings(block))
        if command:
            raise ValueError(f"Intervening Lean command {command.group(0)!r} cannot be safely extracted")
        statement, proof = split_statement_proof(block)
        result.append(Declaration(theorem_name(statement), statement, proof, block))
    return header, result


def referenced_names(proof: str, names: set[str]) -> list[str]:
    masked = mask_comments_and_strings(proof)
    # Lean identifiers may contain apostrophes, Unicode and namespace dots.
    # A following dot can be proof term field notation (`lemma_name.symm`).
    # Preserve that conservative dependency while excluding unrelated names
    # qualified by a different namespace (`Nat.lemma_name`).
    return sorted(name for name in names if re.search(rf"(?<![\w'.]){re.escape(name)}(?![\w'])", masked))
