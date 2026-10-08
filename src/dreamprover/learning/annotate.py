"""Annotate an extracted library for sleep-stage clustering (paper §3.2/F.3)."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from dreamprover.clients.completion import query_llm, run_informal_llm

ANNOTATION_PROMPT = """You are a mathematician and an expert in Lean. Below is a proved theorem:
```lean
{statement}
```
Generate a natural language description. Identify its mathematical sub-domain
and difficulty, whether it is a trivial equivalence transformation, and any
well-known theorem or variant useful in its proof. Summarize in 1 to 5 sentences.
Enclose the final description in one pair of <description> tags.
"""


def parse_description(response: str) -> str:
    matches = re.findall(r"<description>\s*(.*?)\s*</description>", response, re.DOTALL)
    if len(matches) != 1 or not matches[0].strip():
        raise ValueError("Expected one nonempty <description>...</description> block")
    return matches[0].strip()


def annotate_library(library: dict, complete) -> dict:
    result = {}
    for name, record in library.items():
        copied = dict(record)
        if not copied.get("description"):
            response = complete(ANNOTATION_PROMPT.format(statement=copied["statement"]))
            copied["description"] = parse_description(response)
        result[name] = copied
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="full_library.json from dreamprover.learning.extract")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--descriptions", type=Path, help="JSON mapping from theorem names to descriptions")
    parser.add_argument("--base-url", help="OpenAI-compatible API base URL; otherwise use Hydra config")
    parser.add_argument("--model", help="Model name when --base-url is supplied")
    parser.add_argument("--max-tokens", type=int, default=1200)
    args = parser.parse_args()
    if bool(args.base_url) != bool(args.model):
        parser.error("--base-url and --model must be supplied together")
    library = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(library, dict):
        parser.error("Input must be a library object keyed by theorem name")
    if args.descriptions:
        descriptions = json.loads(args.descriptions.read_text(encoding="utf-8"))
        for name, record in library.items():
            if name not in descriptions or not isinstance(descriptions[name], str) or not descriptions[name].strip():
                raise ValueError(f"Missing nonempty description for {name}")
            record["description"] = descriptions[name]
            record["annotation_source"] = str(args.descriptions)
    def complete(prompt):
        if args.base_url:
            return query_llm(prompt, api_url=args.base_url, model_name=args.model, max_tokens=args.max_tokens)["text"]
        return run_informal_llm(prompt, max_tokens=args.max_tokens)
    result = annotate_library(library, complete)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Annotated {len(result)} declarations into {args.output}")


if __name__ == "__main__":
    main()
