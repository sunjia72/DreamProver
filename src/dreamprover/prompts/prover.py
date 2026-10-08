#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
##############################################################
#               Formal to Formal 
##############################################################
COT_PROMPT = """
Complete the following Lean 4 code:
```lean4
{formal_statement}
```
Before producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan outlining the main proof steps and strategies. The plan should highlight key ideas, intermediate lemmas, and proof structures that will guide the construction of the final formal proof.
"""

NON_COT_PROMPT = """
Prove the final theorem in the following Lean 4 context:
```lean
{formal_statement}
```
Return exactly one complete Lean code block containing the original target theorem
declaration and its proof. Preserve the theorem name, binders, assumptions, and
conclusion exactly. Return the whole theorem rather than only a proof fragment.
Use local `have` statements if needed. Omit imports and previously supplied
context declarations. Do not use sorry, admit, or introduce axioms.
"""
