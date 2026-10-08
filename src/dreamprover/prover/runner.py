# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
"""Run independent proof workers with bounded concurrency and ordered results."""

import asyncio
import logging
from typing import Any, Dict, Optional

from dreamprover.prover.worker import HILBERTWorker
from dreamprover.lean.library import LemmaLibrary
from dreamprover.lean.text import extract_jsonl_contents
from dreamprover.prover.config import ProofAttemptConfig

logger = logging.getLogger(__name__)


class AsyncHILBERT:
    def __init__(self, prover_llm_factory, informal_llm_client_factory,
                 lean_verifier_factory, search_engine, return_proofs: bool,
                 proof_attempt_config=None, max_depth=3, max_concurrent_problems=2,
                 verify_each_subgoal_separately=True, complexity_proof_length_cutoff=20,
                 max_tokens=16384, proof_save_dir: Optional[str] = None,
                 sequential_processing=False, run_proof_attempts_sequentially=False,
                 enable_retrieval=True, lemma_library_path=None,
                 validate_lemma_library=True):
        if max_concurrent_problems < 1:
            raise ValueError("max_concurrent_problems must be positive")
        if enable_retrieval and search_engine is None:
            raise ValueError("Retrieval requires a search engine; set enable_retrieval=false to disable it")
        self.prover_llm_factory = prover_llm_factory
        self.informal_llm_client_factory = informal_llm_client_factory
        self.lean_verifier_factory = lean_verifier_factory
        self.search_engine = search_engine
        self.return_proofs = return_proofs
        self.proof_attempt_config = (
            ProofAttemptConfig.from_dict(proof_attempt_config)
            if proof_attempt_config is not None and not isinstance(proof_attempt_config, ProofAttemptConfig)
            else proof_attempt_config or ProofAttemptConfig()
        )
        self.max_depth = max_depth
        self.max_concurrent_problems = 1 if sequential_processing else max_concurrent_problems
        self.verify_each_subgoal_separately = verify_each_subgoal_separately
        self.complexity_proof_length_cutoff = complexity_proof_length_cutoff
        self.max_tokens = max_tokens
        self.proof_save_dir = proof_save_dir
        self.run_proof_attempts_sequentially = run_proof_attempts_sequentially
        self.enable_retrieval = enable_retrieval
        self.library = LemmaLibrary.from_file(lemma_library_path) if lemma_library_path else LemmaLibrary()
        self.validate_lemma_library = validate_lemma_library

    def run_from_file(self, file_path: str) -> Dict[str, Any]:
        return asyncio.run(self._run_from_file_async(file_path))

    async def _run_from_file_async(self, file_path: str) -> Dict[str, Any]:
        examples = extract_jsonl_contents(file_path)
        for i, example in enumerate(examples):
            if not isinstance(example, dict) or not isinstance(example.get("formal_statement"), str):
                raise ValueError(f"Example {i + 1} requires a string formal_statement")
            if not isinstance(example.get("header", ""), str):
                raise ValueError(f"Example {i + 1} header must be a string")
        problem_ids = [example.get("name", example.get("id", i)) for i, example in enumerate(examples)]
        if len({str(value) for value in problem_ids}) != len(problem_ids):
            raise ValueError("Dataset problem IDs must be unique")

        if self.library.source and examples and self.validate_lemma_library:
            verifier = self.lean_verifier_factory()
            try:
                # Context may differ between samples; validate every distinct one once.
                for header in dict.fromkeys(example.get("header", "") for example in examples):
                    valid, error = await verifier.verify_proof(
                        self.library.with_header(header),
                        timeout=self.proof_attempt_config.proof_verification_timeout,
                        return_error_message=True, is_sorry_ok=False,
                    )
                    if not valid:
                        raise ValueError(f"Lemma library failed Lean verification: {error}")
            finally:
                await verifier.close()

        semaphore = asyncio.Semaphore(self.max_concurrent_problems)

        async def solve(example, problem_id):
            async with semaphore:
                worker = None
                resources = []
                try:
                    prover = self.prover_llm_factory() if self.prover_llm_factory else None
                    if prover is not None:
                        resources.append(getattr(prover, "llm_client", prover))
                    reasoner = self.informal_llm_client_factory()
                    resources.append(reasoner)
                    verifier = self.lean_verifier_factory()
                    resources.append(verifier)
                    worker = HILBERTWorker(
                        prover_llm=prover, informal_llm_client=reasoner, lean_verifier=verifier,
                        search_engine=self.search_engine, proof_attempt_config=self.proof_attempt_config,
                        max_depth=self.max_depth,
                        verify_each_subgoal_separately=self.verify_each_subgoal_separately,
                        complexity_proof_length_cutoff=self.complexity_proof_length_cutoff,
                        max_tokens=self.max_tokens, proof_save_dir=self.proof_save_dir,
                        run_proof_attempts_sequentially=self.run_proof_attempts_sequentially,
                        enable_retrieval=self.enable_retrieval, lemma_library=self.library,
                    )
                    success, proof = await worker.generate_single_proof(
                        example["formal_statement"], example.get("header", ""), str(problem_id)
                    )
                    return bool(success), proof, None
                except Exception as exc:
                    logger.exception("Problem %s failed", problem_id)
                    return False, None, f"{type(exc).__name__}: {exc}"
                finally:
                    if worker is not None:
                        await worker.close()
                    else:
                        # Also clean up partially constructed workers.
                        for resource in resources:
                            if hasattr(resource, "close"):
                                await resource.close()

        completed = await asyncio.gather(*(solve(example, problem_id) for example, problem_id in zip(examples, problem_ids)))
        results = [success for success, _, _ in completed]
        output = {
            "problem_ids": problem_ids,
            "results": results,
            "failure_cases": [pid for pid, success in zip(problem_ids, results) if not success],
            "errors": [error for _, _, error in completed],
            "pass_rate": sum(results) / len(results) if results else 0.0,
        }
        if self.return_proofs:
            output["proofs"] = [proof for _, proof, _ in completed]
            output["formal_statements"] = [example["formal_statement"] for example in examples]
        return output
