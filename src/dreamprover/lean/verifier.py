#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""Lean verification with separate mathematical, resource, and infrastructure outcomes."""
import asyncio
import logging
from contextvars import ContextVar
from dataclasses import replace
from types import SimpleNamespace
from typing import List, Tuple, Optional

from dreamprover.lean.kimina import DiagnosticAsyncKiminaClient as AsyncKiminaClient, verifier_failure_message
from kimina_client.models import Snippet
from dreamprover.lean.proof import (
    VerificationInfrastructureError, VerificationOutcome, bounded_verification_diagnostic,
    can_reuse_lean_environment, classify_client_result, is_verification_resource_failure,
    order_client_response, prepare_lean_verification, proof_axiom_audit_error,
)
from dreamprover.lean.helpers import extract_all_error_messages

logger = logging.getLogger(__name__)


class AsyncLeanVerifier:
    """A failed server check never becomes a completed, unsuccessful proof."""

    def __init__(self, base_url: str, max_concurrent_requests: int = 10,
                 default_batch_size: int = 8, http_timeout: int = 600):
        if max_concurrent_requests < 1 or default_batch_size < 1:
            raise ValueError("Verifier concurrency and batch size must be positive")
        self.client = AsyncKiminaClient(api_url=base_url, http_timeout=http_timeout, n_retries=1)
        self.max_concurrent = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.default_batch_size = default_batch_size
        self._closed = False
        # Multiple workers share a verifier. The awaiting caller must see its own
        # verdict, not whichever concurrent request finished most recently.
        self._outcomes = ContextVar(f"lean_verifier_outcomes_{id(self)}", default=())

    @property
    def last_outcomes(self):
        return self._outcomes.get()

    @property
    def last_outcome(self):
        outcomes = self.last_outcomes
        return outcomes[0] if len(outcomes) == 1 else None

    def get_last_verification_outcomes(self):
        return self.last_outcomes

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    @staticmethod
    def _classify(result, names, is_sorry_ok):
        outcome = classify_client_result(result)
        if is_sorry_ok and outcome.diagnostic == "Lean declaration uses 'sorry'":
            return VerificationOutcome("valid", proof_valid=True)
        if outcome.proof_valid and not is_sorry_ok:
            audit_error = proof_axiom_audit_error(result, names)
            if audit_error:
                kind = "infrastructure_failure" if audit_error.startswith("Missing Lean axiom audits for:") else "mathematical_invalid"
                return VerificationOutcome(kind, bounded_verification_diagnostic(audit_error))
        return outcome

    async def _check(self, snippets, names, timeout, is_sorry_ok, batch_size, show_progress, reuse):
        """Return one explicit outcome per snippet, including SDK-safe failures."""
        try:
            async with self._semaphore:
                response = await self.client.check(
                    snippets, timeout=timeout, show_progress=show_progress,
                    batch_size=batch_size, max_workers=self.max_concurrent, reuse=reuse,
                )
            response = order_client_response(response, [snippet.id for snippet in snippets])
            results = response.results
            return results, [self._classify(result, audit_names, is_sorry_ok)
                             for result, audit_names in zip(results, names)]
        except Exception as error:
            raw_diagnostic = verifier_failure_message(error)
            kind = "resource_exhausted" if is_verification_resource_failure(raw_diagnostic) else "infrastructure_failure"
            diagnostic = bounded_verification_diagnostic(raw_diagnostic)
            return [None] * len(snippets), [VerificationOutcome(kind, diagnostic)] * len(snippets)

    async def _checked_outcomes(self, snippets, names, timeout, is_sorry_ok,
                                batch_size, show_progress, reuse):
        results, outcomes = await self._check(
            snippets, names, timeout, is_sorry_ok, batch_size, show_progress, reuse,
        )
        retry_indices = [i for i, outcome in enumerate(outcomes) if outcome.outcome == "infrastructure_failure"]
        if retry_indices:
            logger.warning("Verifier infrastructure failed for %d proof(s), retrying once in fresh Lean environments", len(retry_indices))
            retried_results, retried_outcomes = await self._check(
                [snippets[i] for i in retry_indices], [names[i] for i in retry_indices],
                timeout, is_sorry_ok, batch_size, show_progress, False,
            )
            for i, result, outcome in zip(retry_indices, retried_results, retried_outcomes):
                diagnostic = bounded_verification_diagnostic(
                    f"Initial verification infrastructure failure:\n{outcomes[i].diagnostic}\n"
                    f"Fresh-environment retry ({outcome.outcome}):\n{outcome.diagnostic}"
                )
                results[i] = result
                outcomes[i] = replace(outcome, diagnostic=diagnostic, retry_count=1)
        return results, outcomes

    def _publish_outcomes(self, outcomes):
        self._outcomes.set(tuple(outcomes))
        failures = [outcome for outcome in outcomes if outcome.outcome == "infrastructure_failure"]
        if failures:
            raise VerificationInfrastructureError(
                "\n".join(outcome.diagnostic for outcome in failures), outcomes,
            )

    @staticmethod
    def _error_message(proof, result, outcome):
        if outcome.proof_valid:
            return None
        # Preserve the existing line-context feedback used in correction prompts.
        # Metadata additionally stores complete bounded server diagnostics.
        if result is not None and outcome.outcome == "mathematical_invalid":
            try:
                message = extract_all_error_messages(SimpleNamespace(results=[result]), [proof])[0]
                if outcome.diagnostic and ("axiom" in outcome.diagnostic.lower() or "sorry" in outcome.diagnostic.lower()):
                    return f"{message}\nError: {outcome.diagnostic}"
                return message
            except (KeyError, IndexError, TypeError, ValueError):
                pass
        return f"Proof: {proof}\nError: {outcome.diagnostic}"

    async def verify_proof(self, proof: str, timeout: int = 30,
                           return_error_message: bool = False,
                           is_sorry_ok: bool = False, **metadata) -> bool | Tuple[bool, Optional[str]]:
        """Return a proof verdict, or raise if the verifier did not finish reliably.

        Resource exhaustion consumes the configured allowance and returns False.
        Infrastructure failures get at most one fresh retry before they propagate.
        """
        if self._closed:
            raise RuntimeError("AsyncLeanVerifier is closed")
        self._outcomes.set(())
        proof = proof.strip()
        source, names, preparation_error = prepare_lean_verification(proof, is_sorry_ok)
        if preparation_error:
            outcome = VerificationOutcome("mathematical_invalid", bounded_verification_diagnostic(preparation_error))
            self._publish_outcomes([outcome])
            return (False, preparation_error) if return_error_message else False
        snippet = Snippet.from_code(source)
        results, outcomes = await self._checked_outcomes(
            [snippet], [names], timeout, is_sorry_ok,
            self.max_concurrent, False, can_reuse_lean_environment(proof),
        )
        self._publish_outcomes(outcomes)
        outcome = outcomes[0]
        if return_error_message:
            return outcome.proof_valid, self._error_message(proof, results[0], outcome)
        return outcome.proof_valid

    async def batch_verify_proofs(self, proofs: List[str], return_error_messages: bool = False,
                                  timeout: int = 30, is_sorry_ok: bool = False,
                                  batch_size: Optional[int] = None,
                                  show_progress: bool = True) -> List[bool] | Tuple[List[bool], List[str]]:
        """Verify a batch with ordered outcomes and retry only interrupted snippets."""
        if self._closed:
            raise RuntimeError("AsyncLeanVerifier is closed")
        self._outcomes.set(())
        if not proofs:
            return ([], []) if return_error_messages else []
        batch_sz = batch_size or self.default_batch_size
        if batch_sz < 1:
            raise ValueError("Verifier batch size must be positive")
        prepared = [prepare_lean_verification(proof, is_sorry_ok) for proof in proofs]
        outcomes = [VerificationOutcome("mathematical_invalid", bounded_verification_diagnostic(error or ""))
                    for _, _, error in prepared]
        results = [None] * len(proofs)
        submitted_indices = [i for i, (_, _, error) in enumerate(prepared) if error is None]
        if submitted_indices:
            snippets = [Snippet.from_code(prepared[i][0]) for i in submitted_indices]
            checked_results, checked_outcomes = await self._checked_outcomes(
                snippets, [prepared[i][1] for i in submitted_indices], timeout, is_sorry_ok,
                batch_sz, show_progress,
                all(can_reuse_lean_environment(proofs[i]) for i in submitted_indices),
            )
            for i, result, outcome in zip(submitted_indices, checked_results, checked_outcomes):
                results[i], outcomes[i] = result, outcome
        self._publish_outcomes(outcomes)
        validity = [outcome.proof_valid for outcome in outcomes]
        if return_error_messages:
            return validity, [self._error_message(proof, result, outcome) or ""
                              for proof, result, outcome in zip(proofs, results, outcomes)]
        return validity

    async def close(self):
        if not self._closed:
            await self.client.close()
            self._closed = True

    def __del__(self):
        if hasattr(self, '_closed') and not self._closed:
            import warnings
            warnings.warn(
                "AsyncLeanVerifier was not properly closed. Use 'async with AsyncLeanVerifier(...)' "
                "or call 'await verifier.close()' explicitly.", ResourceWarning, stacklevel=2,
            )
