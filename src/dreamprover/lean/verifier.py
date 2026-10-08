#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
import traceback
import asyncio
from typing import List, Tuple, Optional
from dreamprover.lean.kimina import DiagnosticAsyncKiminaClient as AsyncKiminaClient
from kimina_client.models import Snippet
from dreamprover.lean.proof import (
    read_client_response, can_reuse_lean_environment, response_has_server_failure, order_client_response,
    prepare_lean_verification, proof_axiom_audit_error,
)
from dreamprover.lean.helpers import extract_all_error_messages
from dreamprover.lean.library import assert_complete_source
import logging
logger = logging.getLogger(__name__)

class AsyncLeanVerifier:
    """
    Lean verification optimized for throughput with async/await patterns
    and leverages AsyncKiminaClient's built-in concurrency management.
    """

    def __init__(self, 
                 base_url: str,
                 max_concurrent_requests: int = 10,
                 default_batch_size: int = 8,
                 http_timeout: int = 600):
        """
        Initialize AsyncLeanVerifier with async client.
        
        Args:
            base_url: The base URL for the Lean server API
            max_concurrent_requests: Maximum number of concurrent operations (default: 10)
            default_batch_size: Default batch size for operations (default: 8)
            http_timeout: HTTP timeout in seconds (default: 600)
        """
        if max_concurrent_requests < 1 or default_batch_size < 1:
            raise ValueError("Verifier concurrency and batch size must be positive")
        self.client = AsyncKiminaClient(
            api_url=base_url,
            http_timeout=http_timeout,
            n_retries=1
        )
        self.max_concurrent = max_concurrent_requests
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self.default_batch_size = default_batch_size
        self._closed = False

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit with proper cleanup."""
        await self.close()

    async def verify_proof(self, proof: str, timeout: int = 30,
                          return_error_message: bool = False, 
                          is_sorry_ok: bool = False, **metadata) -> bool | Tuple[bool, Optional[str]]:
        """
        Verify a single proof asynchronously.
        
        Args:
            proof: The proof to verify
            timeout: The timeout for the verification request (default: 30s)
            return_error_message: Whether to return the error message if the proof is invalid (default: False)
            is_sorry_ok: Whether to allow the proof to be valid even if it contains a "sorry" term (default: False)
            
        Returns:
            bool: True if the proof is valid, False otherwise
            Tuple[bool, Optional[str]]: If return_error_message=True, returns (is_valid, error_message)
        """
        if self._closed:
            raise RuntimeError("AsyncLeanVerifier is closed")
            
        proof = proof.strip()
        if not proof:
            return (False, "Empty Lean source") if return_error_message else False
        audited, audited_names, preparation_error = prepare_lean_verification(proof, is_sorry_ok)
        if preparation_error:
            return (False, preparation_error) if return_error_message else False
        
        # AsyncKiminaClient handles concurrency internally
        async with self._semaphore:
            response = await self.client.check(
                audited, timeout=timeout, show_progress=False,
                batch_size=self.max_concurrent, max_workers=self.max_concurrent,
                reuse=can_reuse_lean_environment(proof),
            )
            if response_has_server_failure(response):
                logger.warning("Verifier server failed; retrying once in a fresh Lean environment")
                response = await self.client.check(
                    audited, timeout=timeout, show_progress=False, reuse=False,
                    batch_size=self.max_concurrent, max_workers=self.max_concurrent,
                )
        
        parsed = read_client_response(response)
        if len(parsed) != 1:
            raise RuntimeError(f"Verifier returned {len(parsed)} results for one proof")
        verification_results = parsed[0]

        if is_sorry_ok:
            is_proof_valid = verification_results['is_correct_with_sorry']
        else:
            is_proof_valid = verification_results['is_correct_no_sorry']
        audit_error = None if is_sorry_ok else proof_axiom_audit_error(response.results[0], audited_names)
        if is_proof_valid and audit_error:
            return (False, audit_error) if return_error_message else False

        if return_error_message:
            if not is_proof_valid:
                try:
                    error_messages = extract_all_error_messages(response, [proof])
                except Exception:
                    logger.info("Proof is invalid...")
                    traceback.print_exc()
                    return False, f"Proof:\n{proof}\n\nError: Most likely timed out"
                return is_proof_valid, error_messages[0]
            else:
                return is_proof_valid, None
        else:
            return is_proof_valid

    async def batch_verify_proofs(self, proofs: List[str], 
                                 return_error_messages: bool = False, 
                                 timeout: int = 30, 
                                 is_sorry_ok: bool = False,
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True) -> List[bool] | Tuple[List[bool], List[str]]:
        """
        Verify a batch of proofs asynchronously with optimized concurrency.
        
        Args:
            proofs: A list of proofs to verify
            return_error_messages: Whether to return the error messages if the proofs are invalid (default: False)
            timeout: The timeout for the verification request (default: 30s)
            is_sorry_ok: Whether to allow proofs that contain a "sorry" expression (default: False)
            batch_size: Override the batch size for this operation
            show_progress: Whether to show progress bar (default: True)
            
        Returns:
            List[bool]: A list of booleans indicating the validity of each proof
            Tuple[List[bool], List[str]]: If return_error_messages=True, returns (validity_list, error_messages_list)
        """
        if self._closed:
            raise RuntimeError("AsyncLeanVerifier is closed")
            
        if not proofs:
            return ([], []) if return_error_messages else []
            
        # Use provided parameters or fall back to instance defaults
        batch_sz = batch_size or self.default_batch_size
        
        # Create verification list with custom IDs

        # Use AsyncKiminaClient's check method which handles concurrency internally
        prepared = [prepare_lean_verification(proof, is_sorry_ok) for proof in proofs]
        snippets = [Snippet.from_code(source) for source, _, _ in prepared]
        responses = await self.client.check(
            snippets,
            timeout=timeout,
            batch_size=batch_sz,
            max_workers=self.max_concurrent,
            show_progress=show_progress,
            reuse=all(can_reuse_lean_environment(proof) for proof in proofs),
        )
        responses = order_client_response(responses, [snippet.id for snippet in snippets])
        
        # Parse results
        parse_results = read_client_response(responses)
        if len(parse_results) != len(proofs):
            raise RuntimeError("Verifier response count does not match submitted proof count")

        if is_sorry_ok:
            verification_results = [
                item['is_correct_with_sorry']
                for item in parse_results
            ]
        else:
            verification_results = [
                item['is_correct_no_sorry']
                for item in parse_results
            ]
            for i, proof in enumerate(proofs):
                try:
                    if not proof.strip():
                        raise ValueError("Empty Lean source")
                    assert_complete_source(proof)
                except ValueError:
                    verification_results[i] = False
                if prepared[i][2] or proof_axiom_audit_error(responses.results[i], prepared[i][1]):
                    verification_results[i] = False

        if return_error_messages:
            all_error_messages = extract_all_error_messages(responses, proofs)
            if not is_sorry_ok:
                for i, (_, names, error) in enumerate(prepared):
                    audit_error = error or proof_axiom_audit_error(responses.results[i], names)
                    if audit_error and not all_error_messages[i]:
                        all_error_messages[i] = audit_error
            return verification_results, all_error_messages
        else:
            return verification_results

    async def close(self):
        """
        Explicitly close the AsyncLeanVerifier and clean up resources.
        
        This should be called when you're done using the verifier, or use it as an async context manager.
        """
        if not self._closed:
            await self.client.close()
            self._closed = True

    def __del__(self):
        """
        Destructor to ensure resources are cleaned up if close() wasn't called explicitly.
        
        Note: This will log a warning if close() wasn't called properly.
        """
        if hasattr(self, '_closed') and not self._closed:
            import warnings
            warnings.warn(
                "AsyncLeanVerifier was not properly closed. Use 'async with AsyncLeanVerifier(...)' "
                "or call 'await verifier.close()' explicitly.",
                ResourceWarning,
                stacklevel=2
            )
