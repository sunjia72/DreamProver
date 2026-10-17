#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
"""
Wrapper classes for automatic LLM call tracking in HILBERT.

These wrappers intercept LLM calls to track token usage and performance metrics.
"""

import time
from dreamprover.lean.proof import bounded_verification_diagnostic
from typing import Dict, List, Optional
from dreamprover.runtime.statistics import ProofStatistics, LLMType, StrategyType, PromptType, VerificationType
import logging

logger = logging.getLogger(__name__)

class MaxLLMCallsExceeded(Exception):
    """Exception raised when the maximum number of LLM calls is exceeded."""
    pass

class TrackedAsyncLLMClient:
    """Wrapper for AsyncLLMClient that tracks all calls."""
    
    def __init__(self, 
                 client, 
                 stats: ProofStatistics, 
                 llm_type: LLMType,
                 max_num_llm_calls: int = None):
        self.client = client
        self.stats = stats
        self.llm_type = llm_type
        self.max_num_llm_calls = max_num_llm_calls # None means no limit
        self.llm_count = 0

    async def _increment_call_count(self):
        """Increment the LLM call count and enforce max limit if set."""
        if self.max_num_llm_calls is not None and self.llm_count >= self.max_num_llm_calls:
            raise MaxLLMCallsExceeded(f"Exceeded maximum number of LLM calls: {self.max_num_llm_calls}")
        self.llm_count += 1

    def set_max_num_llm_calls(self, max_calls: int):
        """Set the maximum number of LLM calls."""
        self.max_num_llm_calls = max_calls

    async def simple_chat(self, prompt: str, max_tokens: int = None,
                         reasoning: Dict = None, prompt_type: str = None,
                         context: str = "", **kwargs) -> str:
        """Track a simple chat call."""
        start_time = time.time()
        
        # Use provided prompt_type or default to SIMPLE_CHAT
        if prompt_type is None:
            prompt_type = PromptType.SIMPLE_CHAT.value
        
        await self._increment_call_count()
        try:

            response = await self.client.simple_chat(
                prompt, max_tokens=max_tokens, reasoning=reasoning, **kwargs
            )
            
            # Extract token usage from the last response
            input_tokens, output_tokens, cached_input_tokens = self._extract_token_usage()
            call_duration = time.time() - start_time
            
            self.stats.add_llm_call(
                llm_type=self.llm_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                call_duration=call_duration,
                prompt_type=prompt_type,
                context=context,
                success=True
            )
            return response
            
        except Exception as e:
            call_duration = time.time() - start_time
            # Use fallback estimation for failed calls
            input_tokens = self._estimate_tokens(prompt)
            
            self.stats.add_llm_call(
                llm_type=self.llm_type,
                input_tokens=input_tokens,
                output_tokens=0,
                call_duration=call_duration,
                prompt_type=prompt_type,
                context=context,
                success=False
            )
            raise e
    
    async def chat_completion(self, messages: List[Dict[str, str]], 
                             max_tokens: int = None, reasoning: Dict = None, 
                             prompt_type: str = None, context: str = "", **kwargs) -> str:
        """Track a chat completion call."""
        start_time = time.time()
        
        # Use provided prompt_type or default to CHAT_COMPLETION
        if prompt_type is None:
            prompt_type = PromptType.CHAT_COMPLETION.value
        
        await self._increment_call_count()
        try:
            response = await self.client.chat_completion(
                messages, max_tokens=max_tokens, reasoning=reasoning, **kwargs
            )
            
            # Extract token usage from the last response
            input_tokens, output_tokens, cached_input_tokens = self._extract_token_usage()
            call_duration = time.time() - start_time
            
            self.stats.add_llm_call(
                llm_type=self.llm_type,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                call_duration=call_duration,
                prompt_type=prompt_type,
                context=context,
                success=True
            )
            return response
            
        except Exception as e:
            call_duration = time.time() - start_time
            # Use fallback estimation for failed calls
            input_text = "\n".join([msg.get('content', '') for msg in messages])
            input_tokens = self._estimate_tokens(input_text)
            
            self.stats.add_llm_call(
                llm_type=self.llm_type,
                input_tokens=input_tokens,
                output_tokens=0,
                call_duration=call_duration,
                prompt_type=prompt_type,
                context=context,
                success=False
            )
            raise e
    
    def _extract_token_usage(self) -> tuple[int, int, int]:
        """Extract token usage from the last response object.
        Returns (input_tokens, output_tokens, cached_input_tokens)."""
        getter = getattr(self.client, 'get_last_response_raw', None)
        last_response = getter() if getter else None
        if not last_response:
            return 0, 0, 0
        
        try:
            return self._extract_openai_tokens(last_response)
        except Exception as e:
            # Fallback to estimation if extraction fails
            logger.info(f"Warning: Failed to extract tokens from OpenAI response: {e}")
            return 0, 0, 0
        
        return 0, 0, 0
    
    def _extract_openai_tokens(self, response) -> tuple[int, int, int]:
        """Extract tokens from OpenAI response.
        Returns (input_tokens, output_tokens, cached_input_tokens)."""
        if hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            
            # Extract cached tokens from prompt_tokens_details
            cached_input_tokens = 0
            if hasattr(usage, 'prompt_tokens_details'):
                input_details = usage.prompt_tokens_details
                cached_input_tokens = getattr(input_details, 'cached_tokens', 0)
            
            return input_tokens, output_tokens, cached_input_tokens
        return 0, 0, 0
    
    
    def _estimate_tokens(self, text: str) -> int:
        """Fallback token estimation (4 characters ≈ 1 token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def __getattr__(self, name):
        """Delegate other methods to the wrapped client."""
        return getattr(self.client, name)

class TrackedAsyncProverLLM:
    """Wrapper for AsyncProverLLM that tracks all calls."""
    
    def __init__(self, 
                 prover_llm, 
                 stats: ProofStatistics,
                 max_num_llm_calls: int = None):
        self.prover_llm = prover_llm
        self.stats = stats
        self.max_num_llm_calls = max_num_llm_calls
        self.llm_count = 0

    async def _increment_call_count(self):
        """Increment the LLM call count and enforce max limit if set."""
        if self.max_num_llm_calls is not None and self.llm_count >= self.max_num_llm_calls:
            raise MaxLLMCallsExceeded(f"Exceeded maximum number of LLM calls: {self.max_num_llm_calls}")
        self.llm_count += 1
    
    def set_max_num_llm_calls(self, max_calls: int):
        """Set the maximum number of LLM calls."""
        self.max_num_llm_calls = max_calls
        
    async def generate_proof(self, theorem: str, context: str = "", **kwargs) -> str:
        """Track a proof generation call."""
        start_time = time.time()
        
        await self._increment_call_count()
        try:

            proof = await self.prover_llm.generate_proof(theorem, **kwargs)
            
            # Extract token usage from the underlying LLM client
            input_tokens, output_tokens, cached_input_tokens = self._extract_token_usage()
            call_duration = time.time() - start_time
            
            self.stats.add_llm_call(
                llm_type=LLMType.PROVER_LLM,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cached_input_tokens=cached_input_tokens,
                call_duration=call_duration,
                prompt_type=PromptType.GENERATE_PROOF.value,
                context=context,
                success=proof is not None
            )
            return proof
            
        except Exception as e:
            call_duration = time.time() - start_time
            # Use fallback for failed calls
            input_tokens = self._estimate_tokens(theorem)
            
            self.stats.add_llm_call(
                llm_type=LLMType.PROVER_LLM,
                input_tokens=input_tokens,
                output_tokens=0,
                call_duration=call_duration,
                prompt_type=PromptType.GENERATE_PROOF.value,
                context=context,
                success=False
            )
            raise e
    
    def _extract_token_usage(self) -> tuple[int, int, int]:
        """Extract token usage from the underlying LLM client.
        Returns (input_tokens, output_tokens, cached_input_tokens)."""
        if hasattr(self.prover_llm, 'llm_client'):
            llm_client = self.prover_llm.llm_client
            getter = getattr(llm_client, 'get_last_response_raw', None)
            last_response = getter() if getter else None
            if not last_response:
                return 0, 0, 0
            
            
            try:
                return self._extract_openai_tokens(last_response)
            except Exception as e:
                logger.info(f"Warning: Failed to extract tokens from OpenAI response: {e}")
                return 0, 0, 0
        
        return 0, 0, 0
    
    def _extract_openai_tokens(self, response) -> tuple[int, int, int]:
        """Extract tokens from OpenAI response.
        Returns (input_tokens, output_tokens, cached_input_tokens)."""
        if hasattr(response, 'usage'):
            usage = response.usage
            input_tokens = getattr(usage, 'prompt_tokens', 0)
            output_tokens = getattr(usage, 'completion_tokens', 0)
            
            # Extract cached tokens from prompt_tokens_details
            cached_input_tokens = 0
            if hasattr(usage, 'prompt_tokens_details'):
                input_details = usage.prompt_tokens_details
                cached_input_tokens = getattr(input_details, 'cached_tokens', 0)
            
            return input_tokens, output_tokens, cached_input_tokens
        return 0, 0, 0
    
    def _estimate_tokens(self, text: str) -> int:
        """Fallback token estimation (4 characters ≈ 1 token)."""
        if not text:
            return 0
        return max(1, len(text) // 4)
    
    def __getattr__(self, name):
        """Delegate other methods to the wrapped prover."""
        return getattr(self.prover_llm, name)

class StrategyTracker:
    """Context manager for tracking strategy execution."""
    
    def __init__(self, 
                 stats: ProofStatistics,
                 strategy: StrategyType, 
                 depth: int = 0, 
                 theorem_name: str = ""):
        self.stats = stats
        self.strategy = strategy
        self.depth = depth
        self.theorem_name = theorem_name
        self.attempt_index = None
        self._explicit_outcome = None
    
    def __enter__(self):
        self.attempt_index = self.stats.start_strategy_attempt(
            self.strategy, self.depth, self.theorem_name
        )
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.attempt_index is not None:
            successfully_completed, error_message = (
                (False, str(exc_val)) if exc_type is not None
                else self._explicit_outcome or (True, None)
            )
            self.stats.complete_strategy_attempt(
                self.attempt_index, successfully_completed, error_message
            )
    
    def mark_success(self):
        """Manually mark the strategy as successful."""
        self._explicit_outcome = (True, None)
    
    def mark_failure(self, error_message: str = ""):
        """Manually mark the strategy as failed."""
        self._explicit_outcome = (False, error_message)
    
    def increment_attempts(self):
        """Increment the attempt counter for this strategy."""
        if self.attempt_index is not None:
            self.stats.increment_attempt_count(self.attempt_index)

class TrackedSearchEngine:
    """Wrapper for SemanticSearchEngine that tracks all search operations."""
    
    def __init__(self, search_engine, stats: ProofStatistics):
        self.search_engine = search_engine
        self.stats = stats
    
    def search(self, query: str, top_k: int = None):
        """Track a semantic search operation."""
        start_time = time.time()
        
        try:
            results = self.search_engine.search(query, top_k)
            search_duration = time.time() - start_time
            
            self.stats.add_search_operation(
                search_type="semantic_search",
                query=query,
                num_results=len(results) if results else 0,
                search_duration=search_duration,
                success=True
            )
            
            return results
            
        except Exception as e:
            search_duration = time.time() - start_time
            
            self.stats.add_search_operation(
                search_type="semantic_search",
                query=query,
                num_results=0,
                search_duration=search_duration,
                success=False
            )
            raise e
    
    def get_search_results(self, query: str, top_k: int = 5) -> str:
        """Track a formatted search results operation."""
        start_time = time.time()
        
        try:
            results = self.search_engine.get_search_results(query, top_k)
            search_duration = time.time() - start_time
            
            # Estimate number of results from formatted string
            num_results = results.count('\n1. ') if results else 0
            
            self.stats.add_search_operation(
                search_type="semantic_search_formatted",
                query=query,
                num_results=num_results,
                search_duration=search_duration,
                success=True
            )
            
            return results
            
        except Exception as e:
            search_duration = time.time() - start_time
            
            self.stats.add_search_operation(
                search_type="semantic_search_formatted",
                query=query,
                num_results=0,
                search_duration=search_duration,
                success=False
            )
            raise e
    
    def search_by_name(self, names, exact_match=False):
        """Track a name-based search operation."""
        start_time = time.time()
        
        try:
            results = self.search_engine.search_by_name(names, exact_match)
            search_duration = time.time() - start_time
            
            # Estimate number of results from formatted string
            num_results = results.count('\n1. ') if results else 0
            
            # Convert names to string for query tracking
            query_str = str(names) if not isinstance(names, str) else names
            
            self.stats.add_search_operation(
                search_type="name_search",
                query=query_str,
                num_results=num_results,
                search_duration=search_duration,
                success=True
            )
            
            return results
            
        except Exception as e:
            search_duration = time.time() - start_time
            query_str = str(names) if not isinstance(names, str) else names
            
            self.stats.add_search_operation(
                search_type="name_search",
                query=query_str,
                num_results=0,
                search_duration=search_duration,
                success=False
            )
            raise e
    
    def __getattr__(self, name):
        """Delegate other methods to the wrapped search engine."""
        return getattr(self.search_engine, name)

class TrackedAsyncLeanVerifier:
    """Wrapper for AsyncLeanVerifier that tracks all verification operations."""
    
    def __init__(self, lean_verifier, stats: ProofStatistics):
        self.lean_verifier = lean_verifier
        self.stats = stats
    
    def _track_verification(self, verification_type: VerificationType, verification_duration: float,
                           timeout: int, is_sorry_ok: bool, return_error_message: bool,
                           success: bool, proof_valid: bool, has_error_message: bool,
                           proof_length: int, is_batch: bool, batch_size: int, context: str,
                           exception: Optional[Exception] = None):
        """Keep classified, bounded diagnostics instead of only an error flag."""
        if exception is not None:
            raw_outcomes = getattr(exception, "outcomes", ())
        else:
            getter = getattr(self.lean_verifier, "get_last_verification_outcomes", None)
            raw_outcomes = getter() if callable(getter) else ()
        outcomes = [item.to_dict() if hasattr(item, "to_dict") else dict(item)
                    for item in raw_outcomes]
        if outcomes:
            kinds = {item["outcome"] for item in outcomes}
            if "infrastructure_failure" in kinds:
                outcome = "infrastructure_failure"
            elif "resource_exhausted" in kinds:
                outcome = "resource_exhausted"
            else:
                outcome = next(iter(kinds)) if len(kinds) == 1 else "mixed"
            diagnostic = bounded_verification_diagnostic("\n".join(
                f"Snippet {i + 1} ({item['outcome']}): {item['diagnostic']}"
                for i, item in enumerate(outcomes) if item.get("diagnostic")
            ))
            retry_count = max((item.get("retry_count", 0) for item in outcomes), default=0)
            success = not bool(kinds & {"resource_exhausted", "infrastructure_failure"})
        else:
            outcome = "infrastructure_failure" if exception is not None else ("valid" if proof_valid else "mathematical_invalid")
            diagnostic = bounded_verification_diagnostic(exception) if exception is not None else ""
            retry_count = 0
        has_error_message = has_error_message or bool(diagnostic)
        self.stats.add_verification_operation(
            verification_type=verification_type,
            verification_duration=verification_duration,
            timeout=timeout,
            is_sorry_ok=is_sorry_ok,
            return_error_message=return_error_message,
            success=success,
            proof_valid=proof_valid,
            has_error_message=has_error_message,
            proof_length=proof_length,
            is_batch=is_batch,
            batch_size=batch_size,
            context=context,
            outcome=outcome,
            diagnostic=diagnostic,
            retry_count=retry_count,
            outcomes=outcomes,
        )
    
    async def verify_proof(self, proof: str, timeout: int = 30, 
                          return_error_message: bool = False, 
                          is_sorry_ok: bool = False,
                          verification_type: VerificationType = VerificationType.UNKNOWN_VERIFICATION,
                          context: str = ""):
        """Track a single proof verification call."""
        start_time = time.time()
        proof_length = len(proof.split('\n')) if proof else 0
        
        try:
            result = await self.lean_verifier.verify_proof(
                proof, timeout=timeout, return_error_message=return_error_message, 
                is_sorry_ok=is_sorry_ok
            )
            verification_duration = time.time() - start_time
            
            # Handle both return formats
            if return_error_message:
                proof_valid, error_message = result
                has_error_message = error_message is not None
            else:
                proof_valid = result
                has_error_message = False
            
            self._track_verification(
                verification_type, verification_duration, timeout, is_sorry_ok,
                return_error_message, True, proof_valid, has_error_message,
                proof_length, False, 1, context
            )
            
            return result
                
        except Exception as e:
            verification_duration = time.time() - start_time
            self._track_verification(
                verification_type, verification_duration, timeout, is_sorry_ok,
                return_error_message, False, False, False,
                proof_length, False, 1, context, exception=e
            )
            raise
    
    async def batch_verify_proofs(self, proofs: List[str], 
                                 return_error_messages: bool = False, 
                                 timeout: int = 30, 
                                 is_sorry_ok: bool = False,
                                 batch_size: Optional[int] = None,
                                 show_progress: bool = True,
                                 verification_type: VerificationType = VerificationType.BATCH_PROOF_VERIFICATION,
                                 context: str = ""):
        """Track a batch proof verification call."""
        start_time = time.time()
        actual_batch_size = len(proofs) if proofs else 0
        total_proof_length = sum(len(proof.split('\n')) for proof in proofs) if proofs else 0
        
        try:
            result = await self.lean_verifier.batch_verify_proofs(
                proofs, return_error_messages=return_error_messages, timeout=timeout,
                is_sorry_ok=is_sorry_ok, batch_size=batch_size, show_progress=show_progress
            )
            verification_duration = time.time() - start_time
            
            # Handle both return formats
            if return_error_messages:
                results, error_messages = result
                proof_valid = any(results) if results else False
                has_error_message = any(error_messages) if error_messages else False
            else:
                results = result
                proof_valid = any(results) if results else False
                has_error_message = False
            
            self._track_verification(
                verification_type, verification_duration, timeout, is_sorry_ok,
                return_error_messages, True, proof_valid, has_error_message,
                total_proof_length, True, actual_batch_size, context
            )
            
            return result
                
        except Exception as e:
            verification_duration = time.time() - start_time
            self._track_verification(
                verification_type, verification_duration, timeout, is_sorry_ok,
                return_error_messages, False, False, False,
                total_proof_length, True, actual_batch_size, context, exception=e
            )
            raise
    
    def __getattr__(self, name):
        """Delegate other methods to the wrapped verifier."""
        return getattr(self.lean_verifier, name)
