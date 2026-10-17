#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2025 Apple Inc. All Rights Reserved.
#
from dreamprover.prover.runner import AsyncHILBERT
from dreamprover.prover.config import ProofAttemptConfig
from dreamprover.prover.generation import AsyncProverLLM
from dreamprover.clients.openai import AsyncLLMClient
from dreamprover.lean.verifier import AsyncLeanVerifier
from logging import getLogger

logger = getLogger(__name__)

def run_async_hilbert(cfg):
    """
    Run AsyncHILBERT experiment
    The async handling is done internally by AsyncHILBERT.run_from_file().
    
    Args:
        cfg: Configuration object containing experiment parameters
        
    Returns:
        Dictionary with experiment results
    """
    # Extract config sections for cleaner access
    exp_cfg = cfg.experiment
    prover_cfg = exp_cfg.prover_llm
    informal_cfg = exp_cfg.informal_llm
    enable_retrieval = exp_cfg.get('enable_retrieval', True)
    
    max_concurrent_requests = exp_cfg.max_concurrent_requests
    logger.info("Starting AsyncHILBERT experiment...")
    
    # Create factory functions for AsyncHILBERT
    # These ensure each worker gets fresh client instances
    
    def create_prover_llm():
        """Factory function to create AsyncProverLLM instances."""
        # Get fresh AsyncLLMClient instance from task_id
        base_url = prover_cfg.base_url
        prover_llm_client = AsyncLLMClient(base_url=base_url, model_name=prover_cfg.llm_name,
                                        api_key=prover_cfg.get('api_key'),
                                        timeout=prover_cfg.get('timeout', 2400))

        return AsyncProverLLM(
            llm_client=prover_llm_client,
            model_name=prover_cfg.llm_name,
            prompt_strategy=prover_cfg.prompt_strategy,
            max_tokens=prover_cfg.max_tokens
        )
    
    def create_informal_llm_client():
        """Factory function to create AsyncLLMClient instances for informal reasoning."""
        params = dict(informal_cfg)
        provider = params.pop('provider', 'openai')
        params.pop('max_tokens', None)
        if provider != 'openai':
            raise ValueError(
                f"Unsupported informal_llm.provider: {provider!r}; "
                "use 'openai' with an OpenAI-compatible base_url"
            )
        return AsyncLLMClient(**params)
    
    def create_lean_verifier():
        """Factory function to create AsyncLeanVerifier instances."""
        verifier_base_url = exp_cfg.verifier_base_url
        return AsyncLeanVerifier(base_url=verifier_base_url,
                        max_concurrent_requests=max_concurrent_requests)
    
    # Create the shared semantic search engine (thread-safe)
    search_engine = None
    if enable_retrieval:
        from dreamprover.lean.retrieval import SemanticSearchEngine
        search_engine = SemanticSearchEngine(**dict(exp_cfg.search_engine))

    # Create AsyncHILBERT instance with factory functions
    experiment = AsyncHILBERT(
        prover_llm_factory=create_prover_llm,
        informal_llm_client_factory=create_informal_llm_client,
        lean_verifier_factory=create_lean_verifier,
        search_engine=search_engine,
        verify_each_subgoal_separately=exp_cfg.verify_each_subgoal_separately,
        proof_attempt_config=ProofAttemptConfig.from_dict(dict(exp_cfg.proof_attempt_config)),
        complexity_proof_length_cutoff=exp_cfg.complexity_proof_length_cutoff,
        return_proofs=exp_cfg.save_proofs_to_disk,
        max_depth=exp_cfg.max_depth,
        max_concurrent_problems=exp_cfg.max_concurrent_problems,
        proof_save_dir=exp_cfg.proof_save_dir if exp_cfg.save_proofs_to_disk else None,
        max_tokens=exp_cfg.get('max_tokens', informal_cfg.get('max_tokens', 16384)),
        sequential_processing=exp_cfg.get('sequential_processing', False),
        run_proof_attempts_sequentially=exp_cfg.run_proof_attempts_sequentially,
        enable_retrieval=enable_retrieval,
        lemma_library_path=exp_cfg.get('lemma_library_path'),
        validate_lemma_library=exp_cfg.get('validate_lemma_library', True),
    )
    
    # Run the experiment - run_from_file handles asyncio.run internally
    logger.info(f"Processing problems from: {cfg.data.file_path}")
    results = experiment.run_from_file(cfg.data.file_path)
    
    return results
