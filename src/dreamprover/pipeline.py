"""Resumable, budgeted wake/sleep training and matched held-out evaluation.

Each completed problem and stage has a durable checkpoint. API response replay
allows an interrupted problem/stage to resume without purchasing its successful
requests again. Only evolved candidates with actual Lean proofs enter the fixed
library used for evaluation; held-out problems never update that library.
"""
from __future__ import annotations

import asyncio
import json
import logging
import random
import re
import os
import math
import hashlib
import signal
import socket
import threading
from functools import lru_cache
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime, timezone
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from dreamprover.runtime.budget import BudgetExhausted, BudgetLedger, BudgetedClient, RunStopped, atomic_json, fingerprint
from dreamprover.learning.extract import build_library, library_lean_source, write_library
from dreamprover.lean.source import extract_declarations, mask_comments_and_strings, referenced_names, split_statement_proof
from dreamprover.learning.update import _statement_key
from dreamprover.learning.checkpoints import load_cycle_library
from dreamprover.paths import config_directory, repository_root, working_directory
from dreamprover.prover.worker import HILBERTWorker
from dreamprover.runtime.io import safe_problem_filename
from dreamprover.lean.library import LemmaLibrary
from dreamprover.lean.proof import VerificationInfrastructureError
from dreamprover.prover.config import ProofAttemptConfig
from dreamprover.runtime.tracking import MaxLLMCallsExceeded

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    train_data: str = "data/inequalities/train.jsonl"
    eval_data: str = "data/inequalities/eval.jsonl"
    output_dir: str = "runs/inequalities/gpt-6-luna"
    lean_project: str = "vendor/kimina-lean-server/mathlib4-v4.15.0"
    verifier_url: str = "http://127.0.0.1:10002"
    model: str = "gpt-6-luna"
    base_url: str = "https://api.openai.com/v1"
    domain: str = "inequalities"
    cycles: int = 5
    train_limit: int = 100
    eval_limit: int | None = None
    seed: int = 0
    capacity: int = 99
    wake_depth: int = 3
    inference_depth: int = 1
    direct_attempts: int = 4
    corrections: int = 6
    wake_sketch_attempts: int = 1
    inference_sketch_attempts: int = 4
    max_completion_tokens: int = 16384
    problem_concurrency: int = 2
    max_prover_calls_per_problem: int | None = None
    max_reasoner_calls_per_problem: int | None = None
    verification_timeout: int = 120
    embedding_model: str = "artifacts/models/all-mpnet-base-v2"
    embedding_revision: str = "e8c3b32edf5434bc2275fc9bab85f82640a19130"
    max_clusters: int = 12
    relevance_threshold: float = 0.5
    duplicate_threshold: float = 0.95
    structural_parser_dir: str | None = "artifacts/tbps-v4.15/parser"
    tbps_root: str = "vendor/tbps"
    max_cost_usd: float | None = None
    max_calls: int | None = None
    max_total_tokens: int | None = None
    input_per_million: float = 0.10
    output_per_million: float = 0.50
    cached_input_per_million: float = 0.01

    @classmethod
    def from_mapping(cls, value):
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown pipeline options: {sorted(unknown)}")
        config = cls(**value)
        for name in ("cycles", "train_limit", "capacity", "direct_attempts", "max_completion_tokens", "max_clusters", "problem_concurrency", "verification_timeout"):
            if not isinstance(getattr(config, name), int) or isinstance(getattr(config, name), bool) or getattr(config, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("wake_depth", "inference_depth", "corrections", "wake_sketch_attempts", "inference_sketch_attempts"):
            if not isinstance(getattr(config, name), int) or isinstance(getattr(config, name), bool) or getattr(config, name) < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        for name in ("relevance_threshold", "duplicate_threshold"):
            if not math.isfinite(getattr(config, name)) or not 0 <= getattr(config, name) <= 1:
                raise ValueError(f"{name} must be finite and between zero and one")
        for name in ("max_prover_calls_per_problem", "max_reasoner_calls_per_problem"):
            value = getattr(config, name)
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise ValueError(f"{name} must be a positive integer")
        if config.eval_limit is not None and (not isinstance(config.eval_limit, int) or isinstance(config.eval_limit, bool) or config.eval_limit < 1):
            raise ValueError("eval_limit must be a positive integer")
        return config

    def identity(self):
        # Only run-wide ceilings can increase on resume. Every scientific or
        # per-problem compute choice remains fixed in the manifest.
        return {name: value for name, value in asdict(self).items()
                if name not in {"max_cost_usd", "max_calls", "max_total_tokens"}}

    def proof_config(self, training: bool):
        return ProofAttemptConfig(
            formal_proof_attempts=self.direct_attempts,
            main_theorem_error_corrections=self.corrections,
            subgoal_error_corrections=self.corrections,
            subgoal_decomp_attempts=self.wake_sketch_attempts if training else self.inference_sketch_attempts,
            parallel_subgoal_proof_attempts=self.direct_attempts,
            proof_sketch_corrections=self.corrections,
            proof_verification_timeout=self.verification_timeout,
            max_prover_llm_calls=self.max_prover_calls_per_problem,
            max_reasoner_llm_calls=self.max_reasoner_calls_per_problem,
        )


def load_problems(path, *, limit=None, seed=0) -> list[dict]:
    path = Path(path)
    if path.suffix == ".jsonl":
        raw = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    else:
        raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"Dataset must contain a nonempty list of problems: {path}")
    problems = []
    for record in raw:
        statement = record.get("formal_statement")
        identifier = record.get("id", record.get("name"))
        if not isinstance(statement, str) or not statement.strip() or identifier is None:
            raise ValueError("Every problem needs id/name and a nonempty formal_statement")
        # Validate a single named target, including quoted names and := binder
        # defaults. Input problems must not supply their answer as a proof.
        try:
            split_statement_proof(statement)
            source = statement
        except ValueError:
            source = statement + " := by sorry"
        preamble, declarations = extract_declarations(source)
        if len(declarations) != 1 or mask_comments_and_strings(preamble).strip():
            raise ValueError("formal_statement must contain exactly one theorem and no preamble")
        body = mask_comments_and_strings(declarations[0].proof).strip()
        if not re.fullmatch(r"(?:by(?:\s+sorry)?|sorry)?", body):
            raise ValueError("Dataset target already contains a proof; provide its statement only")
        problems.append(dict(record, id=str(identifier), header=record.get("header", "")))
    if len({record["id"] for record in problems}) != len(problems):
        raise ValueError("Duplicate problem IDs in dataset")
    problems.sort(key=lambda record: record["id"])
    random.Random(seed).shuffle(problems)
    if limit is not None:
        if len(problems) < limit:
            raise ValueError(f"Requested {limit} problems, but dataset contains only {len(problems)}")
        problems = problems[:limit]
    return problems


def problem_statement_key(problem):
    try:
        statement, _ = split_statement_proof(problem["formal_statement"])
    except ValueError:
        statement = problem["formal_statement"]
    return _statement_key(statement)


def assert_disjoint(train: list[dict], evaluation: list[dict], library: dict | None = None):
    overlap = {record["id"] for record in train} & {record["id"] for record in evaluation}
    if overlap:
        raise ValueError(f"Train/evaluation ID leakage: {sorted(overlap)}")
    keys = {problem_statement_key(record) for record in train}
    eval_keys = {problem_statement_key(record) for record in evaluation}
    if keys & eval_keys:
        raise ValueError("Train/evaluation proposition leakage (theorem names ignored)")
    if library and {_statement_key(record["statement"]) for record in library.values()} & eval_keys:
        raise ValueError("The learned library contains a held-out target proposition")


def _read(path: Path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


@lru_cache(maxsize=256)
def _file_hash(path: str, size: int, modified_ns: int):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_if_present(path: Path):
    if not path.is_file():
        return None
    info = path.stat()
    return _file_hash(str(path.resolve()), info.st_size, info.st_mtime_ns)


def runtime_identity(config: PipelineConfig):
    project = Path(config.lean_project)
    model = Path(config.embedding_model)
    parser = Path(config.structural_parser_dir) if config.structural_parser_dir else None
    toolchain = (project / "lean-toolchain").read_text().strip() if (project / "lean-toolchain").is_file() else None
    version = toolchain.rsplit(":", 1)[-1] if toolchain else None
    kimina_candidates = [Path("artifacts") / ("kimina-" + version if version else "kimina") / "setup.json"]
    if version and version.endswith(".0"):
        kimina_candidates.append(Path("artifacts") / ("kimina-" + version[:-2]) / "setup.json")
    kimina_metadata = next((path for path in kimina_candidates if path.exists()), kimina_candidates[0])
    model_files = {str(path.relative_to(model)): _hash_if_present(path)
                   for path in sorted(model.rglob("*")) if path.is_file()
                   and path.suffix in {".safetensors", ".bin", ".json", ".txt", ".model"}} if model.is_dir() else {}
    return dict(lean_toolchain=toolchain, lake_manifest_sha256=_hash_if_present(project / "lake-manifest.json"),
                lakefile_sha256=_hash_if_present(project / "lakefile.lean"),
                kimina_setup=_read(kimina_metadata),
                embedding_setup=_read(model.parent / "embedding-model.json"),
                embedding_revision=config.embedding_revision, embedding_files_sha256=model_files,
                structural_parser_sha256={name: _hash_if_present(parser / name)
                    for name in ("Mathlib_Construction.lean", "Mathlib_Construction.olean")} if parser else {},
                tbps_setup=_read(parser.parent / "setup.json") if parser else None)


@contextmanager
def run_lock(directory):
    """Reject a second CLI writer immediately, instead of corrupting a run."""
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".run.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process is writing this run: {directory}") from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()) + "\n")
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _replace_names(source: str, names: dict[str, str]) -> str:
    code = mask_comments_and_strings(source)
    replacements = []
    for old, new in names.items():
        for match in re.finditer(r"(?<![\w'.])" + re.escape(old) + r"(?![\w'])", code):
            replacements.append((match.start(), match.end(), new))
    for start, end, new in sorted(replacements, reverse=True):
        source = source[:start] + new + source[end:]
    return source


class ExperienceWorker(HILBERTWorker):
    def __init__(self, *args, experience_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.experience_dir = experience_dir
        # Retain captures from interrupted attempts. Their proof bodies also
        # supply wake-stage usage counts for library retention after resume.
        self.experiences = _read(self.experience_dir / "index.json", [])

    def _record_verified_direct_proof(self, theorem, header, proof):
        source = header + "\n" + proof
        digest = fingerprint(source)
        if any(record["digest"] == digest for record in self.experiences):
            return
        self.experience_dir.mkdir(parents=True, exist_ok=True)
        path = self.experience_dir / (digest + ".lean")
        path.write_text(source, encoding="utf-8")
        self.experiences.append(dict(digest=digest, path=str(path), theorem=theorem, proof=proof))
        atomic_json(self.experience_dir / "index.json", self.experiences)


def _saved_experience_proofs(directory: Path, header: str) -> list[str]:
    """Recover usage from durable captures, including old incomplete indexes.

    The exact trusted header contains the injected library. Remove it before
    counting references so declarations never masquerade as lemma uses. Wake
    collection independently verifies these complete exports before sleep.
    """
    proofs = []
    prefix = header + "\n"
    for path in sorted(directory.glob("*.lean")):
        # Preserve CRLF inside model output when checking the original digest.
        source = path.read_bytes().decode("utf-8")
        if path.stem != fingerprint(source):
            raise ValueError(f"Wake experience content does not match its digest: {path}")
        if not source.startswith(prefix):
            raise ValueError(f"Wake experience does not match the trusted preamble: {path}")
        proofs.append(source[len(prefix):])
    return proofs


class Pipeline:
    def __init__(self, config: PipelineConfig, *, phase: str | None = None, stop_event=None):
        config = PipelineConfig.from_mapping(asdict(config))
        self.config = config
        self.stop_event = stop_event if stop_event is not None else threading.Event()
        self.root = Path(config.output_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.train = load_problems(config.train_data, limit=config.train_limit, seed=config.seed)
        self.evaluation = load_problems(config.eval_data, limit=config.eval_limit, seed=config.seed)
        # Check overlap against the entire held-out dataset.
        assert_disjoint(self.train, load_problems(config.eval_data, seed=config.seed))
        headers = {record["header"].strip() for record in self.train}
        if len(headers) != 1:
            raise ValueError("Domain training requires one common Lean preamble")
        self.header = next(iter(headers))
        identity = dict(config=config.identity(), train=self.train, evaluation=self.evaluation,
                        runtime=runtime_identity(config),
                        train_file_sha256=fingerprint(Path(config.train_data).read_text()),
                        eval_file_sha256=fingerprint(Path(config.eval_data).read_text()))
        manifest = _read(self.root / "manifest.json")
        def scientific_identity(value):
            copied = dict(value, config=dict(value["config"]))
            # Parallel admission changes throughput, while every worker retains
            # its own ordered prompts, replay keys and fixed compute limits.
            # Normalize stored manifests without rewriting them.
            copied["config"].pop("problem_concurrency", None)
            return copied
        if manifest is not None and scientific_identity(manifest["identity"]) != scientific_identity(identity):
            raise ValueError("Resume configuration/dataset differs from this run's immutable manifest")
        if manifest is None:
            atomic_json(self.root / "manifest.json", dict(schema_version=1, identity=identity,
                         model=config.model, training_ids=[p["id"] for p in self.train],
                         evaluation_ids=[p["id"] for p in self.evaluation],
                         paper_defaults=dict(training_problems=100, cycles=5, wake_depth=3,
                         inference_depth=1, direct_attempts=4, corrections=6, wake_sketches=1,
                         inference_sketches=4, capacity=99),
                         leakage_policy="disjoint IDs and name-independent propositions; frozen evaluation library"))
        self.ledger = BudgetLedger(self.root, max_cost_usd=config.max_cost_usd,
                                  max_calls=config.max_calls, max_tokens=config.max_total_tokens,
                                  input_per_million=config.input_per_million,
                                  output_per_million=config.output_per_million,
                                  cached_input_per_million=config.cached_input_per_million)
        execution_path = self.root / "execution_history.json"
        execution_history = _read(execution_path, [])
        execution = dict(phase=phase, pid=os.getpid(), hostname=socket.gethostname(),
            started_at=datetime.now(timezone.utc).isoformat(), problem_concurrency=config.problem_concurrency,
            verification_timeout=config.verification_timeout, budget_limits=self.ledger.limits,
            implementation_sha256={"src/dreamprover/" + str(path.relative_to(Path(__file__).resolve().parent)): _hash_if_present(path)
                for path in sorted(Path(__file__).resolve().parent.rglob("*.py"))})
        if os.environ.get("SLURM_JOB_ID"):
            execution["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
        execution_history.append(execution)
        atomic_json(execution_path, execution_history)
        self._embedding_model = None
        self._comparator = None

    def _client(self, scope):
        from dreamprover.clients.openai import AsyncLLMClient
        client = AsyncLLMClient(base_url=self.config.base_url, model_name=self.config.model,
                                timeout=600, max_retries=0, temperature_fallback=False)
        return BudgetedClient(client, self.ledger, scope, stop_event=self.stop_event)

    def _check_stop(self):
        event = getattr(self, "stop_event", None)
        if event is not None and event.is_set():
            raise RunStopped("Stop requested; unfinished work remains resumable")

    def complete(self, scope, validator=None):
        # One adapter per stage preserves repeated-prompt indices across calls.
        client = self._client(scope)
        def callback(prompt):
            async def request():
                try:
                    current = prompt
                    for correction in range(self.config.corrections + 1):
                        response = await client.simple_chat(current, max_tokens=self.config.max_completion_tokens)
                        if validator is None:
                            return response
                        try:
                            validator(response)
                            return response
                        except ValueError as exc:
                            if correction == self.config.corrections:
                                raise
                            current = (prompt + "\nThe previous response did not follow the requested format.\n"
                                       + f"Previous response:\n{response}\nFeedback:\n{exc}\nReturn a corrected response.")
                finally:
                    # Each synchronous call has its own event loop. Close the
                    # HTTP session before leaving that loop, keeping replay state.
                    await client.close()
            return asyncio.run(request())
        return callback

    @property
    def comparator(self):
        if self._comparator is None:
            from dreamprover.lean.structural import StructuralComparator
            self._comparator = StructuralComparator(self.config.lean_project,
                parser_dir=self.config.structural_parser_dir, tbps_root=self.config.tbps_root,
                timeout=self.config.verification_timeout, cache_dir=self.root / "structural_cache")
        return self._comparator

    def _status(self, phase, status, **extra):
        atomic_json(self.root / "status.json", dict(phase=phase, status=status,
                    budget=self.ledger.summary(), **extra))

    def run_problem(self, problem, library, directory: Path, *, training: bool,
                    proof_config=None, max_depth=None):
        checkpoint = directory / "result.json"
        if checkpoint.exists():
            return _read(checkpoint)
        directory.mkdir(parents=True, exist_ok=True)
        async def run():
            from dreamprover.prover.generation import AsyncProverLLM
            from dreamprover.lean.verifier import AsyncLeanVerifier
            scope = str(directory.relative_to(self.root))
            prover_client = self._client(scope + "/prover")
            reasoner_client = self._client(scope + "/reasoner")
            source = library_lean_source(library) if library else ""
            worker = ExperienceWorker(
                AsyncProverLLM(prover_client, self.config.model, max_tokens=self.config.max_completion_tokens),
                reasoner_client,
                AsyncLeanVerifier(self.config.verifier_url, max_concurrent_requests=2), None,
                proof_attempt_config=proof_config if proof_config is not None else self.config.proof_config(training),
                max_depth=max_depth if max_depth is not None else (self.config.wake_depth if training else self.config.inference_depth),
                enable_retrieval=False, enable_statistics=True,
                lemma_library=LemmaLibrary(source, "\n".join(record["statement"] for record in library.values())),
                max_tokens=self.config.max_completion_tokens, proof_save_dir=str(directory / "proofs"),
                run_proof_attempts_sequentially=True, experience_dir=directory / "experiences")
            try:
                success, proof = await worker.generate_single_proof(problem["formal_statement"], problem["header"], problem["id"])
                result = dict(id=problem["id"], success=success, proof=proof,
                              stop_reason="completed", statistics=worker.stats.get_summary_stats())
            except BudgetExhausted:
                # Keep direct experiences and cached calls, but leave result.json
                # absent so the interrupted problem resumes after a cap increase.
                raise
            except VerificationInfrastructureError as exc:
                atomic_json(directory / "incomplete.json", dict(
                    id=problem["id"], stop_reason="verification_infrastructure_failure",
                    resumable=True, diagnostic=exc.diagnostic,
                    verification_outcomes=[outcome.to_dict() for outcome in exc.outcomes],
                    statistics=worker.stats.get_summary_stats(),
                    api_usage=self.ledger.summary(scope + "/")))
                raise
            except MaxLLMCallsExceeded as exc:
                result = dict(id=problem["id"], success=False, proof=None,
                              stop_reason="per_problem_call_limit", error=str(exc),
                              statistics=worker.stats.get_summary_stats())
            finally:
                await worker.close()
            if not result["success"]:
                self._check_stop()
            result["api_usage"] = self.ledger.summary(scope + "/")
            result["library_lemmas_used"] = referenced_names(result.get("proof") or "", set(library))
            atomic_json(checkpoint, result)
            (directory / "incomplete.json").unlink(missing_ok=True)
            return result
        return asyncio.run(run())

    def run_jobs(self, jobs: list[tuple], *, training: bool, proof_config=None, max_depth=None):
        """Continuously fill a bounded set of active problems; checkpoint each.

        Admission order is deterministic. Completion and remote token use may
        vary under concurrency, so the ledger records the actual request order.
        Already admitted work finishes before a budget stop returns control.
        """
        concurrency = self.config.problem_concurrency
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending = {}
            iterator = iter(enumerate(jobs))
            def admit_next():
                self._check_stop()
                try:
                    index, (problem, library, directory) = next(iterator)
                except StopIteration:
                    return False
                future = executor.submit(self.run_problem, problem, library, directory,
                                         training=training, proof_config=proof_config, max_depth=max_depth)
                pending[future] = index
                return True
            for _ in range(concurrency):
                if not admit_next():
                    break
            while pending:
                done, _ = wait(pending, return_when=FIRST_COMPLETED)
                error = None
                for future in sorted(done, key=pending.get):
                    pending.pop(future)
                    try:
                        future.result()
                    except BaseException as exc:
                        if error is None:
                            error = exc
                if error is not None:
                    if isinstance(error, VerificationInfrastructureError):
                        self.stop_event.set()
                    # No queued tail: already admitted jobs finish before the
                    # original error is raised. Their checkpoints remain usable.
                    for future in sorted(pending, key=pending.get):
                        try:
                            future.result()
                        except BaseException:
                            pass
                    raise error
                while len(pending) < concurrency and admit_next():
                    pass
        self._check_stop()

    def collect_experiences(self, cycle: int, directory: Path, library: dict):
        checkpoint = directory / "experiences.json"
        # Reconstruct the aggregate from individually validated exports on each
        # resume. An aggregate alone cannot authenticate changed wake captures.
        records = {}
        captures = []
        for problem in self.train:
            problem_dir = directory / "wake" / safe_problem_filename(problem["id"])
            files = sorted((problem_dir / "experiences").glob("*.lean"))
            for capture_index, path in enumerate(files):
                captures.append((problem, capture_index, path))
        compiler_digest = _hash_if_present(Path(__file__).resolve().parent / "lean/compiler.py")

        def extract_capture(capture):
            problem, capture_index, path = capture
            source = path.read_bytes().decode("utf-8")
            if path.stem != fingerprint(source):
                raise ValueError(f"Wake experience content does not match its digest: {path}")
            _, declarations = extract_declarations(source)
            new = [declaration for declaration in declarations if declaration.name not in library]
            prefix = f"wake_c{cycle}_{fingerprint(problem['id'])[:10]}_{capture_index}_"
            names = {decl.name: prefix + str(index) for index, decl in enumerate(new)}
            renamed = _replace_names(source, names)
            export = directory / "extracted" / safe_problem_filename(problem["id"]) / str(capture_index)
            export.mkdir(parents=True, exist_ok=True)
            identity = dict(source_sha256=fingerprint(renamed), compiler_sha256=compiler_digest,
                            verification_timeout=self.config.verification_timeout,
                            runtime_sha256=fingerprint(_read(self.root / "manifest.json")["identity"]["runtime"]))
            try:
                saved = _read(export / "verified.json")
            except (json.JSONDecodeError, UnicodeDecodeError):
                saved = None
            if (isinstance(saved, dict) and isinstance(saved.get("records"), dict)
                    and all(isinstance(value, dict) for value in saved["records"].values())
                    and saved.get("identity") == identity
                    and saved.get("records_sha256") == fingerprint(saved.get("records"))):
                extracted = saved["records"]
            else:
                (export / "proof.lean").write_text(renamed)
                # Each export still receives an independent native Lean check.
                extracted = build_library(export, verify=True, project_root=self.config.lean_project,
                                          timeout=self.config.verification_timeout)
                atomic_json(export / "verified.json", dict(identity=identity, records=extracted,
                            records_sha256=fingerprint(extracted)))
            return {name: dict(record, training_problem_id=problem["id"], cycle=cycle,
                               original_wake_source=str(path))
                    for name, record in extracted.items() if name not in library}

        self._map_stage(captures, extract_capture, records.update)
        atomic_json(checkpoint, records)
        return records

    def _map_stage(self, items, function, consume):
        """Process independent stage items with bounded workers and stable order."""
        def execute(item):
            self._check_stop()
            try:
                return function(item)
            except BaseException:
                self.stop_event.set()
                raise
        with ThreadPoolExecutor(max_workers=self.config.problem_concurrency) as executor:
            try:
                for value in executor.map(execute, items):
                    consume(value)
            except BaseException:
                # Queued work stops before new paid calls. Admitted calls settle
                # and independently verified exports keep their own checkpoints.
                self.stop_event.set()
                raise

    def embeddings(self, library: dict, directory: Path):
        import numpy as np
        path = directory / "embeddings.npy"
        descriptions = [record["description"] for record in library.values()]
        identity = dict(model=self.config.embedding_model, revision=self.config.embedding_revision,
                        descriptions=fingerprint(descriptions), names=list(library))
        metadata = _read(directory / "embedding_metadata.json")
        if path.exists() and metadata == identity:
            return np.load(path, allow_pickle=False)
        if self._embedding_model is None:
            from sentence_transformers import SentenceTransformer
            self._embedding_model = SentenceTransformer(self.config.embedding_model, revision=self.config.embedding_revision)
        values = self._embedding_model.encode(descriptions, normalize_embeddings=True, show_progress_bar=True)
        np.save(path, values, allow_pickle=False)
        atomic_json(directory / "embedding_metadata.json", identity)
        return values

    def sleep_cycle(self, cycle: int, directory: Path, library: dict, experiences: dict):
        from dreamprover.learning.annotate import annotate_library, parse_description
        from dreamprover.learning.sleep import propose_from_clusters, parse_candidates
        from dreamprover.learning.filtering import filter_candidates, deduplicate_candidates
        from dreamprover.learning.prove import prove_candidates
        from dreamprover.learning.update import update_library
        completed = load_cycle_library(directory)
        if completed is not None:
            return completed
        annotations = _read(directory / "annotations.json", {})
        pending_annotations = [(name, record) for name, record in experiences.items() if name not in annotations]
        def annotate(item):
            name, record = item
            return annotate_library({name: record}, self.complete(f"cycles/cycle_{cycle:02d}/annotation/{name}", parse_description))
        def save_annotation(value):
            annotations.update(value)
            atomic_json(directory / "annotations.json", annotations)
        self._map_stage(pending_annotations, annotate, save_annotation)
        proposals = _read(directory / "proposals.json")
        if proposals is None:
            if annotations:
                proposals = propose_from_clusters(annotations, self.embeddings(annotations, directory),
                    self.complete(f"cycles/cycle_{cycle:02d}/abstraction", parse_candidates), domain=self.config.domain,
                    max_clusters=self.config.max_clusters, seed=self.config.seed, name_prefix=f"dream_c{cycle}_")
                proposals["clustering"]["embedding_source"] = self.config.embedding_model
            else:
                proposals = dict(domain=self.config.domain, candidates=[], clusters={},
                                 clustering=dict(clusters=0, reason="no_strictly_proved_wake_experiences"))
            atomic_json(directory / "proposals.json", proposals)
        filtered = _read(directory / "filtered_proposals.json")
        if filtered is None:
            filtered = filter_candidates(proposals, annotations, self.comparator,
                                         relevance_threshold=self.config.relevance_threshold)
            filtered = deduplicate_candidates(filtered, library, self.comparator,
                                             duplicate_threshold=self.config.duplicate_threshold)
            atomic_json(directory / "filtered_proposals.json", filtered)
        proved, proof_reports = {}, []
        for index, candidate in enumerate(filtered["candidates"]):
            path = directory / "candidate_proofs" / f"{index:04d}.json"
            cached = _read(path)
            if cached is None:
                new, reports = prove_candidates([candidate], self.complete(f"cycles/cycle_{cycle:02d}/candidate/{index}"),
                    header=self.header, library=dict(library, **proved), project_root=self.config.lean_project,
                    attempts=self.config.direct_attempts, corrections=self.config.corrections,
                    timeout=self.config.verification_timeout)
                cached = dict(proofs=new, reports=reports)
                atomic_json(path, cached)
            proved.update(cached["proofs"])
            proof_reports.extend(cached["reports"])
        atomic_json(directory / "verified_candidates.json", proved)
        atomic_json(directory / "candidate_proof_report.json", proof_reports)
        # Count uses in generated proof bodies only: injected library declarations
        # would otherwise make every lemma appear used on every wake attempt.
        usage_sources = []
        wake_library = LemmaLibrary(library_lean_source(library) if library else "")
        for problem in self.train:
            problem_dir = directory / "wake" / safe_problem_filename(problem["id"])
            result = _read(problem_dir / "result.json", {})
            header = wake_library.with_header(problem["header"])
            direct = _saved_experience_proofs(problem_dir / "experiences", header)
            sources = [result.get("proof") or ""] + direct
            if any(sources):
                usage_sources.append("\n".join(sources))
        updated, report = update_library(library, proved, capacity=self.config.capacity, cycle=cycle,
            usage_sources=usage_sources, project_root=self.config.lean_project,
            timeout=self.config.verification_timeout, structural_comparator=self.comparator,
            duplicate_threshold=self.config.duplicate_threshold)
        counts = dict(wake_problems=len(self.train), wake_proved=sum(
            _read(directory / "wake" / safe_problem_filename(p["id"]) / "result.json", {}).get("success", False)
            for p in self.train), learnable_experiences=len(experiences), annotated=len(annotations),
            proposed=len(proposals["candidates"]), retained_for_proving=len(filtered["candidates"]),
            verified_candidates=len(proved), final_library=len(updated))
        atomic_json(directory / "update_report.json", dict(report, stage_counts=counts,
                    budget=self.ledger.summary(f"cycles/cycle_{cycle:02d}/")))
        write_library(updated, directory / "library", export_lean=True)
        atomic_json(directory / "complete.json", dict(cycle=cycle, library_sha256=fingerprint(updated)))
        return updated

    def load_cycle_library(self, cycle: int):
        if not 1 <= cycle <= self.config.cycles:
            raise ValueError("Cycle must be within the configured training range")
        library = load_cycle_library(self.root / "cycles" / f"cycle_{cycle:02d}")
        if library is None:
            raise ValueError(f"Cycle {cycle} has not completed")
        return library

    def train_run(self, *, stop_after_cycle=None):
        last_cycle = self.config.cycles if stop_after_cycle is None else stop_after_cycle
        if not isinstance(last_cycle, int) or isinstance(last_cycle, bool) or not 1 <= last_cycle <= self.config.cycles:
            raise ValueError("stop_after_cycle must be within the configured training range")
        self._status("training", "running")
        library = {}
        try:
            for cycle in range(1, last_cycle + 1):
                self._check_stop()
                directory = self.root / "cycles" / f"cycle_{cycle:02d}"
                completed = load_cycle_library(directory)
                if completed is not None:
                    library = completed
                    continue
                directory.mkdir(parents=True, exist_ok=True)
                atomic_json(directory / "input_library.json", library)
                self._status("training", "running", cycle=cycle)
                self.run_jobs([(problem, library, directory / "wake" / safe_problem_filename(problem["id"]))
                               for problem in self.train], training=True)
                experiences = self.collect_experiences(cycle, directory, library)
                library = self.sleep_cycle(cycle, directory, library, experiences)
                assert_disjoint(self.train, load_problems(self.config.eval_data, seed=self.config.seed), library)
            self._check_stop()
            if last_cycle < self.config.cycles:
                self._status("training", "cycle_complete", completed_cycles=last_cycle,
                             planned_cycles=self.config.cycles, library_size=len(library))
                return _read(self.root / "status.json")
            atomic_json(self.root / "final_library.json", library)
            if library:
                (self.root / "final_library.lean").write_text(library_lean_source(library))
            self._status("training", "trained", completed_cycles=self.config.cycles, library_size=len(library))
        except BudgetExhausted as exc:
            self._status("training", "budget_exhausted", reason=str(exc))
        except RunStopped as exc:
            self._status("training", "stopped", reason=str(exc))
        except VerificationInfrastructureError as exc:
            self._status("training", "infrastructure_failed", error_type=type(exc).__name__,
                         reason=exc.diagnostic, resumable=True)
            raise
        except Exception as exc:
            self._status("training", "failed", error_type=type(exc).__name__, reason=str(exc))
            raise
        return _read(self.root / "status.json")

    def evaluate_run(self):
        final = self.root / "final_library.json"
        if not final.exists():
            raise ValueError("Complete training before evaluating the frozen learned library")
        library = _read(final)
        assert_disjoint(self.train, load_problems(self.config.eval_data, seed=self.config.seed), library)
        fixed_hash = fingerprint(library)
        evaluation_dir = self.root / "evaluation"
        evaluation_dir.mkdir(exist_ok=True)
        contract = dict(library_sha256=fixed_hash, problem_ids=[p["id"] for p in self.evaluation],
                        proof_config=self.config.proof_config(False).to_dict(),
                        max_depth=self.config.inference_depth, max_completion_tokens=self.config.max_completion_tokens,
                        arm_policy="same inference configuration; empty vs frozen learned library; no updates")
        existing = _read(evaluation_dir / "manifest.json")
        if existing is not None and existing != contract:
            raise ValueError("Frozen evaluation library or matched budgets changed on resume")
        atomic_json(evaluation_dir / "manifest.json", contract)
        self._status("evaluation", "running")
        interrupted = None
        interruption_status = None
        try:
            jobs = []
            for index, problem in enumerate(self.evaluation):
                # Alternate the first arm to avoid giving every scarce final
                # global-budget request to the same arm. Only complete pairs are
                # used in the paired success-rate comparison.
                arms = ["baseline", "learned_library"] if index % 2 == 0 else ["learned_library", "baseline"]
                for arm in arms:
                    jobs.append((problem, library if arm == "learned_library" else {},
                                 evaluation_dir / arm / safe_problem_filename(problem["id"])))
            self.run_jobs(jobs, training=False)
        except BudgetExhausted as exc:
            interrupted = str(exc)
            interruption_status = "budget_exhausted"
        except RunStopped as exc:
            interrupted = str(exc)
            interruption_status = "stopped"
        except Exception as exc:
            self._status("evaluation", "failed", error_type=type(exc).__name__, reason=str(exc))
            raise
        rows = []
        for problem in self.evaluation:
            row = dict(id=problem["id"], benchmark=problem.get("benchmark", "unspecified"))
            for arm in ("baseline", "learned_library"):
                row[arm] = _read(evaluation_dir / arm / safe_problem_filename(problem["id"]) / "result.json")
            rows.append(row)
        paired = [row for row in rows if row["baseline"] is not None and row["learned_library"] is not None]
        summary = dict(total_problems=len(rows), paired_completed=len(paired),
            completed_by_arm={arm: sum(row[arm] is not None for row in rows) for arm in ("baseline", "learned_library")},
            paired_successes={arm: sum(row[arm]["success"] for row in paired) for arm in ("baseline", "learned_library")},
            paired_pass_rate={arm: sum(row[arm]["success"] for row in paired) / len(paired) if paired else None
                              for arm in ("baseline", "learned_library")},
            usage_by_arm={arm: self.ledger.summary(f"evaluation/{arm}/") for arm in ("baseline", "learned_library")},
            paired_statistics={arm: dict(
                total_llm_calls=sum(row[arm].get("statistics", {}).get("total_llm_calls", 0) for row in paired),
                total_tokens_used=sum(row[arm].get("statistics", {}).get("total_tokens_used", 0) for row in paired),
                problems_using_library=sum(bool(row[arm].get("library_lemmas_used")) for row in paired))
                for arm in ("baseline", "learned_library")},
            library_sha256=fixed_hash, budget=self.ledger.summary(), results=rows)
        summary.update(evaluation_metrics(rows))
        atomic_json(evaluation_dir / "summary.json", summary)
        self._status("evaluation", interruption_status or "evaluated",
                     reason=interrupted, paired_completed=len(paired))
        return summary


def evaluation_metrics(rows: list[dict]) -> dict:
    """Compare completed pairs; measure generated proof bodies explicitly.

    Count nonempty Lean proof-body lines and characters, excluding comments,
    theorem signatures, imports, and the injected library. Shared-success
    lengths compare the same targets. All-success lengths can reflect difficulty.
    """
    from statistics import mean, median

    arms = ("baseline", "learned_library")
    paired = [row for row in rows if all(row.get(arm) is not None for arm in arms)]
    common = [row for row in paired if all(row[arm]["success"] for arm in arms)]

    def outcomes(group):
        counts = dict(both=0, baseline_only=0, learned_library_only=0, neither=0)
        for row in group:
            baseline, learned = (bool(row[arm]["success"]) for arm in arms)
            key = "both" if baseline and learned else "baseline_only" if baseline else "learned_library_only" if learned else "neither"
            counts[key] += 1
        return counts

    def lengths(group, arm):
        samples = []
        for row in group:
            result = row[arm]
            if not result["success"]:
                continue
            source = result.get("proof") or ""
            try:
                _, declarations = extract_declarations(source)
                body = "\n".join(declaration.proof for declaration in declarations) if declarations else source
            except ValueError:
                body = source
            code = mask_comments_and_strings(body)
            lines = [line.strip() for line in code.splitlines() if line.strip()]
            samples.append(dict(nonempty_lines=len(lines), characters=sum(len(line) for line in lines)))
        return dict(proofs=len(samples), **{
            measure: dict(mean=mean(sample[measure] for sample in samples) if samples else None,
                          median=median(sample[measure] for sample in samples) if samples else None)
            for measure in ("nonempty_lines", "characters")})

    benchmarks = {}
    for benchmark in sorted({row.get("benchmark", "unspecified") for row in rows}):
        all_rows = [row for row in rows if row.get("benchmark", "unspecified") == benchmark]
        group = [row for row in paired if row.get("benchmark", "unspecified") == benchmark]
        benchmarks[benchmark] = dict(total_problems=len(all_rows), paired_completed=len(group),
            completed_by_arm={arm: sum(row.get(arm) is not None for row in all_rows) for arm in arms},
            paired_successes={arm: sum(row[arm]["success"] for row in group) for arm in arms},
            paired_pass_rate={arm: sum(row[arm]["success"] for row in group) / len(group) if group else None for arm in arms},
            paired_outcomes=outcomes(group))

    return dict(by_benchmark=benchmarks, paired_outcomes=outcomes(paired),
        proof_length_definition="Nonempty generated Lean proof-body lines/characters after comment masking; excludes signatures, imports and injected library",
        proof_lengths={selection: {arm: lengths(group, arm) for arm in arms}
                       for selection, group in [("all_paired_successes", paired), ("shared_successes", common)]},
        successful_library_reuse={arm: dict(
            problems=sum(row[arm]["success"] and bool(row[arm].get("library_lemmas_used")) for row in paired),
            distinct_lemmas=sorted({name for row in paired if row[arm]["success"]
                                   for name in row[arm].get("library_lemmas_used", [])})) for arm in arms})


@contextmanager
def cli_stop_signals(stop_event):
    """Install cooperative handlers only while an explicit CLI run is active."""
    received = []
    def request_stop(signum, frame):
        if not received:
            received.append(signum)
        stop_event.set()
    previous = {signum: signal.signal(signum, request_stop)
                for signum in (signal.SIGTERM, signal.SIGINT)}
    try:
        yield received
    finally:
        for signum, handler in previous.items():
            signal.signal(signum, handler)


def command_line(phase: str, argv=None):
    import argparse
    import yaml
    parser = argparse.ArgumentParser(description=f"Resumable DreamProver inequality {phase}")
    parser.add_argument("--config", type=Path, default=config_directory() / "pipeline/inequalities.yaml")
    for option in ("train_data", "eval_data", "output_dir", "lean_project", "verifier_url", "model",
                   "base_url", "embedding_model", "embedding_revision", "structural_parser_dir", "tbps_root"):
        parser.add_argument("--" + option.replace("_", "-"))
    for option in ("cycles", "train_limit", "eval_limit", "max_calls", "max_total_tokens", "max_completion_tokens",
                   "max_prover_calls_per_problem", "max_reasoner_calls_per_problem", "problem_concurrency", "wake_depth", "inference_depth",
                   "direct_attempts", "corrections", "wake_sketch_attempts", "inference_sketch_attempts", "seed"):
        parser.add_argument("--" + option.replace("_", "-"), type=int)
    parser.add_argument("--max-cost-usd", type=float)
    if phase == "training":
        parser.add_argument("--stop-after-cycle", type=int,
                            help="Finish this cycle without changing the configured experiment")
    args = vars(parser.parse_args(argv))
    stop_after_cycle = args.pop("stop_after_cycle", None)
    config_path = args.pop("config").expanduser().resolve()
    if not config_path.is_file():
        parser.error(f"Configuration not found: {config_path}")
    values = yaml.safe_load(config_path.read_text()) or {}
    values.update({key: value for key, value in args.items() if value is not None})
    config = PipelineConfig.from_mapping(values)
    if config.max_cost_usd is None:
        parser.error("Set --max-cost-usd (or max_cost_usd in the config) before a paid experiment")
    stop_event = threading.Event()
    with working_directory(repository_root(config_path)):
        with cli_stop_signals(stop_event) as received:
            try:
                with run_lock(config.output_dir):
                    pipeline = Pipeline(config, phase=phase, stop_event=stop_event)
                    result = pipeline.train_run(stop_after_cycle=stop_after_cycle) if phase == "training" else pipeline.evaluate_run()
            except BaseException as exc:
                # A concurrent transport failure still uses the requested signal
                # exit code once current work and its cleanup have drained.
                if received:
                    if isinstance(exc, Exception):
                        logger.exception("Run failed while draining after signal %s", received[0])
                    raise SystemExit(128 + received[0])
                raise
    print(json.dumps({key: value for key, value in result.items() if key != "results"}, indent=2))
    if received:
        raise SystemExit(128 + received[0])
    if _read(pipeline.root / "status.json")["status"] == "budget_exhausted":
        raise SystemExit(2)
