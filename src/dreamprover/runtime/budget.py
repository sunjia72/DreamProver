"""Durable request accounting and replay for resumable, bounded experiments.

The ledger reserves the full request ceiling before sending a network request.
Unsettled requests keep their reservation, including after a process crash.
Successful requests settle against API usage; reasoning tokens are included in
completion_tokens. Replay is scoped to the same stage and repeated-request index.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import threading
import math
from pathlib import Path
from types import SimpleNamespace

from dreamprover.runtime.tracking import MaxLLMCallsExceeded


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


class BudgetExhausted(MaxLLMCallsExceeded):
    """A configured run budget would be exceeded by the next request."""


class RunStopped(BaseException):
    """An operator stop must bypass proof-correction exception handlers."""


class BudgetLedger:
    def __init__(self, directory, *, max_cost_usd: float | None, max_calls: int | None = None,
                 max_tokens: int | None = None, input_per_million: float = 0.10,
                 output_per_million: float = 0.50, cached_input_per_million: float = 0.01):
        self.directory = Path(directory)
        self.path = self.directory / "budget.json"
        self.cache = self.directory / "responses"
        self.limits = dict(max_cost_usd=max_cost_usd, max_calls=max_calls, max_tokens=max_tokens)
        for name, limit in self.limits.items():
            if limit is not None and (not math.isfinite(limit) or limit <= 0):
                raise ValueError(f"{name} must be positive and finite")
            if limit is not None and name != "max_cost_usd" and (not isinstance(limit, int) or isinstance(limit, bool)):
                raise ValueError(f"{name} must be a positive integer")
        self.prices = dict(input_per_million=input_per_million, output_per_million=output_per_million,
                           cached_input_per_million=cached_input_per_million)
        if any(not math.isfinite(value) or value < 0 for value in self.prices.values()):
            raise ValueError("Token prices must be finite and nonnegative")
        self.lock = threading.RLock()
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {"schema_version": 1, "requests": []}
        if self.data.get("prices", self.prices) != self.prices:
            raise ValueError("Cannot resume a budget ledger with changed token prices")
        self.data.update(prices=self.prices, limits=self.limits)
        atomic_json(self.path, self.data)

    def cost(self, usage) -> float:
        cached = min(usage.get("cached_input_tokens", 0), usage["input_tokens"])
        return ((usage["input_tokens"] - cached) * self.prices["input_per_million"]
                + cached * self.prices["cached_input_per_million"]
                + usage["output_tokens"] * self.prices["output_per_million"]) / 1_000_000

    def summary(self, scope_prefix: str | None = None) -> dict:
        with self.lock:
            actual = {name: 0 for name in ("input_tokens", "output_tokens", "cached_input_tokens", "total_tokens")}
            cost = reserved_cost = 0.0
            reserved_tokens = 0
            requests = [request for request in self.data["requests"]
                        if scope_prefix is None or request["scope"].startswith(scope_prefix)]
            for request in requests:
                if request["state"] == "complete":
                    usage = request["usage"]
                    for name in actual:
                        actual[name] += usage.get(name, 0)
                    cost += request["cost_usd"]
                else:
                    reserved_cost += request["reserved_cost_usd"]
                    reserved_tokens += request["reserved_tokens"]
            return dict(actual, actual_cost_usd=cost, reserved_cost_usd=reserved_cost,
                        charged_cost_usd=cost + reserved_cost,
                        reserved_tokens=reserved_tokens, charged_tokens=actual["total_tokens"] + reserved_tokens,
                        api_calls=len(requests), completed_calls=sum(r["state"] == "complete" for r in requests),
                        uncertain_calls=sum(r["state"] != "complete" for r in requests))

    def reserve(self, request: dict, scope: str, *, stop_event=None) -> int:
        # A token is at least one UTF-8 byte. JSON bytes plus generous message
        # framing cover tokenization without a model-specific tokenizer guess.
        input_ceiling = len(json.dumps(request["messages"], ensure_ascii=False).encode("utf-8")) + 4096
        output_ceiling = request["max_tokens"]
        reserved_cost = (input_ceiling * max(self.prices["input_per_million"], self.prices["cached_input_per_million"])
                         + output_ceiling * self.prices["output_per_million"]) / 1_000_000
        reserved_tokens = input_ceiling + output_ceiling
        with self.lock:
            if stop_event is not None and stop_event.is_set():
                raise RunStopped("Stop requested; unfinished work remains resumable")
            current = self.summary()
            additions = dict(max_cost_usd=current["charged_cost_usd"] + reserved_cost,
                             max_calls=current["api_calls"] + 1,
                             max_tokens=current["charged_tokens"] + reserved_tokens)
            for name, value in additions.items():
                if self.limits[name] is not None and value > self.limits[name] + 1e-12:
                    raise BudgetExhausted(f"Next {scope} request would exceed {name}={self.limits[name]} (ceiling {value})")
            index = len(self.data["requests"])
            self.data["requests"].append(dict(scope=scope, request_fingerprint=fingerprint(request),
                                              state="reserved", reserved_cost_usd=reserved_cost,
                                              reserved_tokens=reserved_tokens))
            atomic_json(self.path, self.data)
            return index

    def settle(self, index: int, usage: dict) -> None:
        with self.lock:
            request = self.data["requests"][index]
            request.update(state="complete", usage=usage, cost_usd=self.cost(usage))
            atomic_json(self.path, self.data)

    def failed(self, index: int, error: BaseException) -> None:
        with self.lock:
            # A disconnected/failed request may have consumed tokens remotely.
            # Retain its ceiling; never silently refund an unknown usage amount.
            self.data["requests"][index].update(state="uncertain", error_type=type(error).__name__)
            atomic_json(self.path, self.data)


def response_usage(response) -> dict:
    usage = getattr(response, "usage", None)
    if usage is None:
        raise RuntimeError("The model endpoint returned no usage; cannot settle a bounded experiment")
    get = lambda obj, name, default=0: obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    input_tokens = int(get(usage, "prompt_tokens"))
    output_tokens = int(get(usage, "completion_tokens"))
    details = get(usage, "prompt_tokens_details", None)
    cached = int(get(details, "cached_tokens")) if details else 0
    total = int(get(usage, "total_tokens", input_tokens + output_tokens))
    if min(input_tokens, output_tokens, cached, total) < 0 or cached > input_tokens or total < input_tokens + output_tokens:
        raise RuntimeError("Invalid token usage returned by model endpoint")
    return dict(input_tokens=input_tokens, output_tokens=output_tokens, cached_input_tokens=cached,
                total_tokens=total)


def response_diagnostics(response, text: str) -> dict:
    get = lambda obj, name, default=None: obj.get(name, default) if isinstance(obj, dict) else getattr(obj, name, default)
    choices = get(response, "choices", []) or []
    usage = get(response, "usage")
    details = get(usage, "completion_tokens_details")
    return dict(finish_reasons=[get(choice, "finish_reason") for choice in choices],
                reasoning_tokens=get(details, "reasoning_tokens"),
                visible_text_characters=len(text), visible_text_utf8_bytes=len(text.encode("utf-8")))


class BudgetedClient:
    """AsyncLLMClient adapter sharing one durable ledger across all stages."""
    def __init__(self, client, ledger: BudgetLedger, scope: str, *, stop_event=None):
        self.client, self.ledger, self.scope = client, ledger, scope
        self.stop_event = stop_event
        self.occurrences = {}
        self._last_response = None
        self.lock = asyncio.Lock()

    async def chat_completion(self, messages, max_tokens=4096, temperature=0.6, **kwargs):
        stopping = self.stop_event is not None and self.stop_event.is_set()
        # Metadata does not change a remote request and is not a cache key.
        kwargs = {key: value for key, value in kwargs.items() if key not in {"context", "prompt_type"} and value is not None}
        request = dict(model=self.client.model_name, base_url=self.client.base_url,
                       messages=messages, max_tokens=max_tokens, temperature=temperature, kwargs=kwargs)
        digest = fingerprint(request)
        async with self.lock:
            ordinal = self.occurrences.get(digest, 0)
            self.occurrences[digest] = ordinal + 1
            key = fingerprint(dict(scope=self.scope, request=digest, occurrence=ordinal))
            path = self.ledger.cache / (key + ".json")
            if path.exists():
                cached = json.loads(path.read_text())
                usage = cached["usage"]
                index = cached["request_index"]
                if self.ledger.data["requests"][index]["state"] != "complete":
                    self.ledger.settle(index, usage)
                if stopping or self.stop_event is not None and self.stop_event.is_set():
                    raise RunStopped("Stop requested; unfinished work remains resumable")
                self._last_response = SimpleNamespace(usage=SimpleNamespace(
                    prompt_tokens=usage["input_tokens"], completion_tokens=usage["output_tokens"],
                    total_tokens=usage["total_tokens"],
                    prompt_tokens_details=SimpleNamespace(cached_tokens=usage["cached_input_tokens"])))
                return cached["text"]
            if stopping:
                raise RunStopped("Stop requested; unfinished work remains resumable")
            # Already reserved requests may dispatch and settle after a signal;
            # the shared event prevents subsequent requests from being reserved.
            index = self.ledger.reserve(request, self.scope, stop_event=self.stop_event)
            try:
                text = await self.client.chat_completion(messages, max_tokens=max_tokens,
                                                         temperature=temperature, **kwargs)
                self._last_response = self.client.get_last_response_raw()
                usage = response_usage(self._last_response)
                # Cache before settling: a crash after the response is safely
                # replayable even if its reservation remains conservatively held.
                atomic_json(path, dict(text=text, usage=usage, request_index=index,
                                       diagnostics=response_diagnostics(self._last_response, text)))
                self.ledger.settle(index, usage)
                return text
            except BaseException as exc:
                self.ledger.failed(index, exc)
                raise

    async def simple_chat(self, prompt, system_message=None, **kwargs):
        messages = [{"role": "system", "content": system_message}] if system_message else []
        messages.append({"role": "user", "content": prompt})
        return await self.chat_completion(messages, **kwargs)

    def get_last_response_raw(self):
        return self._last_response

    async def close(self):
        await self.client.close()

    async def health_check(self):
        # Health checking must not send an unaccounted model request.
        return True
