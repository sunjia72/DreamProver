"""Small synchronous client used by sleep-stage scripts.

The endpoint uses the OpenAI chat-completions schema. No model client or
network request is constructed when this module is imported.
"""
from __future__ import annotations

import os
from typing import Any

import requests

from dreamprover.paths import config_directory


def load_default_cfg():
    from hydra import compose, initialize_config_dir

    config_dir = config_directory()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name="run")


def get_informal_llm_config(cfg=None):
    if cfg is None:
        cfg = load_default_cfg()
    informal = cfg.experiment.informal_llm
    return informal.base_url, informal.model_name, cfg


def query_llm(
    prompt: str,
    max_tokens: int = 10000,
    temperature: float = 0.2,
    api_url: str = "",
    model_name: str = "",
    *,
    timeout: float = 240,
    api_key: str | None = None,
) -> dict[str, Any]:
    """Return response content; ordinary replies need no reasoning field."""
    if not api_url or not model_name:
        raise ValueError("Both api_url and model_name are required")
    endpoint = api_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint += "/chat/completions"
    headers = {"Content-Type": "application/json"}
    key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    short_name = model_name.lower().rsplit("/", 1)[-1]
    reasoning_model = short_name.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_completion_tokens" if reasoning_model else "max_tokens": max_tokens,
    }
    if not reasoning_model:
        payload["temperature"] = temperature
    response = requests.post(
        endpoint,
        headers=headers,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    try:
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("LLM response has no choices[0].message.content") from exc
    if not isinstance(content, str) or not content.strip():
        raise ValueError("LLM returned empty or non-text content")
    return {"status_code": response.status_code, "text": content}


def run_informal_llm(
    prompt: str,
    cfg=None,
    *,
    temperature: float = 0.2,
    max_tokens: int = 10000,
    timeout: float = 240,
) -> str:
    base_url, model_name, _ = get_informal_llm_config(cfg)
    return query_llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        api_url=base_url,
        model_name=model_name,
        timeout=timeout,
    )["text"]
