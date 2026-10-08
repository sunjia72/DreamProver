"""Preserve verifier HTTP diagnostics through Kimina's safe response adapter."""
from __future__ import annotations

import json

from kimina_client.async_client import AsyncKiminaClient
from kimina_client.sync_client import KiminaClient


def verifier_failure_message(error: Exception) -> str:
    """Recover HTTP response bodies hidden by the SDK's Tenacity wrapper."""
    pending, visited = [error], set()
    while pending:
        current = pending.pop()
        if id(current) in visited:
            continue
        visited.add(id(current))
        response = getattr(current, "response", None)
        if response is not None and getattr(response, "status_code", 0) >= 400:
            try:
                payload = response.json()
                detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
                body = detail if isinstance(detail, str) else json.dumps(detail, ensure_ascii=False)
            except ValueError:
                body = response.text
            if body:
                return f"{error}\nVerifier HTTP {response.status_code} diagnostic:\n{body[:16384]}"
        pending.extend(value for value in (getattr(current, "__cause__", None),
                                           getattr(current, "__context__", None)) if value is not None)
        attempt = getattr(current, "last_attempt", None)
        if attempt is not None:
            nested = attempt.exception()
            if nested is not None:
                pending.append(nested)
    return str(error)


class DiagnosticAsyncKiminaClient(AsyncKiminaClient):
    async def _query(self, *args, **kwargs):
        try:
            return await super()._query(*args, **kwargs)
        except Exception as error:
            message = verifier_failure_message(error)
            if message != str(error):
                raise RuntimeError(message) from error
            raise


class DiagnosticKiminaClient(KiminaClient):
    def _query(self, *args, **kwargs):
        try:
            return super()._query(*args, **kwargs)
        except Exception as error:
            message = verifier_failure_message(error)
            if message != str(error):
                raise RuntimeError(message) from error
            raise
