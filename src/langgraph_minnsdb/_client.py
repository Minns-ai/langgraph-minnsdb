"""A small HTTP client for the MinnsDB REST API.

Paths are the server's own (``/api/...``), so the same client works against a
local MinnsDB (``http://localhost:3000``) and the hosted one
(``https://minns.ai``).
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_URL = "http://localhost:3000"


class MinnsDBError(Exception):
    """MinnsDB answered with an error status."""

    def __init__(self, status_code: int, detail: str, method: str, path: str) -> None:
        super().__init__(f"MinnsDB {method} {path} failed ({status_code}): {detail}")
        self.status_code = status_code
        self.detail = detail


class MinnsDBClient:
    """Synchronous MinnsDB client.

    Args:
        url: Base URL of the server. Defaults to ``MINNSDB_URL`` or
            ``http://localhost:3000``.
        api_key: A ``mndb_...`` key. Defaults to ``MINNSDB_API_KEY``. Leave
            unset for a server started with ``MINNS_AUTH_DISABLED=true``.
        timeout: Seconds per request. Conversation ingestion runs the LLM
            pipeline before it answers, so keep this generous.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        *,
        timeout: float = 120.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.url = (url or os.environ.get("MINNSDB_URL") or DEFAULT_URL).rstrip("/")
        key = api_key if api_key is not None else os.environ.get("MINNSDB_API_KEY")
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        self._http = http or httpx.Client(base_url=self.url, headers=headers, timeout=timeout)

    def request(
        self,
        method: str,
        path: str,
        *,
        json: Any = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        res = self._http.request(method, path, json=json, params=params)
        if res.status_code >= 400:
            try:
                body = res.json()
                detail = body.get("details") or body.get("error") or res.text
            except ValueError:
                detail = res.text
            raise MinnsDBError(res.status_code, str(detail), method, path)
        if not res.content:
            return None
        return res.json()

    def query(self, minnsql: str) -> dict[str, Any]:
        """Run a MinnsQL statement (``POST /api/query``)."""
        return self.request("POST", "/api/query", json={"query": minnsql})

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> MinnsDBClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
