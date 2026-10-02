"""MinnsDBMemory: conversation memory in MinnsDB's temporal graph.

``remember`` sends conversation turns to MinnsDB, which extracts facts with
its LLM pipeline and writes them as graph edges with validity times, so a new
fact ("I moved to Berlin") supersedes an old one ("I live in London") instead
of sitting beside it. ``recall`` asks the graph a question in plain English.

The server needs an LLM configured for ingestion (``LLM_API_KEY``).
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import BaseMessage
from langchain_core.tools import BaseTool, tool

from ._client import MinnsDBClient
from .store import _when

_ROLES = {"human": "user", "user": "user", "ai": "assistant", "assistant": "assistant"}
# Names go into a MinnsQL string literal; keep to characters that need no escaping.
_SAFE_NAME = re.compile(r"^[\w .,'&@:/+-]{1,200}$")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
_CASE_ID = re.compile(r"^[\w.@:-]{1,128}$")


def _date_text(at: str | datetime) -> str:
    if isinstance(at, datetime):
        if at.tzinfo is None:
            raise ValueError("pass a timezone-aware datetime")
        return at.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if not _DATE.match(at):
        raise ValueError(f"dates look like 2025-03-01, not {at!r}")
    return at


def _turns(messages: Iterable[BaseMessage | Mapping[str, Any] | str]) -> list[dict[str, str]]:
    turns = []
    for m in messages:
        if isinstance(m, str):
            role, content = "user", m
        elif isinstance(m, BaseMessage):
            text = m.text  # a property in langchain-core 1.x, a method before
            role, content = _ROLES.get(m.type, ""), text if isinstance(text, str) else text()
        else:
            role, content = _ROLES.get(str(m.get("role", "")), ""), str(m.get("content", ""))
        # Tool calls and system prompts are not things a person said.
        if role and content.strip():
            turns.append({"role": role, "content": content})
    return turns


class MinnsDBMemory:
    """Conversation memory for one case: a user, an account, a project.

    Everything is written and read under the case, so one user's facts are
    never used to answer for another. (Raw :meth:`query` is not scoped.)

    Args:
        case_id: The case, for example ``"user-123"``. Letters, digits and
            ``_ - . @ :`` only.
        client: An existing :class:`MinnsDBClient`, or pass ``url`` and
            ``api_key``.
        include_assistant_facts: Also extract facts from the agent's own
            replies. Off by default, so the agent cannot remember its own
            guesses as facts.
    """

    def __init__(
        self,
        case_id: str,
        client: MinnsDBClient | None = None,
        *,
        url: str | None = None,
        api_key: str | None = None,
        include_assistant_facts: bool = False,
    ) -> None:
        if not _CASE_ID.match(case_id):
            raise ValueError(f"unsupported characters in case_id {case_id!r}")
        self.case_id = case_id
        self.client = client or MinnsDBClient(url, api_key)
        self.include_assistant_facts = include_assistant_facts

    def remember(
        self,
        messages: Iterable[BaseMessage | Mapping[str, Any] | str] | str,
        *,
        session_id: str | None = None,
        topic: str | None = None,
        wait: bool = True,
    ) -> dict[str, Any]:
        """Extract facts from conversation turns and write them to the graph.

        With ``wait=True`` (the default) this returns once MinnsDB has run its
        extraction, with the report (``compaction.facts_extracted`` and so on).
        With ``wait=False`` it returns a background job straight away
        (``job_id``); check it with :meth:`job`. The server skips an exact
        repeat (same case, session and messages), so pass a fixed
        ``session_id`` if you may send the same turns twice.
        """
        turns = _turns([messages] if isinstance(messages, str) else messages)
        if not turns:
            return {"case_id": self.case_id, "messages_processed": 0}
        session: dict[str, Any] = {"session_id": session_id or uuid.uuid4().hex, "messages": turns}
        if topic:
            session["topic"] = topic
        return self.client.request(
            "POST",
            "/api/conversations/ingest",
            params={"wait": "true"} if wait else None,
            json={
                "case_id": self.case_id,
                "group_id": self.case_id,
                "sessions": [session],
                "include_assistant_facts": self.include_assistant_facts,
            },
        )

    def job(self, job_id: str) -> dict[str, Any]:
        """A background ingestion, e.g. ``{"state": "done", "response": {...}}``.

        ``state`` is queued, processing, done or failed (with ``error``)."""
        return self.client.request("GET", f"/api/jobs/{job_id}")["state"]

    def recall(self, question: str, *, limit: int = 10, session_id: str | None = None) -> dict[str, Any]:
        """Ask the graph a question in plain English (``POST /api/nlq``).

        The answer is in ``["answer"]``; ``["explanation"]`` says how it was found.
        """
        body: dict[str, Any] = {"question": question, "limit": limit, "group_id": self.case_id}
        if session_id:
            body["session_id"] = session_id
        return self.client.request("POST", "/api/nlq", json=body)

    def facts_about(
        self,
        name: str,
        *,
        at: str | datetime | None = None,
        history: bool = False,
    ) -> list[dict[str, Any]]:
        """What the graph holds about ``name``: now, on a date, or every version.

        Reads the graph directly with MinnsQL, so it needs no LLM. Each fact is
        ``{"relation", "value", "valid_from", "valid_until"}`` (datetimes in UTC;
        ``valid_until`` is None while the fact still holds).

        Args:
            at: A date ("2025-03-01") or a timezone-aware datetime. Returns
                the facts that held then.
            history: Return every version, including superseded ones.
        """
        if not _SAFE_NAME.match(name):
            raise ValueError(f"unsupported characters in name {name!r}")
        if history and at is not None:
            raise ValueError("pass either at or history, not both")
        when = ""
        if history:
            when = "WHEN ALL "
        elif at is not None:
            when = f'WHEN "{_date_text(at)}" '
        res = self.client.query(
            f'MATCH (a)-[r]->(b) {when}WHERE a.name = "{name}" AND r.group_id = "{self.case_id}" '
            "RETURN type(r), b.name, valid_from(r), valid_until(r)"
        )
        facts = [
            {"relation": rel, "value": value, "valid_from": _when(start), "valid_until": _when(end)}
            for rel, value, start, end in res.get("rows") or []
        ]
        facts.sort(key=lambda f: (f["relation"] or "", f["valid_from"] or _EPOCH))
        return facts

    def query(self, minnsql: str) -> dict[str, Any]:
        """Run MinnsQL directly, for example a time-travel query::

            memory.query('MATCH (a)-[r]->(b) WHEN ALL RETURN a.name, type(r), b.name, valid_from(r), valid_until(r)')
        """
        return self.client.query(minnsql)

    def as_tools(self) -> list[BaseTool]:
        """``save_memory``, ``search_memory`` and ``facts_about`` tools for an agent."""
        memory = self

        @tool("facts_about")
        def facts_about_tool(name: str, date: str | None = None) -> str:
            """List what is known about a person, company or thing. Leave date empty for what is true now,
            or pass a date like 2025-03-01 for what was true then."""
            facts = memory.facts_about(name, at=date or None)
            if not facts:
                return f"Nothing known about {name}" + (f" on {date}." if date else ".")
            return "\n".join(
                f"{f['relation']}: {f['value']}"
                + (f" (since {f['valid_from']:%Y-%m-%d})" if f["valid_from"] else "")
                for f in facts
            )

        @tool
        def save_memory(fact: str) -> str:
            """Save something the user told you that is worth remembering later, such as where they live,
            where they work, their preferences or a change to any of these. Write it as the user said it."""
            report = memory.remember([{"role": "user", "content": fact}])
            facts = (report.get("compaction") or {}).get("facts_extracted", 0)
            return f"Saved. {facts} fact(s) extracted."

        @tool
        def search_memory(question: str) -> str:
            """Look up what you know about the user. Ask a plain-English question,
            for example 'Where does the user live?'."""
            res = memory.recall(question)
            return str(res.get("answer") or "Nothing found.")

        return [save_memory, search_memory, facts_about_tool]
