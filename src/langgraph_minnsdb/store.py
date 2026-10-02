"""MinnsDBStore: LangGraph long-term memory in a MinnsDB temporal table.

Every item is one row. Writing an item again closes the old row version and
opens a new one, and deleting an item closes it, so nothing is lost: the
store can tell you what an item holds now, what it held at any earlier moment
(``get_as_of``) and every value it has had (``history``).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from langgraph.store.base import (
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    MatchCondition,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
)

from ._client import MinnsDBClient, MinnsDBError

_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,62}$")
_COLUMNS = [
    {"name": "id", "col_type": "Int64", "nullable": False},
    {"name": "ns", "col_type": "String", "nullable": False},
    {"name": "item_key", "col_type": "String", "nullable": False},
    {"name": "item_value", "col_type": "Json", "nullable": False},
    {"name": "created_ns", "col_type": "Int64", "nullable": False},
]
_WORD = re.compile(r"\w+")
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)
# The server returns at most 10,000 rows per call and will not page past 100,000.
_PAGE = 10_000
_SCAN_CAP = 100_000


@dataclass(frozen=True)
class ItemVersion:
    """One version of an item: its value and when that value held."""

    value: dict[str, Any]
    valid_from: datetime
    valid_until: datetime | None

    @property
    def current(self) -> bool:
        return self.valid_until is None


def _row_id_of(namespace: tuple[str, ...], key: str) -> int:
    """Stable 63-bit primary key for (namespace, key)."""
    raw = json.dumps([list(namespace), key], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big") & ((1 << 63) - 1)


def _ns_text(namespace: tuple[str, ...]) -> str:
    return json.dumps(list(namespace), separators=(",", ":"))


def _when(ns: int | None) -> datetime | None:
    # Integer maths: a float loses the microseconds at today's epoch values.
    return None if ns is None else _EPOCH + timedelta(microseconds=ns // 1000)


def _now_ns() -> int:
    return time.time_ns()


def _ns_of(at: datetime) -> int:
    """The last nanosecond of ``at``'s microsecond.

    Python datetimes stop at microseconds and MinnsDB counts nanoseconds, so
    a time read back from an item (``updated_at``) is up to 999ns before the
    real write. Rounding up makes ``get_as_of(item.updated_at)`` see that write.
    """
    if at.tzinfo is None:
        raise ValueError("pass a timezone-aware datetime")
    return (at - _EPOCH) // timedelta(microseconds=1) * 1000 + 999


def _compare(value: Any, condition: Any) -> bool:
    """LangGraph filter semantics: equality, or a dict of $eq/$ne/$gt/$gte/$lt/$lte."""
    if isinstance(condition, dict) and condition and all(k.startswith("$") for k in condition):
        for op, target in condition.items():
            try:
                ok = {
                    "$eq": lambda: value == target,
                    "$ne": lambda: value != target,
                    "$gt": lambda: value is not None and value > target,
                    "$gte": lambda: value is not None and value >= target,
                    "$lt": lambda: value is not None and value < target,
                    "$lte": lambda: value is not None and value <= target,
                }[op]()
            except KeyError:
                raise ValueError(f"unsupported filter operator {op}") from None
            except TypeError:
                ok = False
            if not ok:
                return False
        return True
    if isinstance(condition, dict) and isinstance(value, dict):
        return all(_compare(value.get(k), v) for k, v in condition.items())
    return value == condition


def _matches(condition: MatchCondition, namespace: tuple[str, ...]) -> bool:
    path = tuple(condition.path)
    if len(namespace) < len(path):
        return False
    part = namespace[: len(path)] if condition.match_type == "prefix" else namespace[-len(path) :]
    return all(p == "*" or p == n for p, n in zip(path, part, strict=True))


def _score(query: str, value: dict[str, Any]) -> float:
    words = set(_WORD.findall(query.lower()))
    if not words:
        return 0.0
    text = set(_WORD.findall(json.dumps(value, ensure_ascii=False).lower()))
    return len(words & text) / len(words)


class MinnsDBStore(BaseStore):
    """A LangGraph ``BaseStore`` backed by a MinnsDB temporal table.

    Use it anywhere LangGraph takes a store::

        store = MinnsDBStore(url="http://localhost:3000")
        graph = builder.compile(store=store)

    Args:
        client: An existing :class:`MinnsDBClient`. If omitted, one is made
            from ``url`` and ``api_key`` (or ``MINNSDB_URL`` and
            ``MINNSDB_API_KEY``).
        table: Table to keep items in. Created on first use.

    ``search(query=...)`` ranks items by the share of query words found in
    their value. It does not embed anything; for meaning-based recall over
    conversations, use :class:`~langgraph_minnsdb.MinnsDBMemory`.
    """

    supports_ttl = False

    def __init__(
        self,
        client: MinnsDBClient | None = None,
        *,
        url: str | None = None,
        api_key: str | None = None,
        table: str = "langgraph_store",
    ) -> None:
        if not _TABLE_NAME.match(table):
            raise ValueError(f"invalid table name {table!r}")
        self.client = client or MinnsDBClient(url, api_key)
        self.table = table
        self._ready = False
        self._lock = threading.RLock()
        self._row_ids: dict[int, int] = {}

    # Setup

    def setup(self) -> None:
        """Create the table if it is missing. Called on first use."""
        with self._lock:
            if self._ready:
                return
            try:
                self.client.request("GET", f"/api/tables/{self.table}/schema")
            except MinnsDBError as e:
                if e.status_code != 404:
                    raise
                try:
                    self.client.request(
                        "POST",
                        "/api/tables",
                        json={"name": self.table, "columns": _COLUMNS, "constraints": [{"PrimaryKey": ["id"]}]},
                    )
                except MinnsDBError as create_err:
                    # Another process created it between our check and create.
                    if "exist" not in create_err.detail.lower():
                        raise
            self._ready = True

    # Row access

    def _scan(self, *, all_versions: bool = False, as_of: int | None = None) -> list[dict[str, Any]]:
        """Every row, paged. Raises rather than return a partial table."""
        params: dict[str, Any] = {"when": "all" if all_versions else "active", "limit": _PAGE}
        if as_of is not None:
            params["as_of"] = as_of
        for _ in range(2):  # pages have no fixed order; a write between pages can shift them, so try twice
            rows: dict[int, dict[str, Any]] = {}
            total = 0
            offset = 0
            while True:
                res = self.client.request("GET", f"/api/tables/{self.table}/rows", params={**params, "offset": offset})
                total = res["count"]
                if total > _SCAN_CAP:
                    raise MinnsDBError(
                        413, f"table {self.table} has {total} rows; scans stop at {_SCAN_CAP}", "GET", self.table
                    )
                for r in res["rows"]:
                    rows[r["version_id"]] = r
                offset += len(res["rows"])
                if not res["rows"] or offset >= total:
                    break
            if len(rows) == total:
                break
        else:
            raise MinnsDBError(409, f"table {self.table} changed during the scan; try again", "GET", self.table)
        found = list(rows.values())
        if not all_versions and as_of is None:
            self._row_ids = {r["values"][0]: r["row_id"] for r in found}
        return found

    def _item(self, row: dict[str, Any]) -> Item:
        _, ns, key, value, created = row["values"]
        return Item(
            value=value,
            key=key,
            namespace=tuple(json.loads(ns)),
            created_at=_when(created),
            updated_at=_when(row["valid_from"]),
        )

    def _current(self, row_id: int) -> tuple[str, str, dict[str, Any], int, int] | None:
        t = self.table
        res = self.client.query(
            f"FROM {t} WHERE {t}.id = {row_id} "
            f"RETURN {t}.ns, {t}.item_key, {t}.item_value, {t}.created_ns, {t}.valid_from"
        )
        rows = res.get("rows") or []
        return tuple(rows[0]) if rows else None  # type: ignore[return-value]

    def _storage_row_id(self, row_id: int, *, fresh: bool = False) -> int | None:
        if fresh or row_id not in self._row_ids:
            self._scan()
        return self._row_ids.get(row_id)

    def _write_row(self, method: str, row_id: int, body: dict[str, Any]) -> None:
        """PUT or DELETE the server row for ``row_id``. Another process may have
        replaced it since we cached its server id, so on 404/409 look it up again
        and retry once."""
        for fresh in (False, True):
            storage_id = self._storage_row_id(row_id, fresh=fresh)
            if storage_id is None:
                if method == "DELETE":
                    return  # already gone
                raise MinnsDBError(404, "item was deleted during the update", method, self.table)
            try:
                self.client.request(method, f"/api/tables/{self.table}/rows/{storage_id}", json=body)
                return
            except MinnsDBError as e:
                if fresh or e.status_code not in (404, 409):
                    raise
                self._row_ids.pop(row_id, None)

    # Operations

    def _get(self, op: GetOp) -> Item | None:
        found = self._current(_row_id_of(op.namespace, op.key))
        if found is None:
            return None
        ns, key, value, created, valid_from = found
        return Item(
            value=value,
            key=key,
            namespace=tuple(json.loads(ns)),
            created_at=_when(created),
            updated_at=_when(valid_from),
        )

    def _put(self, op: PutOp) -> None:
        row_id = _row_id_of(op.namespace, op.key)
        existing = self._current(row_id)
        if op.value is None:
            if existing is not None:
                self._write_row("DELETE", row_id, {})
                self._row_ids.pop(row_id, None)
            return
        if existing is None:
            values = [row_id, _ns_text(op.namespace), op.key, op.value, _now_ns()]
            try:
                res = self.client.request("POST", f"/api/tables/{self.table}/rows", json={"values": values})
                self._row_ids[row_id] = res["row_id"]
                return
            except MinnsDBError as e:
                # Another process inserted the same item first: update it instead.
                if e.status_code != 409 or "unique" not in e.detail.lower():
                    raise
                existing = self._current(row_id)
                if existing is None:
                    raise
        values = [row_id, _ns_text(op.namespace), op.key, op.value, existing[3]]
        self._write_row("PUT", row_id, {"values": values})

    def _search(self, op: SearchOp, rows: list[dict[str, Any]]) -> list[SearchItem]:
        prefix = tuple(op.namespace_prefix)
        found: list[SearchItem] = []
        for row in rows:
            item = self._item(row)
            if item.namespace[: len(prefix)] != prefix:
                continue
            if op.filter and not all(_compare(item.value.get(k), v) for k, v in op.filter.items()):
                continue
            score = _score(op.query, item.value) if op.query else None
            found.append(SearchItem(item.namespace, item.key, item.value, item.created_at, item.updated_at, score))
        if op.query:
            found.sort(key=lambda i: (i.score or 0.0, i.updated_at), reverse=True)
        else:
            found.sort(key=lambda i: i.updated_at, reverse=True)
        return found[op.offset : op.offset + op.limit]

    def _list_namespaces(self, op: ListNamespacesOp, rows: list[dict[str, Any]]) -> list[tuple[str, ...]]:
        namespaces = {tuple(json.loads(r["values"][1])) for r in rows}
        if op.match_conditions:
            namespaces = {ns for ns in namespaces if all(_matches(c, ns) for c in op.match_conditions)}
        if op.max_depth is not None:
            namespaces = {ns[: op.max_depth] for ns in namespaces}
        return sorted(namespaces)[op.offset : op.offset + op.limit]

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        self.setup()
        results: list[Result] = []
        with self._lock:
            active: list[dict[str, Any]] | None = None
            for op in ops:
                if isinstance(op, GetOp):
                    results.append(self._get(op))
                elif isinstance(op, PutOp):
                    self._put(op)
                    active = None
                    results.append(None)
                elif isinstance(op, SearchOp):
                    if active is None:
                        active = self._scan()
                    results.append(self._search(op, active))
                elif isinstance(op, ListNamespacesOp):
                    if active is None:
                        active = self._scan()
                    results.append(self._list_namespaces(op, active))
                else:
                    raise ValueError(f"unknown operation {type(op).__name__}")
        return results

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, list(ops))

    # Time

    def history(self, namespace: tuple[str, ...], key: str) -> list[ItemVersion]:
        """Every value the item has had, oldest first."""
        self.setup()
        row_id = _row_id_of(tuple(namespace), key)
        rows = [r for r in self._scan(all_versions=True) if r["values"][0] == row_id]
        rows.sort(key=lambda r: r["valid_from"])
        return [ItemVersion(r["values"][3], _when(r["valid_from"]), _when(r["valid_until"])) for r in rows]

    def get_as_of(self, namespace: tuple[str, ...], key: str, at: datetime) -> Item | None:
        """The item as it stood at ``at`` (timezone-aware), or None if it did not exist then."""
        self.setup()
        row_id = _row_id_of(tuple(namespace), key)
        for row in self._scan(as_of=_ns_of(at)):
            if row["values"][0] == row_id:
                return self._item(row)
        return None
