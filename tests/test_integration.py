"""Against a real MinnsDB. Set MINNSDB_URL (and MINNSDB_API_KEY if auth is on).

    MINNSDB_URL=http://localhost:3000 pytest -m integration
"""

import os
import time
import uuid
from datetime import datetime, timezone

import pytest
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

from langgraph_minnsdb import MinnsDBError, MinnsDBMemory, MinnsDBStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not os.environ.get("MINNSDB_URL"), reason="MINNSDB_URL not set"),
]


@pytest.fixture
def store():
    s = MinnsDBStore(table=f"lg_test_{uuid.uuid4().hex[:10]}")
    yield s
    try:
        s.client.request("DELETE", f"/api/tables/{s.table}")
    except MinnsDBError as e:
        if e.status_code != 404:  # never created: the test failed before first use
            raise


def test_put_get_update_delete(store):
    ns = ("users", "u1")
    assert store.get(ns, "prefs") is None

    store.put(ns, "prefs", {"city": "London", "tags": ["a", "b"], "nested": {"x": 1}})
    first = store.get(ns, "prefs")
    assert first.value == {"city": "London", "tags": ["a", "b"], "nested": {"x": 1}}
    assert first.namespace == ns and first.key == "prefs"

    time.sleep(0.01)
    store.put(ns, "prefs", {"city": "Berlin"})
    second = store.get(ns, "prefs")
    assert second.value == {"city": "Berlin"}
    assert second.created_at == first.created_at
    assert second.updated_at > first.updated_at

    store.delete(ns, "prefs")
    assert store.get(ns, "prefs") is None
    store.delete(ns, "prefs")  # deleting twice is fine


def test_history_and_as_of(store):
    ns = ("users", "u2")
    store.put(ns, "home", {"city": "London"})
    time.sleep(0.05)
    between = datetime.now(timezone.utc)
    time.sleep(0.05)
    store.put(ns, "home", {"city": "Berlin"})

    versions = store.history(ns, "home")
    assert [v.value["city"] for v in versions] == ["London", "Berlin"]
    assert versions[0].valid_until is not None and versions[1].current

    assert store.get_as_of(ns, "home", between).value == {"city": "London"}
    assert store.get_as_of(ns, "home", datetime.now(timezone.utc)).value == {"city": "Berlin"}
    assert store.get_as_of(ns, "home", datetime(2000, 1, 1, tzinfo=timezone.utc)) is None

    store.delete(ns, "home")
    assert store.history(ns, "home")[-1].valid_until is not None


def test_search_filter_query_and_namespaces(store):
    store.put(("users", "u1", "notes"), "n1", {"text": "Moved to Berlin for a job", "kind": "move", "n": 1})
    store.put(("users", "u1", "notes"), "n2", {"text": "Likes running", "kind": "hobby", "n": 2})
    store.put(("users", "u2", "notes"), "n3", {"text": "Lives in Paris", "kind": "move", "n": 3})

    assert {i.key for i in store.search(("users",))} == {"n1", "n2", "n3"}
    assert {i.key for i in store.search(("users", "u1"))} == {"n1", "n2"}
    assert {i.key for i in store.search(("users",), filter={"kind": "move"})} == {"n1", "n3"}
    assert {i.key for i in store.search(("users",), filter={"n": {"$gte": 2}})} == {"n2", "n3"}

    ranked = store.search(("users",), query="berlin job")
    assert ranked[0].key == "n1" and ranked[0].score == 1.0
    assert len(store.search(("users",), limit=2)) == 2

    assert store.list_namespaces(prefix=("users",)) == [("users", "u1", "notes"), ("users", "u2", "notes")]
    assert store.list_namespaces(max_depth=2) == [("users", "u1"), ("users", "u2")]
    assert store.list_namespaces(suffix=("notes",), limit=1) == [("users", "u1", "notes")]


def test_survives_a_new_store_instance(store):
    store.put(("a",), "k", {"v": 1})
    other = MinnsDBStore(table=store.table)
    other.put(("a",), "k", {"v": 2})  # update through a fresh row-id cache
    assert store.get(("a",), "k").value == {"v": 2}
    assert [v.value for v in store.history(("a",), "k")] == [{"v": 1}, {"v": 2}]


@pytest.mark.asyncio
async def test_async_api(store):
    await store.aput(("x",), "k", {"v": 1})
    assert (await store.aget(("x",), "k")).value == {"v": 1}
    assert [i.key for i in await store.asearch(("x",))] == ["k"]


def test_as_a_langgraph_store(store):
    def remember_city(state: MessagesState, *, store: BaseStore):
        last = state["messages"][-1].content
        store.put(("users", "u9"), "last_said", {"text": last})
        return {"messages": [("ai", f"Stored: {last}")]}

    graph = StateGraph(MessagesState).add_node("remember", remember_city).add_edge(START, "remember")
    app = graph.compile(store=store)
    app.invoke({"messages": [("user", "I live in Lisbon")]})
    assert store.get(("users", "u9"), "last_said").value == {"text": "I live in Lisbon"}


def _seed_person(memory, name):
    """Two homes for one person, the first closed when the second began (no LLM needed)."""
    moved = 1780272000000000000  # 2026-06-01
    memory.client.request(
        "POST",
        "/api/graph/import",
        json={
            "group_id": memory.case_id,
            "nodes": [{"name": name, "type": "concept"}, {"name": "London", "type": "concept"}, {"name": "Berlin", "type": "concept"}],
            "edges": [
                {"source": name, "target": "London", "label": "location:lives_in", "valid_from": 1704067200000000000, "valid_until": moved},
                {"source": name, "target": "Berlin", "label": "location:lives_in", "valid_from": moved},
            ],
        },
    )


def test_facts_about_now_then_and_history():
    memory = MinnsDBMemory(case_id=f"facts-{uuid.uuid4().hex[:8]}")
    name = f"Priya{uuid.uuid4().hex[:6]}"
    _seed_person(memory, name)

    # Another case never sees these facts.
    assert MinnsDBMemory(case_id=f"other-{uuid.uuid4().hex[:8]}").facts_about(name) == []

    now = memory.facts_about(name)
    assert [(f["relation"], f["value"]) for f in now] == [("location:lives_in", "berlin")]
    assert now[0]["valid_until"] is None

    then = memory.facts_about(name, at="2025-03-01")
    assert [f["value"] for f in then] == ["london"]
    assert [f["value"] for f in memory.facts_about(name, at=datetime(2025, 3, 1, tzinfo=timezone.utc))] == ["london"]

    every = memory.facts_about(name, history=True)
    assert [f["value"] for f in every] == ["london", "berlin"]
    assert every[0]["valid_until"] == every[1]["valid_from"]

    tool = {t.name: t for t in memory.as_tools()}["facts_about"]
    assert "berlin" in tool.invoke({"name": name})
    assert "london" in tool.invoke({"name": name, "date": "2025-03-01"})
    assert tool.invoke({"name": "nobody-" + name}).startswith("Nothing known")

    with pytest.raises(ValueError):
        memory.facts_about('x" OR 1=1')
    with pytest.raises(ValueError):
        memory.facts_about(name, at="March 2025")


def test_case_id_is_validated():
    with pytest.raises(ValueError):
        MinnsDBMemory(case_id='x" OR 1=1')


def test_more_rows_than_one_server_page(store):
    for i in range(1005):
        store.put(("bulk", str(i % 3)), f"k{i}", {"i": i})
    assert len(store.search(("bulk",), limit=2000)) == 1005
    assert len(store.list_namespaces(prefix=("bulk",))) == 3
    fresh = MinnsDBStore(table=store.table)
    for i in (0, 500, 1004):
        fresh.put(("bulk", str(i % 3)), f"k{i}", {"i": -i})
        assert store.get(("bulk", str(i % 3)), f"k{i}").value == {"i": -i}
    fresh.delete(("bulk", "1"), "k1003")
    assert store.get(("bulk", "1"), "k1003") is None


def test_row_replaced_by_another_process(store):
    store.put(("p",), "k", {"v": 1})
    other = MinnsDBStore(table=store.table)
    other.delete(("p",), "k")
    other.put(("p",), "k", {"v": 2})  # a new server row; store's cache still has the old one
    store.put(("p",), "k", {"v": 3})
    assert other.get(("p",), "k").value == {"v": 3}
    store.delete(("p",), "k")
    assert other.get(("p",), "k") is None


def test_first_write_race_becomes_an_update(store, monkeypatch):
    other = MinnsDBStore(table=store.table)
    store.put(("r",), "k", {"v": 1})
    real = other._current
    calls = {"n": 0}

    def stale_once(row_id):
        calls["n"] += 1
        return None if calls["n"] == 1 else real(row_id)  # saw no row, then lost the race

    monkeypatch.setattr(other, "_current", stale_once)
    other.put(("r",), "k", {"v": 2})
    assert store.get(("r",), "k").value == {"v": 2}


def test_as_of_an_items_own_timestamp(store):
    for i in range(20):
        store.put(("t",), "k", {"v": i})
        item = store.get(("t",), "k")
        assert store.get_as_of(("t",), "k", item.updated_at).value == {"v": i}


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("MINNSDB_LLM"), reason="set MINNSDB_LLM=1 when the server has an LLM configured")
def test_memory_supersedes_and_recalls():
    memory = MinnsDBMemory(case_id=f"lg-test-{uuid.uuid4().hex[:8]}")
    first = memory.remember([{"role": "user", "content": "Hi, I'm Priya. I live in London and work at Monzo."}])
    assert first["compaction"]["llm_success"]
    second = memory.remember([{"role": "user", "content": "Big news: I've moved to Berlin and started at Stripe."}])
    assert second["compaction"]["facts_extracted"] >= 1

    answer = memory.recall("Where does Priya live?")["answer"]
    assert "Berlin" in answer

    history = [f["value"] for f in memory.facts_about("Priya", history=True)]
    assert "london" in history and "berlin" in history
    assert [f["value"] for f in memory.facts_about("Priya") if f["relation"].startswith("location")] == ["berlin"]
    tools = {t.name: t for t in memory.as_tools()}
    assert "Berlin" in tools["search_memory"].invoke({"question": "Where does Priya live?"})
