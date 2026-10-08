import warnings

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.store.base import MatchCondition

from langgraph_minnsdb import MinnsDBMemory, NoFactsExtracted
from langgraph_minnsdb.memory import _turns
from langgraph_minnsdb.store import _compare, _matches, _row_id_of, _score


def test_row_id_is_stable_positive_and_distinct():
    a = _row_id_of(("users", "u1"), "prefs")
    assert a == _row_id_of(("users", "u1"), "prefs")
    assert 0 <= a < 2**63
    assert a != _row_id_of(("users", "u1"), "prefs2")
    # Joining parts must not collide: ("a", "b.c") vs ("a.b", "c").
    assert _row_id_of(("a", "b.c"), "k") != _row_id_of(("a.b", "c"), "k")


def test_compare_operators():
    assert _compare("x", "x")
    assert not _compare("x", "y")
    assert _compare(5, {"$gt": 3, "$lte": 5})
    assert not _compare(None, {"$gt": 3})
    assert _compare("a", {"$ne": "b"})
    assert _compare({"city": "Berlin"}, {"city": "Berlin"})
    assert not _compare("text", {"$gt": 3})


def test_namespace_match_conditions():
    ns = ("users", "u1", "prefs")
    assert _matches(MatchCondition("prefix", ("users",)), ns)
    assert _matches(MatchCondition("prefix", ("users", "*")), ns)
    assert _matches(MatchCondition("suffix", ("prefs",)), ns)
    assert not _matches(MatchCondition("prefix", ("teams",)), ns)
    assert not _matches(MatchCondition("prefix", ("users", "u1", "prefs", "x")), ns)


def test_score_counts_query_words():
    assert _score("berlin job", {"text": "Moved to Berlin for a job"}) == 1.0
    assert _score("berlin london", {"text": "Moved to Berlin"}) == 0.5
    assert _score("", {"text": "x"}) == 0.0


def test_turns_keep_what_people_said():
    turns = _turns(
        [
            SystemMessage("You are helpful"),
            HumanMessage("I moved to Berlin"),
            AIMessage("Noted"),
            ToolMessage("ok", tool_call_id="1"),
            {"role": "user", "content": "  "},
            "plain text counts as the user",
        ]
    )
    assert turns == [
        {"role": "user", "content": "I moved to Berlin"},
        {"role": "assistant", "content": "Noted"},
        {"role": "user", "content": "plain text counts as the user"},
    ]


class _FakeClient:
    def __init__(self, facts):
        self.facts = facts

    def request(self, method, path, **kwargs):
        return {"case_id": "u", "compaction": {"facts_extracted": self.facts, "llm_success": True}}


def test_remember_warns_when_no_facts_extracted():
    memory = MinnsDBMemory("u", client=_FakeClient(0))
    with pytest.warns(NoFactsExtracted, match="LLM_API_KEY"):
        memory.remember("I live in London")


def test_remember_quiet_when_facts_extracted_or_nothing_to_extract():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        MinnsDBMemory("u", client=_FakeClient(2)).remember("I live in London")
        MinnsDBMemory("u", client=_FakeClient(0)).remember([AIMessage("Noted")])
        MinnsDBMemory("u", client=_FakeClient(0)).remember("I live in London", wait=False)


def test_save_memory_tool_says_when_nothing_saved():
    save = {t.name: t for t in MinnsDBMemory("u", client=_FakeClient(0)).as_tools()}["save_memory"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert save.invoke({"fact": "hello"}).startswith("Nothing saved")
