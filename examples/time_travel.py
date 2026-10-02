"""LangGraph memory that keeps every version, on MinnsDB.

Start MinnsDB first (see the README), then:

    MINNSDB_URL=http://localhost:3000 python examples/time_travel.py
"""

import time
from datetime import datetime, timezone

from langgraph.graph import START, MessagesState, StateGraph
from langgraph.store.base import BaseStore

from langgraph_minnsdb import MinnsDBStore


def save_home(state: MessagesState, *, store: BaseStore):
    city = state["messages"][-1].content
    store.put(("users", "priya"), "home", {"city": city})
    return {"messages": [("ai", f"Noted: you live in {city}.")]}


store = MinnsDBStore(table="example_time_travel")
app = StateGraph(MessagesState).add_node("save_home", save_home).add_edge(START, "save_home").compile(store=store)

app.invoke({"messages": [("user", "London")]})
time.sleep(0.1)
before_move = datetime.now(timezone.utc)
time.sleep(0.1)
app.invoke({"messages": [("user", "Berlin")]})

print("Now:        ", store.get(("users", "priya"), "home").value)
print("Before move:", store.get_as_of(("users", "priya"), "home", before_move).value)
print("History:")
for v in store.history(("users", "priya"), "home"):
    until = f"{v.valid_until:%H:%M:%S.%f}" if v.valid_until else "now"
    print(f"  {v.value['city']:<8} from {v.valid_from:%H:%M:%S.%f} until {until}")
