# langgraph-minnsdb

LangGraph long-term memory on [MinnsDB](https://github.com/Minns-ai/MinnsDB), the temporal memory database for agents.

LangGraph's built-in stores keep the latest value of each item. When a user moves from London to Berlin, London is gone, and the agent can't answer "where did they live in March?". MinnsDB keeps every version with the time it held, so you can.

Two pieces:

- **`MinnsDBStore`**: a drop-in LangGraph `BaseStore`. Same `put` / `get` / `search` / `list_namespaces` as any store, plus `history()` and `get_as_of()`.
- **`MinnsDBMemory`**: conversation memory in MinnsDB's temporal graph. Send it chat turns and MinnsDB extracts facts, superseding old ones rather than piling them up. Ask what's true about someone now or on any date. Comes as agent tools.

## Install

```bash
pip install langgraph-minnsdb
```

You need a MinnsDB server. To run one locally:

```bash
git clone https://github.com/Minns-ai/MinnsDB
cd MinnsDB && docker compose up --build
```

It listens on `http://localhost:3000`. Point the package at it with `MINNSDB_URL`, and `MINNSDB_API_KEY` if the server has auth turned on.

## The store

```python
from langgraph_minnsdb import MinnsDBStore

store = MinnsDBStore()                 # MINNSDB_URL, or http://localhost:3000
graph = builder.compile(store=store)   # use it like any LangGraph store
```

Inside a node it behaves like `InMemoryStore` or `PostgresStore`:

```python
store.put(("users", "priya"), "home", {"city": "London"})
store.put(("users", "priya"), "home", {"city": "Berlin"})

store.get(("users", "priya"), "home").value             # {'city': 'Berlin'}
store.search(("users",), filter={"city": "Berlin"})
store.list_namespaces(prefix=("users",))
```

What only this store can do:

```python
store.get_as_of(("users", "priya"), "home", some_time_before_the_move).value
# {'city': 'London'}

for v in store.history(("users", "priya"), "home"):
    print(v.value, v.valid_from, v.valid_until)
# {'city': 'London'} <when it was written> <when Berlin replaced it>
# {'city': 'Berlin'} <when it was written> None
```

`get_as_of` takes a timezone-aware `datetime`. Deleting an item closes its last version, and `history()` still shows it. The async methods (`aput`, `aget`, `asearch`, ...) work too. [examples/time_travel.py](examples/time_travel.py) runs the whole thing inside a LangGraph graph.

Items live in one MinnsDB table (`langgraph_store` by default, created on first use; pass `table=` to change it).

## Graph memory for agents

```python
from langgraph_minnsdb import MinnsDBMemory

memory = MinnsDBMemory(case_id="user-priya")

memory.remember([{"role": "user", "content": "I'm Priya. I live in London and work at Monzo."}])
# months later
memory.remember([{"role": "user", "content": "I've moved to Berlin and started at Stripe."}])

memory.facts_about("Priya")                     # what holds now
memory.facts_about("Priya", at="2026-03-01")    # what held on that date
memory.facts_about("Priya", history=True)       # every version, with valid_from and valid_until
memory.recall("Where does Priya work?")["answer"]
```

`remember` also takes LangChain messages, so you can pass `state["messages"]` after each turn. System prompts and tool messages are skipped, and the agent's own replies are ignored unless you set `include_assistant_facts=True`. It waits for extraction to finish; pass `wait=False` to get a background job instead and check it with `memory.job(job_id)`.

Give an agent the tools, for example with LangChain's `create_agent`:

```python
from langchain.agents import create_agent

agent = create_agent("anthropic:claude-sonnet-5-5", tools=memory.as_tools())
```

| Tool | What it does | Needs an LLM on the MinnsDB server |
| --- | --- | --- |
| `save_memory` | Extracts facts from what the user said and writes them to the graph | Yes |
| `search_memory` | Answers a plain-English question from the graph | Yes |
| `facts_about` | Lists what's true about a person or thing, now or on a date | No |

The server-side LLM is MinnsDB's own (`LLM_API_KEY` in its environment), not your agent's model.

For anything else, run MinnsQL directly:

```python
memory.query('MATCH (a)-[r]->(b) WHEN "2026-03-01" WHERE a.name = "Priya" RETURN type(r), b.name')
```

## Limits

- `search(query=...)` ranks by the share of query words found in each item. It does not use embeddings. For meaning-based recall, use `MinnsDBMemory`.
- `search` and `list_namespaces` read the whole table. Fine for thousands of items; past 100,000 rows MinnsDB stops paging and the store raises an error rather than return part of the table.
- TTL is not supported.
- `MinnsDBMemory` writes and reads under its `case_id`, so one user's facts never answer for another. Raw `query()` is not scoped.

## Run the tests

```bash
pip install -e ".[test]"
pytest                                          # unit tests
MINNSDB_URL=http://localhost:3000 pytest        # plus integration tests against your server
MINNSDB_URL=... MINNSDB_LLM=1 pytest            # plus fact extraction, if the server has an LLM key
```

## Licence

MIT. MinnsDB itself is AGPL-3.0; this package only talks to it over HTTP.
