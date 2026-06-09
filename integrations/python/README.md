# DuxxDB Integrations

Connect DuxxDB to the agent ecosystem: a thin **RESP client** plus **LangChain**
and **LlamaIndex** adapters. Pure Python — no native build — talks to a running
`duxx-server` over the Redis wire protocol.

```bash
pip install duxxdb-integrations              # client only (redis-py)
pip install 'duxxdb-integrations[langchain]'
pip install 'duxxdb-integrations[llama-index]'
```

## Quickstart

```python
from duxxdb_integrations import DuxxDBClient

db = DuxxDBClient("redis://localhost:6379", token="optional-token-or-jwt")
db.remember("the user prefers dark mode", key="user-42")
hits = db.recall("ui preferences", k=5, key="user-42")   # vector + BM25 + RRF
```

### LangChain

```python
from duxxdb_integrations import DuxxDBClient
from duxxdb_integrations.langchain import DuxxDBVectorStore, DuxxDBChatMessageHistory

db = DuxxDBClient()
store = DuxxDBVectorStore(db, namespace="kb")
store.add_texts(["Paris is the capital of France."], metadatas=[{"src": "wiki"}])
docs = store.similarity_search("French capital", k=3)

history = DuxxDBChatMessageHistory("conversation-1", db)   # for RunnableWithMessageHistory
```

### LlamaIndex

```python
from llama_index.core import VectorStoreIndex, StorageContext
from duxxdb_integrations import DuxxDBClient
from duxxdb_integrations.llama_index import DuxxDBVectorStore

vs = DuxxDBVectorStore(DuxxDBClient(), namespace="docs")
index = VectorStoreIndex.from_documents(
    docs, storage_context=StorageContext.from_defaults(vector_store=vs)
)
# Query in text mode (DuxxDB embeds server-side):
index.as_query_engine(vector_store_query_mode="text_search").query("...")
```

## Bonus: drop-in Redis replacement

Because DuxxDB speaks RESP, **anything that uses Redis can point at DuxxDB by
changing a URL** — you inherit the Redis ecosystem for free:

```python
# LangChain's Redis-backed chat history works unchanged against duxx-server:
from langchain_community.chat_message_histories import RedisChatMessageHistory
RedisChatMessageHistory("session-1", url="redis://localhost:6379")

# redis-py / redisvl / go-redis / node-redis all connect the same way.
```

Use the DuxxDB-native adapters above when you want its **agent** features
(hybrid recall, importance decay, semantic tool cache); use the Redis-compatible
path when you just want a faster drop-in for cache/session.

## Status & testing

Verified against real `langchain-core` and `llama-index-core` — the adapter
suite (`tests/`) stubs the client (no server/redis needed) and exercises the
LangChain `VectorStore` + `ChatMessageHistory` and the LlamaIndex
`VectorStore`:

```bash
pip install -e '.[langchain,llama-index,dev]'
pytest        # 3 passed
```

> Pin your `langchain-core` / `llama-index-core` versions — those interfaces
> evolve. For integration testing against a live server, run `duxx-server` and
> point `DuxxDBClient` at it.
