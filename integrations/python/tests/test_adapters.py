"""Adapter unit tests that stub the DuxxDB client — no server or redis needed.

Run with: pip install -e '.[langchain,dev]' && pytest
"""

from __future__ import annotations

from duxxdb_integrations.client import Recall


class FakeClient:
    """Duck-typed stand-in for DuxxDBClient (same method surface)."""

    def __init__(self) -> None:
        self.memories: dict[int, str] = {}
        self.kv: dict[str, str] = {}
        self._next = 0

    def remember(self, text: str, *, key: str = "default") -> int:
        self._next += 1
        self.memories[self._next] = text
        return self._next

    def recall(self, query: str, k: int = 10, *, key: str = "default"):
        # Naive substring relevance, newest first — enough for adapter tests.
        hits = [
            Recall(mid, 1.0, text)
            for mid, text in sorted(self.memories.items(), reverse=True)
            if any(w in text.lower() for w in query.lower().split())
        ]
        return hits[:k]

    def kv_set(self, key: str, value: str) -> None:
        self.kv[key] = value

    def kv_get(self, key: str):
        return self.kv.get(key)

    def kv_del(self, key: str) -> int:
        return 1 if self.kv.pop(key, None) is not None else 0


def test_langchain_vectorstore_roundtrip() -> None:
    from duxxdb_integrations.langchain import DuxxDBVectorStore

    c = FakeClient()
    store = DuxxDBVectorStore(c, namespace="kb")
    ids = store.add_texts(
        ["Paris is the capital of France", "the sky is blue"],
        metadatas=[{"src": "wiki"}, {"src": "obs"}],
    )
    assert len(ids) == 2

    docs = store.similarity_search("capital France", k=3)
    assert any("Paris" in d.page_content for d in docs)
    # Metadata round-trips and id is attached.
    top = docs[0]
    assert "id" in top.metadata
    assert top.metadata.get("src") == "wiki"


def test_llama_index_vectorstore_roundtrip() -> None:
    import pytest

    pytest.importorskip("llama_index.core")
    from llama_index.core.schema import TextNode
    from llama_index.core.vector_stores.types import VectorStoreQuery

    from duxxdb_integrations.llama_index import DuxxDBVectorStore

    c = FakeClient()
    vs = DuxxDBVectorStore(c, namespace="docs")
    ids = vs.add(
        [TextNode(text="Paris is the capital of France", id_="n1", metadata={"src": "wiki"})]
    )
    assert ids == ["n1"]

    res = vs.query(VectorStoreQuery(query_str="capital France", similarity_top_k=3))
    assert res.nodes is not None and len(res.nodes) >= 1
    assert "Paris" in res.nodes[0].get_content()
    assert res.nodes[0].metadata.get("src") == "wiki"


def test_langchain_chat_history_roundtrip() -> None:
    from langchain_core.messages import AIMessage, HumanMessage

    from duxxdb_integrations.langchain import DuxxDBChatMessageHistory

    c = FakeClient()
    hist = DuxxDBChatMessageHistory("s1", c)
    assert hist.messages == []
    hist.add_message(HumanMessage(content="hi"))
    hist.add_message(AIMessage(content="hello"))
    msgs = hist.messages
    assert [m.content for m in msgs] == ["hi", "hello"]
    hist.clear()
    assert hist.messages == []
