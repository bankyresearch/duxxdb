"""LangChain adapters for DuxxDB.

- :class:`DuxxDBVectorStore` — a LangChain ``VectorStore`` backed by DuxxDB's
  hybrid retrieval (vector + BM25 fused with RRF). DuxxDB embeds text
  server-side, so a LangChain ``Embeddings`` object is **optional**.
- :class:`DuxxDBChatMessageHistory` — conversation history persisted in
  DuxxDB's session/KV store.

    pip install duxxdb-integrations[langchain]
"""

from __future__ import annotations

import json
from typing import Any, Iterable, List, Optional, Tuple

from .client import DuxxDBClient

try:
    from langchain_core.chat_history import BaseChatMessageHistory
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.messages import (
        BaseMessage,
        messages_from_dict,
        messages_to_dict,
    )
    from langchain_core.vectorstores import VectorStore
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LangChain support requires langchain-core: "
        "pip install 'duxxdb-integrations[langchain]'"
    ) from e


class DuxxDBVectorStore(VectorStore):
    """A LangChain ``VectorStore`` over DuxxDB hybrid retrieval.

    Args:
        client: a :class:`DuxxDBClient`.
        namespace: partition key for stored memories (one logical collection).
        embedding: optional; DuxxDB embeds server-side, so this is only used if
            a caller explicitly wants client-side embeddings (not required).
    """

    def __init__(
        self,
        client: DuxxDBClient,
        *,
        namespace: str = "default",
        embedding: Optional[Embeddings] = None,
    ) -> None:
        self._c = client
        self._ns = namespace
        self._embedding = embedding

    @property
    def embeddings(self) -> Optional[Embeddings]:
        return self._embedding

    def _meta_key(self, mid: int) -> str:
        return f"_meta:{self._ns}:{mid}"

    def add_texts(
        self,
        texts: Iterable[str],
        metadatas: Optional[List[dict]] = None,
        **kwargs: Any,
    ) -> List[str]:
        texts = list(texts)
        if not metadatas or len(metadatas) != len(texts):
            metadatas = [{} for _ in texts]
        ids: List[str] = []
        for text, meta in zip(texts, metadatas):
            mid = self._c.remember(text, key=self._ns)
            if meta:
                self._c.kv_set(self._meta_key(mid), json.dumps(meta))
            ids.append(str(mid))
        return ids

    def similarity_search(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Document]:
        return [doc for doc, _ in self.similarity_search_with_score(query, k, **kwargs)]

    def similarity_search_with_score(
        self, query: str, k: int = 4, **kwargs: Any
    ) -> List[Tuple[Document, float]]:
        out: List[Tuple[Document, float]] = []
        for hit in self._c.recall(query, k, key=self._ns):
            raw = self._c.kv_get(self._meta_key(hit.id))
            meta = json.loads(raw) if raw else {}
            meta["id"] = hit.id
            out.append((Document(page_content=hit.text, metadata=meta), hit.score))
        return out

    @classmethod
    def from_texts(
        cls,
        texts: List[str],
        embedding: Optional[Embeddings] = None,
        metadatas: Optional[List[dict]] = None,
        *,
        client: Optional[DuxxDBClient] = None,
        namespace: str = "default",
        **kwargs: Any,
    ) -> "DuxxDBVectorStore":
        if client is None:
            raise ValueError("pass client=DuxxDBClient(...) to from_texts")
        store = cls(client, namespace=namespace, embedding=embedding)
        store.add_texts(texts, metadatas)
        return store


class DuxxDBChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat history persisted in DuxxDB's session/KV store."""

    def __init__(self, session_id: str, client: DuxxDBClient) -> None:
        self._c = client
        self._key = f"_chat:{session_id}"

    @property
    def messages(self) -> List[BaseMessage]:
        raw = self._c.kv_get(self._key)
        if not raw:
            return []
        return messages_from_dict(json.loads(raw))

    def add_message(self, message: BaseMessage) -> None:
        serialized = messages_to_dict(self.messages)
        serialized.append(messages_to_dict([message])[0])
        self._c.kv_set(self._key, json.dumps(serialized))

    def clear(self) -> None:
        self._c.kv_del(self._key)
