"""LlamaIndex adapter for DuxxDB.

:class:`DuxxDBVectorStore` implements LlamaIndex's ``BasePydanticVectorStore``
over DuxxDB hybrid retrieval. Because DuxxDB embeds text **server-side**, this
store operates in *text-query* mode — it uses ``VectorStoreQuery.query_str``
(set ``vector_store_query_mode`` accordingly on your retriever, or pass a text
query). Node metadata is round-tripped through DuxxDB's KV store.

    pip install duxxdb-integrations[llama-index]
"""

from __future__ import annotations

import json
from typing import Any, List, Optional

from .client import DuxxDBClient

try:
    from llama_index.core.bridge.pydantic import PrivateAttr
    from llama_index.core.schema import BaseNode, MetadataMode, TextNode
    from llama_index.core.vector_stores.types import (
        BasePydanticVectorStore,
        VectorStoreQuery,
        VectorStoreQueryResult,
    )
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "LlamaIndex support requires llama-index-core: "
        "pip install 'duxxdb-integrations[llama-index]'"
    ) from e


class DuxxDBVectorStore(BasePydanticVectorStore):
    """A LlamaIndex vector store backed by DuxxDB hybrid retrieval."""

    stores_text: bool = True
    namespace: str = "default"

    _client: DuxxDBClient = PrivateAttr()

    def __init__(
        self, client: DuxxDBClient, namespace: str = "default", **kwargs: Any
    ) -> None:
        super().__init__(namespace=namespace, **kwargs)
        self._client = client

    @property
    def client(self) -> DuxxDBClient:
        return self._client

    def _meta_key(self, mid: int) -> str:
        return f"_li:{self.namespace}:{mid}"

    def add(self, nodes: List[BaseNode], **kwargs: Any) -> List[str]:
        ids: List[str] = []
        for node in nodes:
            text = node.get_content(metadata_mode=MetadataMode.NONE)
            mid = self._client.remember(text, key=self.namespace)
            self._client.kv_set(
                self._meta_key(mid),
                json.dumps(
                    {
                        "node_id": node.node_id,
                        "metadata": node.metadata,
                        "ref_doc_id": node.ref_doc_id,
                    }
                ),
            )
            ids.append(node.node_id)
        return ids

    def delete(self, ref_doc_id: str, **kwargs: Any) -> None:
        # DuxxDB exposes delete-by-memory-id, not by ref_doc_id; a production
        # build would maintain a reverse index. No-op here (documented).
        return None

    def query(
        self, query: VectorStoreQuery, **kwargs: Any
    ) -> VectorStoreQueryResult:
        query_text = query.query_str
        if not query_text:
            raise ValueError(
                "DuxxDBVectorStore embeds server-side and needs a text query; "
                "set query.query_str (text-query mode)."
            )
        top_k = query.similarity_top_k or 10
        hits = self._client.recall(query_text, top_k, key=self.namespace)

        nodes: List[TextNode] = []
        ids: List[str] = []
        similarities: List[float] = []
        for hit in hits:
            raw = self._client.kv_get(self._meta_key(hit.id))
            meta = json.loads(raw) if raw else {}
            node = TextNode(
                text=hit.text,
                id_=meta.get("node_id"),
                metadata=meta.get("metadata", {}),
            )
            nodes.append(node)
            ids.append(node.node_id)
            similarities.append(hit.score)

        return VectorStoreQueryResult(
            nodes=nodes, ids=ids, similarities=similarities
        )
