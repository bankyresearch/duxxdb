"""DuxxDB - the database built for AI agents.

A hybrid (vector + BM25 + structured) embedded database with
agent-native primitives: ``MemoryStore``, ``ToolCache``, ``SessionStore``.

Quickstart
----------

.. code-block:: python

    import duxxdb

    store = duxxdb.MemoryStore(dim=4)
    store.remember(key="alice", text="hello world", embedding=[1.0, 0.0, 0.0, 0.0])
    hits = store.recall(key="alice", query="hello",
                        embedding=[1.0, 0.0, 0.0, 0.0], k=5)
    for hit in hits:
        print(f"{hit.score:.4f}  {hit.text}")
"""

from ._native import (  # type: ignore[import]
    MemoryStore,
    MemoryHit,
    ToolCache,
    ToolCacheHit,
    SessionStore,
    __version__,
)

__all__ = [
    "MemoryStore",
    "MemoryHit",
    "ToolCache",
    "ToolCacheHit",
    "SessionStore",
    "__version__",
]
