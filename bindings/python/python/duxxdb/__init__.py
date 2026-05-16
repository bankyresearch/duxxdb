"""DuxxDB - the database built for AI agents.

A hybrid (vector + BM25 + structured) embedded database with
agent-native primitives: ``MemoryStore``, ``ToolCache``, ``SessionStore``.

Quickstart (embedded)
---------------------

.. code-block:: python

    import duxxdb

    store = duxxdb.MemoryStore(dim=4)
    store.remember(key="alice", text="hello world", embedding=[1.0, 0.0, 0.0, 0.0])
    hits = store.recall(key="alice", query="hello",
                        embedding=[1.0, 0.0, 0.0, 0.0], k=5)
    for hit in hits:
        print(f"{hit.score:.4f}  {hit.text}")

Quickstart (talking to a running ``duxx-server``)
-------------------------------------------------

For Phase 7 primitives (traces, prompts, datasets, evals, replay,
cost) reach for :class:`duxxdb.server.ServerClient`. It wraps
``redis-py`` and gives you typed namespaces, so you write
``client.evals.start(...)`` instead of ``client.execute_command(
"EVAL.START", ...)``:

.. code-block:: python

    # pip install 'duxxdb[server]'
    from duxxdb.server import ServerClient

    client = ServerClient(url="redis://:<token>@localhost:6379")
    run_id = client.evals.start(
        dataset_name="refund_examples",
        dataset_version=3,
        model="gpt-4o-mini",
        scorer="llm_judge_v1",
    )
"""

from typing import TYPE_CHECKING

# The native PyO3 module hosts the embedded primitives. Importing it
# at package load time is the fast path for users who do
# ``import duxxdb``. We tolerate the import failing (e.g. when only
# the pure-Python server facade is being used in a dev tree without a
# built wheel) by falling back to a deferred error: anyone who
# actually touches MemoryStore / ToolCache / SessionStore gets a
# helpful message; anyone importing only ``duxxdb.server`` proceeds.
try:
    from ._native import (  # type: ignore[import]
        MemoryStore,
        MemoryHit,
        ToolCache,
        ToolCacheHit,
        SessionStore,
        __version__,
    )

    _NATIVE_AVAILABLE = True
except ImportError as _native_err:  # pragma: no cover
    _NATIVE_AVAILABLE = False
    _NATIVE_ERR = _native_err
    __version__ = "0.0.0+unbuilt"  # placeholder until the wheel is built

    def _missing_native(name: str):
        def _raise(*_a, **_kw):
            raise ImportError(
                f"duxxdb.{name} requires the native extension, which was not "
                f"built in this environment. Original error: {_NATIVE_ERR}"
            )

        return _raise

    MemoryStore = _missing_native("MemoryStore")  # type: ignore[assignment]
    MemoryHit = _missing_native("MemoryHit")  # type: ignore[assignment]
    ToolCache = _missing_native("ToolCache")  # type: ignore[assignment]
    ToolCacheHit = _missing_native("ToolCacheHit")  # type: ignore[assignment]
    SessionStore = _missing_native("SessionStore")  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover
    from .server import ServerClient

__all__ = [
    "MemoryStore",
    "MemoryHit",
    "ToolCache",
    "ToolCacheHit",
    "SessionStore",
    "ServerClient",
    "__version__",
]


def __getattr__(name: str):
    """Lazy import of :class:`ServerClient`.

    Keeps ``import duxxdb`` cheap when the user only needs the
    embedded primitives. ``ServerClient`` pulls in ``redis-py``, which
    we don't want to make a hard dependency of the base wheel.
    """
    if name == "ServerClient":
        from .server import ServerClient  # noqa: PLC0415
        return ServerClient
    raise AttributeError(f"module 'duxxdb' has no attribute {name!r}")
