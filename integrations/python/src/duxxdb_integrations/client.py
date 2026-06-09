"""Thin DuxxDB client over the RESP protocol (the Redis wire format).

DuxxDB speaks RESP, so we reuse the battle-tested ``redis`` client and issue
DuxxDB's agent commands directly — no native build, no extra dependency beyond
``redis-py``. Point it at a running ``duxx-server``.

Commands used:
    REMEMBER <key> <text...>   -> memory id (int)
    RECALL   <key> <query> <k> -> [[id, score, text], ...]
    SET/GET/DEL <key> ...      -> session/KV store
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

try:
    import redis
except ImportError as e:  # pragma: no cover
    raise ImportError("duxxdb-integrations requires redis-py: pip install 'redis>=5'") from e


@dataclass
class Recall:
    """One hybrid-recall hit."""

    id: int
    score: float
    text: str


class DuxxDBClient:
    """A minimal, synchronous DuxxDB client.

    Args:
        url: RESP URL, e.g. ``redis://localhost:6379``.
        token: optional auth token or workspace JWT (sent via ``AUTH``).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        *,
        token: Optional[str] = None,
        **redis_kwargs: Any,
    ) -> None:
        # decode_responses=True so RESP bulk strings come back as ``str``.
        self._r = redis.from_url(
            url,
            decode_responses=True,
            password=token,  # redis-py sends AUTH <password> on connect
            **redis_kwargs,
        )

    # ----- health --------------------------------------------------------
    def ping(self) -> bool:
        return self._r.execute_command("PING") == "PONG"

    # ----- memory (hybrid: vector + BM25 + RRF, embedded server-side) ----
    def remember(self, text: str, *, key: str = "default") -> int:
        """Store ``text`` under ``key`` and return its memory id."""
        return int(self._r.execute_command("REMEMBER", key, text))

    def recall(self, query: str, k: int = 10, *, key: str = "default") -> List[Recall]:
        """Top-``k`` hybrid-recall hits for ``query``."""
        rows = self._r.execute_command("RECALL", key, query, int(k)) or []
        out: List[Recall] = []
        for row in rows:
            # row = [id (int), score (str), text (str)]
            out.append(Recall(int(row[0]), float(row[1]), str(row[2])))
        return out

    def forget(self, memory_id: int) -> bool:
        """Best-effort delete (no-op if the server build lacks FORGET-by-id)."""
        try:
            return int(self._r.execute_command("FORGET", int(memory_id))) > 0
        except redis.ResponseError:
            return False

    # ----- session / KV --------------------------------------------------
    def kv_set(self, key: str, value: str) -> None:
        self._r.execute_command("SET", key, value)

    def kv_get(self, key: str) -> Optional[str]:
        v = self._r.execute_command("GET", key)
        return v if v is None else str(v)

    def kv_del(self, key: str) -> int:
        return int(self._r.execute_command("DEL", key))

    @property
    def raw(self) -> "redis.Redis":
        """Escape hatch: the underlying redis-py client for custom commands."""
        return self._r
