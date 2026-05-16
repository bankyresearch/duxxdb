"""Typed Python facade over the DuxxDB RESP wire protocol.

Use this module when you're running ``duxx-server`` (the daemon) and
want a friendlier Python surface than raw
``redis_client.execute_command("EVAL.START", ...)`` calls.

Quickstart
----------

.. code-block:: python

    from duxxdb.server import ServerClient

    client = ServerClient(url="redis://:<token>@localhost:6379")
    client.ping()

    # ---- Phase 7.2: prompts --------------------------------------
    v1 = client.prompts.put(
        name="refund_classifier",
        content="Classify the message as REFUND or NOT_REFUND...",
    )
    p = client.prompts.get("refund_classifier", v1)

    # ---- Phase 7.4: evals ----------------------------------------
    run_id = client.evals.start(
        dataset_name="refund_examples",
        dataset_version=3,
        prompt_name="refund_classifier",
        prompt_version=v1,
        model="gpt-4o-mini",
        scorer="llm_judge_v1",
    )
    client.evals.score(run_id, row_id="row-0", score=0.92, output_text="REFUND")
    summary = client.evals.complete(run_id)
    print(summary.mean, summary.p99)

The facade is intentionally a thin layer: every method maps 1:1 to
one RESP command, return types are plain ``dataclasses`` decoded from
the server's JSON responses, and the underlying ``redis.Redis``
client is exposed as ``client.raw`` for anything the facade doesn't
cover yet.

It is **not** required for using DuxxDB from Python. The embedded
``MemoryStore`` / ``ToolCache`` / ``SessionStore`` in
:mod:`duxxdb` still work standalone. Reach for ``ServerClient``
specifically when you're talking to a running ``duxx-server``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

if TYPE_CHECKING:  # pragma: no cover
    import redis as _redis

__all__ = [
    "ServerClient",
    # Dataclasses surfaced as return types.
    "Span",
    "Prompt",
    "PromptHit",
    "Dataset",
    "DatasetRow",
    "DatasetHit",
    "EvalRun",
    "EvalScore",
    "EvalSummary",
    "EvalComparison",
    "FailureCluster",
    "ReplaySession",
    "ReplayInvocation",
    "ReplayRun",
    "ReplayDiff",
    "CostEntry",
    "CostBucket",
    "CostBudget",
    "CostAlert",
    "CostCluster",
]

# ----------------------------------------------------------------- helpers


_NONE_SENTINEL = "-"


def _arg(value: Any) -> str:
    """Serialize a single Python value into a RESP argument string.

    ``None`` → ``"-"`` (the server's sentinel for "absent")
    ``dict`` / ``list`` → ``json.dumps``
    bool / int / float / str → ``str``
    """
    if value is None:
        return _NONE_SENTINEL
    if isinstance(value, (dict, list)):
        return json.dumps(value, separators=(",", ":"))
    if isinstance(value, bool):
        # bools first, because bool is a subclass of int.
        return "1" if value else "0"
    return str(value)


def _decode(blob: Any) -> Any:
    """Decode one server reply payload.

    The server returns ``BulkString`` (bytes/str) holding JSON for
    everything except plain status, integer, and null replies. We
    accept bytes or str interchangeably (redis-py's
    ``decode_responses`` may or may not be set).
    """
    if blob is None:
        return None
    if isinstance(blob, (bytes, bytearray)):
        blob = blob.decode("utf-8")
    if isinstance(blob, str):
        return json.loads(blob)
    return blob


def _decode_each(seq: Sequence[Any]) -> list[Any]:
    return [_decode(b) for b in seq]


def _as_str(blob: Any) -> str:
    if blob is None:
        return ""
    if isinstance(blob, (bytes, bytearray)):
        return blob.decode("utf-8")
    return str(blob)


# ----------------------------------------------------------------- dataclasses


def _dc_from(cls, data: Mapping[str, Any] | None):
    if data is None:
        return None
    fields = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
    return cls(**{k: v for k, v in data.items() if k in fields})


@dataclass
class Span:
    trace_id: str
    span_id: str
    parent_span_id: str | None = None
    thread_id: str | None = None
    name: str = ""
    kind: str = "internal"
    start_unix_ns: int = 0
    end_unix_ns: int | None = None
    status: str = "unset"
    attributes: Any = None


@dataclass
class Prompt:
    name: str
    version: int
    content: str
    metadata: Any = None
    created_at_unix_ns: int = 0
    tags: list[str] = field(default_factory=list)


@dataclass
class PromptHit:
    name: str
    version: int
    score: float
    content: str


@dataclass
class DatasetRow:
    id: str = ""
    text: str = ""
    data: Any = None
    split: str = ""
    annotations: Any = None


@dataclass
class Dataset:
    name: str
    version: int
    rows: list[DatasetRow] = field(default_factory=list)
    metadata: Any = None
    created_at_unix_ns: int = 0
    tags: list[str] = field(default_factory=list)
    schema: Any = None


@dataclass
class DatasetHit:
    dataset: str
    version: int
    row_id: str
    score: float
    text: str


@dataclass
class EvalSummary:
    total_scored: int = 0
    mean: float = 0.0
    p50: float = 0.0
    p90: float = 0.0
    p99: float = 0.0
    min: float = 0.0
    max: float = 0.0
    pass_rate_50: float = 0.0


@dataclass
class EvalRun:
    id: str
    name: str = ""
    dataset_name: str = ""
    dataset_version: int = 0
    prompt_name: str | None = None
    prompt_version: int | None = None
    model: str = ""
    scorer: str = ""
    status: str = "pending"
    metadata: Any = None
    created_at_unix_ns: int = 0
    completed_at_unix_ns: int | None = None
    summary: EvalSummary | None = None


@dataclass
class EvalScore:
    run_id: str
    row_id: str
    score: float
    notes: Any = None
    output_text: str = ""
    recorded_at_unix_ns: int = 0


@dataclass
class EvalComparison:
    run_a: str
    run_b: str
    mean_delta: float = 0.0
    pass_rate_50_delta: float = 0.0
    regressed: list[Any] = field(default_factory=list)
    improved: list[Any] = field(default_factory=list)
    new_rows: list[str] = field(default_factory=list)
    dropped_rows: list[str] = field(default_factory=list)


@dataclass
class FailureCluster:
    representative_row_id: str
    representative_text: str
    members: list[Any] = field(default_factory=list)
    mean_score: float = 0.0


@dataclass
class ReplayInvocation:
    idx: int = 0
    span_id: str = ""
    kind: Any = None
    model: str | None = None
    prompt_name: str | None = None
    prompt_version: int | None = None
    input: Any = None
    output: Any = None
    metadata: Any = None
    recorded_at_unix_ns: int = 0


@dataclass
class ReplaySession:
    trace_id: str
    invocations: list[ReplayInvocation] = field(default_factory=list)
    fingerprint: str = ""
    captured_at_unix_ns: int = 0


@dataclass
class ReplayRun:
    id: str
    source_trace_id: str = ""
    mode: str = "live"
    overrides: list[Any] = field(default_factory=list)
    status: str = "pending"
    current_idx: int = 0
    outputs: Any = None
    replay_trace_id: str | None = None
    metadata: Any = None
    started_at_unix_ns: int = 0
    completed_at_unix_ns: int | None = None


@dataclass
class ReplayDiff:
    source_trace_id: str
    replay_run_id: str
    per_step: list[Any] = field(default_factory=list)
    differing_count: int = 0


@dataclass
class CostEntry:
    id: str
    tenant: str = ""
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    trace_id: str | None = None
    run_id: str | None = None
    prompt_name: str | None = None
    prompt_version: int | None = None
    input_text: str = ""
    metadata: Any = None
    recorded_at_unix_ns: int = 0


@dataclass
class CostBucket:
    key: str
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    entries: int = 0


@dataclass
class CostBudget:
    tenant: str
    period: Any = None
    amount_usd: float = 0.0
    warn_pct: float = 0.8
    metadata: Any = None
    created_at_unix_ns: int = 0


@dataclass
class CostAlert:
    tenant: str
    status: str
    spent_usd: float = 0.0
    budget_usd: float = 0.0
    period: Any = None
    raised_at_unix_ns: int = 0


@dataclass
class CostCluster:
    representative_id: str
    representative_text: str
    members: list[Any] = field(default_factory=list)
    total_cost_usd: float = 0.0


# ----------------------------------------------------------------- namespaces


class _Namespace:
    """Shared base for all six per-primitive namespaces."""

    __slots__ = ("_client",)

    def __init__(self, client: "ServerClient") -> None:
        self._client = client

    def _exec(self, *args: Any) -> Any:
        return self._client.raw.execute_command(*[_arg(a) for a in args])


class TraceAPI(_Namespace):
    """Wraps Phase 7.1 ``TRACE.*`` commands."""

    def record(
        self,
        trace_id: str,
        span_id: str,
        name: str,
        *,
        parent_span_id: str | None = None,
        attributes: Mapping[str, Any] | None = None,
        start_unix_ns: int | None = None,
        end_unix_ns: int | None = None,
        status: str | None = None,
        kind: str | None = None,
        thread_id: str | None = None,
    ) -> None:
        self._exec(
            "TRACE.RECORD",
            trace_id,
            span_id,
            parent_span_id,
            name,
            attributes,
            start_unix_ns,
            end_unix_ns,
            status,
            kind,
            thread_id,
        )

    def close(self, span_id: str, end_unix_ns: int, status: str | None = None) -> None:
        self._exec("TRACE.CLOSE", span_id, end_unix_ns, status)

    def get(self, trace_id: str) -> list[Span]:
        reply = self._exec("TRACE.GET", trace_id) or []
        return [_dc_from(Span, _decode(b)) for b in reply]

    def subtree(self, span_id: str) -> list[Span]:
        reply = self._exec("TRACE.SUBTREE", span_id) or []
        return [_dc_from(Span, _decode(b)) for b in reply]

    def thread(self, thread_id: str) -> list[Span]:
        reply = self._exec("TRACE.THREAD", thread_id) or []
        return [_dc_from(Span, _decode(b)) for b in reply]

    def search(self, filter_json: Mapping[str, Any]) -> list[Span]:
        reply = self._exec("TRACE.SEARCH", dict(filter_json)) or []
        return [_dc_from(Span, _decode(b)) for b in reply]


class PromptsAPI(_Namespace):
    """Wraps Phase 7.2 ``PROMPT.*`` commands."""

    def put(
        self,
        name: str,
        content: str,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        version = self._exec("PROMPT.PUT", name, content, metadata)
        return int(version)

    def get(self, name: str, version_or_tag: int | str | None = None) -> Prompt | None:
        reply = self._exec("PROMPT.GET", name, version_or_tag) if version_or_tag is not None else self._exec("PROMPT.GET", name)
        return _dc_from(Prompt, _decode(reply)) if reply else None

    def list(self, name: str) -> list[Prompt]:
        reply = self._exec("PROMPT.LIST", name) or []
        return [_dc_from(Prompt, _decode(b)) for b in reply]

    def names(self) -> list[str]:
        reply = self._exec("PROMPT.NAMES") or []
        return [_as_str(b) for b in reply]

    def tag(self, name: str, version: int, tag: str) -> None:
        self._exec("PROMPT.TAG", name, version, tag)

    def untag(self, name: str, tag: str) -> bool:
        return bool(self._exec("PROMPT.UNTAG", name, tag))

    def delete(self, name: str, version: int) -> bool:
        return bool(self._exec("PROMPT.DELETE", name, version))

    def search(self, query: str, k: int = 10) -> list[PromptHit]:
        reply = self._exec("PROMPT.SEARCH", query, k) or []
        out: list[PromptHit] = []
        for row in reply:
            if isinstance(row, (list, tuple)) and len(row) >= 4:
                out.append(
                    PromptHit(
                        name=_as_str(row[0]),
                        version=int(row[1]),
                        score=float(_as_str(row[2])),
                        content=_as_str(row[3]),
                    )
                )
        return out

    def diff(self, name: str, version_a: int, version_b: int) -> str:
        return _as_str(self._exec("PROMPT.DIFF", name, version_a, version_b))


class DatasetsAPI(_Namespace):
    """Wraps Phase 7.3 ``DATASET.*`` commands."""

    def create(self, name: str, schema: Mapping[str, Any] | None = None) -> None:
        self._exec("DATASET.CREATE", name, schema)

    def add(
        self,
        name: str,
        rows: Iterable[Mapping[str, Any] | str],
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> int:
        version = self._exec("DATASET.ADD", name, list(rows), metadata)
        return int(version)

    def get(self, name: str, version_or_tag: int | str | None = None) -> Dataset | None:
        reply = self._exec("DATASET.GET", name, version_or_tag) if version_or_tag is not None else self._exec("DATASET.GET", name)
        decoded = _decode(reply) if reply else None
        if decoded is None:
            return None
        rows = [_dc_from(DatasetRow, r) for r in decoded.get("rows", [])]
        ds = _dc_from(Dataset, decoded)
        if ds is not None:
            ds.rows = [r for r in rows if r is not None]
        return ds

    def list(self, name: str) -> list[Dataset]:
        reply = self._exec("DATASET.LIST", name) or []
        return [_dc_from(Dataset, _decode(b)) for b in reply]

    def names(self) -> list[str]:
        reply = self._exec("DATASET.NAMES") or []
        return [_as_str(b) for b in reply]

    def tag(self, name: str, version: int, tag: str) -> None:
        self._exec("DATASET.TAG", name, version, tag)

    def untag(self, name: str, tag: str) -> bool:
        return bool(self._exec("DATASET.UNTAG", name, tag))

    def delete(self, name: str, version: int) -> bool:
        return bool(self._exec("DATASET.DELETE", name, version))

    def sample(self, name: str, version: int, n: int, split: str | None = None) -> list[DatasetRow]:
        reply = self._exec("DATASET.SAMPLE", name, version, n, split) or []
        return [_dc_from(DatasetRow, _decode(b)) for b in reply]

    def size(self, name: str, version: int, split: str | None = None) -> int:
        return int(self._exec("DATASET.SIZE", name, version, split))

    def splits(self, name: str, version: int) -> list[str]:
        reply = self._exec("DATASET.SPLITS", name, version) or []
        return [_as_str(b) for b in reply]

    def search(self, query: str, k: int = 10, name_filter: str | None = None) -> list[DatasetHit]:
        reply = self._exec("DATASET.SEARCH", query, k, name_filter) or []
        out: list[DatasetHit] = []
        for row in reply:
            if isinstance(row, (list, tuple)) and len(row) >= 5:
                out.append(
                    DatasetHit(
                        dataset=_as_str(row[0]),
                        version=int(row[1]),
                        row_id=_as_str(row[2]),
                        score=float(_as_str(row[3])),
                        text=_as_str(row[4]),
                    )
                )
        return out

    def from_recall(self, key: str, query: str, k: int, dataset_name: str, split: str | None = None) -> int:
        return int(self._exec("DATASET.FROM_RECALL", key, query, k, dataset_name, split))


class EvalsAPI(_Namespace):
    """Wraps Phase 7.4 ``EVAL.*`` commands."""

    def start(
        self,
        dataset_name: str,
        dataset_version: int,
        model: str,
        scorer: str,
        *,
        prompt_name: str | None = None,
        prompt_version: int | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return _as_str(
            self._exec(
                "EVAL.START",
                dataset_name,
                dataset_version,
                prompt_name,
                prompt_version,
                model,
                scorer,
                metadata,
            )
        )

    def score(
        self,
        run_id: str,
        *,
        row_id: str,
        score: float,
        output_text: str | None = None,
        notes: Mapping[str, Any] | None = None,
    ) -> None:
        self._exec("EVAL.SCORE", run_id, row_id, score, output_text, notes)

    def complete(self, run_id: str) -> EvalSummary | None:
        reply = self._exec("EVAL.COMPLETE", run_id)
        return _dc_from(EvalSummary, _decode(reply)) if reply else None

    def fail(self, run_id: str, reason: str) -> None:
        self._exec("EVAL.FAIL", run_id, reason)

    def get(self, run_id: str) -> EvalRun | None:
        decoded = _decode(self._exec("EVAL.GET", run_id))
        if decoded is None:
            return None
        run = _dc_from(EvalRun, decoded)
        if run is not None and isinstance(decoded.get("summary"), Mapping):
            run.summary = _dc_from(EvalSummary, decoded["summary"])
        return run

    def scores(self, run_id: str) -> list[EvalScore]:
        reply = self._exec("EVAL.SCORES", run_id) or []
        return [_dc_from(EvalScore, _decode(b)) for b in reply]

    def list(self, dataset_name: str | None = None, dataset_version: int | None = None) -> list[EvalRun]:
        if dataset_name is not None and dataset_version is not None:
            reply = self._exec("EVAL.LIST", dataset_name, dataset_version) or []
        else:
            reply = self._exec("EVAL.LIST") or []
        out: list[EvalRun] = []
        for b in reply:
            decoded = _decode(b)
            if decoded is None:
                continue
            run = _dc_from(EvalRun, decoded)
            if run is not None and isinstance(decoded.get("summary"), Mapping):
                run.summary = _dc_from(EvalSummary, decoded["summary"])
            if run is not None:
                out.append(run)
        return out

    def compare(self, run_a: str, run_b: str) -> EvalComparison | None:
        reply = self._exec("EVAL.COMPARE", run_a, run_b)
        return _dc_from(EvalComparison, _decode(reply)) if reply else None

    def cluster_failures(
        self,
        run_id: str,
        *,
        score_threshold: float = 0.5,
        sim_threshold: float = 0.8,
        max_clusters: int = 10,
    ) -> list[FailureCluster]:
        reply = self._exec(
            "EVAL.CLUSTER_FAILURES",
            run_id,
            score_threshold,
            sim_threshold,
            max_clusters,
        ) or []
        return [_dc_from(FailureCluster, _decode(b)) for b in reply]


class ReplayAPI(_Namespace):
    """Wraps Phase 7.5 ``REPLAY.*`` commands."""

    def capture(self, trace_id: str, invocation: Mapping[str, Any]) -> int:
        return int(self._exec("REPLAY.CAPTURE", trace_id, dict(invocation)))

    def start(
        self,
        source_trace_id: str,
        *,
        mode: str = "live",
        overrides: Iterable[Mapping[str, Any]] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return _as_str(
            self._exec(
                "REPLAY.START",
                source_trace_id,
                mode,
                list(overrides) if overrides is not None else None,
                metadata,
            )
        )

    def step(self, run_id: str) -> ReplayInvocation | None:
        reply = self._exec("REPLAY.STEP", run_id)
        return _dc_from(ReplayInvocation, _decode(reply)) if reply else None

    def record(self, run_id: str, invocation_idx: int, output: Any) -> None:
        # ``output`` may be any JSON-encodable value; force JSON encoding.
        payload = output if isinstance(output, (dict, list)) else {"value": output}
        self._exec("REPLAY.RECORD", run_id, invocation_idx, payload)

    def complete(self, run_id: str) -> None:
        self._exec("REPLAY.COMPLETE", run_id)

    def fail(self, run_id: str, reason: str) -> None:
        self._exec("REPLAY.FAIL", run_id, reason)

    def get_session(self, trace_id: str) -> ReplaySession | None:
        decoded = _decode(self._exec("REPLAY.GET_SESSION", trace_id))
        if decoded is None:
            return None
        session = _dc_from(ReplaySession, decoded)
        if session is not None:
            session.invocations = [
                _dc_from(ReplayInvocation, inv) for inv in decoded.get("invocations", [])
            ]
            session.invocations = [i for i in session.invocations if i is not None]
        return session

    def get_run(self, run_id: str) -> ReplayRun | None:
        return _dc_from(ReplayRun, _decode(self._exec("REPLAY.GET_RUN", run_id)))

    def list_sessions(self) -> list[ReplaySession]:
        reply = self._exec("REPLAY.LIST_SESSIONS") or []
        return [_dc_from(ReplaySession, _decode(b)) for b in reply]

    def list_runs(self, source_trace_id: str | None = None) -> list[ReplayRun]:
        reply = self._exec("REPLAY.LIST_RUNS", source_trace_id) or []
        return [_dc_from(ReplayRun, _decode(b)) for b in reply]

    def diff(self, source_trace_id: str, replay_run_id: str) -> ReplayDiff | None:
        return _dc_from(ReplayDiff, _decode(self._exec("REPLAY.DIFF", source_trace_id, replay_run_id)))

    def set_trace(self, run_id: str, replay_trace_id: str) -> None:
        self._exec("REPLAY.SET_TRACE", run_id, replay_trace_id)


class CostAPI(_Namespace):
    """Wraps Phase 7.6 ``COST.*`` commands."""

    def record(
        self,
        tenant: str,
        model: str,
        *,
        tokens_in: int,
        tokens_out: int,
        cost_usd: float,
        input_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        return _as_str(
            self._exec(
                "COST.RECORD",
                tenant,
                model,
                tokens_in,
                tokens_out,
                cost_usd,
                input_text,
                metadata,
            )
        )

    def query(self, filter: Mapping[str, Any] | None = None) -> list[CostEntry]:
        reply = self._exec("COST.QUERY", filter) or []
        return [_dc_from(CostEntry, _decode(b)) for b in reply]

    def aggregate(
        self,
        group_by: str,
        filter: Mapping[str, Any] | None = None,
    ) -> list[CostBucket]:
        reply = self._exec("COST.AGGREGATE", group_by, filter) or []
        return [_dc_from(CostBucket, _decode(b)) for b in reply]

    def total(
        self,
        tenant: str,
        *,
        since_unix_ns: int | None = None,
        until_unix_ns: int | None = None,
    ) -> float:
        return float(_as_str(self._exec("COST.TOTAL", tenant, since_unix_ns, until_unix_ns)))

    def set_budget(
        self,
        tenant: str,
        period: str,
        amount_usd: float,
        *,
        warn_pct: float | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self._exec("COST.SET_BUDGET", tenant, period, amount_usd, warn_pct, metadata)

    def get_budget(self, tenant: str) -> CostBudget | None:
        return _dc_from(CostBudget, _decode(self._exec("COST.GET_BUDGET", tenant)))

    def delete_budget(self, tenant: str) -> bool:
        return bool(self._exec("COST.DELETE_BUDGET", tenant))

    def status(self, tenant: str) -> str:
        return _as_str(self._exec("COST.STATUS", tenant))

    def alerts(self) -> list[CostAlert]:
        reply = self._exec("COST.ALERTS") or []
        return [_dc_from(CostAlert, _decode(b)) for b in reply]

    def cluster_expensive(
        self,
        *,
        filter: Mapping[str, Any] | None = None,
        sim_threshold: float = 0.7,
        max_clusters: int = 10,
        top_n: int = 50,
    ) -> list[CostCluster]:
        reply = self._exec(
            "COST.CLUSTER_EXPENSIVE",
            filter,
            sim_threshold,
            max_clusters,
            top_n,
        ) or []
        return [_dc_from(CostCluster, _decode(b)) for b in reply]


# ----------------------------------------------------------------- client


class ServerClient:
    """Typed Python facade over the DuxxDB RESP wire protocol.

    Wraps a ``redis.Redis`` client and exposes six per-primitive
    namespaces:

    * :attr:`trace` — Phase 7.1 distributed traces
    * :attr:`prompts` — Phase 7.2 prompt versioning + tagging
    * :attr:`datasets` — Phase 7.3 dataset versioning + sampling
    * :attr:`evals` — Phase 7.4 eval runs + comparisons + clustering
    * :attr:`replay` — Phase 7.5 deterministic replay
    * :attr:`cost` — Phase 7.6 cost ledger + budgets

    The raw ``redis.Redis`` is exposed as :attr:`raw` for anything
    the facade does not cover yet.

    :param url: A standard ``redis://`` URL. Use
        ``redis://:<token>@host:port`` when DuxxDB token auth is
        enabled.
    :param redis_client: Optional pre-built ``redis.Redis`` instance.
        When supplied, ``url`` is ignored. Useful for advanced setups
        (TLS, connection pools).
    """

    __slots__ = ("raw", "trace", "prompts", "datasets", "evals", "replay", "cost")

    def __init__(
        self,
        url: str | None = None,
        *,
        redis_client: "_redis.Redis | None" = None,
    ) -> None:
        if redis_client is None:
            try:
                import redis  # type: ignore[import]
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "duxxdb.server.ServerClient requires the redis-py package. "
                    "Install duxxdb with the 'server' extra: pip install 'duxxdb[server]'"
                ) from exc
            if url is None:
                url = "redis://localhost:6379"
            redis_client = redis.Redis.from_url(url, decode_responses=False)
        self.raw = redis_client
        self.trace = TraceAPI(self)
        self.prompts = PromptsAPI(self)
        self.datasets = DatasetsAPI(self)
        self.evals = EvalsAPI(self)
        self.replay = ReplayAPI(self)
        self.cost = CostAPI(self)

    def ping(self) -> bool:
        return bool(self.raw.ping())

    def close(self) -> None:
        try:
            self.raw.close()
        except Exception:  # pragma: no cover
            pass

    def __enter__(self) -> "ServerClient":
        return self

    def __exit__(self, *_exc_info: Any) -> None:
        self.close()
