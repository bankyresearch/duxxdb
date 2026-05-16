"""Unit tests for the ``duxxdb.server.ServerClient`` facade.

These tests exercise the *encoding* side of the facade — every public
method translates its Python arguments into the correct sequence of
RESP arguments for the target command. They use an in-process mock
redis client that captures every ``execute_command`` call, so they do
not require a running ``duxx-server``.

End-to-end tests against a real ``duxx-server`` live in
``tests/test_server_e2e.py`` and are gated on ``DUXXDB_E2E=1`` because
they spawn the daemon as a subprocess.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from duxxdb.server import (
    CostBudget,
    CostEntry,
    Dataset,
    DatasetRow,
    EvalRun,
    EvalSummary,
    Prompt,
    ServerClient,
    Span,
)


class MockRedis:
    """Captures every ``execute_command`` call and returns a canned reply.

    Tests set ``self.reply`` before invoking a facade method, then
    assert against ``self.commands[-1]`` afterwards.
    """

    def __init__(self) -> None:
        self.commands: list[tuple[Any, ...]] = []
        self.reply: Any = None

    def execute_command(self, *args: Any) -> Any:
        self.commands.append(args)
        # The facade encodes every argument to ``str`` via ``_arg``,
        # so we assert that here too — the wire never sees bytes.
        for a in args:
            assert isinstance(a, str), f"non-str argument escaped to RESP: {a!r}"
        # If the caller pre-populated a list of replies, pop the next one.
        if isinstance(self.reply, list) and self.reply and isinstance(self.reply[0], _NextReply):
            return self.reply.pop(0).value
        return self.reply

    def ping(self) -> bool:
        return True

    def close(self) -> None:
        pass


class _NextReply:
    def __init__(self, value: Any) -> None:
        self.value = value


@pytest.fixture
def client() -> tuple[ServerClient, MockRedis]:
    fake = MockRedis()
    return ServerClient(redis_client=fake), fake  # type: ignore[arg-type]


# ----------------------------------------------------------------- traces


def test_trace_record_serializes_attributes_as_json(client):
    sc, fake = client
    sc.trace.record(
        trace_id="t1",
        span_id="s1",
        name="run.turn",
        parent_span_id=None,
        attributes={"user": "alice", "tokens": 42},
        start_unix_ns=1000,
        end_unix_ns=2000,
        status="ok",
        kind="internal",
        thread_id=None,
    )
    cmd = fake.commands[-1]
    assert cmd[0] == "TRACE.RECORD"
    assert cmd[1:5] == ("t1", "s1", "-", "run.turn")
    # attributes encoded as JSON
    assert json.loads(cmd[5]) == {"user": "alice", "tokens": 42}
    assert cmd[6:11] == ("1000", "2000", "ok", "internal", "-")


def test_trace_get_decodes_array_of_spans(client):
    sc, fake = client
    fake.reply = [
        json.dumps({"trace_id": "t1", "span_id": "s1", "name": "root"}).encode(),
        json.dumps({"trace_id": "t1", "span_id": "s2", "name": "child", "parent_span_id": "s1"}).encode(),
    ]
    spans = sc.trace.get("t1")
    assert len(spans) == 2
    assert all(isinstance(s, Span) for s in spans)
    assert spans[0].span_id == "s1"
    assert spans[1].parent_span_id == "s1"


# ----------------------------------------------------------------- prompts


def test_prompt_put_returns_int_version(client):
    sc, fake = client
    fake.reply = 7
    v = sc.prompts.put("classifier", "you are a refund agent", metadata={"author": "alice"})
    assert v == 7
    cmd = fake.commands[-1]
    assert cmd[0] == "PROMPT.PUT"
    assert cmd[1:3] == ("classifier", "you are a refund agent")
    assert json.loads(cmd[3]) == {"author": "alice"}


def test_prompt_get_by_tag(client):
    sc, fake = client
    fake.reply = json.dumps({"name": "c", "version": 5, "content": "x"}).encode()
    p = sc.prompts.get("c", "prod")
    assert isinstance(p, Prompt)
    assert p.version == 5
    cmd = fake.commands[-1]
    assert cmd == ("PROMPT.GET", "c", "prod")


def test_prompt_get_latest_no_version_arg(client):
    sc, fake = client
    fake.reply = json.dumps({"name": "c", "version": 9, "content": "x"}).encode()
    sc.prompts.get("c")
    cmd = fake.commands[-1]
    assert cmd == ("PROMPT.GET", "c")  # no third arg


def test_prompt_search_decodes_nested_array(client):
    sc, fake = client
    fake.reply = [
        ["greeter", 3, "0.812345", "Hello, how can I help?"],
        [b"farewell", 1, b"0.701234", b"Goodbye!"],  # mixed bytes/str
    ]
    hits = sc.prompts.search("hello", k=5)
    assert len(hits) == 2
    assert hits[0].name == "greeter"
    assert hits[0].version == 3
    assert abs(hits[0].score - 0.812345) < 1e-6
    assert hits[1].name == "farewell"  # bytes → str
    cmd = fake.commands[-1]
    assert cmd == ("PROMPT.SEARCH", "hello", "5")


# ----------------------------------------------------------------- datasets


def test_dataset_add_serializes_rows_as_json(client):
    sc, fake = client
    fake.reply = 2
    rows = [{"text": "refund please", "split": "train"}, "another row"]
    v = sc.datasets.add("refunds", rows, metadata={"source": "csv"})
    assert v == 2
    cmd = fake.commands[-1]
    assert cmd[0] == "DATASET.ADD"
    assert cmd[1] == "refunds"
    decoded = json.loads(cmd[2])
    assert decoded == [{"text": "refund please", "split": "train"}, "another row"]
    assert json.loads(cmd[3]) == {"source": "csv"}


def test_dataset_get_decodes_nested_rows(client):
    sc, fake = client
    payload = {
        "name": "ds",
        "version": 1,
        "rows": [
            {"id": "r1", "text": "hi", "split": "train"},
            {"id": "r2", "text": "bye", "split": "test"},
        ],
        "metadata": {"src": "demo"},
    }
    fake.reply = json.dumps(payload).encode()
    ds = sc.datasets.get("ds", 1)
    assert isinstance(ds, Dataset)
    assert ds.version == 1
    assert len(ds.rows) == 2
    assert all(isinstance(r, DatasetRow) for r in ds.rows)
    assert ds.rows[0].id == "r1"


def test_dataset_size_returns_int(client):
    sc, fake = client
    fake.reply = 100
    assert sc.datasets.size("ds", 1, split="train") == 100
    assert fake.commands[-1] == ("DATASET.SIZE", "ds", "1", "train")


def test_dataset_search_decodes_5tuple(client):
    sc, fake = client
    fake.reply = [["ds", 1, "row-7", "0.91", "refund text"]]
    hits = sc.datasets.search("refund", k=3, name_filter="ds")
    assert len(hits) == 1
    assert hits[0].dataset == "ds"
    assert hits[0].row_id == "row-7"
    assert abs(hits[0].score - 0.91) < 1e-6
    cmd = fake.commands[-1]
    assert cmd == ("DATASET.SEARCH", "refund", "3", "ds")


# ----------------------------------------------------------------- evals


def test_eval_start_with_all_kwargs(client):
    sc, fake = client
    fake.reply = b"run-abc-123"
    run_id = sc.evals.start(
        dataset_name="ds",
        dataset_version=4,
        model="gpt-4o-mini",
        scorer="llm_judge",
        prompt_name="classifier",
        prompt_version=7,
        metadata={"experiment": "v8"},
    )
    assert run_id == "run-abc-123"
    cmd = fake.commands[-1]
    assert cmd[0] == "EVAL.START"
    # canonical ordering: dataset_name, dataset_version, prompt_name,
    # prompt_version, model, scorer, metadata
    assert cmd[1:7] == ("ds", "4", "classifier", "7", "gpt-4o-mini", "llm_judge")
    assert json.loads(cmd[7]) == {"experiment": "v8"}


def test_eval_start_without_prompt_uses_sentinels(client):
    sc, fake = client
    fake.reply = b"run-1"
    sc.evals.start(
        dataset_name="ds",
        dataset_version=1,
        model="m",
        scorer="s",
    )
    cmd = fake.commands[-1]
    # prompt_name + prompt_version → "-" so the server treats them as None
    assert cmd[3] == "-"
    assert cmd[4] == "-"
    # metadata absent → also "-"
    assert cmd[7] == "-"


def test_eval_score_with_optional_notes(client):
    sc, fake = client
    sc.evals.score("run-1", row_id="row-0", score=0.875, output_text="REFUND", notes={"latency_ms": 42})
    cmd = fake.commands[-1]
    assert cmd[0] == "EVAL.SCORE"
    assert cmd[1:5] == ("run-1", "row-0", "0.875", "REFUND")
    assert json.loads(cmd[5]) == {"latency_ms": 42}


def test_eval_complete_decodes_summary(client):
    sc, fake = client
    fake.reply = json.dumps(
        {"total_scored": 10, "mean": 0.81, "p50": 0.84, "p90": 0.92, "p99": 0.98,
         "min": 0.2, "max": 1.0, "pass_rate_50": 0.9}
    ).encode()
    summary = sc.evals.complete("run-1")
    assert isinstance(summary, EvalSummary)
    assert summary.total_scored == 10
    assert summary.pass_rate_50 == pytest.approx(0.9)


def test_eval_get_inflates_nested_summary(client):
    sc, fake = client
    fake.reply = json.dumps(
        {
            "id": "run-1",
            "dataset_name": "ds",
            "model": "m",
            "scorer": "s",
            "status": "completed",
            "summary": {"total_scored": 5, "mean": 0.5},
        }
    ).encode()
    run = sc.evals.get("run-1")
    assert isinstance(run, EvalRun)
    assert run.status == "completed"
    assert isinstance(run.summary, EvalSummary)
    assert run.summary.total_scored == 5


def test_eval_cluster_failures_thresholds(client):
    sc, fake = client
    fake.reply = []
    sc.evals.cluster_failures("run-1", score_threshold=0.4, sim_threshold=0.85, max_clusters=20)
    cmd = fake.commands[-1]
    assert cmd == ("EVAL.CLUSTER_FAILURES", "run-1", "0.4", "0.85", "20")


# ----------------------------------------------------------------- replay


def test_replay_start_serializes_overrides_and_mode(client):
    sc, fake = client
    fake.reply = b"replay-run-7"
    overrides = [
        {"at_idx": 0, "kind": {"kind": "swap_model", "model": "gpt-4o"}},
        {"at_idx": 2, "kind": {"kind": "skip"}},
    ]
    run_id = sc.replay.start("trace-1", mode="stepped", overrides=overrides, metadata={"why": "regression"})
    assert run_id == "replay-run-7"
    cmd = fake.commands[-1]
    assert cmd[0] == "REPLAY.START"
    assert cmd[1:3] == ("trace-1", "stepped")
    assert json.loads(cmd[3]) == overrides
    assert json.loads(cmd[4]) == {"why": "regression"}


def test_replay_record_wraps_scalar_output_in_value(client):
    sc, fake = client
    sc.replay.record("run-1", invocation_idx=3, output="answer-string")
    cmd = fake.commands[-1]
    assert cmd[0] == "REPLAY.RECORD"
    assert cmd[1:3] == ("run-1", "3")
    assert json.loads(cmd[3]) == {"value": "answer-string"}


def test_replay_record_passes_dict_through(client):
    sc, fake = client
    sc.replay.record("run-1", invocation_idx=0, output={"text": "hi"})
    cmd = fake.commands[-1]
    assert json.loads(cmd[3]) == {"text": "hi"}


# ----------------------------------------------------------------- cost


def test_cost_record_all_optional_fields(client):
    sc, fake = client
    fake.reply = b"cost-id-9"
    eid = sc.cost.record(
        tenant="acme",
        model="gpt-4o",
        tokens_in=120,
        tokens_out=80,
        cost_usd=0.0042,
        input_text="please refund",
        metadata={"trace_id": "t1"},
    )
    assert eid == "cost-id-9"
    cmd = fake.commands[-1]
    assert cmd[0] == "COST.RECORD"
    assert cmd[1:6] == ("acme", "gpt-4o", "120", "80", "0.0042")
    assert cmd[6] == "please refund"
    assert json.loads(cmd[7]) == {"trace_id": "t1"}


def test_cost_total_returns_float(client):
    sc, fake = client
    fake.reply = b"12.345678"
    total = sc.cost.total("acme", since_unix_ns=1000, until_unix_ns=9999)
    assert total == pytest.approx(12.345678)
    assert fake.commands[-1] == ("COST.TOTAL", "acme", "1000", "9999")


def test_cost_set_budget_with_warn_pct(client):
    sc, fake = client
    sc.cost.set_budget("acme", "monthly", 1000.0, warn_pct=0.75, metadata={"approver": "cfo"})
    cmd = fake.commands[-1]
    assert cmd[0] == "COST.SET_BUDGET"
    assert cmd[1:5] == ("acme", "monthly", "1000.0", "0.75")
    assert json.loads(cmd[5]) == {"approver": "cfo"}


def test_cost_get_budget_decodes_dataclass(client):
    sc, fake = client
    fake.reply = json.dumps(
        {"tenant": "acme", "amount_usd": 500.0, "warn_pct": 0.8, "period": "monthly"}
    ).encode()
    b = sc.cost.get_budget("acme")
    assert isinstance(b, CostBudget)
    assert b.tenant == "acme"
    assert b.amount_usd == 500.0


def test_cost_query_returns_list_of_entries(client):
    sc, fake = client
    fake.reply = [
        json.dumps({"id": "c1", "tenant": "acme", "model": "m", "cost_usd": 0.01}).encode(),
        json.dumps({"id": "c2", "tenant": "acme", "model": "m", "cost_usd": 0.02}).encode(),
    ]
    entries = sc.cost.query({"tenant": "acme"})
    assert len(entries) == 2
    assert all(isinstance(e, CostEntry) for e in entries)
    assert entries[0].id == "c1"
    cmd = fake.commands[-1]
    assert cmd[0] == "COST.QUERY"
    assert json.loads(cmd[1]) == {"tenant": "acme"}


def test_cost_aggregate_with_filter(client):
    sc, fake = client
    fake.reply = []
    sc.cost.aggregate("model", filter={"since_unix_ns": 1000})
    cmd = fake.commands[-1]
    assert cmd[0] == "COST.AGGREGATE"
    assert cmd[1] == "model"
    assert json.loads(cmd[2]) == {"since_unix_ns": 1000}


# ----------------------------------------------------------------- general


def test_none_becomes_dash_sentinel(client):
    sc, fake = client
    # COST.TOTAL has two optional integer args -- passing None should send "-"
    fake.reply = b"0.0"
    sc.cost.total("acme")
    cmd = fake.commands[-1]
    assert cmd == ("COST.TOTAL", "acme", "-", "-")


def test_client_works_as_context_manager(client):
    sc, fake = client
    with sc as c:
        assert c is sc
    # close() is a no-op on MockRedis; just verify no exception.


def test_raw_client_is_exposed(client):
    sc, fake = client
    assert sc.raw is fake
