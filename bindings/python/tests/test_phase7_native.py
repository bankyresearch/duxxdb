"""Tests for the v0.2.1 native PyO3 bindings of the remaining five
Phase 7 primitives:

* :class:`duxxdb.CostLedger`
* :class:`duxxdb.DatasetRegistry`
* :class:`duxxdb.EvalRegistry`
* :class:`duxxdb.ReplayRegistry`
* :class:`duxxdb.TraceStore`

Each test covers the in-memory path plus the redb persistence
round-trip (open / write / drop / reopen / verify).
"""

from __future__ import annotations

import pytest

import duxxdb


# ---------------------------------------------------------------- CostLedger


@pytest.fixture
def cost(tmp_path) -> duxxdb.CostLedger:
    return duxxdb.CostLedger(dim=16)


def test_cost_record_and_total(cost):
    cost.record(tenant="acme", model="gpt-4o", tokens_in=100, tokens_out=50, cost_usd=0.0023)
    cost.record(
        tenant="acme", model="gpt-4o-mini", tokens_in=200, tokens_out=80, cost_usd=0.0011
    )
    assert cost.total("acme") == pytest.approx(0.0034, abs=1e-9)


def test_cost_query_returns_entries(cost):
    cost.record(tenant="acme", model="gpt-4o", tokens_in=10, tokens_out=5, cost_usd=0.001)
    entries = cost.query(tenant="acme")
    assert len(entries) == 1
    e = entries[0]
    assert e.tenant == "acme"
    assert e.model == "gpt-4o"
    assert e.cost_usd == pytest.approx(0.001)


def test_cost_metadata_round_trips(cost):
    cost.record(
        tenant="acme",
        model="gpt-4o",
        tokens_in=10,
        tokens_out=5,
        cost_usd=0.001,
        metadata={"trace_id": "t1", "tags": ["a", "b"]},
    )
    e = cost.query(tenant="acme")[0]
    assert e.metadata == {"trace_id": "t1", "tags": ["a", "b"]}


def test_cost_aggregate_by_model(cost):
    for m in ["gpt-4o", "gpt-4o", "gpt-4o-mini"]:
        cost.record(tenant="acme", model=m, tokens_in=10, tokens_out=5, cost_usd=0.001)
    buckets = cost.aggregate("model", tenant="acme")
    by_key = {b[0]: b for b in buckets}
    assert by_key["gpt-4o"][1] == 2  # count
    assert by_key["gpt-4o-mini"][1] == 1


def test_cost_budget_lifecycle(cost):
    cost.set_budget("acme", "monthly", 100.0, warn_pct=0.75)
    b = cost.get_budget("acme")
    assert b is not None
    assert b.amount_usd == 100.0
    assert b.period == "monthly"
    assert b.warn_pct == pytest.approx(0.75)
    assert cost.status("acme") in {"ok", "warning", "exceeded", "no_budget"}
    assert cost.delete_budget("acme") is True
    assert cost.get_budget("acme") is None


def test_cost_budget_custom_period(cost):
    cost.set_budget("acme", "custom:3600", 5.0)
    b = cost.get_budget("acme")
    assert b.period == "custom:3600"


def test_cost_redb_round_trip(tmp_path):
    db = str(tmp_path / "cost.redb")
    c1 = duxxdb.CostLedger(dim=16, storage=f"redb:{db}")
    c1.record(tenant="acme", model="gpt-4o", tokens_in=10, tokens_out=5, cost_usd=0.05)
    c1.set_budget("acme", "monthly", 100.0)
    del c1

    c2 = duxxdb.CostLedger(dim=16, storage=f"redb:{db}")
    assert c2.total("acme") == pytest.approx(0.05)
    assert c2.get_budget("acme").amount_usd == 100.0


# ---------------------------------------------------------------- DatasetRegistry


@pytest.fixture
def ds() -> duxxdb.DatasetRegistry:
    return duxxdb.DatasetRegistry(dim=16)


def test_dataset_create_and_add(ds):
    ds.create("refunds", schema={"columns": ["text"]})
    v = ds.add(
        "refunds",
        [
            {"id": "r1", "text": "I want a refund", "split": "train"},
            {"id": "r2", "text": "Where is my package?", "split": "test"},
        ],
    )
    assert v == 1
    assert ds.size("refunds", v) == 2
    assert sorted(ds.splits("refunds", v)) == ["test", "train"]


def test_dataset_string_rows_get_auto_ids(ds):
    v = ds.add("qa", ["hello", "goodbye"])
    got = ds.get("qa", v)
    assert len(got.rows) == 2
    assert all(r.id for r in got.rows)


def test_dataset_tag_and_get_by_tag(ds):
    ds.add("c", [{"id": "r1", "text": "v1 only", "split": "train"}])
    v2 = ds.add("c", [{"id": "r1", "text": "v2 update", "split": "train"}])
    ds.tag("c", v2, "golden")
    got = ds.get("c", "golden")
    assert got.version == v2
    assert got.rows[0].text == "v2 update"


def test_dataset_delete_keeps_counter(ds):
    ds.add("c", [{"id": "r1", "text": "x", "split": "train"}])
    ds.add("c", [{"id": "r2", "text": "y", "split": "train"}])
    assert ds.delete("c", 1) is True
    v3 = ds.add("c", [{"id": "r3", "text": "z", "split": "train"}])
    assert v3 == 3  # v1 not reused


def test_dataset_sample_with_split(ds):
    ds.add(
        "c",
        [
            {"id": f"r{i}", "text": f"row {i}", "split": "train"} for i in range(3)
        ]
        + [{"id": "ev1", "text": "eval row", "split": "eval"}],
    )
    sampled = ds.sample("c", 1, n=5, split="train")
    assert len(sampled) == 3
    assert all(r.split == "train" for r in sampled)


def test_dataset_search(ds):
    ds.add(
        "qa",
        [
            {"id": "h1", "text": "hello world how are you today", "split": "train"},
            {"id": "g1", "text": "goodbye see you soon", "split": "train"},
        ],
    )
    hits = ds.search("hello", k=2)
    assert hits, "search returned nothing"
    # tuples: (dataset, version, row_id, score, text)
    assert hits[0][4].startswith("hello")


def test_dataset_redb_round_trip(tmp_path):
    db = str(tmp_path / "ds.redb")
    d1 = duxxdb.DatasetRegistry(dim=16, storage=f"redb:{db}")
    d1.create("c", schema={"v": 1})
    v = d1.add("c", [{"id": "r1", "text": "stays", "split": "train"}])
    d1.tag("c", v, "golden")
    del d1

    d2 = duxxdb.DatasetRegistry(dim=16, storage=f"redb:{db}")
    got = d2.get("c", "golden")
    assert got is not None
    assert got.rows[0].text == "stays"


# ---------------------------------------------------------------- EvalRegistry


@pytest.fixture
def evals() -> duxxdb.EvalRegistry:
    return duxxdb.EvalRegistry(dim=16)


def test_eval_full_loop(evals):
    rid = evals.start(
        dataset_name="ds",
        dataset_version=1,
        model="gpt-4o-mini",
        scorer="judge",
        prompt_name="classifier",
        prompt_version=3,
        metadata={"experiment": "v8"},
    )
    for i in range(5):
        evals.score(rid, row_id=f"r{i}", score=1.0 if i < 4 else 0.0, output_text=f"out {i}")
    summary = evals.complete(rid)
    assert summary.total_scored == 5
    assert summary.pass_rate_50 == pytest.approx(0.8)

    run = evals.get(rid)
    assert run.status == "completed"
    assert run.summary.total_scored == 5
    assert run.metadata == {"experiment": "v8"}


def test_eval_scores_and_list(evals):
    rid = evals.start(dataset_name="ds", dataset_version=1, model="m", scorer="s")
    for i in range(3):
        evals.score(rid, row_id=f"r{i}", score=0.5)
    scores = evals.scores(rid)
    assert len(scores) == 3
    assert all(s.score == 0.5 for s in scores)
    runs = evals.list(dataset_name="ds", dataset_version=1)
    assert any(r.id == rid for r in runs)


def test_eval_fail_stashes_reason(evals):
    rid = evals.start(dataset_name="ds", dataset_version=1, model="m", scorer="s")
    evals.fail(rid, reason="OOM at row 12")
    run = evals.get(rid)
    assert run.status == "failed"
    assert run.metadata["failure_reason"] == "OOM at row 12"


def test_eval_compare(evals):
    a = evals.start(dataset_name="ds", dataset_version=1, model="ma", scorer="s")
    b = evals.start(dataset_name="ds", dataset_version=1, model="mb", scorer="s")
    for i in range(3):
        evals.score(a, row_id=f"r{i}", score=0.9)
        evals.score(b, row_id=f"r{i}", score=0.5 if i == 0 else 0.95)
    mean_delta, prate_delta, regressed, improved, new, dropped = evals.compare(a, b)
    assert regressed >= 1  # row 0 went 0.9 -> 0.5
    assert improved >= 1  # row 1, 2 went 0.9 -> 0.95


def test_eval_cluster_failures(evals):
    rid = evals.start(dataset_name="ds", dataset_version=1, model="m", scorer="s")
    for i in range(5):
        evals.score(rid, row_id=f"r{i}", score=0.2, output_text=f"wrong answer {i}")
    clusters = evals.cluster_failures(rid, score_threshold=0.5, sim_threshold=0.0, max_clusters=3)
    assert clusters, "expected at least one failure cluster"


def test_eval_redb_round_trip(tmp_path):
    db = str(tmp_path / "evals.redb")
    e1 = duxxdb.EvalRegistry(dim=16, storage=f"redb:{db}")
    rid = e1.start(dataset_name="ds", dataset_version=1, model="m", scorer="s")
    e1.score(rid, row_id="r1", score=0.9, output_text="good")
    e1.complete(rid)
    del e1

    e2 = duxxdb.EvalRegistry(dim=16, storage=f"redb:{db}")
    run = e2.get(rid)
    assert run.status == "completed"
    assert len(e2.scores(rid)) == 1


# ---------------------------------------------------------------- ReplayRegistry


@pytest.fixture
def replay() -> duxxdb.ReplayRegistry:
    return duxxdb.ReplayRegistry()


def test_replay_capture_and_get_session(replay):
    replay.capture("t1", kind="llm_call", input={"msg": "hi"}, model="gpt-4o",
                   output={"reply": "hello"})
    replay.capture("t1", kind="tool_call:search", input={"q": "weather"})
    session = replay.get_session("t1")
    assert len(session.invocations) == 2
    assert session.invocations[0].kind == "llm_call"
    assert session.invocations[1].kind == "tool_call:search"
    assert session.invocations[0].output == {"reply": "hello"}
    assert session.invocations[1].output is None


def test_replay_run_lifecycle(replay):
    replay.capture("t1", kind="llm_call", input={"msg": "hi"}, output={"reply": "first"})
    replay.capture("t1", kind="llm_call", input={"msg": "again"}, output={"reply": "second"})
    run_id = replay.start("t1", mode="live")
    inv = replay.step(run_id)
    assert inv is not None
    replay.record_output(run_id, invocation_idx=0, output={"reply": "new"})
    inv = replay.step(run_id)
    assert inv is not None
    replay.record_output(run_id, invocation_idx=1, output={"reply": "new2"})
    # Exhaust the session.
    assert replay.step(run_id) is None
    replay.complete(run_id)
    run = replay.get_run(run_id)
    assert run.status == "completed"


def test_replay_set_replay_trace_id(replay):
    replay.capture("t1", kind="llm_call", input={"msg": "hi"}, output={"reply": "x"})
    run_id = replay.start("t1", mode="live")
    replay.set_replay_trace_id(run_id, "t2")
    assert replay.get_run(run_id).replay_trace_id == "t2"


def test_replay_list_runs_filter(replay):
    replay.capture("t1", kind="llm_call", input={"a": 1})
    replay.capture("t2", kind="llm_call", input={"a": 2})
    r1 = replay.start("t1", mode="live")
    r2 = replay.start("t2", mode="live")
    runs_t1 = replay.list_runs(source_trace_id="t1")
    assert any(r.id == r1 for r in runs_t1)
    assert not any(r.id == r2 for r in runs_t1)


def test_replay_redb_round_trip(tmp_path):
    db = str(tmp_path / "replay.redb")
    r1 = duxxdb.ReplayRegistry(storage=f"redb:{db}")
    r1.capture("t1", kind="llm_call", input={"msg": "hi"}, output={"reply": "x"})
    run_id = r1.start("t1", mode="live")
    r1.record_output(run_id, invocation_idx=0, output={"reply": "y"})
    r1.complete(run_id)
    del r1

    r2 = duxxdb.ReplayRegistry(storage=f"redb:{db}")
    session = r2.get_session("t1")
    assert session is not None
    assert len(session.invocations) == 1
    run = r2.get_run(run_id)
    assert run.status == "completed"


# ---------------------------------------------------------------- TraceStore


@pytest.fixture
def trace() -> duxxdb.TraceStore:
    return duxxdb.TraceStore()


def test_trace_record_and_get(trace):
    trace.record_span(
        trace_id="t1",
        span_id="root",
        name="agent.turn",
        attributes={"user": "alice"},
        start_unix_ns=1_000_000_000,
        status="ok",
    )
    trace.record_span(
        trace_id="t1",
        span_id="child",
        parent_span_id="root",
        name="llm.call",
        attributes={"model": "gpt-4o"},
        start_unix_ns=1_500_000_000,
    )
    spans = trace.get_trace("t1")
    names = {s.name for s in spans}
    assert {"agent.turn", "llm.call"}.issubset(names)
    # attributes round-trip via JSON
    root = next(s for s in spans if s.name == "agent.turn")
    assert root.attributes == {"user": "alice"}
    assert root.status == "ok"


def test_trace_close_span(trace):
    trace.record_span(
        trace_id="t1",
        span_id="s1",
        name="x",
        start_unix_ns=1_000_000_000,
        status="unset",
    )
    trace.close_span("s1", end_unix_ns=2_000_000_000, status="ok")
    spans = trace.get_trace("t1")
    assert spans[0].end_unix_ns == 2_000_000_000
    assert spans[0].status == "ok"


def test_trace_thread_groups(trace):
    trace.record_span(
        trace_id="t1",
        span_id="s1",
        name="x",
        thread_id="user-42",
        start_unix_ns=1_000_000_000,
    )
    trace.record_span(
        trace_id="t2",
        span_id="s2",
        name="y",
        thread_id="user-42",
        start_unix_ns=1_500_000_000,
    )
    grouped = trace.thread("user-42")
    trace_ids = {s.trace_id for s in grouped}
    assert trace_ids == {"t1", "t2"}


def test_trace_redb_round_trip(tmp_path):
    db = str(tmp_path / "trace.redb")
    t1 = duxxdb.TraceStore(storage=f"redb:{db}")
    t1.record_span(
        trace_id="t1",
        span_id="s1",
        name="agent.turn",
        attributes={"user": "alice"},
        start_unix_ns=1_000_000_000,
    )
    t1.close_span("s1", end_unix_ns=2_000_000_000, status="ok")
    del t1

    t2 = duxxdb.TraceStore(storage=f"redb:{db}")
    spans = t2.get_trace("t1")
    assert len(spans) == 1
    assert spans[0].end_unix_ns == 2_000_000_000
    assert spans[0].status == "ok"
    assert spans[0].attributes == {"user": "alice"}
