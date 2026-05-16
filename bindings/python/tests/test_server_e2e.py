"""End-to-end tests for ``duxxdb.server.ServerClient`` against a live
``duxx-server`` daemon.

These are gated on ``DUXXDB_E2E=1`` because they:

* spawn ``duxx-server`` as a subprocess
* listen on a real TCP port
* take seconds to run

They exist to catch protocol-level skew between the Python facade and
the Rust server that unit tests with a mock client cannot detect:
argument ordering, optional-arg sentinels, response decoding for every
real command. The unit tests in ``test_server_facade.py`` cover the
encoding side fast; this file is the cross-language integration
safety net.

Run them locally with::

    cargo build -p duxx-server --bin duxx-server --release
    DUXXDB_E2E=1 pytest bindings/python/tests/test_server_e2e.py -v
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

if os.environ.get("DUXXDB_E2E") != "1":
    pytest.skip("Set DUXXDB_E2E=1 to run end-to-end tests", allow_module_level=True)

from duxxdb.server import (  # noqa: E402
    EvalRun,
    EvalSummary,
    Prompt,
    ServerClient,
)


# ----------------------------------------------------------------- helpers


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _wait_for_port(host: str, port: int, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.25):
                return
        except OSError:
            time.sleep(0.05)
    raise RuntimeError(f"duxx-server never opened {host}:{port}")


def _server_binary() -> Path:
    # Prefer release if present (faster startup, smaller memory),
    # fall back to debug.
    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "target" / "release" / "duxx-server.exe",
        repo_root / "target" / "release" / "duxx-server",
        repo_root / "target" / "debug" / "duxx-server.exe",
        repo_root / "target" / "debug" / "duxx-server",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise RuntimeError(
        "duxx-server binary not found. Run: "
        "cargo build -p duxx-server --bin duxx-server"
    )


@pytest.fixture(scope="module")
def server():
    """Spawn one ``duxx-server`` per test module."""
    port = _free_port()
    binary = _server_binary()
    proc = subprocess.Popen(
        [str(binary), "--addr", f"127.0.0.1:{port}", "--embedder", "hash:32"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        _wait_for_port("127.0.0.1", port)
        yield f"redis://localhost:{port}"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture
def client(server):
    sc = ServerClient(url=server)
    try:
        yield sc
    finally:
        sc.close()


# ----------------------------------------------------------------- tests


def test_ping(client):
    assert client.ping() is True


def test_prompt_lifecycle(client):
    v1 = client.prompts.put("classifier", "version one content", metadata={"author": "alice"})
    v2 = client.prompts.put("classifier", "version two content")
    assert v2 == v1 + 1

    latest = client.prompts.get("classifier")
    assert isinstance(latest, Prompt)
    assert latest.version == v2
    assert latest.content == "version two content"

    earlier = client.prompts.get("classifier", v1)
    assert earlier.content == "version one content"

    client.prompts.tag("classifier", v1, "prod")
    by_tag = client.prompts.get("classifier", "prod")
    assert by_tag.version == v1

    names = client.prompts.names()
    assert "classifier" in names

    diff_text = client.prompts.diff("classifier", v1, v2)
    assert isinstance(diff_text, str)
    assert "version" in diff_text


def test_dataset_lifecycle(client):
    client.datasets.create("refunds")
    v1 = client.datasets.add(
        "refunds",
        [
            {"id": "r1", "text": "I want a refund", "split": "train"},
            {"id": "r2", "text": "Please return my money", "split": "train"},
            {"id": "r3", "text": "Where is my package?", "split": "test"},
        ],
    )
    assert v1 >= 1

    size_all = client.datasets.size("refunds", v1)
    assert size_all == 3
    assert client.datasets.size("refunds", v1, split="train") == 2
    assert sorted(client.datasets.splits("refunds", v1)) == ["test", "train"]

    sample = client.datasets.sample("refunds", v1, n=2, split="train")
    assert len(sample) == 2
    assert all(r.split == "train" for r in sample)


def test_eval_full_loop(client):
    client.datasets.create("eval_ds")
    ds_v = client.datasets.add(
        "eval_ds",
        [{"id": f"row-{i}", "text": f"example {i}", "split": "test"} for i in range(5)],
    )
    pv = client.prompts.put("eval_prompt", "you are an evaluator")

    run_id = client.evals.start(
        dataset_name="eval_ds",
        dataset_version=ds_v,
        model="gpt-4o-mini",
        scorer="exact_match",
        prompt_name="eval_prompt",
        prompt_version=pv,
    )
    assert run_id

    for i in range(5):
        score = 1.0 if i < 4 else 0.0  # one failure
        client.evals.score(
            run_id,
            row_id=f"row-{i}",
            score=score,
            output_text=f"output {i}",
        )

    summary = client.evals.complete(run_id)
    assert isinstance(summary, EvalSummary)
    assert summary.total_scored == 5
    assert summary.pass_rate_50 == pytest.approx(0.8)

    run = client.evals.get(run_id)
    assert isinstance(run, EvalRun)
    assert run.status == "completed"
    assert run.summary is not None
    assert run.summary.total_scored == 5

    scores = client.evals.scores(run_id)
    assert len(scores) == 5

    runs = client.evals.list(dataset_name="eval_ds", dataset_version=ds_v)
    assert any(r.id == run_id for r in runs)


def test_cost_lifecycle(client):
    client.cost.record(
        tenant="acme",
        model="gpt-4o",
        tokens_in=100,
        tokens_out=50,
        cost_usd=0.0023,
        input_text="test prompt",
    )
    client.cost.record(
        tenant="acme",
        model="gpt-4o-mini",
        tokens_in=200,
        tokens_out=80,
        cost_usd=0.0011,
    )

    total = client.cost.total("acme")
    assert total == pytest.approx(0.0034, abs=1e-9)

    entries = client.cost.query({"tenant": "acme"})
    assert len(entries) >= 2

    buckets = client.cost.aggregate("model", filter={"tenant": "acme"})
    assert len(buckets) >= 2

    client.cost.set_budget("acme", "monthly", 100.0)
    budget = client.cost.get_budget("acme")
    assert budget is not None
    assert budget.amount_usd == 100.0

    status = client.cost.status("acme")
    assert status in {"ok", "warning", "exceeded", "no_budget"}

    assert client.cost.delete_budget("acme") is True
    assert client.cost.get_budget("acme") is None


def test_trace_record_and_get(client):
    client.trace.record(
        trace_id="t-1",
        span_id="s-root",
        name="agent.turn",
        attributes={"user": "alice"},
        start_unix_ns=1_000_000_000,
        status="ok",
    )
    client.trace.record(
        trace_id="t-1",
        span_id="s-child",
        parent_span_id="s-root",
        name="llm.call",
        attributes={"model": "gpt-4o"},
        start_unix_ns=1_500_000_000,
        end_unix_ns=2_000_000_000,
        status="ok",
    )

    spans = client.trace.get("t-1")
    assert len(spans) >= 2
    names = {s.name for s in spans}
    assert {"agent.turn", "llm.call"}.issubset(names)
