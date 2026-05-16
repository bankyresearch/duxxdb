"""Tests for the embedded ``duxxdb.PromptRegistry`` Python class.

Exercises both the in-memory and the redb-backed storage modes,
plus the round-trip behavior critical to the durability contract.
"""

from __future__ import annotations

import json
import os
import tempfile

import pytest

import duxxdb


@pytest.fixture
def registry() -> duxxdb.PromptRegistry:
    return duxxdb.PromptRegistry(dim=16)


# ---------------------------------------------------------------- basics


def test_put_returns_monotonic_versions(registry):
    v1 = registry.put("classifier", "first")
    v2 = registry.put("classifier", "second")
    v3 = registry.put("classifier", "third")
    assert (v1, v2, v3) == (1, 2, 3)


def test_put_under_different_names_does_not_share_versions(registry):
    a1 = registry.put("classifier", "x")
    b1 = registry.put("greeter", "y")
    assert (a1, b1) == (1, 1)


def test_get_returns_latest_when_no_version_or_tag(registry):
    registry.put("c", "v1")
    registry.put("c", "v2")
    p = registry.get("c")
    assert p.version == 2
    assert p.content == "v2"


def test_get_by_int_version(registry):
    registry.put("c", "v1")
    registry.put("c", "v2")
    p = registry.get("c", 1)
    assert p.content == "v1"


def test_get_by_tag(registry):
    registry.put("c", "v1")
    v2 = registry.put("c", "v2")
    registry.tag("c", v2, "prod")
    p = registry.get("c", "prod")
    assert p.version == v2
    assert p.tags == ["prod"]


def test_get_returns_none_for_unknown(registry):
    registry.put("c", "v1")
    assert registry.get("missing") is None
    assert registry.get("c", 99) is None
    assert registry.get("c", "nonexistent_tag") is None


def test_metadata_round_trips_dict(registry):
    registry.put("c", "x", metadata={"author": "alice", "tokens": 42, "nested": {"k": [1, 2]}})
    p = registry.get("c")
    assert p.metadata == {"author": "alice", "tokens": 42, "nested": {"k": [1, 2]}}


def test_metadata_defaults_to_null(registry):
    registry.put("c", "x")
    p = registry.get("c")
    assert p.metadata is None


# ---------------------------------------------------------------- tags


def test_tag_moves_atomically(registry):
    registry.put("c", "v1")
    v2 = registry.put("c", "v2")
    registry.tag("c", 1, "prod")
    assert registry.get("c", "prod").version == 1
    # Re-tag: prod jumps from v1 to v2.
    registry.tag("c", v2, "prod")
    assert registry.get("c", "prod").version == 2


def test_untag_returns_true_only_when_tag_existed(registry):
    registry.put("c", "v1")
    registry.tag("c", 1, "prod")
    assert registry.untag("c", "prod") is True
    assert registry.untag("c", "prod") is False
    assert registry.get("c", "prod") is None


def test_tag_unknown_version_raises(registry):
    registry.put("c", "v1")
    with pytest.raises(RuntimeError):
        registry.tag("c", 99, "prod")


# ---------------------------------------------------------------- delete


def test_delete_removes_version_and_associated_tags(registry):
    registry.put("c", "v1")
    v2 = registry.put("c", "v2")
    registry.tag("c", v2, "prod")
    assert registry.delete("c", v2) is True
    assert registry.get("c", v2) is None
    assert registry.get("c", "prod") is None
    # Counter must not reuse v2.
    v3 = registry.put("c", "v3")
    assert v3 == 3


def test_delete_returns_false_for_unknown(registry):
    registry.put("c", "v1")
    assert registry.delete("c", 99) is False


# ---------------------------------------------------------------- search


def test_search_finds_semantically_close_prompts(registry):
    registry.put("greeting", "hello world how are you")
    registry.put("farewell", "goodbye see you tomorrow")
    registry.put("support", "i can help with your issue")
    hits = registry.search("hello", k=2)
    assert len(hits) >= 1
    assert hits[0].prompt.content.lower().startswith("hello")
    assert 0.0 <= hits[0].score <= 1.0


def test_search_with_empty_registry_returns_empty(registry):
    hits = registry.search("anything", k=5)
    assert hits == []


# ---------------------------------------------------------------- diff


def test_diff_marks_added_and_removed_lines(registry):
    registry.put("c", "line a\nshared\nline c")
    registry.put("c", "line a\nshared\nNEW line")
    d = registry.diff("c", 1, 2)
    assert "-line c" in d
    assert "+NEW line" in d


# ---------------------------------------------------------------- repr / introspection


def test_registry_repr_includes_counts(registry):
    registry.put("a", "x")
    registry.put("b", "y")
    registry.tag("a", 1, "prod")
    text = repr(registry)
    assert "names=2" in text
    assert "versions=2" in text
    assert "tags=1" in text


def test_prompt_repr_is_informative(registry):
    registry.put("c", "hello there")
    p = registry.get("c")
    text = repr(p)
    assert "name=" in text and "version=" in text


def test_dim_attribute(registry):
    assert registry.dim == 16


# ---------------------------------------------------------------- persistence


@pytest.fixture
def redb_path(tmp_path) -> str:
    return str(tmp_path / "prompts.redb")


def test_redb_round_trip(redb_path):
    r = duxxdb.PromptRegistry(dim=16, storage=f"redb:{redb_path}")
    r.put("c", "v1", metadata={"author": "alice"})
    v2 = r.put("c", "v2", metadata={"author": "bob"})
    r.tag("c", v2, "prod")
    del r

    r2 = duxxdb.PromptRegistry(dim=16, storage=f"redb:{redb_path}")
    p1 = r2.get("c", 1)
    assert p1.content == "v1"
    assert p1.metadata == {"author": "alice"}

    prod = r2.get("c", "prod")
    assert prod.version == 2
    assert prod.metadata == {"author": "bob"}
    assert prod.tags == ["prod"]


def test_redb_preserves_monotonic_counter_across_reopen(redb_path):
    r = duxxdb.PromptRegistry(dim=16, storage=f"redb:{redb_path}")
    r.put("c", "v1")
    r.put("c", "v2")
    r.delete("c", 1)
    del r

    r2 = duxxdb.PromptRegistry(dim=16, storage=f"redb:{redb_path}")
    # v1 stays deleted.
    assert r2.get("c", 1) is None
    # Next put MUST be v3, not v1 reused.
    next_v = r2.put("c", "v3")
    assert next_v == 3


def test_redb_preserves_search_index_across_reopen(redb_path):
    r = duxxdb.PromptRegistry(dim=16, storage=f"redb:{redb_path}")
    r.put("greeting", "hello world how are you")
    r.put("farewell", "goodbye see you tomorrow")
    del r

    r2 = duxxdb.PromptRegistry(dim=16, storage=f"redb:{redb_path}")
    hits = r2.search("hello", k=2)
    assert hits, "search returned nothing after reopen"
    assert hits[0].prompt.content.lower().startswith("hello")


def test_memory_backend_explicit(tmp_path):
    # "memory" spec is equivalent to no spec — both produce a fresh
    # in-memory registry.
    r1 = duxxdb.PromptRegistry(dim=16, storage="memory")
    r1.put("c", "x")
    assert r1.names() == ["c"]

    r2 = duxxdb.PromptRegistry(dim=16, storage="memory")
    assert r2.names() == []  # separate process-local instances


def test_unknown_storage_spec_raises():
    with pytest.raises(RuntimeError):
        duxxdb.PromptRegistry(dim=16, storage="rocksdb:./nope")
