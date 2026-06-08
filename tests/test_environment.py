from __future__ import annotations

from pathlib import Path

import pytest

from rlm_gepa_retrieval.environment import RetrievalEnvironment
from rlm_gepa_retrieval.retrieval import load_markdown_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_search_open_find_and_trace() -> None:
    env = RetrievalEnvironment(load_markdown_corpus(ROOT / "examples" / "demo_corpus"), max_searches=2)

    hits = env.search("evidence ledger provenance", k=3)
    assert hits
    assert hits[0]["passage_id"].startswith("evidence_ledger#")

    doc = env.open_doc("evidence_ledger")
    assert "ledger" in doc.lower()

    matches = env.find_in_doc("evidence_ledger", "provenance")
    assert matches

    trace = env.trace()
    assert [step["tool"] for step in trace] == ["search", "open_doc", "find_in_doc"]


def test_search_budget_is_enforced() -> None:
    env = RetrievalEnvironment(load_markdown_corpus(ROOT / "examples" / "demo_corpus"), max_searches=1)
    env.search("agentic retrieval", k=1)

    with pytest.raises(RuntimeError, match="search budget"):
        env.search("evaluation", k=1)


def test_find_in_doc_invalid_regex_is_reported_in_trace() -> None:
    env = RetrievalEnvironment(load_markdown_corpus(ROOT / "examples" / "demo_corpus"), max_searches=1)

    matches = env.find_in_doc("evidence_ledger", "[")

    assert matches == []
    assert env.trace()[-1]["tool"] == "find_in_doc"
    assert "invalid regex" in env.trace()[-1]["error"]
