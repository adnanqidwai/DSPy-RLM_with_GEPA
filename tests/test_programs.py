from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from rlm_gepa_retrieval.programs import (
    HeuristicRetrievalProgram,
    SingleShotRAGProgram,
    build_program,
    make_rlm_retrieval_program,
)
from rlm_gepa_retrieval.retrieval import load_markdown_corpus
from rlm_gepa_retrieval.tasks import generate_examples


ROOT = Path(__file__).resolve().parents[1]


def test_heuristic_program_returns_required_citation_for_easy_task() -> None:
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")
    example = generate_examples(n=1, seed=2)[0]

    pred = HeuristicRetrievalProgram(passages)(example)

    assert pred.answer
    assert set(example.required_passage_ids) & set(pred.citation_ids)
    assert any(step.get("tool") == "search" for step in pred.trace)


def test_build_program_supports_heuristic() -> None:
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")
    program = build_program("heuristic", passages=passages)

    assert isinstance(program, HeuristicRetrievalProgram)


def test_single_shot_rag_baseline_retrieves_once_and_cites_evidence() -> None:
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")
    example = generate_examples(n=1, seed=2)[0]

    pred = SingleShotRAGProgram(passages)(example)

    assert pred.answer
    assert set(example.required_passage_ids) & set(pred.citation_ids)
    assert [step.get("tool") for step in pred.trace] == ["search"]


def test_build_program_supports_single_shot_rag() -> None:
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")
    program = build_program("single_shot_rag", passages=passages)

    assert isinstance(program, SingleShotRAGProgram)


def test_to_dspy_examples_uses_expected_inputs() -> None:
    pytest.importorskip("dspy")
    from rlm_gepa_retrieval.programs import to_dspy_examples

    examples = generate_examples(n=2, seed=2)
    dspy_examples = to_dspy_examples(examples, corpus_json="[]")

    assert len(dspy_examples) == 2
    assert hasattr(dspy_examples[0], "question")
    assert hasattr(dspy_examples[0], "corpus_json")


def test_rlm_program_uses_host_environment_trace(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("dspy")
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")
    program = make_rlm_retrieval_program(passages=passages, max_iterations=1, max_llm_calls=1)

    class FakeRetrieve:
        def __call__(self, **kwargs: object) -> SimpleNamespace:
            del kwargs
            hits = program.search("evidence ledger provenance", k=2)
            return SimpleNamespace(
                answer="The evidence ledger stores provenance.",
                citation_ids=[hits[0]["passage_id"]],
                trace=[{"tool": "model_claimed_search"}],
            )

    monkeypatch.setattr(program, "retrieve", FakeRetrieve())
    pred = program(question="What stores provenance?", corpus_json="[]", tool_budget=2)

    assert pred.trace
    assert pred.trace[0]["tool"] == "search"
    assert pred.model_reported_trace == [{"tool": "model_claimed_search"}]
