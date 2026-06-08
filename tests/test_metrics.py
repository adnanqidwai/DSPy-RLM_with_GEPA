from __future__ import annotations

from rlm_gepa_retrieval.metrics import RetrievalPrediction, score_retrieval
from rlm_gepa_retrieval.tasks import RetrievalExample


def example() -> RetrievalExample:
    return RetrievalExample(
        example_id="single_hop.0000.000",
        split="test",
        family="single_hop",
        question="What stores evidence?",
        expected_answer="The evidence ledger stores provenance.",
        required_terms=["evidence", "ledger", "provenance"],
        required_doc_ids=["evidence_ledger"],
        required_passage_ids=["evidence_ledger#p1"],
        max_searches=4,
    )


def test_score_rewards_answer_evidence_precision_and_budget() -> None:
    pred = RetrievalPrediction(
        answer="The evidence ledger stores provenance.",
        citation_ids=["evidence_ledger#p1"],
        trace=[{"tool": "search"}, {"tool": "search"}],
    )

    result = score_retrieval(example(), pred)

    assert result.score == 1.0
    assert result.components["answer_correctness"] == 1.0
    assert result.components["evidence_recall"] == 1.0
    assert result.components["citation_precision"] == 1.0
    assert result.components["budget_efficiency"] == 1.0


def test_budget_efficiency_counts_all_retrieval_tools() -> None:
    pred = RetrievalPrediction(
        answer="The evidence ledger stores provenance.",
        citation_ids=["evidence_ledger#p1"],
        trace=[
            {"tool": "search"},
            {"tool": "open_doc"},
            {"tool": "find_in_doc"},
            {"tool": "find_in_doc"},
        ],
    )

    result = score_retrieval(example(), pred)

    assert result.components["budget_efficiency"] == 1.0
    assert "tool calls used: 4; budget: 4" in result.feedback


def test_budget_efficiency_penalizes_open_and_find_over_budget() -> None:
    pred = RetrievalPrediction(
        answer="The evidence ledger stores provenance.",
        citation_ids=["evidence_ledger#p1"],
        trace=[
            {"tool": "search"},
            {"tool": "open_doc"},
            {"tool": "find_in_doc"},
            {"tool": "find_in_doc"},
            {"tool": "find_in_doc"},
            {"tool": "find_in_doc"},
            {"tool": "find_in_doc"},
            {"tool": "find_in_doc"},
        ],
    )

    result = score_retrieval(example(), pred)

    assert result.components["budget_efficiency"] == 0.5
    assert result.score == 0.925
    assert "tool calls used: 8; budget: 4" in result.feedback


def test_budget_efficiency_requires_metered_retrieval_tools() -> None:
    pred = RetrievalPrediction(
        answer="The evidence ledger stores provenance.",
        citation_ids=["evidence_ledger#p1"],
        trace=[{"tool": "model_claimed_search"}],
    )

    result = score_retrieval(example(), pred)

    assert result.components["budget_efficiency"] == 0.0
    assert "tool calls used: 0; budget: 4" in result.feedback


def test_feedback_names_missed_and_wrong_evidence() -> None:
    pred = RetrievalPrediction(answer="The answer mentions evidence only.", citation_ids=["wrong#p1"], trace=[])

    result = score_retrieval(example(), pred)

    assert result.score < 0.5
    assert "missed support passage IDs" in result.feedback
    assert "wrong#p1" in result.feedback
    assert result.components["budget_efficiency"] == 0.0
    assert "missing execution trace" in result.feedback


def test_prediction_from_obj_normalizes_non_dict_trace_steps() -> None:
    pred = RetrievalPrediction.from_obj({"answer": "x", "citation_ids": [], "trace": ["search evidence"]})

    assert pred.trace == [{"raw_trace": "search evidence"}]
