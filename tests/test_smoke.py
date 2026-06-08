from pathlib import Path

from rlm_gepa_retrieval.eval import evaluate
from rlm_gepa_retrieval.pipeline import answer_question
from rlm_gepa_retrieval.retrieval import load_markdown_corpus


ROOT = Path(__file__).resolve().parents[1]


def test_answer_returns_evidence_with_citations() -> None:
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")
    result = answer_question("How does RLM-GEPA Retrieval make retrieval auditable?", passages)

    assert result.evidence
    assert result.evidence[0].citation.startswith("E")
    assert "[" in result.answer
    assert all(not item.text.startswith("#") for item in result.evidence)


def test_demo_eval_has_support_signal() -> None:
    metrics = evaluate(
        ROOT / "examples" / "demo_corpus",
        ROOT / "examples" / "demo_questions.jsonl",
        program_name="heuristic",
    )

    assert metrics["num_questions"] == 3
    assert metrics["mean_score"] > 0.0
    assert metrics["mean_components"]["evidence_recall"] >= 0.0
