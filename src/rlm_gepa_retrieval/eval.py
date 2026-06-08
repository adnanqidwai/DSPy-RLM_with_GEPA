from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .metrics import score_retrieval
from .programs import build_program, passages_to_corpus_json, run_program
from .retrieval import load_markdown_corpus
from .tasks import RetrievalExample


def load_questions(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _examples_from_questions(path: Path) -> list[RetrievalExample]:
    rows = load_questions(path)
    if rows and "example_id" in rows[0]:
        return [RetrievalExample.from_record(row) for row in rows]
    return [
        RetrievalExample(
            example_id=f"manual.{idx:03d}",
            split="test",
            family="manual",
            question=row["question"],
            expected_answer=" ".join(row.get("required_terms", [])),
            required_terms=[str(item) for item in row.get("required_terms", [])],
            required_doc_ids=[str(item) for item in row.get("support_doc_ids", [])],
            required_passage_ids=[
                str(item)
                for item in row.get(
                    "required_passage_ids",
                    [f"{doc_id}#p1" for doc_id in row.get("support_doc_ids", [])],
                )
            ],
        )
        for idx, row in enumerate(rows)
    ]


def evaluate(
    corpus_dir: Path,
    questions_path: Path,
    *,
    program_name: str = "heuristic",
    model: str | None = None,
    api_base: str | None = None,
    artifact: str | None = None,
) -> dict[str, Any]:
    passages = load_markdown_corpus(corpus_dir)
    examples = _examples_from_questions(questions_path)
    validate_examples_against_corpus(examples, passages)
    corpus_json = passages_to_corpus_json(passages)
    program = build_program(
        program_name,
        passages=passages,
        model=model,
        api_base=api_base,
        artifact=artifact,
    )
    scored = []
    for example in examples:
        pred = run_program(program, example, corpus_json)
        metric = score_retrieval(example, pred)
        scored.append(
            {
                "example": example.to_record(),
                "prediction": {
                    "answer": pred.answer,
                    "citation_ids": pred.citation_ids,
                    "trace": pred.trace,
                },
                "score": metric.score,
                "components": metric.components,
                "feedback": metric.feedback,
            }
        )
    mean_score = sum(item["score"] for item in scored) / max(len(scored), 1)
    component_names = ["answer_correctness", "evidence_recall", "citation_precision", "budget_efficiency"]
    return {
        "program": program_name,
        "model": model,
        "num_questions": len(scored),
        "mean_score": mean_score,
        "mean_components": {
            name: sum(item["components"][name] for item in scored) / max(len(scored), 1)
            for name in component_names
        },
        "examples": scored,
    }


def validate_examples_against_corpus(examples: list[RetrievalExample], passages: list[Any]) -> None:
    passage_ids = {passage.passage_id for passage in passages}
    missing: list[str] = []
    for example in examples:
        for passage_id in example.required_passage_ids:
            if passage_id not in passage_ids:
                missing.append(f"{example.example_id}:{passage_id}")
    if missing:
        preview = ", ".join(missing[:8])
        suffix = "" if len(missing) <= 8 else f", ... ({len(missing)} total)"
        raise ValueError(f"Questions reference passage IDs not present in corpus: {preview}{suffix}")
