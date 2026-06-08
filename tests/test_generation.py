from __future__ import annotations

from pathlib import Path

import pytest

from rlm_gepa_retrieval.eval import validate_examples_against_corpus
from rlm_gepa_retrieval.retrieval import load_markdown_corpus
from rlm_gepa_retrieval.tasks import generate_examples, load_examples, save_examples


ROOT = Path(__file__).resolve().parents[1]


def _content_signature(example) -> tuple:
    return (
        example.family,
        example.question,
        example.expected_answer,
        tuple(example.required_doc_ids),
        tuple(example.required_passage_ids),
    )


def test_generation_is_deterministic() -> None:
    first = generate_examples(n=12, seed=123)
    second = generate_examples(n=12, seed=123)

    assert [item.to_record() for item in first] == [item.to_record() for item in second]
    assert {item.family for item in first} >= {"single_hop", "multi_hop_bridge", "conflict_resolution"}


def test_jsonl_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    examples = generate_examples(n=5, seed=4)

    save_examples(path, examples)
    loaded = load_examples(path)

    assert [item.example_id for item in loaded] == [item.example_id for item in examples]
    assert all(item.required_passage_ids for item in loaded)


def test_generated_examples_reference_existing_demo_passages() -> None:
    examples = generate_examples(n=12, seed=4)
    passages = load_markdown_corpus(ROOT / "examples" / "demo_corpus")

    validate_examples_against_corpus(examples, passages)


def test_generated_examples_have_unique_content_signatures() -> None:
    examples = generate_examples(n=24, seed=7)

    signatures = [_content_signature(example) for example in examples]

    assert len(signatures) == len(set(signatures))


def test_generated_splits_do_not_share_gold_passages() -> None:
    examples = generate_examples(n=24, seed=7)
    passages_by_split: dict[str, set[str]] = {}
    for example in examples:
        passages_by_split.setdefault(example.split, set()).update(example.required_passage_ids)

    assert set(passages_by_split) == {"train", "dev", "test"}
    assert passages_by_split["train"].isdisjoint(passages_by_split["dev"])
    assert passages_by_split["train"].isdisjoint(passages_by_split["test"])
    assert passages_by_split["dev"].isdisjoint(passages_by_split["test"])


def test_generation_rejects_duplicate_task_bank_cycles() -> None:
    with pytest.raises(ValueError, match="Bundled task bank contains 24 examples"):
        generate_examples(n=25, seed=7)
