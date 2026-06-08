from __future__ import annotations

import importlib
import json
import re
from typing import Any

from .environment import RetrievalEnvironment
from .metrics import RetrievalPrediction
from .openai_compatible import DEFAULT_OPENAI_COMPATIBLE_MODEL, require_openai_compatible_config
from .retrieval import tokenize
from .schemas import Passage
from .tasks import RetrievalExample


RLM_SIGNATURE = (
    "question: str, corpus_json: str, tool_budget: int -> "
    "answer: str, citation_ids: list[str], trace: list[dict]"
)


def require_dspy() -> Any:
    try:
        return importlib.import_module("dspy")
    except ModuleNotFoundError as exc:
        raise RuntimeError("Install DSPy with `pip install -e .[dspy]` to use RLM or GEPA.") from exc


def make_dspy_lm(
    model: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> Any:
    dspy = require_dspy()
    config = require_openai_compatible_config(
        model or DEFAULT_OPENAI_COMPATIBLE_MODEL,
        api_key=api_key,
        api_base=api_base,
    )
    return dspy.LM(**config.to_dspy_lm_kwargs(max_tokens=max_tokens, temperature=temperature))


def configure_dspy(
    model: str | None = None,
    *,
    api_key: str | None = None,
    api_base: str | None = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
) -> None:
    dspy = require_dspy()
    dspy.configure(
        lm=make_dspy_lm(
            model,
            api_key=api_key,
            api_base=api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    )


class HeuristicRetrievalProgram:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages

    def __call__(self, example: RetrievalExample | Any) -> RetrievalPrediction:
        retrieval_example = _coerce_example(example)
        env = RetrievalEnvironment(self.passages, max_searches=retrieval_example.max_searches)
        query = retrieval_example.question
        hits = env.search(query, k=8)
        citation_ids = [hit["passage_id"] for hit in hits[:4]]
        answer = _extractive_answer(hits)
        return RetrievalPrediction(answer=answer, citation_ids=citation_ids, trace=env.trace())


class SingleShotRAGProgram:
    """Deterministic retrieve-once baseline that mirrors standard RAG control flow."""

    def __init__(self, passages: list[Passage], *, top_k: int = 4) -> None:
        self.passages = passages
        self.top_k = top_k

    def __call__(self, example: RetrievalExample | Any) -> RetrievalPrediction:
        retrieval_example = _coerce_example(example)
        env = RetrievalEnvironment(self.passages, max_searches=retrieval_example.max_searches)
        hits = env.search(retrieval_example.question, k=self.top_k)
        citation_ids = [hit["passage_id"] for hit in hits]
        answer = _single_shot_rag_answer(retrieval_example.question, hits)
        return RetrievalPrediction(answer=answer, citation_ids=citation_ids, trace=env.trace())


def make_rlm_retrieval_program(
    *,
    passages: list[Passage] | None = None,
    max_iterations: int = 8,
    max_llm_calls: int = 12,
    verbose: bool = False,
) -> Any:
    dspy = require_dspy()

    class RLMRetrievalProgram(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.passages = passages or []
            self._active_env: RetrievalEnvironment | None = None
            self.retrieve = dspy.RLM(
                RLM_SIGNATURE,
                max_iterations=max_iterations,
                max_llm_calls=max_llm_calls,
                verbose=verbose,
                tools=[self.search, self.open_doc, self.find_in_doc],
            )

        def forward(self, question: str, corpus_json: str, tool_budget: int = 6) -> Any:
            self._active_env = RetrievalEnvironment(self.passages, max_searches=tool_budget)
            try:
                pred = self.retrieve(question=question, corpus_json=corpus_json, tool_budget=tool_budget)
                normalized = RetrievalPrediction.from_obj(pred)
                return dspy.Prediction(
                    answer=normalized.answer,
                    citation_ids=normalized.citation_ids,
                    trace=self._active_env.trace(),
                    model_reported_trace=normalized.trace,
                )
            finally:
                self._active_env = None

        def search(self, query: str, k: int = 5) -> list[dict]:
            """Search the local corpus for passages relevant to query."""
            return self._require_env().search(query, k=k)

        def open_doc(self, doc_id: str) -> str:
            """Return all available text for a local corpus document ID."""
            return self._require_env().open_doc(doc_id)

        def find_in_doc(self, doc_id: str, pattern: str) -> list[dict]:
            """Find regex snippets inside a local corpus document."""
            return self._require_env().find_in_doc(doc_id, pattern)

        def _require_env(self) -> RetrievalEnvironment:
            if self._active_env is None:
                raise RuntimeError("Retrieval tools can only be used during an RLM forward pass.")
            return self._active_env

    return RLMRetrievalProgram()


def build_program(
    program: str,
    *,
    passages: list[Passage],
    model: str | None = None,
    api_base: str | None = None,
    max_tokens: int = 4000,
    rlm_max_iterations: int = 8,
    rlm_max_llm_calls: int = 12,
    artifact: str | None = None,
) -> Any:
    if program == "heuristic":
        return HeuristicRetrievalProgram(passages)
    if program == "single_shot_rag":
        return SingleShotRAGProgram(passages)
    configure_dspy(model, api_base=api_base, max_tokens=max_tokens)
    if program == "rlm":
        return make_rlm_retrieval_program(passages=passages, max_iterations=rlm_max_iterations, max_llm_calls=rlm_max_llm_calls)
    if program == "optimized":
        if not artifact:
            raise ValueError("--artifact is required for --program optimized")
        rlm = make_rlm_retrieval_program(passages=passages, max_iterations=rlm_max_iterations, max_llm_calls=rlm_max_llm_calls)
        rlm.load(artifact)
        return rlm
    raise ValueError(f"Unsupported program: {program}")


def passages_to_corpus_json(passages: list[Passage]) -> str:
    manifest = [
        {
            "passage_id": passage.passage_id,
            "doc_id": passage.doc_id,
            "source": passage.source,
            "preview": passage.text[:160],
        }
        for passage in passages
    ]
    return json.dumps(manifest, ensure_ascii=False)


def to_dspy_examples(examples: list[RetrievalExample], corpus_json: str) -> list[Any]:
    dspy = require_dspy()
    rows = []
    for example in examples:
        rows.append(
            dspy.Example(
                **example.to_record(),
                corpus_json=corpus_json,
                tool_budget=example.max_searches,
            ).with_inputs("question", "corpus_json", "tool_budget")
        )
    return rows


def run_program(program: Any, example: RetrievalExample, corpus_json: str) -> RetrievalPrediction:
    if isinstance(program, (HeuristicRetrievalProgram, SingleShotRAGProgram)):
        return program(example)
    pred = program(question=example.question, corpus_json=corpus_json, tool_budget=example.max_searches)
    return RetrievalPrediction.from_obj(pred)


def _coerce_example(example: RetrievalExample | Any) -> RetrievalExample:
    if isinstance(example, RetrievalExample):
        return example
    if hasattr(example, "toDict"):
        return RetrievalExample.from_record(example.toDict())
    return RetrievalExample.from_record(dict(example))


def _extractive_answer(hits: list[dict], *, max_chars: int = 700) -> str:
    snippets = [str(hit.get("text", "")).strip() for hit in hits[:2] if hit.get("text")]
    if not snippets:
        return ""
    answer = " ".join(snippets)
    return answer[:max_chars].rstrip()


def _single_shot_rag_answer(question: str, hits: list[dict], *, max_chars: int = 700) -> str:
    question_terms = set(tokenize(question))
    candidates: list[tuple[int, int, str]] = []
    for hit_index, hit in enumerate(hits):
        text = str(hit.get("text", "")).strip()
        for sentence_index, sentence in enumerate(_sentences(text)):
            overlap = len(question_terms & set(tokenize(sentence)))
            candidates.append((overlap, -(hit_index * 100 + sentence_index), sentence))
    if not candidates:
        return _extractive_answer(hits, max_chars=max_chars)
    candidates.sort(reverse=True)
    selected = [sentence for overlap, _, sentence in candidates[:2] if overlap > 0]
    if not selected:
        selected = [candidates[0][2]]
    return " ".join(selected)[:max_chars].rstrip()


def _sentences(text: str) -> list[str]:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    return sentences or ([text] if text else [])
