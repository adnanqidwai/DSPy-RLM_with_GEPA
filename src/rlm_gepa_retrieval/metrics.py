from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .retrieval import tokenize
from .tasks import RetrievalExample


WEIGHTS = {
    "answer_correctness": 0.35,
    "evidence_recall": 0.35,
    "citation_precision": 0.15,
    "budget_efficiency": 0.15,
}


@dataclass(frozen=True)
class RetrievalPrediction:
    answer: str
    citation_ids: list[str]
    trace: list[dict] = field(default_factory=list)

    @classmethod
    def from_obj(cls, obj: Any) -> "RetrievalPrediction":
        if isinstance(obj, cls):
            return obj
        if isinstance(obj, dict):
            answer = obj.get("answer", "")
            citation_ids = obj.get("citation_ids", obj.get("citations", []))
            trace = obj.get("trace", [])
        else:
            answer = getattr(obj, "answer", "")
            citation_ids = getattr(obj, "citation_ids", getattr(obj, "citations", []))
            trace = getattr(obj, "trace", [])
        if isinstance(citation_ids, str):
            try:
                citation_ids = json.loads(citation_ids)
            except json.JSONDecodeError:
                citation_ids = [part.strip() for part in citation_ids.split(",") if part.strip()]
        if isinstance(trace, str):
            try:
                trace = json.loads(trace)
            except json.JSONDecodeError:
                trace = [{"raw_trace": trace}]
        return cls(answer=str(answer), citation_ids=[str(item) for item in citation_ids], trace=_normalize_trace(trace))


@dataclass(frozen=True)
class MetricResult:
    score: float
    feedback: str
    components: dict[str, float]

    def to_record(self) -> dict:
        return {"score": self.score, "feedback": self.feedback, "components": self.components}


def score_retrieval(example: RetrievalExample, pred: Any) -> MetricResult:
    prediction = RetrievalPrediction.from_obj(pred)
    answer_terms = set(tokenize(prediction.answer))
    required_terms = {term.lower() for term in example.required_terms}
    required_passages = set(example.required_passage_ids)
    cited = set(prediction.citation_ids)

    answer_correctness = len(answer_terms & required_terms) / max(len(required_terms), 1)
    evidence_recall = len(cited & required_passages) / max(len(required_passages), 1)
    citation_precision = len(cited & required_passages) / max(len(cited), 1) if cited else 0.0
    tool_call_count = _retrieval_tool_call_count(prediction.trace)
    if not prediction.trace or tool_call_count == 0:
        budget_efficiency = 0.0
    elif tool_call_count <= example.max_searches:
        budget_efficiency = 1.0
    else:
        budget_efficiency = max(0.0, example.max_searches / tool_call_count)

    components = {
        "answer_correctness": round(answer_correctness, 4),
        "evidence_recall": round(evidence_recall, 4),
        "citation_precision": round(citation_precision, 4),
        "budget_efficiency": round(budget_efficiency, 4),
    }
    score = round(sum(WEIGHTS[name] * value for name, value in components.items()), 4)
    feedback = _feedback(example, prediction, components, tool_call_count)
    return MetricResult(score=score, feedback=feedback, components=components)


def retrieval_gepa_metric(example: Any, pred: Any, trace: Any | None = None, pred_name: str | None = None, pred_trace: Any | None = None) -> Any:
    del trace, pred_name, pred_trace
    retrieval_example = example if isinstance(example, RetrievalExample) else RetrievalExample.from_record(example.toDict() if hasattr(example, "toDict") else dict(example))
    result = score_retrieval(retrieval_example, pred)
    try:
        import dspy  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        return result
    return dspy.Prediction(score=result.score, feedback=result.feedback)


def _feedback(example: RetrievalExample, prediction: RetrievalPrediction, components: dict[str, float], tool_call_count: int) -> str:
    required = set(example.required_passage_ids)
    cited = set(prediction.citation_ids)
    missing = sorted(required - cited)
    extra = sorted(cited - required)
    missing_terms = sorted({term.lower() for term in example.required_terms} - set(tokenize(prediction.answer)))
    parts = [
        f"example_id: {example.example_id}",
        f"family: {example.family}",
        f"score components: {components}",
        f"tool calls used: {tool_call_count}; budget: {example.max_searches}",
        f"expected answer: {example.expected_answer}",
        f"predicted answer: {prediction.answer}",
    ]
    if missing:
        parts.append(f"missed support passage IDs: {missing}")
    if extra:
        parts.append(f"unsupported or unnecessary citation IDs: {extra}")
    if missing_terms:
        parts.append(f"missing required answer terms: {missing_terms}")
    if prediction.trace:
        parts.append(f"trace: {json.dumps(prediction.trace, ensure_ascii=False)[:3000]}")
    else:
        parts.append("missing execution trace; budget efficiency scored as 0.0")
    return "\n".join(parts)


def _retrieval_tool_call_count(trace: list[dict]) -> int:
    metered_tools = {"search", "open_doc", "find_in_doc"}
    return sum(1 for step in trace if step.get("tool") in metered_tools)


def _normalize_trace(trace: Any) -> list[dict]:
    if trace is None:
        return []
    if isinstance(trace, dict):
        trace = [trace]
    if isinstance(trace, (str, bytes)):
        trace = [{"raw_trace": trace.decode() if isinstance(trace, bytes) else trace}]
    normalized = []
    try:
        iterator = iter(trace)
    except TypeError:
        iterator = iter([trace])
    for step in iterator:
        if isinstance(step, dict):
            normalized.append({str(key): value for key, value in step.items()})
        else:
            normalized.append({"raw_trace": str(step)})
    return normalized
