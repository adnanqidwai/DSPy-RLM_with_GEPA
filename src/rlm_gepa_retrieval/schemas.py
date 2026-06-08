from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class Passage:
    passage_id: str
    doc_id: str
    source: str
    text: str


@dataclass(frozen=True)
class RetrievalIntent:
    role: str
    query: str
    rationale: str
    expected_evidence: str


@dataclass(frozen=True)
class RetrievalHit:
    intent: RetrievalIntent
    passage: Passage
    rank: int
    score: float


@dataclass
class EvidenceItem:
    citation: str
    passage_id: str
    doc_id: str
    source: str
    text: str
    roles: list[str] = field(default_factory=list)
    best_rank: int = 9999
    fused_score: float = 0.0
    support_tokens: list[str] = field(default_factory=list)


@dataclass
class AnswerResult:
    question: str
    answer: str
    intents: list[RetrievalIntent]
    evidence: list[EvidenceItem]

    def to_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.answer,
            "intents": [asdict(intent) for intent in self.intents],
            "evidence": [asdict(item) for item in self.evidence],
        }

