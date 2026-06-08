from __future__ import annotations

from collections import defaultdict

from .agents import OpenAICompatibleClient, PlannerAgent, SynthesisAgent
from .retrieval import LexicalRetriever, tokenize
from .schemas import AnswerResult, EvidenceItem, Passage, RetrievalHit


def answer_question(
    question: str,
    passages: list[Passage],
    top_k_per_agent: int = 5,
    evidence_k: int = 8,
    model: str | None = None,
    api_base: str | None = None,
    use_api: bool = False,
) -> AnswerResult:
    planner = PlannerAgent()
    intents = planner.plan(question)
    retriever = LexicalRetriever(passages)

    hits: list[RetrievalHit] = []
    for intent in intents:
        hits.extend(retriever.search(intent, top_k=top_k_per_agent))

    evidence = fuse_hits(question, hits, evidence_k=evidence_k)
    client = OpenAICompatibleClient(model=model, base_url=api_base) if use_api and model else None
    synthesis = SynthesisAgent(client=client)
    answer = synthesis.synthesize(question, evidence)
    return AnswerResult(question=question, answer=answer, intents=intents, evidence=evidence)


def fuse_hits(question: str, hits: list[RetrievalHit], evidence_k: int = 8) -> list[EvidenceItem]:
    by_passage: dict[str, EvidenceItem] = {}
    role_counts: dict[str, set[str]] = defaultdict(set)
    question_tokens = set(tokenize(question))

    for hit in hits:
        item = by_passage.get(hit.passage.passage_id)
        if item is None:
            item = EvidenceItem(
                citation=f"E{len(by_passage) + 1}",
                passage_id=hit.passage.passage_id,
                doc_id=hit.passage.doc_id,
                source=hit.passage.source,
                text=hit.passage.text,
            )
            by_passage[hit.passage.passage_id] = item
        item.roles.append(hit.intent.role)
        item.best_rank = min(item.best_rank, hit.rank)
        item.fused_score += 1.0 / (60 + hit.rank)
        role_counts[hit.passage.passage_id].add(hit.intent.role)

    for passage_id, item in by_passage.items():
        passage_tokens = set(tokenize(item.text))
        item.support_tokens = sorted(question_tokens & passage_tokens)
        item.fused_score += 0.04 * len(role_counts[passage_id])
        item.fused_score += 0.01 * min(len(item.support_tokens), 10)
        item.roles = sorted(set(item.roles))

    ranked = sorted(
        by_passage.values(),
        key=lambda item: (item.fused_score, -item.best_rank, len(item.roles)),
        reverse=True,
    )
    for idx, item in enumerate(ranked[:evidence_k], start=1):
        item.citation = f"E{idx}"
    return ranked[:evidence_k]
