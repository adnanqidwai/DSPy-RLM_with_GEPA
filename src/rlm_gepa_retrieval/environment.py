from __future__ import annotations

import re
from collections import defaultdict

from .retrieval import LexicalRetriever
from .schemas import Passage, RetrievalIntent


class RetrievalEnvironment:
    def __init__(self, passages: list[Passage], max_searches: int = 6) -> None:
        self.passages = passages
        self.max_searches = max_searches
        self._searches_used = 0
        self._retriever = LexicalRetriever(passages)
        self._doc_text = _group_docs(passages)
        self._trace: list[dict] = []

    @property
    def searches_used(self) -> int:
        return self._searches_used

    def search(self, query: str, k: int = 5) -> list[dict]:
        if self._searches_used >= self.max_searches:
            raise RuntimeError(f"search budget exhausted: {self._searches_used}/{self.max_searches}")
        self._searches_used += 1
        intent = RetrievalIntent(
            role="rlm_search",
            query=query,
            rationale="RLM-requested lexical search",
            expected_evidence="supporting passages",
        )
        hits = self._retriever.search(intent, top_k=k)
        payload = [
            {
                "passage_id": hit.passage.passage_id,
                "doc_id": hit.passage.doc_id,
                "source": hit.passage.source,
                "rank": hit.rank,
                "score": hit.score,
                "text": hit.passage.text,
            }
            for hit in hits
        ]
        self._trace.append({"tool": "search", "query": query, "k": k, "results": payload})
        return payload

    def open_doc(self, doc_id: str) -> str:
        text = self._doc_text.get(doc_id, "")
        self._trace.append({"tool": "open_doc", "doc_id": doc_id, "found": bool(text)})
        return text

    def find_in_doc(self, doc_id: str, pattern: str) -> list[dict]:
        text = self._doc_text.get(doc_id, "")
        matches: list[dict] = []
        error: str | None = None
        if text:
            if len(pattern) > 200:
                error = "pattern too long"
            else:
                try:
                    regex = re.compile(pattern, re.IGNORECASE)
                except re.error as exc:
                    error = f"invalid regex: {exc}"
                else:
                    for match in regex.finditer(text):
                        start = max(match.start() - 120, 0)
                        end = min(match.end() + 120, len(text))
                        matches.append({"doc_id": doc_id, "pattern": pattern, "snippet": text[start:end]})
                        if len(matches) >= 20:
                            break
        trace_step = {"tool": "find_in_doc", "doc_id": doc_id, "pattern": pattern, "matches": matches}
        if error:
            trace_step["error"] = error
        self._trace.append(trace_step)
        return matches

    def trace(self) -> list[dict]:
        return list(self._trace)


def _group_docs(passages: list[Passage]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for passage in passages:
        grouped[passage.doc_id].append(passage.text)
    return {doc_id: "\n\n".join(parts) for doc_id, parts in grouped.items()}
