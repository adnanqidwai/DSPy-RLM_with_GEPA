from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

from .schemas import Passage, RetrievalHit, RetrievalIntent

TOKEN_RE = re.compile(r"[a-zA-Z0-9][a-zA-Z0-9_-]*")


def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_RE.finditer(text)]


def load_markdown_corpus(corpus_dir: Path) -> list[Passage]:
    passages: list[Passage] = []
    for path in sorted(corpus_dir.glob("*.md")):
        doc_id = path.stem
        raw = path.read_text(encoding="utf-8")
        chunks = _markdown_chunks(raw)
        for idx, chunk in enumerate(chunks, start=1):
            passages.append(
                Passage(
                    passage_id=f"{doc_id}#p{idx}",
                    doc_id=doc_id,
                    source=path.name,
                    text=" ".join(chunk.split()),
                )
            )
    return passages


def _markdown_chunks(raw: str) -> list[str]:
    chunks: list[str] = []
    pending_heading: str | None = None
    for chunk in [part.strip() for part in re.split(r"\n\s*\n", raw) if part.strip()]:
        if chunk.startswith("#"):
            pending_heading = chunk.lstrip("#").strip()
            continue
        if pending_heading:
            chunk = f"{pending_heading}. {chunk}"
            pending_heading = None
        chunks.append(chunk)
    return chunks


class LexicalRetriever:
    def __init__(self, passages: list[Passage]) -> None:
        self.passages = passages
        self.doc_tokens = [tokenize(passage.text) for passage in passages]
        self.doc_tf = [Counter(tokens) for tokens in self.doc_tokens]
        self.avg_len = sum(len(tokens) for tokens in self.doc_tokens) / max(len(self.doc_tokens), 1)
        self.df = Counter()
        for tokens in self.doc_tokens:
            self.df.update(set(tokens))

    def search(self, intent: RetrievalIntent, top_k: int = 5) -> list[RetrievalHit]:
        query_terms = tokenize(intent.query)
        scored: list[tuple[float, Passage]] = []
        for passage, tf, tokens in zip(self.passages, self.doc_tf, self.doc_tokens):
            score = self._bm25_score(query_terms, tf, len(tokens))
            if score > 0:
                scored.append((score, passage))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [
            RetrievalHit(intent=intent, passage=passage, rank=rank, score=score)
            for rank, (score, passage) in enumerate(scored[:top_k], start=1)
        ]

    def _bm25_score(self, query_terms: list[str], tf: Counter[str], doc_len: int) -> float:
        k1 = 1.5
        b = 0.75
        total = 0.0
        n_docs = max(len(self.doc_tokens), 1)
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = math.log(1 + (n_docs - self.df[term] + 0.5) / (self.df[term] + 0.5))
            denom = freq + k1 * (1 - b + b * doc_len / max(self.avg_len, 1))
            total += idf * (freq * (k1 + 1) / denom)
        return total
