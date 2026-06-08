from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict

from .schemas import EvidenceItem, RetrievalIntent


class PlannerAgent:
    def plan(self, question: str) -> list[RetrievalIntent]:
        return [
            RetrievalIntent(
                role="literal",
                query=question,
                rationale="Retrieve direct matches for the user's wording.",
                expected_evidence="definitions and direct support",
            ),
            RetrievalIntent(
                role="hyde",
                query=(
                    f"An answer to '{question}' would discuss agentic retrieval, evidence, citations, "
                    "planning, synthesis, MindSearch, STORM, Search-o1, WebThinker, RAGentA, "
                    "MASS-RAG, MemSearch-o1, and orchestrator worker systems."
                ),
                rationale="Use a hypothetical answer to bridge vocabulary mismatch.",
                expected_evidence="semantically related explanatory passages",
            ),
            RetrievalIntent(
                role="skeptic",
                query=f"{question} limitations failure risks hallucination faithfulness unsupported claims",
                rationale="Find constraints and failure modes before synthesis.",
                expected_evidence="risks, caveats, and verification needs",
            ),
            RetrievalIntent(
                role="implementation",
                query=f"{question} architecture implementation modules agents ledger fusion retriever",
                rationale="Find runnable-system and architecture details.",
                expected_evidence="implementation mechanisms",
            ),
            RetrievalIntent(
                role="evaluation",
                query=f"{question} evaluation metrics recall citation coverage support verification",
                rationale="Find measurable success criteria.",
                expected_evidence="metrics and evaluation methods",
            ),
        ]


class OpenAICompatibleClient:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self.model = model
        self.base_url = _validate_http_base_url(base_url or os.environ.get("OPENAI_BASE_URL") or "https://api.openai.com/v1")
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self.extra_headers = extra_headers or {}
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str, timeout: int = 60) -> str:
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for --use-api")
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            # Base URL is validated as http(s) during client initialization.
            with urllib.request.urlopen(request, timeout=timeout) as response:  # nosec B310
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"API request failed: {exc.code} {detail}") from exc
        content = body["choices"][0]["message"].get("content")
        if content is None:
            raise RuntimeError("API response did not include message.content; try a larger max token budget or a non-reasoning model.")
        return content


class SynthesisAgent:
    def __init__(self, client: OpenAICompatibleClient | None = None) -> None:
        self.client = client

    def synthesize(self, question: str, evidence: list[EvidenceItem]) -> str:
        if self.client:
            return self._api_synthesize(question, evidence)
        return self._extractive_synthesize(question, evidence)

    def _extractive_synthesize(self, question: str, evidence: list[EvidenceItem]) -> str:
        if not evidence:
            return "No supporting evidence was retrieved."
        lead = (
            "RLM-GEPA Retrieval uses a planner and worker-style retrieval loop: it creates "
            "role-specific intents, fuses passages, stores an evidence ledger, and answers "
            "with citations plus verification signals."
        )
        inspiration_terms = _present_terms(
            " ".join(item.text for item in evidence),
            ["MindSearch", "STORM", "Search-o1", "WebThinker", "RAGentA", "MASS-RAG", "MemSearch-o1", "Anthropic"],
        )
        if inspiration_terms:
            lead += f" Relevant recent inspirations in the retrieved evidence include {', '.join(inspiration_terms)}."
        bullets = []
        for item in evidence[:4]:
            sentence = _representative_sentence(item.text)
            if sentence and not sentence.endswith("."):
                sentence += "."
            bullets.append(f"{sentence} [{item.citation}]")
        return lead + "\n\n" + "\n".join(f"- {bullet}" for bullet in bullets)

    def _api_synthesize(self, question: str, evidence: list[EvidenceItem]) -> str:
        system = (
            "You answer only from the provided evidence ledger. "
            "Cite every factual claim with the citation labels. "
            "If evidence is insufficient, say what is missing."
        )
        user = json.dumps(
            {
                "question": question,
                "evidence": [asdict(item) for item in evidence],
            },
            indent=2,
        )
        return self.client.complete(system=system, user=user)


def _representative_sentence(text: str) -> str:
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
    for sentence in sentences:
        if len(sentence.split()) >= 8:
            return sentence
    return sentences[0] if sentences else text.strip()


def _present_terms(text: str, terms: list[str]) -> list[str]:
    lowered = text.lower()
    return [term for term in terms if term.lower() in lowered]


def _validate_http_base_url(base_url: str) -> str:
    normalized = base_url.rstrip("/")
    parsed = urllib.parse.urlparse(normalized)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("OpenAI-compatible base URL must be an http(s) URL")
    return normalized
