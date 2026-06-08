from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


FAMILIES = (
    "single_hop",
    "multi_hop_bridge",
    "conflict_resolution",
    "acronym_alias",
    "needle_section",
    "aggregation",
)


@dataclass(frozen=True)
class RetrievalExample:
    example_id: str
    split: str
    family: str
    question: str
    expected_answer: str
    required_terms: list[str]
    required_doc_ids: list[str]
    required_passage_ids: list[str]
    max_searches: int = 6
    notes: str = ""

    def to_record(self) -> dict:
        return asdict(self)

    @classmethod
    def from_record(cls, record: dict) -> "RetrievalExample":
        return cls(
            example_id=str(record["example_id"]),
            split=str(record.get("split", "test")),
            family=str(record["family"]),
            question=str(record["question"]),
            expected_answer=str(record["expected_answer"]),
            required_terms=[str(item) for item in record.get("required_terms", [])],
            required_doc_ids=[str(item) for item in record.get("required_doc_ids", [])],
            required_passage_ids=[str(item) for item in record.get("required_passage_ids", [])],
            max_searches=int(record.get("max_searches", 6)),
            notes=str(record.get("notes", "")),
        )


_TASK_TEMPLATES: tuple[dict[str, Any], ...] = (
    {
        "split": "train",
        "family": "single_hop",
        "question": "How does agentic retrieval differ from static RAG?",
        "expected_answer": "Agentic retrieval can plan, call tools, reformulate queries, and inspect intermediate evidence instead of running one static query.",
        "required_terms": ["plan", "tools", "reformulate", "evidence"],
        "required_doc_ids": ["agentic_rag"],
        "required_passage_ids": ["agentic_rag#p1"],
    },
    {
        "split": "dev",
        "family": "aggregation",
        "question": "Which extra metrics complement evidence support recall?",
        "expected_answer": "Additional metrics include cited passage count, unsupported citation count, claim-term coverage, and citation use from the evidence ledger.",
        "required_terms": ["unsupported", "citation", "coverage", "ledger"],
        "required_doc_ids": ["fusion_and_eval"],
        "required_passage_ids": ["fusion_and_eval#p3"],
    },
    {
        "split": "test",
        "family": "conflict_resolution",
        "question": "What information should an adverse-event reviewer track before stopping a trial?",
        "expected_answer": "The reviewer should track severity, attribution, monitoring signals, and stopping rules.",
        "required_terms": ["severity", "attribution", "monitoring", "stopping"],
        "required_doc_ids": ["clinical_trials"],
        "required_passage_ids": ["clinical_trials#p2"],
    },
    {
        "split": "train",
        "family": "multi_hop_bridge",
        "question": "Why does a multi-intent question benefit from planner and worker roles?",
        "expected_answer": "A planner can split the question into subquestions while worker agents search for direct facts, background, counterevidence, implementation details, and evaluation needs.",
        "required_terms": ["planner", "subquestions", "worker", "counterevidence"],
        "required_doc_ids": ["agentic_rag"],
        "required_passage_ids": ["agentic_rag#p2"],
    },
    {
        "split": "dev",
        "family": "acronym_alias",
        "question": "Which modern systems motivate perspective-driven attributed retrieval?",
        "expected_answer": "MindSearch motivates dynamic search graphs, and STORM motivates multi-perspective question asking.",
        "required_terms": ["MindSearch", "STORM", "dynamic", "perspective"],
        "required_doc_ids": ["modern_sources"],
        "required_passage_ids": ["modern_sources#p1"],
    },
    {
        "split": "test",
        "family": "single_hop",
        "question": "Why can systematic reviews still be distorted by observational evidence?",
        "expected_answer": "Systematic reviews must account for cohort evidence, bias, and confounding when interpreting observational studies.",
        "required_terms": ["systematic", "cohort", "bias", "confounding"],
        "required_doc_ids": ["clinical_trials"],
        "required_passage_ids": ["clinical_trials#p3"],
    },
    {
        "split": "train",
        "family": "conflict_resolution",
        "question": "What must a trustworthy agentic retrieval system verify before final synthesis?",
        "expected_answer": "It must verify evidence provenance, source-aware synthesis, and grounding of final claims in retrieved passages.",
        "required_terms": ["provenance", "source-aware", "grounded", "passages"],
        "required_doc_ids": ["agentic_rag"],
        "required_passage_ids": ["agentic_rag#p3"],
    },
    {
        "split": "dev",
        "family": "multi_hop_bridge",
        "question": "How do specialized evidence-processing roles affect a small attributed-answering prototype?",
        "expected_answer": "RAGentA and MASS-RAG motivate separating retrieval, extraction, reasoning, verification, and synthesis into specialized roles.",
        "required_terms": ["retrieval", "extraction", "verification", "synthesis"],
        "required_doc_ids": ["modern_sources"],
        "required_passage_ids": ["modern_sources#p2"],
    },
    {
        "split": "test",
        "family": "needle_section",
        "question": "Which flood signals should trigger an evacuation workflow?",
        "expected_answer": "Rainfall, river gauge readings, threshold crossings, and evacuation status should be tracked together.",
        "required_terms": ["rainfall", "gauge", "threshold", "evacuation"],
        "required_doc_ids": ["climate_infrastructure"],
        "required_passage_ids": ["climate_infrastructure#p1"],
    },
    {
        "split": "train",
        "family": "single_hop",
        "question": "What fields make an evidence ledger auditable?",
        "expected_answer": "An evidence ledger stores source documents, passage IDs, retrieving roles, ranks, fused scores, support-token overlap, and provenance metadata.",
        "required_terms": ["source", "passage", "rank", "provenance"],
        "required_doc_ids": ["evidence_ledger"],
        "required_passage_ids": ["evidence_ledger#p1"],
    },
    {
        "split": "dev",
        "family": "needle_section",
        "question": "Which orchestrator-worker design ideas transfer to retrieval handoffs?",
        "expected_answer": "Orchestrator-worker coordination benefits from end-state evaluation, external memory, and filesystem-style handoffs.",
        "required_terms": ["orchestrator", "external", "memory", "handoffs"],
        "required_doc_ids": ["modern_sources"],
        "required_passage_ids": ["modern_sources#p3"],
    },
    {
        "split": "test",
        "family": "aggregation",
        "question": "What infrastructure lets a microgrid preserve service during a grid outage?",
        "expected_answer": "A resilient microgrid combines batteries, solar generation, islanding controls, and resilience planning.",
        "required_terms": ["battery", "solar", "islanding", "resilience"],
        "required_doc_ids": ["climate_infrastructure"],
        "required_passage_ids": ["climate_infrastructure#p2"],
    },
    {
        "split": "train",
        "family": "needle_section",
        "question": "How does the ledger reduce memory pressure in long retrieval loops?",
        "expected_answer": "Workers write durable artifacts, and the coordinator passes compact references instead of copying large passages through every agent message.",
        "required_terms": ["durable", "artifacts", "coordinator", "references"],
        "required_doc_ids": ["evidence_ledger"],
        "required_passage_ids": ["evidence_ledger#p2"],
    },
    {
        "split": "dev",
        "family": "single_hop",
        "question": "What should an incident commander capture in the first response record?",
        "expected_answer": "The record should capture the owner, timeline, mitigation, and rollback decision for the incident.",
        "required_terms": ["owner", "timeline", "mitigation", "rollback"],
        "required_doc_ids": ["incident_response"],
        "required_passage_ids": ["incident_response#p1"],
    },
    {
        "split": "test",
        "family": "conflict_resolution",
        "question": "Which heat-response actions target vulnerable residents before a forecasted heat wave?",
        "expected_answer": "Cooling centers, outreach, vulnerability mapping, and weather forecasts guide a heat-response plan.",
        "required_terms": ["cooling", "outreach", "vulnerability", "forecast"],
        "required_doc_ids": ["climate_infrastructure"],
        "required_passage_ids": ["climate_infrastructure#p3"],
    },
    {
        "split": "train",
        "family": "acronym_alias",
        "question": "What can a developer inspect after an auditable retrieval run?",
        "expected_answer": "A developer can inspect which role found each passage, which passages were fused, and whether the answer cites the selected evidence.",
        "required_terms": ["role", "passages", "fused", "cites"],
        "required_doc_ids": ["evidence_ledger"],
        "required_passage_ids": ["evidence_ledger#p3"],
    },
    {
        "split": "dev",
        "family": "conflict_resolution",
        "question": "Which postmortem fields separate incident impact from follow-up work?",
        "expected_answer": "A postmortem separates impact, detection, root cause, and follow-up actions.",
        "required_terms": ["impact", "detection", "root", "follow-up"],
        "required_doc_ids": ["incident_response"],
        "required_passage_ids": ["incident_response#p2"],
    },
    {
        "split": "test",
        "family": "single_hop",
        "question": "Which rollout signals should a canary deployment compare before expanding exposure?",
        "expected_answer": "A canary compares baseline behavior, error rate, rollback criteria, and the exposed cohort before widening rollout.",
        "required_terms": ["baseline", "error", "rollback", "cohort"],
        "required_doc_ids": ["software_delivery"],
        "required_passage_ids": ["software_delivery#p1"],
    },
    {
        "split": "train",
        "family": "aggregation",
        "question": "How does reciprocal-rank fusion reward repeated evidence across agents?",
        "expected_answer": "Reciprocal-rank fusion gives credit to passages that appear near the top of result lists and adds diversity credit when multiple roles find them.",
        "required_terms": ["reciprocal", "rank", "diversity", "roles"],
        "required_doc_ids": ["fusion_and_eval"],
        "required_passage_ids": ["fusion_and_eval#p1"],
    },
    {
        "split": "dev",
        "family": "aggregation",
        "question": "Which incident artifacts help an audit connect alerts to response procedures?",
        "expected_answer": "Dashboards, alerts, runbooks, and escalation records connect monitoring evidence to response procedures.",
        "required_terms": ["dashboard", "alert", "runbook", "escalation"],
        "required_doc_ids": ["incident_response"],
        "required_passage_ids": ["incident_response#p3"],
    },
    {
        "split": "test",
        "family": "acronym_alias",
        "question": "Which supply-chain artifacts show where a vulnerable dependency came from?",
        "expected_answer": "An SBOM, provenance record, vulnerability entry, and patch status show dependency risk.",
        "required_terms": ["SBOM", "provenance", "vulnerability", "patch"],
        "required_doc_ids": ["software_delivery"],
        "required_passage_ids": ["software_delivery#p2"],
    },
    {
        "split": "train",
        "family": "conflict_resolution",
        "question": "Why does the first deterministic evaluation target avoid an LLM judge?",
        "expected_answer": "Evidence support recall uses gold support document IDs to keep the MVP deterministic without requiring an LLM judge.",
        "required_terms": ["support", "recall", "deterministic", "judge"],
        "required_doc_ids": ["fusion_and_eval"],
        "required_passage_ids": ["fusion_and_eval#p2"],
    },
    {
        "split": "dev",
        "family": "multi_hop_bridge",
        "question": "What elements define a randomized trial comparison?",
        "expected_answer": "A randomized trial defines eligibility, an intervention, a control group, and an outcome.",
        "required_terms": ["eligibility", "intervention", "control", "outcome"],
        "required_doc_ids": ["clinical_trials"],
        "required_passage_ids": ["clinical_trials#p1"],
    },
    {
        "split": "test",
        "family": "multi_hop_bridge",
        "question": "How do traces connect slow requests to capacity pressure?",
        "expected_answer": "Observability links traces, spans, latency, and saturation to diagnose capacity pressure.",
        "required_terms": ["trace", "span", "latency", "saturation"],
        "required_doc_ids": ["software_delivery"],
        "required_passage_ids": ["software_delivery#p3"],
    },
)


def generate_examples(n: int = 24, seed: int = 0) -> list[RetrievalExample]:
    if n < 0:
        raise ValueError("n must be non-negative")
    if n > len(_TASK_TEMPLATES):
        raise ValueError(f"Bundled task bank contains {len(_TASK_TEMPLATES)} examples; requested {n}.")
    rng = random.Random(seed)  # nosec B311
    examples: list[RetrievalExample] = []
    for idx in range(n):
        template = _TASK_TEMPLATES[idx]
        examples.append(
            RetrievalExample(
                example_id=f"{template['family']}.{template['split']}.{seed:04d}.{idx:03d}",
                split=str(template["split"]),
                family=str(template["family"]),
                question=str(template["question"]),
                expected_answer=str(template["expected_answer"]),
                required_terms=list(template["required_terms"]),
                required_doc_ids=list(template["required_doc_ids"]),
                required_passage_ids=list(template["required_passage_ids"]),
                max_searches=rng.randint(4, 7),
                notes=f"Generated deterministic {template['family']} retrieval task with split-disjoint gold support.",
            )
        )
    return examples


def save_examples(path: Path, examples: list[RetrievalExample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.to_record(), sort_keys=True) + "\n")


def load_examples(path: Path) -> list[RetrievalExample]:
    examples: list[RetrievalExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                examples.append(RetrievalExample.from_record(json.loads(line)))
    return examples
