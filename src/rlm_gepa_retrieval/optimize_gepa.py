from __future__ import annotations

import argparse
import json
from pathlib import Path

from .metrics import retrieval_gepa_metric
from .programs import configure_dspy, make_dspy_lm, make_rlm_retrieval_program, passages_to_corpus_json, require_dspy, to_dspy_examples
from .retrieval import load_markdown_corpus
from .tasks import load_examples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m rlm_gepa_retrieval.optimize_gepa")
    parser.add_argument("--data", required=True)
    parser.add_argument("--corpus", type=Path, default=Path("examples/demo_corpus"))
    parser.add_argument("--artifact", default="artifacts/optimized-rlm.json")
    parser.add_argument("--log-dir", default="runs/gepa")
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--reflection-model", default=None)
    parser.add_argument("--reflection-api-base", default=None)
    parser.add_argument("--train", type=int, default=12)
    parser.add_argument("--dev", type=int, default=6)
    parser.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    parser.add_argument("--max-metric-calls", type=int)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    dspy = require_dspy()
    configure_dspy(args.model, api_base=args.api_base)
    passages = load_markdown_corpus(args.corpus)
    corpus_json = passages_to_corpus_json(passages)
    examples = load_examples(Path(args.data))
    train = [item for item in examples if item.split == "train"][: args.train]
    dev = [item for item in examples if item.split == "dev"][: args.dev]
    if not train:
        raise ValueError("No train examples found")
    if not dev:
        raise ValueError("No dev examples found")

    student = make_rlm_retrieval_program(passages=passages)
    optimizer_kwargs = {
        "metric": retrieval_gepa_metric,
        "reflection_lm": make_dspy_lm(
            args.reflection_model or args.model,
            api_base=args.reflection_api_base or args.api_base,
            temperature=1.0,
            max_tokens=32000,
        ),
        "track_stats": True,
        "log_dir": args.log_dir,
        "seed": args.seed,
    }
    if args.max_metric_calls is None:
        optimizer_kwargs["auto"] = args.auto
    else:
        optimizer_kwargs["max_metric_calls"] = args.max_metric_calls

    optimizer = dspy.GEPA(**optimizer_kwargs)
    optimized = optimizer.compile(
        student,
        trainset=to_dspy_examples(train, corpus_json),
        valset=to_dspy_examples(dev, corpus_json),
    )

    artifact = Path(args.artifact)
    artifact.parent.mkdir(parents=True, exist_ok=True)
    optimized.save(str(artifact))
    metadata = {
        "artifact": str(artifact),
        "log_dir": args.log_dir,
        "train_examples": len(train),
        "dev_examples": len(dev),
        "model": args.model,
        "reflection_model": args.reflection_model or args.model,
    }
    artifact.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
