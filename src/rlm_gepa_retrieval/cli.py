from __future__ import annotations

import argparse
import json
from pathlib import Path

from .eval import evaluate
from .pipeline import answer_question
from .retrieval import load_markdown_corpus


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RLM-GEPA Retrieval DSPy RLM + GEPA retrieval harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run the bundled retrieval inspection question")
    demo.add_argument("--out", type=Path, default=Path("runs/demo-answer.json"))

    answer = subparsers.add_parser("answer", help="Inspect retrieval behavior on one Markdown-corpus question")
    answer.add_argument("--corpus", type=Path, required=True)
    answer.add_argument("--question", required=True)
    answer.add_argument("--out", type=Path)
    answer.add_argument("--top-k-per-agent", type=int, default=5)
    answer.add_argument("--evidence-k", type=int, default=8)
    answer.add_argument("--model")
    answer.add_argument("--api-base")
    answer.add_argument("--use-api", action="store_true")

    generate = subparsers.add_parser("generate", help="Generate deterministic retrieval-task JSONL")
    generate.add_argument("--out", type=Path, required=True)
    generate.add_argument("--n", type=int, default=24)
    generate.add_argument("--seed", type=int, default=0)

    eval_cmd = subparsers.add_parser("eval", help="Run deterministic retrieval-policy eval")
    eval_cmd.add_argument("--corpus", type=Path, required=True)
    eval_cmd.add_argument("--questions", type=Path, required=True)
    eval_cmd.add_argument("--out", type=Path)
    eval_cmd.add_argument("--program", choices=["heuristic", "single_shot_rag", "rlm", "optimized"], default="heuristic")
    eval_cmd.add_argument("--model")
    eval_cmd.add_argument("--api-base")
    eval_cmd.add_argument("--artifact")

    optimize = subparsers.add_parser("optimize-gepa", help="Optimize the DSPy RLM retrieval policy with GEPA")
    optimize.add_argument("--data", required=True)
    optimize.add_argument("--corpus", type=Path, default=Path("examples/demo_corpus"))
    optimize.add_argument("--artifact", default="artifacts/optimized-rlm.json")
    optimize.add_argument("--log-dir", default="runs/gepa")
    optimize.add_argument("--model")
    optimize.add_argument("--api-base")
    optimize.add_argument("--reflection-model")
    optimize.add_argument("--reflection-api-base")
    optimize.add_argument("--train", type=int, default=12)
    optimize.add_argument("--dev", type=int, default=6)
    optimize.add_argument("--auto", choices=["light", "medium", "heavy"], default="light")
    optimize.add_argument("--max-metric-calls", type=int)
    optimize.add_argument("--seed", type=int, default=0)

    report = subparsers.add_parser("report", help="Build Markdown report from run JSON files")
    report.add_argument("--runs", nargs="+", type=Path, required=True)
    report.add_argument("--out", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "demo":
        result = _run_demo(args.out)
    elif args.command == "answer":
        if args.use_api and not args.model:
            parser.error("--model is required when --use-api is set")
        result = _run_answer(args)
    elif args.command == "generate":
        from .tasks import generate_examples, save_examples

        try:
            examples = generate_examples(n=args.n, seed=args.seed)
        except ValueError as exc:
            parser.error(str(exc))
        save_examples(args.out, examples)
        print(json.dumps({"out": str(args.out), "num_examples": len(examples)}, indent=2))
        return
    elif args.command == "eval":
        result = evaluate(
            args.corpus,
            args.questions,
            program_name=args.program,
            model=args.model,
            api_base=args.api_base,
            artifact=args.artifact,
        )
        _write_or_print(result, args.out)
        return
    elif args.command == "optimize-gepa":
        from .optimize_gepa import main as optimize_main

        optimize_args = []
        for name in (
            "data",
            "corpus",
            "artifact",
            "log_dir",
            "model",
            "api_base",
            "reflection_model",
            "reflection_api_base",
            "train",
            "dev",
            "auto",
            "max_metric_calls",
            "seed",
        ):
            value = getattr(args, name)
            if value is None:
                continue
            flag = "--" + name.replace("_", "-")
            optimize_args.extend([flag, str(value)])
        raise SystemExit(optimize_main(optimize_args))
    elif args.command == "report":
        from .report import main as report_main

        report_args = ["--runs", *[str(path) for path in args.runs], "--out", str(args.out)]
        raise SystemExit(report_main(report_args))
    else:
        parser.error(f"unknown command: {args.command}")
        return

    if args.command in {"demo", "answer"}:
        print(result["answer"])
        if "out" in result:
            print(f"\ntrace: {result['out']}")


def _run_demo(out: Path) -> dict:
    root = Path(__file__).resolve().parents[2]
    corpus = root / "examples" / "demo_corpus"
    question = "What makes RLM-GEPA Retrieval different from static RAG?"
    passages = load_markdown_corpus(corpus)
    result = answer_question(question, passages)
    payload = result.to_dict()
    _write_or_print(payload, out)
    return {"answer": result.answer, "out": str(out)}


def _run_answer(args: argparse.Namespace) -> dict:
    passages = load_markdown_corpus(args.corpus)
    result = answer_question(
        args.question,
        passages,
        top_k_per_agent=args.top_k_per_agent,
        evidence_k=args.evidence_k,
        model=args.model,
        api_base=args.api_base,
        use_api=args.use_api,
    )
    payload = result.to_dict()
    _write_or_print(payload, args.out)
    response = {"answer": result.answer}
    if args.out:
        response["out"] = str(args.out)
    return response


def _write_or_print(payload: dict, out: Path | None) -> None:
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    else:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
