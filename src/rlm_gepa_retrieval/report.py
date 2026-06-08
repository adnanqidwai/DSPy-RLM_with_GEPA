from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_report(run_paths: list[Path]) -> str:
    lines = ["# RLM-GEPA Retrieval Evaluation Report", ""]
    for path in run_paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lines.extend(
            [
                f"## {path}",
                "",
                f"- Program: `{payload.get('program')}`",
                f"- Model: `{payload.get('model')}`",
                f"- Questions: `{payload.get('num_questions')}`",
                f"- Mean score: `{payload.get('mean_score', 0):.4f}`",
                "",
                "| Component | Mean |",
                "| --- | ---: |",
            ]
        )
        for name, value in payload.get("mean_components", {}).items():
            lines.append(f"| `{name}` | `{value:.4f}` |")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m rlm_gepa_retrieval.report")
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    text = build_report(args.runs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text + "\n", encoding="utf-8")
    print(str(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
