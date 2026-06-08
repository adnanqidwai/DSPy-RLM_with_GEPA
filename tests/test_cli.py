from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "rlm_gepa_retrieval", *args],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def test_demo_command_writes_trace(tmp_path: Path) -> None:
    out = tmp_path / "demo.json"
    result = run_cli("demo", "--out", str(out))

    assert result.returncode == 0, result.stderr
    assert out.exists()
    assert "trace:" in result.stdout


def test_public_files_do_not_contain_key_material() -> None:
    banned = (
        re.compile(r"(?<![a-z0-9])sk-[a-z0-9_-]{8,}", re.IGNORECASE),
        re.compile(r"bearer\s+ey[a-z0-9_-]{8,}", re.IGNORECASE),
    )
    checked_suffixes = {".py", ".md", ".toml", ".sh", ".yml", ".yaml"}
    offenders: list[str] = []
    for path in ROOT.rglob("*"):
        if any(
            part in {".venv", ".pytest_cache", "__pycache__", "runs", "artifacts", "generated", "reports"}
            for part in path.parts
        ):
            continue
        if path.is_file() and path.suffix in checked_suffixes:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
            for pattern in banned:
                if pattern.search(text):
                    offenders.append(f"{path.relative_to(ROOT)} contains {pattern.pattern}")
    assert offenders == []


def test_repository_ignores_local_env_files() -> None:
    ignore = (ROOT / ".gitignore").read_text()

    assert ".env" in ignore
    assert "*.env" in ignore


def public_markdown_files() -> list[Path]:
    return [
        ROOT / "README.md",
        ROOT / "docs" / "design.md",
        ROOT / "docs" / "literature_review.md",
    ]


def test_public_docs_do_not_use_draft_project_framing() -> None:
    banned_phrases = (
        "res" + "ume",
        "no" + "-api",
        "no " + "api",
        "leg" + "acy",
        "int" + "ernal",
    )
    offenders: list[str] = []
    for path in public_markdown_files():
        text = path.read_text(encoding="utf-8").lower()
        for phrase in banned_phrases:
            if phrase in text:
                offenders.append(f"{path.relative_to(ROOT)} contains {phrase}")

    assert offenders == []


def test_public_docs_keep_rlm_gepa_as_primary_framing() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    literature = (ROOT / "docs" / "literature_review.md").read_text(encoding="utf-8")

    assert "trace-grounded RLM policy" in readme
    assert "host-recorded trace" in design
    assert "trace-grounded retrieval-policy harness" in literature


def test_public_docs_define_honest_results_shape() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    design = (ROOT / "docs" / "design.md").read_text(encoding="utf-8")
    literature = (ROOT / "docs" / "literature_review.md").read_text(encoding="utf-8")

    assert "## Results And Experiments" in readme
    assert "## Results Protocol" in design
    assert "## Baseline And Results Shape" in literature
    assert "| `heuristic` | Deterministic local code | Lexical retrieval baseline" in readme
    assert "| `single_shot_rag` | Deterministic local code | Standard retrieve-once top-k RAG control" in readme
    assert "| `rlm` | OpenAI-compatible chat model | Base DSPy RLM policy" in readme
    assert "| `optimized` | OpenAI-compatible chat model | GEPA-tuned textual retrieval policy" in readme
    assert "Only fill the `rlm` and `optimized` rows after running" in readme
    assert "Do not imply that the RLM or GEPA rows exist" in design


def test_readme_has_public_release_sections() -> None:
    readme = (ROOT / "README.md").read_text()

    assert "## Development" in readme
    assert "## Current Limitations" in readme
    assert "## Repository Hygiene" in readme
    assert "## Related Project" in readme
    assert "optimize-gepa" in readme


def test_ci_workflow_is_present() -> None:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"

    assert workflow.is_file()
    text = workflow.read_text()
    assert "python -m pytest -q" in text


def test_generate_then_eval_heuristic(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    run = tmp_path / "run.json"

    generated = run_cli("generate", "--out", str(tasks), "--n", "6", "--seed", "5")
    assert generated.returncode == 0, generated.stderr

    evaluated = run_cli(
        "eval",
        "--corpus",
        "examples/demo_corpus",
        "--questions",
        str(tasks),
        "--program",
        "heuristic",
        "--out",
        str(run),
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert run.exists()
    assert '"mean_score"' in run.read_text(encoding="utf-8")


def test_generate_then_eval_single_shot_rag(tmp_path: Path) -> None:
    tasks = tmp_path / "tasks.jsonl"
    run = tmp_path / "run.json"

    generated = run_cli("generate", "--out", str(tasks), "--n", "6", "--seed", "5")
    assert generated.returncode == 0, generated.stderr

    evaluated = run_cli(
        "eval",
        "--corpus",
        "examples/demo_corpus",
        "--questions",
        str(tasks),
        "--program",
        "single_shot_rag",
        "--out",
        str(run),
    )
    assert evaluated.returncode == 0, evaluated.stderr
    assert run.exists()
    assert '"program": "single_shot_rag"' in run.read_text(encoding="utf-8")


def test_answer_use_api_requires_model() -> None:
    result = run_cli(
        "answer",
        "--corpus",
        "examples/demo_corpus",
        "--question",
        "What stores evidence?",
        "--use-api",
    )

    assert result.returncode != 0
    assert "--model is required" in result.stderr


def test_report_command(tmp_path: Path) -> None:
    run = tmp_path / "run.json"
    run.write_text(
        '{"program":"heuristic","model":null,"num_questions":1,"mean_score":1.0,'
        '"mean_components":{"answer_correctness":1.0,"evidence_recall":1.0,'
        '"citation_precision":1.0,"budget_efficiency":1.0},"examples":[]}',
        encoding="utf-8",
    )
    out = tmp_path / "report.md"

    result = run_cli("report", "--runs", str(run), "--out", str(out))

    assert result.returncode == 0, result.stderr
    assert "# RLM-GEPA Retrieval Evaluation Report" in out.read_text(encoding="utf-8")
