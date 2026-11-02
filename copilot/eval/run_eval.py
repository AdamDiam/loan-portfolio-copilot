"""Run the evaluation set against the live LLM and write a markdown report.

Usage:
    python -m copilot.eval.run_eval                 # all questions
    python -m copilot.eval.run_eval --ids q01 q05   # a subset
    python -m copilot.eval.run_eval --limit 5
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from copilot.config import EVAL_QUESTIONS_PATH, EVAL_RESULTS_DIR, Settings
from copilot.eval.compare import results_match
from copilot.llm import LLMClient
from copilot.sql.engine import TextToSQL
from copilot.sql.safe_executor import SafeExecutor


def load_questions(path: Path = EVAL_QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def evaluate(engine: TextToSQL, questions: list[dict], verbose: bool = True) -> list[dict]:
    rows = []
    for q in questions:
        gold = engine.executor.run_trusted(q["gold_sql"])
        ans = engine.ask(q["question"], summarize=False)
        correct = bool(ans.ok and results_match(gold, ans.df))
        rows.append(
            {
                "id": q["id"],
                "category": q["category"],
                "question": q["question"],
                "correct": correct,
                "attempts": len(ans.attempts),
                "self_corrected": correct and len(ans.attempts) > 1,
                "latency_s": round(ans.latency_ms / 1000, 2),
                "tokens": ans.usage.total_tokens,
                "sql": ans.sql,
                "error": ans.error or ("not answerable" if not ans.answerable else None),
            }
        )
        if verbose:
            print(f"  {'PASS' if correct else 'FAIL'}  {q['id']}  [{q['category']}]  {q['question']}")
    return rows


def summarize(rows: list[dict]) -> dict:
    by_cat: dict[str, list[bool]] = defaultdict(list)
    for r in rows:
        by_cat[r["category"]].append(r["correct"])
    return {
        "n": len(rows),
        "accuracy_pct": round(100 * sum(r["correct"] for r in rows) / max(len(rows), 1), 1),
        "self_corrected": sum(r["self_corrected"] for r in rows),
        "median_latency_s": round(statistics.median([r["latency_s"] for r in rows]), 2) if rows else 0,
        "avg_tokens": round(statistics.mean([r["tokens"] for r in rows])) if rows else 0,
        "by_category": {c: f"{sum(v)}/{len(v)}" for c, v in sorted(by_cat.items())},
    }


def write_report(rows: list[dict], summary: dict, model: str, out_dir: Path = EVAL_RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    path = out_dir / f"eval_{stamp}_{model.replace('/', '-')}.md"
    lines = [
        f"# Evaluation report — {model}",
        "",
        f"- Questions: **{summary['n']}**",
        f"- Execution accuracy: **{summary['accuracy_pct']}%**",
        f"- Fixed by self-correction: {summary['self_corrected']}",
        f"- Median latency: {summary['median_latency_s']} s · Avg tokens/question: {summary['avg_tokens']}",
        "",
        "| Category | Correct |",
        "|---|---|",
        *[f"| {c} | {v} |" for c, v in summary["by_category"].items()],
        "",
        "| ID | Category | Question | Result | Attempts |",
        "|---|---|---|---|---|",
        *[f"| {r['id']} | {r['category']} | {r['question']} | {'✅' if r['correct'] else '❌'} | {r['attempts']} |" for r in rows],
        "",
        "## Failures",
    ]
    for r in rows:
        if not r["correct"]:
            lines += [f"### {r['id']}: {r['question']}", f"Error: {r['error']}" if r["error"] else "", "```sql", r["sql"] or "-- no SQL", "```", ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate text-to-SQL execution accuracy.")
    parser.add_argument("--ids", nargs="*", help="only run these question ids")
    parser.add_argument("--limit", type=int, help="only run the first N questions")
    args = parser.parse_args()

    from copilot.llm import OpenAIClient

    settings = Settings.from_env()
    llm: LLMClient = OpenAIClient(settings.openai_model)
    engine = TextToSQL(llm, SafeExecutor(settings.db_path, max_rows=100_000), max_retries=settings.max_retries)

    questions = load_questions()
    if args.ids:
        questions = [q for q in questions if q["id"] in set(args.ids)]
    if args.limit:
        questions = questions[: args.limit]

    print(f"Evaluating {len(questions)} questions with {settings.openai_model} ...\n")
    rows = evaluate(engine, questions)
    summary = summarize(rows)
    path = write_report(rows, summary, settings.openai_model)
    print(f"\nExecution accuracy: {summary['accuracy_pct']}%  ({sum(r['correct'] for r in rows)}/{len(rows)})")
    print(f"By category: {summary['by_category']}")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
