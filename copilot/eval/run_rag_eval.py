"""Evaluate routing, retrieval and grounded answers for the regulatory module.

Metrics
  * Router accuracy          : predicted route == expected route
  * Retrieval hit@k          : a retrieved chunk is on an expected page (+/- 1)
  * Answer keyword match     : the answer states the key fact (e.g. "90 days")
  * Cited                    : answerable questions come with >= 1 citation
  * Refusal                  : unanswerable questions are declined

Usage:
    python -m copilot.eval.run_rag_eval
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import yaml

from copilot.config import EVAL_RESULTS_DIR, PROJECT_ROOT, Settings

RAG_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "rag_questions.yaml"


def load(path: Path = RAG_QUESTIONS_PATH) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def retrieval_hit(retrieved, expected: dict | list[dict], tolerance: int = 1) -> bool:
    places = expected if isinstance(expected, list) else [expected]
    return any(
        c.doc_id == place["doc_id"] and any(abs(c.page - p) <= tolerance for p in place["pages"])
        for place in places
        for c in retrieved
    )


def keyword_hit(text: str, keywords: list[str]) -> bool:
    low = text.lower()
    return any(k.lower() in low for k in keywords)


def evaluate(copilot, questions: list[dict], verbose: bool = True) -> list[dict]:
    rows = []
    for q in questions:
        route, _, _ = copilot.route(q["question"])
        row = {"id": q["id"], "question": q["question"], "expected_route": q["route"], "route": route,
               "route_ok": route == q["route"]}
        if q["route"] == "docs":
            ans = copilot.docs.ask(q["question"])
            answerable_expected = q.get("answerable", True)
            if answerable_expected:
                row["retrieval_hit"] = retrieval_hit(ans.retrieved, q["expected"])
                row["keyword_hit"] = keyword_hit(ans.answer_text, q["keywords"])
                row["cited"] = len(ans.citations) > 0
                row["correct"] = row["keyword_hit"] and row["cited"] and ans.answerable
            else:
                row["correct"] = not ans.answerable
                row["refused"] = not ans.answerable
            row["answer"] = ans.answer_text
            row["citations"] = [f"{c.doc_id} p.{c.page}" for c in ans.citations]
            row["latency_s"] = round(ans.latency_ms / 1000, 2)
        rows.append(row)
        if verbose:
            parts = [f"route {'ok' if row['route_ok'] else 'WRONG (' + route + ')'}"]
            if "retrieval_hit" in row:
                parts += [f"retrieval {'hit' if row['retrieval_hit'] else 'MISS'}", f"answer {'ok' if row['correct'] else 'WRONG'}"]
            elif "refused" in row:
                parts.append("refused" if row["refused"] else "DID NOT REFUSE")
            print(f"  {q['id']}  {' | '.join(parts):<55} {q['question'][:70]}")
    return rows


def summarize(rows: list[dict]) -> dict:
    def pct(xs):
        return round(100 * sum(xs) / len(xs), 1) if xs else None

    docs = [r for r in rows if "correct" in r]
    answerable = [r for r in rows if "retrieval_hit" in r]
    return {
        "n_questions": len(rows),
        "router_accuracy_pct": pct([r["route_ok"] for r in rows]),
        "retrieval_hit_pct": pct([r["retrieval_hit"] for r in answerable]),
        "answer_accuracy_pct": pct([r["correct"] for r in docs]),
        "cited_pct": pct([r["cited"] for r in answerable]),
    }


def write_report(rows, summary, model, out_dir: Path = EVAL_RESULTS_DIR) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"rag_eval_{datetime.now():%Y%m%d_%H%M}_{model}.md"
    lines = [
        f"# RAG & routing evaluation — {model}",
        "",
        f"- Router accuracy: **{summary['router_accuracy_pct']}%** ({summary['n_questions']} questions)",
        f"- Retrieval hit@6: **{summary['retrieval_hit_pct']}%**",
        f"- Answer accuracy (key fact + citation, or correct refusal): **{summary['answer_accuracy_pct']}%**",
        f"- Answers with citations: {summary['cited_pct']}%",
        "",
        "| ID | Question | Route | Retrieval | Answer | Citations |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['id']} | {r['question']} | {'✅' if r['route_ok'] else '❌ ' + r['route']} | "
            f"{'' if 'retrieval_hit' not in r else '✅' if r['retrieval_hit'] else '❌'} | "
            f"{'' if 'correct' not in r else '✅' if r['correct'] else '❌'} | {', '.join(r.get('citations', []))} |"
        )
    lines += ["", "## Answers", ""]
    for r in rows:
        if r.get("answer"):
            lines += [f"**{r['id']}. {r['question']}**", "", r["answer"], ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    path.with_suffix(".json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    return path


def main() -> None:
    from copilot.assistant import Copilot
    from copilot.llm import OpenAIClient, OpenAIEmbedder
    from copilot.rag.answerer import DocQA
    from copilot.rag.retriever import HybridRetriever
    from copilot.sql.engine import TextToSQL
    from copilot.sql.safe_executor import SafeExecutor

    s = Settings.from_env()
    llm = OpenAIClient(s.openai_model)
    copilot = Copilot(
        llm,
        TextToSQL(llm, SafeExecutor(s.db_path)),
        DocQA(llm, HybridRetriever(s.docs_db_path, OpenAIEmbedder(s.embedding_model))),
    )
    questions = load()
    print(f"Evaluating {len(questions)} questions with {s.openai_model} ...\n")
    rows = evaluate(copilot, questions)
    summary = summarize(rows)
    path = write_report(rows, summary, s.openai_model)
    print(f"\nRouter accuracy {summary['router_accuracy_pct']}% | retrieval hit@6 {summary['retrieval_hit_pct']}% | "
          f"answer accuracy {summary['answer_accuracy_pct']}%")
    print(f"Report: {path}")


if __name__ == "__main__":
    main()
