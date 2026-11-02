"""Ask the copilot from the terminal.

Usage:
    python -m copilot.cli "What is the NPE ratio by product type?"
    python -m copilot.cli "When does a forborne exposure exit probation?"
    python -m copilot.cli            # interactive mode (follow-ups keep context)
"""
from __future__ import annotations

import sys

import pandas as pd

from copilot.assistant import Copilot, Turn
from copilot.config import Settings
from copilot.llm import OpenAIClient, OpenAIEmbedder
from copilot.rag.answerer import DocQA
from copilot.rag.retriever import HybridRetriever
from copilot.sql.engine import TextToSQL
from copilot.sql.safe_executor import SafeExecutor


def build_copilot(settings: Settings) -> Copilot:
    llm = OpenAIClient(settings.openai_model)
    doc_qa = None
    if settings.docs_db_path.exists():
        doc_qa = DocQA(llm, HybridRetriever(settings.docs_db_path, OpenAIEmbedder(settings.embedding_model)))
    return Copilot(llm, TextToSQL(llm, SafeExecutor(settings.db_path), max_retries=settings.max_retries), doc_qa)


def show(turn: Turn) -> None:
    print(f"\n[source: {turn.route}]")
    if turn.synthesis:
        print(f"\n{turn.synthesis}")
    ans = turn.data
    if ans is not None:
        if ans.error:
            print(f"\n[error] {ans.error}")
        elif not ans.answerable:
            print(f"\n{ans.explanation}")
        else:
            if not turn.synthesis:
                print(f"\n{ans.answer_text or ans.explanation}")
            print("\nSQL:\n" + (ans.sql or ""))
            with pd.option_context("display.max_rows", 30, "display.width", 140, "display.float_format", "{:,.2f}".format):
                print("\n" + ans.df.to_string(index=False, max_rows=30))
    docs = turn.docs
    if docs is not None:
        if docs.error:
            print(f"\n[error] {docs.error}")
        else:
            if not turn.synthesis:
                print(f"\n{docs.answer_text}")
            for c in docs.citations:
                print(f"  [{c.number}] {c.doc_id}, page {c.page}")
    print(f"\n({turn.latency_ms / 1000:.1f}s · {turn.total_tokens:,} tokens)\n")


def main() -> None:
    copilot = build_copilot(Settings.from_env())
    if len(sys.argv) > 1:
        show(copilot.ask(" ".join(sys.argv[1:])))
        return

    history: list[Turn] = []
    print("Loan Portfolio AI Copilot — ask about the portfolio or the regulation, or type 'exit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            turn = copilot.ask(q, history=history)
            history.append(turn)
            show(turn)


if __name__ == "__main__":
    main()
