"""Ask the copilot from the terminal.

Usage:
    python -m copilot.cli "What is the NPE ratio by product type?"
    python -m copilot.cli            # interactive mode (follow-ups keep context)
"""
from __future__ import annotations

import sys

import pandas as pd

from copilot.config import Settings
from copilot.llm import OpenAIClient
from copilot.sql.engine import Answer, TextToSQL
from copilot.sql.safe_executor import SafeExecutor


def show(ans: Answer) -> None:
    if ans.error:
        print(f"\n[error] {ans.error}\n")
        return
    if not ans.answerable:
        print(f"\n{ans.explanation}\n")
        return
    print(f"\n{ans.answer_text or ans.explanation}\n")
    print("SQL:\n" + (ans.sql or ""))
    with pd.option_context("display.max_rows", 30, "display.width", 140, "display.float_format", "{:,.2f}".format):
        print("\n" + ans.df.to_string(index=False, max_rows=30))
    retries = sum(1 for a in ans.attempts if a.error)
    print(f"\n({len(ans.df):,} rows · {ans.latency_ms / 1000:.1f}s · {ans.usage.total_tokens:,} tokens"
          + (f" · self-corrected {retries}x" if retries else "") + ")\n")


def main() -> None:
    settings = Settings.from_env()
    engine = TextToSQL(OpenAIClient(settings.openai_model), SafeExecutor(settings.db_path), max_retries=settings.max_retries)

    if len(sys.argv) > 1:
        show(engine.ask(" ".join(sys.argv[1:])))
        return

    history: list[Answer] = []
    print("Loan Portfolio AI Copilot — type a question, or 'exit'.")
    while True:
        try:
            q = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q.lower() in {"exit", "quit"}:
            break
        if q:
            ans = engine.ask(q, history=history)
            history.append(ans)
            show(ans)


if __name__ == "__main__":
    main()
