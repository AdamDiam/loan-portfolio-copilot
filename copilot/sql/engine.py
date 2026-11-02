"""Text-to-SQL engine: question -> SQL -> safe execution -> answer.

Pipeline
  1. Build a prompt from the semantic layer (schema, glossary, rules, examples)
     plus the last few conversation turns (so follow-ups like "now by region"
     work).
  2. Ask the LLM for a JSON object with the SQL, an explanation and a chart hint.
  3. Execute the SQL in the read-only sandbox.
  4. If the safety layer or the database rejects it, feed the error back to
     the LLM and let it correct itself (up to `max_retries` times).
  5. Optionally ask the LLM to summarise the result in plain language,
     grounded only on the returned rows.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pandas as pd

from copilot.charts import ChartSpec, choose_chart
from copilot.llm import LLMClient, Usage
from copilot.sql.safe_executor import QueryExecutionError, SafeExecutor, UnsafeQueryError
from copilot.sql.semantic import render_context

SYSTEM_PROMPT = """You are LoanCopilot, a senior credit-risk data analyst at a loan servicing company.
You translate business questions about the loan portfolio into ONE correct, read-only SQLite query.

{context}

## Output format
Respond with a single JSON object and nothing else:
{{
  "answerable": true,
  "sql": "<one SELECT statement>",
  "explanation": "<1-2 plain-English sentences: what is measured, scope and assumptions>",
  "chart": {{"type": "bar | line | pie | none", "x": "<column alias>", "y": "<column alias>", "color": "<column alias or null>"}},
  "follow_ups": ["<up to 3 short follow-up questions an analyst might ask next>"]
}}
- If the question cannot be answered from this database (needs external data, is not about the portfolio,
  or asks to modify data), return "answerable": false, "sql": null and use "explanation" to say what you can answer instead.
- Write "explanation" and "follow_ups" in the same language as the user's question.
- Use "chart": {{"type": "none"}} for single numbers or long loan lists.
"""

SUMMARY_PROMPT = """You are a credit-risk analyst presenting results to a portfolio manager.
Given the question, the SQL that was run and the result rows, answer the question in 2-4 sentences.
Rules: use only numbers present in the result; format money as EUR with thousand separators
(use millions, e.g. EUR 12.4m, for amounts above 1,000,000); keep percentages as given; highlight the
key takeaway (largest, trend, outlier). If the result is truncated, say it shows the first rows.
Answer in the language of the question. Respond with JSON: {"answer": "<text>"}"""


@dataclass
class Attempt:
    sql: str | None
    error: str | None = None


@dataclass
class Answer:
    question: str
    answerable: bool = True
    sql: str | None = None
    explanation: str = ""
    answer_text: str = ""
    df: pd.DataFrame | None = None
    truncated: bool = False
    chart: ChartSpec | None = None
    follow_ups: list[str] = field(default_factory=list)
    attempts: list[Attempt] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)

    @property
    def ok(self) -> bool:
        return self.error is None and self.answerable and self.df is not None


class TextToSQL:
    def __init__(self, llm: LLMClient, executor: SafeExecutor, max_retries: int = 2, history_turns: int = 3) -> None:
        self.llm = llm
        self.executor = executor
        self.max_retries = max_retries
        self.history_turns = history_turns
        self._system = SYSTEM_PROMPT.format(context=render_context())

    # ------------------------------------------------------------------
    def _messages(self, question: str, history: list[Answer] | None) -> list[dict]:
        messages = [{"role": "system", "content": self._system}]
        for turn in (history or [])[-self.history_turns :]:
            if turn.sql:
                messages.append({"role": "user", "content": turn.question})
                messages.append(
                    {"role": "assistant", "content": json.dumps({"answerable": True, "sql": turn.sql, "explanation": turn.explanation})}
                )
        messages.append({"role": "user", "content": question})
        return messages

    def ask(self, question: str, history: list[Answer] | None = None, summarize: bool = True) -> Answer:
        t0 = time.perf_counter()
        ans = Answer(question=question)
        messages = self._messages(question, history)

        for attempt_no in range(self.max_retries + 1):
            try:
                resp = self.llm.complete_json(messages)
            except Exception as exc:  # network / auth / rate-limit errors
                ans.error = f"LLM call failed: {exc}"
                break
            ans.usage.add(resp, "sql" if attempt_no == 0 else f"retry_{attempt_no}")
            data = resp.data

            if not data.get("answerable", True) or not data.get("sql"):
                ans.answerable = False
                ans.explanation = data.get("explanation") or "This question cannot be answered from the portfolio data."
                ans.follow_ups = list(data.get("follow_ups") or [])[:3]
                break

            sql = str(data["sql"])
            ans.explanation = data.get("explanation") or ""
            ans.follow_ups = list(data.get("follow_ups") or [])[:3]
            try:
                result = self.executor.run(sql)
            except (UnsafeQueryError, QueryExecutionError) as exc:
                ans.attempts.append(Attempt(sql=sql, error=str(exc)))
                messages += [
                    {"role": "assistant", "content": resp.raw},
                    {
                        "role": "user",
                        "content": f"The query failed with this error:\n{exc}\n"
                        "Fix the SQL (SQLite, read-only, allowed tables only) and return the full JSON object again.",
                    },
                ]
                continue

            ans.attempts.append(Attempt(sql=sql))
            ans.sql = self.executor.precheck(sql)
            ans.df = result.df
            ans.truncated = result.truncated
            ans.chart = choose_chart(result.df, data.get("chart"))
            break
        else:
            ans.error = f"Could not produce a valid query after {self.max_retries + 1} attempts: {ans.attempts[-1].error}"

        if ans.ok and summarize:
            ans.answer_text = self._summarize(ans)
        ans.latency_ms = (time.perf_counter() - t0) * 1000
        return ans

    # ------------------------------------------------------------------
    def _summarize(self, ans: Answer) -> str:
        assert ans.df is not None
        if ans.df.empty:
            return "No records match this question."
        preview = ans.df.head(30).to_csv(index=False)
        note = f"(showing first 30 of {len(ans.df)} rows{'+' if ans.truncated else ''})" if len(ans.df) > 30 else ""
        messages = [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Question: {ans.question}\n\nSQL:\n{ans.sql}\n\nResult {note}:\n{preview}"},
        ]
        try:
            resp = self.llm.complete_json(messages)
        except Exception:  # summary is best-effort; the table is still shown
            return ""
        ans.usage.add(resp, "summary")
        return str(resp.data.get("answer", "")).strip()
