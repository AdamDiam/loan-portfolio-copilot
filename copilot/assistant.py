"""The copilot: routes each question to the portfolio data, the regulation, or both.

    question --> router --+-- "data" --> TextToSQL (portfolio database)
                          +-- "docs" --> DocQA (regulatory documents, cited)
                          +-- "both" --> both, then a short synthesis that
                                         ties the numbers to the rules
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from copilot.llm import LLMClient, Usage
from copilot.rag.answerer import DocAnswer, DocQA
from copilot.sql.engine import Answer, TextToSQL

ROUTER_PROMPT = """You route questions for a loan-servicing copilot with two knowledge sources:

1. "data": a database of the servicer's OWN loan portfolio (loans, borrowers, DPD, NPE flags, IFRS 9 stages,
   provisions, collateral, payments, forbearance measures, legal actions). Use for anything that needs counting,
   summing, listing or comparing the portfolio's loans: "how many", "what is our NPE ratio", "top 10 loans", trends.
2. "docs": EU regulatory guidance (EBA guidelines on NPE and forbearance management, EBA definition of default,
   ECB NPL guidance). Use for definitions, rules, thresholds, requirements, best practice:
   "what is", "when must", "how should", "what do the guidelines say".
3. "both": the question needs portfolio numbers AND what the rules say, e.g. "Which of our loans breach the
   forbearance probation rules and what does the EBA require?", "Is our NPE coverage in line with supervisory
   expectations?".

Follow-up questions inherit the topic of the previous turn unless they clearly switch.
Respond with JSON: {"route": "data" | "docs" | "both", "reason": "<few words>"}"""

SYNTHESIS_PROMPT = """You are a senior credit-risk analyst. Combine the portfolio finding and the regulatory guidance
into one short answer (3-6 sentences) for a portfolio manager: state the relevant numbers, then what the rules say,
then the implication. Keep the regulatory citations exactly as given (e.g. [2]); do not invent numbers or citations.
Answer in the language of the question. Respond with JSON: {"answer": "<text>"}"""

ROUTES = ("data", "docs", "both")


@dataclass
class Turn:
    question: str
    route: str
    route_reason: str = ""
    data: Answer | None = None
    docs: DocAnswer | None = None
    synthesis: str = ""
    error: str | None = None
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)

    @property
    def total_tokens(self) -> int:
        return (
            self.usage.total_tokens
            + (self.data.usage.total_tokens if self.data else 0)
            + (self.docs.usage.total_tokens if self.docs else 0)
        )


class Copilot:
    def __init__(self, llm: LLMClient, sql_engine: TextToSQL, doc_qa: DocQA | None = None) -> None:
        self.llm = llm
        self.sql = sql_engine
        self.docs = doc_qa

    def route(self, question: str, history: list[Turn] | None = None) -> tuple[str, str, Usage]:
        usage = Usage()
        if self.docs is None:
            return "data", "document index not built", usage
        messages = [{"role": "system", "content": ROUTER_PROMPT}]
        for t in (history or [])[-2:]:
            messages.append({"role": "user", "content": t.question})
            messages.append({"role": "assistant", "content": f'{{"route": "{t.route}"}}'})
        messages.append({"role": "user", "content": question})
        try:
            resp = self.llm.complete_json(messages)
        except Exception:
            return "data", "router unavailable, defaulted to data", usage
        usage.add(resp, "router")
        route = str(resp.data.get("route", "data")).lower()
        return (route if route in ROUTES else "data"), str(resp.data.get("reason", "")), usage

    def ask(self, question: str, history: list[Turn] | None = None, summarize: bool = True, route: str | None = None) -> Turn:
        t0 = time.perf_counter()
        history = history or []
        if route is None:
            route, reason, usage = self.route(question, history)
        else:
            reason, usage = "forced", Usage()
        turn = Turn(question=question, route=route, route_reason=reason, usage=usage)

        if route in ("data", "both"):
            data_history = [t.data for t in history if t.data is not None]
            turn.data = self.sql.ask(question, history=data_history, summarize=summarize)
        if route in ("docs", "both") and self.docs is not None:
            doc_history = [t.question for t in history if t.route in ("docs", "both")]
            turn.docs = self.docs.ask(question, history=doc_history)

        # The synthesis sees the data summary, so it only runs when summaries are on
        # (with summaries off, no portfolio data is ever sent to the LLM).
        if route == "both" and summarize and turn.data and turn.data.ok and turn.docs and turn.docs.answer_text:
            turn.synthesis = self._synthesize(turn)
        turn.latency_ms = (time.perf_counter() - t0) * 1000
        return turn

    def _synthesize(self, turn: Turn) -> str:
        data = turn.data
        assert data is not None and turn.docs is not None
        preview = data.df.head(15).to_csv(index=False) if data.df is not None else ""
        content = (
            f"Question: {turn.question}\n\nPortfolio finding:\n{data.answer_text}\n\nResult preview:\n{preview}\n\n"
            f"Regulatory guidance (with citations):\n{turn.docs.answer_text}"
        )
        try:
            resp = self.llm.complete_json([{"role": "system", "content": SYNTHESIS_PROMPT}, {"role": "user", "content": content}])
        except Exception:
            return ""
        turn.usage.add(resp, "synthesis")
        return str(resp.data.get("answer", "")).strip()
