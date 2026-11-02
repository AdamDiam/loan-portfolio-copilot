"""Grounded question answering over the regulatory documents, with citations."""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from copilot.llm import LLMClient, Usage
from copilot.rag.retriever import HybridRetriever, RetrievedChunk

DOC_QA_PROMPT = """You are a regulatory expert on EU banking supervision of non-performing exposures (NPE),
forbearance, definition of default, impairment and collateral valuation.

Answer the user's question using ONLY the numbered sources provided. Rules:
- Every factual sentence must cite its source(s) in square brackets, e.g. [1] or [2][4].
- Quote exact thresholds, periods and conditions from the sources (e.g. "90 days", "one year cure period").
- If the sources do not contain the answer, say so clearly and set "answerable": false. Never use outside knowledge.
- Be concise: 3-8 sentences, or a short list when the source is a list.
- Answer in the language of the question; keep regulatory terms (e.g. "forborne", "unlikely to pay") in English.

Respond with JSON: {"answerable": true|false, "answer": "<text with [n] citations>", "used_sources": [<numbers>]}"""


@dataclass
class Citation:
    number: int
    doc_id: str
    title: str
    page: int
    section: str
    snippet: str


@dataclass
class DocAnswer:
    question: str
    answer_text: str = ""
    answerable: bool = True
    citations: list[Citation] = field(default_factory=list)
    retrieved: list[RetrievedChunk] = field(default_factory=list)
    error: str | None = None
    latency_ms: float = 0.0
    usage: Usage = field(default_factory=Usage)


def format_sources(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] {c.doc_id}, page {c.page}{' - ' + c.section if c.section else ''}\n{c.text}" for i, c in enumerate(chunks, 1)
    )


class DocQA:
    def __init__(self, llm: LLMClient, retriever: HybridRetriever, k: int = 6) -> None:
        self.llm = llm
        self.retriever = retriever
        self.k = k

    def ask(self, question: str, history: list[str] | None = None) -> DocAnswer:
        t0 = time.perf_counter()
        ans = DocAnswer(question=question)
        # follow-ups ("what about retail?") need the previous question for retrieval
        search_text = question if not history else f"{history[-1]}\n{question}"
        try:
            ans.retrieved = self.retriever.search(search_text, k=self.k)
        except Exception as exc:  # embedding API errors etc.
            ans.error = f"Document search failed: {exc}"
            ans.latency_ms = (time.perf_counter() - t0) * 1000
            return ans
        if not ans.retrieved:
            ans.answerable = False
            ans.answer_text = "No relevant passages were found in the indexed regulation."
            ans.latency_ms = (time.perf_counter() - t0) * 1000
            return ans

        messages = [
            {"role": "system", "content": DOC_QA_PROMPT},
            {"role": "user", "content": f"Sources:\n\n{format_sources(ans.retrieved)}\n\nQuestion: {question}"},
        ]
        try:
            resp = self.llm.complete_json(messages)
        except Exception as exc:
            ans.error = f"LLM call failed: {exc}"
            ans.latency_ms = (time.perf_counter() - t0) * 1000
            return ans
        ans.usage.add(resp, "doc_answer")
        ans.answerable = bool(resp.data.get("answerable", True))
        ans.answer_text = str(resp.data.get("answer", "")).strip()
        used = [int(n) for n in resp.data.get("used_sources") or [] if str(n).isdigit()]
        if not ans.answerable:  # a refusal should not point to sources it did not use
            used = []
            ans.answer_text = re.sub(r"\s*(\[\d+\])+", "", ans.answer_text).strip()
        for n in sorted(set(used)):
            if 1 <= n <= len(ans.retrieved):
                c = ans.retrieved[n - 1]
                ans.citations.append(Citation(n, c.doc_id, c.title, c.page, c.section, c.text[:400]))
        ans.latency_ms = (time.perf_counter() - t0) * 1000
        return ans
