"""RAG pipeline tests with a deterministic fake embedder (no API key needed)."""
import json
import re
import zlib

import numpy as np
import pytest

from copilot.assistant import Copilot
from copilot.llm import LLMResponse
from copilot.rag.answerer import DocQA
from copilot.rag.ingest import Chunk, build_index, chunk_page, clean_pages
from copilot.rag.retriever import HybridRetriever, keyword_query
from copilot.rag.sources import Source
from copilot.sql.engine import TextToSQL
from copilot.sql.safe_executor import SafeExecutor


class HashEmbedder:
    """Bag-of-words hashed into a fixed vector: similar wording -> similar vectors."""

    model = "hash-test"

    def embed(self, texts):
        out = []
        for t in texts:
            v = np.zeros(256, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", t.lower()):
                v[zlib.crc32(w.encode()) % 256] += 1
            out.append(v.tolist())
        return out


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, messages):
        self.calls.append(messages)
        data = self.replies.pop(0)
        return LLMResponse(data=data, raw=json.dumps(data), prompt_tokens=50, completion_tokens=10)


DOCS = [
    Chunk("GL-A", 3, "A forborne exposure remains forborne for a probation period of at least two years."),
    Chunk("GL-A", 7, "Collateral of non-performing exposures should be revalued at least annually."),
    Chunk("GL-B", 2, "An obligor is in default when a material credit obligation is more than 90 days past due."),
    Chunk("GL-B", 5, "Indications of unlikeliness to pay include distressed restructuring and bankruptcy."),
]


@pytest.fixture(scope="module")
def docs_db(tmp_path_factory):
    path = tmp_path_factory.mktemp("rag") / "docs.db"
    sources = [(Source("GL-A", "Guideline A", "a.pdf", "http://a"), 10), (Source("GL-B", "Guideline B", "b.pdf", "http://b"), 10)]
    build_index(path, sources, DOCS, HashEmbedder(), progress=False)
    return path


# ---------------------------------------------------------------- ingestion
def test_chunking_respects_size_and_overlap():
    text = ". ".join(f"Sentence number {i} about forbearance" for i in range(200))
    chunks = chunk_page(text, size=300, overlap=50)
    assert len(chunks) > 5
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0][-20:].split()[-1] in chunks[1]  # overlap carries context forward


def test_clean_pages_removes_running_headers_and_page_numbers():
    pages = [f"EBA GUIDELINES ON NPE\nReal content {i} about for-\nbearance case {i}\n{i}" for i in range(10)]
    cleaned = clean_pages(pages)
    assert "EBA GUIDELINES ON NPE" not in cleaned[0]
    assert cleaned[3] == "Real content 3 about for\nbearance case 3".replace("for\nbearance", "forbearance")


def test_keyword_query_is_safe_fts_syntax():
    q = keyword_query('What does "Article 178" say? DROP TABLE; *')
    assert q == '"article" OR "178" OR "say" OR "drop" OR "table"'


# ---------------------------------------------------------------- retrieval
def test_hybrid_retrieval_finds_the_right_page(docs_db):
    r = HybridRetriever(docs_db, HashEmbedder())
    top = r.search("How long is the probation period for forborne exposures?", k=2)
    assert (top[0].doc_id, top[0].page) == ("GL-A", 3)
    top = r.search("when is an obligor in default 90 days past due", k=1)
    assert (top[0].doc_id, top[0].page) == ("GL-B", 2)


def test_retrieval_reports_both_rankings(docs_db):
    r = HybridRetriever(docs_db, HashEmbedder())
    hit = r.search("revalued annually collateral", k=1)[0]
    assert hit.page == 7 and hit.vector_rank == 1
    if r.has_fts:
        assert hit.keyword_rank == 1


# ---------------------------------------------------------------- answering
def test_doc_answer_has_page_citations(docs_db):
    llm = ScriptedLLM([{"answerable": True, "answer": "At least two years [1].", "used_sources": [1]}])
    qa = DocQA(llm, HybridRetriever(docs_db, HashEmbedder()), k=3)
    ans = qa.ask("How long is the forbearance probation period?")
    assert ans.answer_text == "At least two years [1]."
    assert [(c.doc_id, c.page) for c in ans.citations] == [("GL-A", 3)]
    assert "[1] GL-A, page 3" in llm.calls[0][1]["content"]


def test_invalid_citation_numbers_are_ignored(docs_db):
    llm = ScriptedLLM([{"answerable": True, "answer": "x [9]", "used_sources": [9, "abc"]}])
    ans = DocQA(llm, HybridRetriever(docs_db, HashEmbedder()), k=2).ask("probation")
    assert ans.citations == []


# ---------------------------------------------------------------- routing
def test_router_sends_each_question_to_the_right_place(small_db, docs_db):
    llm = ScriptedLLM(
        [
            {"route": "docs"},
            {"answerable": True, "answer": "Two years [1].", "used_sources": [1]},
            {"route": "data"},
            {"answerable": True, "sql": "SELECT COUNT(*) AS n FROM loans WHERE is_forborne = 1"},
            {"route": "both"},
            {"answerable": True, "sql": "SELECT COUNT(*) AS n FROM loans WHERE is_forborne = 1"},
            {"answer": "There are N forborne loans."},
            {"answerable": True, "answer": "Two years [1].", "used_sources": [1]},
            {"answer": "We have N forborne loans; the probation period is two years [1]."},
        ]
    )
    copilot = Copilot(llm, TextToSQL(llm, SafeExecutor(small_db)), DocQA(llm, HybridRetriever(docs_db, HashEmbedder())))

    t1 = copilot.ask("How long is the probation period?", summarize=False)
    assert t1.route == "docs" and t1.data is None and t1.docs.citations

    t2 = copilot.ask("How many loans are forborne?", history=[t1], summarize=False)
    assert t2.route == "data" and t2.data.ok and t2.docs is None

    t3 = copilot.ask("How many forborne loans do we have and what is the probation period?", summarize=True)
    assert t3.route == "both" and t3.data.ok and t3.docs.citations
    assert "[1]" in t3.synthesis


def test_no_synthesis_when_summaries_are_off(small_db, docs_db):
    llm = ScriptedLLM(
        [
            {"answerable": True, "sql": "SELECT COUNT(*) AS n FROM loans"},
            {"answerable": True, "answer": "Two years [1].", "used_sources": [1]},
        ]
    )
    copilot = Copilot(llm, TextToSQL(llm, SafeExecutor(small_db)), DocQA(llm, HybridRetriever(docs_db, HashEmbedder())))
    turn = copilot.ask("both please", summarize=False, route="both")
    assert turn.synthesis == "" and len(llm.calls) == 2  # no data sent for synthesis


def test_without_document_index_everything_goes_to_data(small_db):
    llm = ScriptedLLM([{"answerable": True, "sql": "SELECT 1 AS x"}])
    turn = Copilot(llm, TextToSQL(llm, SafeExecutor(small_db)), None).ask("anything", summarize=False)
    assert turn.route == "data" and turn.data.ok


def test_refusal_has_no_citations(docs_db):
    llm = ScriptedLLM([{"answerable": False, "answer": "Not covered by the sources. [1][2]", "used_sources": [1, 2]}])
    ans = DocQA(llm, HybridRetriever(docs_db, HashEmbedder())).ask("What is the ECB deposit rate?")
    assert not ans.answerable and ans.citations == [] and ans.answer_text == "Not covered by the sources."
