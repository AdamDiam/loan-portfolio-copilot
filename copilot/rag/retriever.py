"""Hybrid retrieval: semantic (embeddings) + keyword (SQLite FTS5 / BM25).

Regulatory text mixes concepts ("unlikely to pay") with exact terms
("Article 178", "90 days", "EBA/GL/2018/06"). Embeddings catch the first,
keyword search the second. The two rankings are merged with Reciprocal Rank
Fusion (RRF), a robust, parameter-light way to combine result lists.
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_STOPWORDS = set(
    """a an and are as at be by can do does for from has have how i in is it its of on or should that the their
    there these this to what when where which who why will with under about into than then they them we you""".split()
)


@dataclass
class RetrievedChunk:
    chunk_id: int
    doc_id: str
    title: str
    page: int
    section: str
    text: str
    score: float = 0.0
    vector_rank: int | None = None
    keyword_rank: int | None = None


def keyword_query(question: str) -> str:
    """Turn a natural-language question into a safe FTS5 OR-query."""
    tokens = [t for t in re.findall(r"[a-z0-9]+", question.lower()) if t not in _STOPWORDS and len(t) > 1]
    return " OR ".join(f'"{t}"' for t in dict.fromkeys(tokens))


class HybridRetriever:
    def __init__(self, db_path: Path, embedder, rrf_k: int = 60, candidates: int = 30) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Document index not found at {self.db_path}. Run `python -m copilot.rag.ingest`.")
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.candidates = candidates
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT c.chunk_id, c.doc_id, d.title, c.page, c.section, c.text, c.embedding "
                "FROM chunks c JOIN documents d ON d.doc_id = c.doc_id ORDER BY c.chunk_id"
            ).fetchall()
            meta = dict(con.execute("SELECT key, value FROM index_metadata").fetchall())
            self.documents = con.execute("SELECT doc_id, title, n_pages FROM documents ORDER BY doc_id").fetchall()
        finally:
            con.close()
        self.has_fts = meta.get("fts5") == "1"
        self.index_model = meta.get("embedding_model", "")
        self._chunks = {r[0]: RetrievedChunk(r[0], r[1], r[2], r[3], r[4] or "", r[5]) for r in rows}
        self._ids = np.array([r[0] for r in rows], dtype=np.int64)
        self._matrix = np.vstack([np.frombuffer(r[6], dtype=np.float32) for r in rows]) if rows else np.zeros((0, 1), np.float32)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)

    @property
    def n_chunks(self) -> int:
        return len(self._ids)

    # ------------------------------------------------------------------
    def vector_search(self, question: str, n: int) -> list[int]:
        if not self.n_chunks:
            return []
        q = np.asarray(self.embedder.embed([question])[0], dtype=np.float32)
        q /= np.linalg.norm(q) or 1.0
        sims = self._matrix @ q
        top = np.argsort(-sims)[:n]
        return [int(self._ids[i]) for i in top]

    def keyword_search(self, question: str, n: int) -> list[int]:
        query = keyword_query(question)
        if not self.has_fts or not query:
            return []
        con = self._connect()
        try:
            rows = con.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?", (query, n)
            ).fetchall()
        except sqlite3.OperationalError:
            return []
        finally:
            con.close()
        return [int(r[0]) for r in rows]

    def search(self, question: str, k: int = 6) -> list[RetrievedChunk]:
        vec = self.vector_search(question, self.candidates)
        kw = self.keyword_search(question, self.candidates)
        scores: dict[int, float] = {}
        for rank, cid in enumerate(vec):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        for rank, cid in enumerate(kw):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (self.rrf_k + rank + 1)
        ranked = sorted(scores, key=scores.get, reverse=True)[:k]
        out = []
        for cid in ranked:
            base = self._chunks[cid]
            out.append(
                RetrievedChunk(
                    **{**base.__dict__,
                       "score": round(scores[cid], 5),
                       "vector_rank": vec.index(cid) + 1 if cid in vec else None,
                       "keyword_rank": kw.index(cid) + 1 if cid in kw else None}
                )
            )
        return out
