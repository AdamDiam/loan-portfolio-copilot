"""Ingest regulatory PDFs into a searchable index.

    PDF -> pages -> cleaned text -> overlapping chunks (never crossing pages,
    so every chunk has an exact page citation) -> OpenAI embeddings + SQLite FTS5

Everything is stored in one SQLite file (data/docs.db):
    documents(doc_id, title, filename, url, n_pages)
    chunks(chunk_id, doc_id, page, section, text, embedding BLOB float32)
    chunks_fts  (FTS5 full-text index over chunks.text, for keyword search)

Usage:
    python -m copilot.rag.ingest
"""
from __future__ import annotations

import argparse
import re
import sqlite3
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from copilot.config import Settings
from copilot.rag.sources import REGULATION_DIR, SOURCES, Source

CHUNK_CHARS = 1200
OVERLAP_CHARS = 200
_HEADING_RE = re.compile(r"^\s*((?:\d{1,2}\.){1,3}\d{0,2}\s+[A-Z][^\n]{3,90})$", re.MULTILINE)


@dataclass
class Chunk:
    doc_id: str
    page: int  # 1-based PDF page number
    text: str
    section: str = ""


# ---------------------------------------------------------------------------
# PDF -> clean page texts
# ---------------------------------------------------------------------------
def extract_pages(pdf_path: Path) -> list[str]:
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    return [(page.extract_text() or "") for page in reader.pages]


def clean_pages(pages: list[str]) -> list[str]:
    """Remove running headers/footers and page numbers, fix hyphenation and whitespace."""
    line_counts: Counter[str] = Counter()
    for text in pages:
        for line in {ln.strip() for ln in text.splitlines() if ln.strip()}:
            line_counts[line] += 1
    threshold = max(3, int(0.3 * len(pages)))
    boilerplate = {line for line, n in line_counts.items() if n >= threshold and len(line) < 120}

    cleaned = []
    for text in pages:
        lines = []
        for ln in text.splitlines():
            s = ln.strip()
            if not s or s in boilerplate or re.fullmatch(r"(page\s*)?\d{1,3}(\s*(/|of)\s*\d{1,3})?", s, re.IGNORECASE):
                continue
            lines.append(s)
        t = "\n".join(lines)
        t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)          # de-hyphenate line breaks
        t = re.sub(r"[ \t]+", " ", t)
        t = re.sub(r"\n{3,}", "\n\n", t)
        cleaned.append(t.strip())
    return cleaned


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------
def chunk_page(text: str, size: int = CHUNK_CHARS, overlap: int = OVERLAP_CHARS) -> list[str]:
    """Split one page into overlapping chunks, preferring sentence/line boundaries."""
    text = text.strip()
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_document(doc_id: str, pages: list[str]) -> list[Chunk]:
    out: list[Chunk] = []
    section = ""
    for page_no, text in enumerate(pages, start=1):
        for piece in chunk_page(text):
            headings = _HEADING_RE.findall(piece)
            if headings:
                section = headings[0].strip()[:100]
            out.append(Chunk(doc_id=doc_id, page=page_no, text=piece, section=section))
    return out


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE documents (doc_id TEXT PRIMARY KEY, title TEXT, filename TEXT, url TEXT, n_pages INTEGER);
CREATE TABLE chunks (
    chunk_id  INTEGER PRIMARY KEY,
    doc_id    TEXT NOT NULL REFERENCES documents (doc_id),
    page      INTEGER NOT NULL,
    section   TEXT,
    text      TEXT NOT NULL,
    embedding BLOB
);
CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT);
"""


def fts5_available() -> bool:
    try:
        sqlite3.connect(":memory:").execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        return True
    except sqlite3.OperationalError:
        return False


def build_index(
    db_path: Path,
    documents: list[tuple[Source, int]],
    chunks: list[Chunk],
    embedder,
    progress: bool = True,
) -> None:
    """Embed `chunks` and write the whole index to `db_path` (replacing it)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = db_path.with_suffix(".tmp")
    if tmp.exists():
        tmp.unlink()

    vectors = np.zeros((0, 0), dtype=np.float32)
    if chunks:
        if progress:
            print(f"Embedding {len(chunks)} chunks with {embedder.model} ...")
        vectors = np.asarray(embedder.embed([c.text for c in chunks]), dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        vectors = vectors / np.where(norms == 0, 1, norms)

    con = sqlite3.connect(tmp)
    try:
        con.executescript(SCHEMA)
        con.executemany(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            [(s.doc_id, s.title, s.filename, s.url, n) for s, n in documents],
        )
        con.executemany(
            "INSERT INTO chunks (doc_id, page, section, text, embedding) VALUES (?, ?, ?, ?, ?)",
            [(c.doc_id, c.page, c.section, c.text, vectors[i].tobytes()) for i, c in enumerate(chunks)],
        )
        has_fts = fts5_available()
        if has_fts:
            con.executescript(
                """
                CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='chunk_id',
                                                       tokenize='porter unicode61');
                INSERT INTO chunks_fts (rowid, text) SELECT chunk_id, text FROM chunks;
                """
            )
        con.executemany(
            "INSERT INTO index_metadata VALUES (?, ?)",
            [
                ("embedding_model", embedder.model),
                ("dimensions", str(vectors.shape[1] if vectors.size else 0)),
                ("fts5", "1" if has_fts else "0"),
                ("built_at", time.strftime("%Y-%m-%d %H:%M:%S")),
            ],
        )
        con.commit()
    finally:
        con.close()
    tmp.replace(db_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index the regulatory PDFs for retrieval.")
    parser.add_argument("--out", type=Path, default=Settings.from_env().docs_db_path)
    args = parser.parse_args()

    available = [s for s in SOURCES if (REGULATION_DIR / s.filename).exists()]
    if not available:
        sys.exit(f"No PDFs found in {REGULATION_DIR}. Run `python scripts/download_docs.py` first.")

    documents, chunks = [], []
    for src in available:
        pages = clean_pages(extract_pages(REGULATION_DIR / src.filename))
        doc_chunks = chunk_document(src.doc_id, pages)
        documents.append((src, len(pages)))
        chunks.extend(doc_chunks)
        print(f"  {src.doc_id:<18} {len(pages):>4} pages  {len(doc_chunks):>5} chunks")

    from copilot.llm import OpenAIEmbedder

    settings = Settings.from_env()
    t0 = time.perf_counter()
    build_index(args.out, documents, chunks, OpenAIEmbedder(settings.embedding_model))
    print(f"Wrote {args.out} ({len(chunks)} chunks) in {time.perf_counter() - t0:.1f}s")


if __name__ == "__main__":
    main()
