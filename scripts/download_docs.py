"""Download the public regulatory documents used by the RAG module.

Usage:  python scripts/download_docs.py

Files are saved to docs/regulation/ (git-ignored). Each document is
official, publicly available EBA / ECB guidance.
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from copilot.rag.sources import SOURCES, REGULATION_DIR  # noqa: E402

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"


def download(url: str, dest: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    if not data.startswith(b"%PDF"):
        raise ValueError("response is not a PDF")
    dest.write_bytes(data)


def main() -> None:
    REGULATION_DIR.mkdir(parents=True, exist_ok=True)
    ok = 0
    for src in SOURCES:
        dest = REGULATION_DIR / src.filename
        if dest.exists() and dest.stat().st_size > 10_000:
            print(f"  [skip] {src.filename} already downloaded")
            ok += 1
            continue
        try:
            download(src.url, dest)
            print(f"  [ok]   {src.filename}  ({dest.stat().st_size / 1e6:.1f} MB)")
            ok += 1
        except Exception as exc:  # noqa: BLE001
            print(f"  [fail] {src.filename}: {exc}\n         Download it manually from {src.url}\n         and save it as {dest}")
    print(f"\n{ok}/{len(SOURCES)} documents ready in {REGULATION_DIR}")


if __name__ == "__main__":
    main()
