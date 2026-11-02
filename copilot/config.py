"""Central configuration: paths, environment variables and defaults."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path

try:  # python-dotenv is optional at import time (tests run without it)
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGE_DIR = Path(__file__).resolve().parent

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

# The portfolio is a point-in-time snapshot ("as-of" / reporting date).
SNAPSHOT_DATE = date(2026, 9, 30)

DATA_DIR = PROJECT_ROOT / "data"
SCHEMA_PATH = PACKAGE_DIR / "data" / "schema.sql"
SEMANTIC_LAYER_PATH = PACKAGE_DIR / "semantic_layer.yaml"
EVAL_QUESTIONS_PATH = PROJECT_ROOT / "eval" / "questions.yaml"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval" / "results"

# Tables (and views) the AI is allowed to read. Anything else is denied
# by the SQLite authorizer, including sqlite_master.
ALLOWED_TABLES = frozenset(
    {
        "portfolio_metadata",
        "borrowers",
        "loans",
        "collaterals",
        "payments",
        "forbearance_measures",
        "legal_actions",
        "v_loan_overview",
    }
)


@dataclass(frozen=True)
class Settings:
    db_path: Path
    openai_model: str
    max_rows: int
    query_timeout_s: float
    max_retries: int
    embedding_model: str
    docs_db_path: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            db_path=Path(os.getenv("COPILOT_DB_PATH", str(DATA_DIR / "portfolio.db"))),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-5.4-mini"),
            max_rows=int(os.getenv("COPILOT_MAX_ROWS", "1000")),
            query_timeout_s=float(os.getenv("COPILOT_QUERY_TIMEOUT_S", "15")),
            max_retries=int(os.getenv("COPILOT_MAX_RETRIES", "2")),
            embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            docs_db_path=Path(os.getenv("COPILOT_DOCS_DB_PATH", str(DATA_DIR / "docs.db"))),
        )
