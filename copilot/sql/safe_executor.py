"""Read-only, sandboxed execution of LLM-generated SQL.

Defence in depth — an LLM-written query can never modify or exfiltrate:
  1. the database is opened with `mode=ro` (SQLite read-only URI);
  2. a SQLite *authorizer* is installed: the engine itself only permits
     SELECT, reads of allow-listed tables/views and plain functions.
     INSERT/UPDATE/DELETE/DDL/PRAGMA/ATTACH, sqlite_master and any table
     not on the allow-list are denied at compile time — no regex
     guessing about what the SQL "looks like";
  3. one statement per call (multi-statement payloads are rejected);
  4. a wall-clock timeout via the progress handler, and a row cap.
"""
from __future__ import annotations

import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from copilot.config import ALLOWED_TABLES

# Authorizer action codes (stable SQLite constants; fall back to raw ints)
_SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
_READ = getattr(sqlite3, "SQLITE_READ", 20)
_FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
_RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)
_DENIED_FUNCTIONS = {"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"}

_COMMENT_RE = re.compile(r"(--[^\n]*)|(/\*.*?\*/)", re.DOTALL)
# CTE names declared in the query: "WITH [RECURSIVE] name [(cols)] AS (" and ", name AS ("
_CTE_RE = re.compile(
    r"(?:\bwith\s+(?:recursive\s+)?|,\s*)([a-z_][a-z0-9_]*)\s*(?:\([^()]*\))?\s+as\s+(?:not\s+)?(?:materialized\s+)?\(",
    re.IGNORECASE,
)


def cte_names(sql: str) -> set[str]:
    names = {m.group(1).lower() for m in _CTE_RE.finditer(_COMMENT_RE.sub(" ", sql))}
    return {n for n in names if not n.startswith(("sqlite_", "pragma_"))}


class UnsafeQueryError(Exception):
    """The query was rejected by the safety layer."""


class QueryExecutionError(Exception):
    """The query is allowed but failed to run (syntax error, bad column, ...)."""


@dataclass
class QueryResult:
    df: pd.DataFrame
    truncated: bool
    elapsed_ms: float


def clean_sql(sql: str) -> str:
    """Strip markdown fences, surrounding whitespace and trailing semicolons."""
    sql = sql.strip()
    sql = re.sub(r"^```(?:sql)?\s*", "", sql, flags=re.IGNORECASE)
    sql = re.sub(r"\s*```$", "", sql)
    return sql.strip().rstrip(";").strip()


class SafeExecutor:
    def __init__(
        self,
        db_path: Path,
        allowed_tables: frozenset[str] = ALLOWED_TABLES,
        max_rows: int = 1000,
        timeout_s: float = 15.0,
    ) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(
                f"Database not found at {self.db_path}. Run `python -m copilot.data.generate` first."
            )
        self.allowed_tables = {t.lower() for t in allowed_tables}
        self.max_rows = max_rows
        self.timeout_s = timeout_s
        self._denied: list[str] = []
        self._query_ctes: set[str] = set()

    # -- SQLite authorizer -------------------------------------------------
    def _authorizer(self, action, arg1, arg2, db_name, trigger):  # noqa: ANN001
        if action == _SELECT or action == _RECURSIVE:
            return sqlite3.SQLITE_OK
        if action == _READ:
            name = (arg1 or "").lower()
            if name in self.allowed_tables or (name in self._query_ctes and db_name is None):
                return sqlite3.SQLITE_OK
            self._denied.append(f"read of table '{arg1}'")
            return sqlite3.SQLITE_DENY
        if action == _FUNCTION:
            if arg2 and arg2.lower() in _DENIED_FUNCTIONS:
                self._denied.append(f"function '{arg2}'")
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        self._denied.append(f"operation code {action}")
        return sqlite3.SQLITE_DENY

    def _connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True, check_same_thread=False)
        con.set_authorizer(self._authorizer)
        return con

    # -- public API --------------------------------------------------------
    def precheck(self, sql: str) -> str:
        """Cheap structural checks with friendly messages (the authorizer is the real guard)."""
        sql = clean_sql(sql)
        if not sql:
            raise UnsafeQueryError("Empty query.")
        body = _COMMENT_RE.sub(" ", sql).strip().lower()
        if not (body.startswith("select") or body.startswith("with")):
            raise UnsafeQueryError("Only SELECT queries (optionally starting with WITH) are allowed.")
        return sql

    def run(self, sql: str) -> QueryResult:
        sql = self.precheck(sql)
        self._denied = []
        self._query_ctes = cte_names(sql)
        con = self._connect()
        deadline = time.monotonic() + self.timeout_s

        def _progress() -> int:  # non-zero aborts the query
            return 1 if time.monotonic() > deadline else 0

        con.set_progress_handler(_progress, 10_000)
        t0 = time.perf_counter()
        try:
            cur = con.execute(sql)
            columns = [d[0] for d in cur.description or []]
            rows = cur.fetchmany(self.max_rows + 1)
        except sqlite3.ProgrammingError as exc:  # e.g. multiple statements
            raise UnsafeQueryError(f"Rejected: {exc}") from exc
        except sqlite3.DatabaseError as exc:
            if self._denied:
                raise UnsafeQueryError("Blocked by read-only policy: " + "; ".join(dict.fromkeys(self._denied))) from exc
            if "interrupted" in str(exc).lower():
                raise QueryExecutionError(f"Query exceeded the {self.timeout_s:.0f}s time limit.") from exc
            raise QueryExecutionError(str(exc)) from exc
        finally:
            con.close()

        truncated = len(rows) > self.max_rows
        df = pd.DataFrame(rows[: self.max_rows], columns=columns)
        return QueryResult(df=df, truncated=truncated, elapsed_ms=(time.perf_counter() - t0) * 1000)

    def run_trusted(self, sql: str) -> pd.DataFrame:
        """Run a developer-written query (KPIs, eval gold SQL) with the same sandbox."""
        return self.run(sql).df
