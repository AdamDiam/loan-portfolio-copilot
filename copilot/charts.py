"""Decide whether and how to chart a query result.

The LLM suggests a chart; we validate it against the actual result and
fall back to simple heuristics when the suggestion does not fit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

_TIME_NAME = re.compile(r"(date|month|quarter|year|period|vintage|week)", re.IGNORECASE)
_TIME_VALUE = re.compile(r"^\d{4}(-\d{2}|-Q[1-4])?(-\d{2})?$")
_DPD_ORDER = ["Current", "1-30", "31-60", "61-90", "91-180", "181-360", "361-720", "720+"]


@dataclass(frozen=True)
class ChartSpec:
    kind: str  # "bar" | "line" | "pie"
    x: str
    y: str
    color: str | None = None
    category_order: tuple[str, ...] | None = None


def _is_numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def _looks_like_time(df: pd.DataFrame, col: str) -> bool:
    if _TIME_NAME.search(col):
        return True
    sample = df[col].dropna().astype(str).head(10)
    return len(sample) > 0 and sample.map(lambda v: bool(_TIME_VALUE.match(v))).all()


def _category_order(df: pd.DataFrame, col: str) -> tuple[str, ...] | None:
    values = set(df[col].dropna().astype(str))
    if values and values <= set(_DPD_ORDER):
        return tuple(v for v in _DPD_ORDER if v in values)
    return None


def choose_chart(df: pd.DataFrame, suggestion: dict | None = None) -> ChartSpec | None:
    if df is None or df.empty or len(df.columns) < 2 or len(df) < 2:
        return None

    suggestion = suggestion or {}
    kind = str(suggestion.get("type") or "").lower()
    x, y, color = suggestion.get("x"), suggestion.get("y"), suggestion.get("color")
    if kind == "none":
        return None
    if kind in {"bar", "line", "pie"} and x in df.columns and y in df.columns and _is_numeric(df[y]):
        if color not in df.columns or color in (x, y):
            color = None
        if kind == "pie" and (len(df) > 8 or (df[y] < 0).any()):
            kind = "bar"
        if len(df) > 60 and kind == "bar" and not _looks_like_time(df, x):
            return None
        return ChartSpec(kind, x, y, color, _category_order(df, x))

    # heuristic fallback: first non-numeric column vs first numeric column
    numeric = [c for c in df.columns if _is_numeric(df[c])]
    dims = [c for c in df.columns if c not in numeric]
    if not numeric:
        return None
    if dims:
        x = dims[0]
    elif len(numeric) >= 2 and _looks_like_time(df, numeric[0]):  # e.g. year as integer
        x = numeric[0]
        numeric = numeric[1:]
    else:
        return None
    y = numeric[0]
    if _looks_like_time(df, x):
        return ChartSpec("line", x, y)
    if len(df) <= 30:
        return ChartSpec("bar", x, y, None, _category_order(df, x))
    return None
