"""Execution-accuracy comparison of two query results.

Two results match when they contain the same rows, ignoring:
  * column names and column order (each gold column must be matched by
    some predicted column; extra predicted columns are allowed),
  * row order,
  * small numeric differences (rounding) and the ratio-vs-percent scale
    (0.4567 vs 45.67).
"""
from __future__ import annotations

import math

import pandas as pd

_SCALES = (1.0, 100.0, 0.01)


def _norm(value):
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value))
    except ValueError:
        return str(value).strip().lower()


def _equal(a, b, scale: float = 1.0) -> bool:
    if a is None or b is None:
        return a is None and b is None
    if isinstance(a, float) and isinstance(b, float):
        b_scaled = b * scale
        return math.isclose(a, b_scaled, rel_tol=1e-3, abs_tol=0.011)
    return a == b


def _col_match(gold: list, pred: list) -> float | None:
    """Return the scale at which two columns hold the same multiset of values, else None."""
    def key(v):
        return (v is None, isinstance(v, str), v if v is not None else 0)

    g_sorted = sorted(gold, key=key)
    for scale in _SCALES:
        p_vals = [v * scale if isinstance(v, float) else v for v in pred]
        p_sorted = sorted(p_vals, key=key)
        if all(_equal(a, b) for a, b in zip(g_sorted, p_sorted)):
            return scale
        if not any(isinstance(v, float) for v in gold):
            break
    return None


def results_match(gold: pd.DataFrame, pred: pd.DataFrame) -> bool:
    if gold is None or pred is None:
        return False
    if len(gold) != len(pred):
        return False
    if gold.empty:
        return True

    g_cols = [[_norm(v) for v in gold.iloc[:, i]] for i in range(gold.shape[1])]
    p_cols = [[_norm(v) for v in pred.iloc[:, j]] for j in range(pred.shape[1])]

    # greedy column assignment
    mapping: list[tuple[int, float]] = []
    used: set[int] = set()
    for gi, gcol in enumerate(g_cols):
        found = None
        for pj, pcol in enumerate(p_cols):
            if pj in used:
                continue
            scale = _col_match(gcol, pcol)
            if scale is not None:
                found = (pj, scale)
                break
        if found is None:
            return False
        used.add(found[0])
        mapping.append(found)

    # row-level check on the matched columns (protects against shuffled pairings)
    def row_key(row):
        return tuple((v is None, isinstance(v, str), round(v, 2) if isinstance(v, float) else v if v is not None else 0) for v in row)

    g_rows = sorted(zip(*g_cols), key=row_key)
    p_rows = sorted(
        (tuple(p_cols[pj][r] * s if isinstance(p_cols[pj][r], float) else p_cols[pj][r] for pj, s in mapping) for r in range(len(pred))),
        key=row_key,
    )
    return all(all(_equal(a, b) for a, b in zip(gr, pr)) for gr, pr in zip(g_rows, p_rows))
