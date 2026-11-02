"""Gold SQL must run, and the result comparer must be fair but strict."""
import pandas as pd
import pytest

from copilot.eval.compare import results_match
from copilot.eval.run_eval import load_questions
from copilot.sql.safe_executor import SafeExecutor


def test_every_gold_query_runs_and_returns_rows(small_db):
    ex = SafeExecutor(small_db, max_rows=100_000)
    for q in load_questions():
        df = ex.run_trusted(q["gold_sql"])
        assert not df.empty, q["id"]


def test_question_ids_are_unique():
    ids = [q["id"] for q in load_questions()]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize(
    "gold, pred, expected",
    [
        # renamed / reordered columns and rows
        (pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}), pd.DataFrame({"n": [2, 1], "k": ["y", "x"]}), True),
        # extra predicted column is fine
        (pd.DataFrame({"a": ["x"], "b": [1.0]}), pd.DataFrame({"k": ["x"], "v": [1.004], "extra": [9]}), True),
        # ratio vs percent
        (pd.DataFrame({"pct": [49.14]}), pd.DataFrame({"ratio": [0.4914]}), True),
        # different value
        (pd.DataFrame({"pct": [49.14]}), pd.DataFrame({"pct": [51.0]}), False),
        # different row count
        (pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1]}), False),
        # same column multisets but wrong pairing of values across rows
        (pd.DataFrame({"a": ["x", "y"], "b": [1, 2]}), pd.DataFrame({"a": ["x", "y"], "b": [2, 1]}), False),
    ],
)
def test_results_match(gold, pred, expected):
    assert results_match(gold, pred) is expected
