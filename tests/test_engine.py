"""Engine behaviour with a scripted fake LLM (no API key needed)."""
import json

from copilot.llm import LLMResponse
from copilot.sql.engine import TextToSQL
from copilot.sql.safe_executor import SafeExecutor


class ScriptedLLM:
    model = "scripted"

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def complete_json(self, messages):
        self.calls.append(messages)
        data = self.replies.pop(0)
        return LLMResponse(data=data, raw=json.dumps(data), prompt_tokens=100, completion_tokens=20)


def _engine(db, replies, **kw):
    llm = ScriptedLLM(replies)
    return TextToSQL(llm, SafeExecutor(db), **kw), llm


def test_happy_path_with_summary(small_db):
    engine, llm = _engine(
        small_db,
        [
            {"answerable": True, "sql": "SELECT product_type, COUNT(*) AS n FROM loans GROUP BY product_type",
             "explanation": "Loans per product", "chart": {"type": "bar", "x": "product_type", "y": "n"}},
            {"answer": "Mortgages are the largest product."},
        ],
    )
    ans = engine.ask("loans per product?")
    assert ans.ok and len(ans.df) == 6
    assert ans.chart and ans.chart.kind == "bar"
    assert ans.answer_text.startswith("Mortgages")
    assert ans.usage.calls == 2 and ans.usage.total_tokens == 240


def test_self_correction_feeds_error_back(small_db):
    engine, llm = _engine(
        small_db,
        [
            {"answerable": True, "sql": "SELECT COUNT(*) FROM loan"},  # wrong table name
            {"answerable": True, "sql": "SELECT COUNT(*) AS n FROM loans"},
        ],
    )
    ans = engine.ask("how many loans?", summarize=False)
    assert ans.ok and len(ans.attempts) == 2
    assert "no such table" in llm.calls[1][-1]["content"]


def test_unsafe_sql_is_never_executed_and_retries_are_bounded(small_db):
    engine, _ = _engine(small_db, [{"answerable": True, "sql": "DELETE FROM loans"}] * 3, max_retries=2)
    ans = engine.ask("delete everything", summarize=False)
    assert not ans.ok and ans.error and len(ans.attempts) == 3
    count = SafeExecutor(small_db).run("SELECT COUNT(*) AS n FROM loans").df.n[0]
    assert count == 3000


def test_unanswerable_question(small_db):
    engine, _ = _engine(small_db, [{"answerable": False, "sql": None, "explanation": "No market data available."}])
    ans = engine.ask("What will the ECB rate be next year?")
    assert not ans.answerable and "market data" in ans.explanation and ans.df is None


def test_history_is_sent_for_follow_ups(small_db):
    engine, llm = _engine(
        small_db,
        [
            {"answerable": True, "sql": "SELECT COUNT(*) AS n FROM loans WHERE is_npe = 1"},
            {"answerable": True, "sql": "SELECT product_type, COUNT(*) AS n FROM loans WHERE is_npe = 1 GROUP BY 1"},
        ],
    )
    first = engine.ask("How many NPE loans?", summarize=False)
    engine.ask("Now split by product", history=[first], summarize=False)
    sent = llm.calls[1]
    assert any("How many NPE loans?" == m["content"] for m in sent if m["role"] == "user")
    assert any("is_npe = 1" in m["content"] for m in sent if m["role"] == "assistant")
