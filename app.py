"""Loan Portfolio AI Copilot — Streamlit UI.

Run:  streamlit run app.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402
import plotly.express as px  # noqa: E402
import streamlit as st  # noqa: E402

from copilot.charts import ChartSpec  # noqa: E402
from copilot.config import SNAPSHOT_DATE, Settings  # noqa: E402
from copilot.llm import OpenAIClient  # noqa: E402
from copilot.sql.engine import Answer, TextToSQL  # noqa: E402
from copilot.sql.safe_executor import SafeExecutor  # noqa: E402

st.set_page_config(page_title="Loan Portfolio AI Copilot", page_icon="🏦", layout="wide")

EXAMPLES: dict[str, list[str]] = {
    "Portfolio": [
        "What is the gross book value and number of active loans by product type?",
        "Which 5 borrowers have the highest exposure?",
    ],
    "Risk & IFRS 9": [
        "What is the NPE ratio by region?",
        "Show GBV, provisions and coverage ratio by IFRS 9 stage",
        "How many loans became non-performing each quarter since 2024?",
    ],
    "Delinquency": [
        "Show the DPD bucket distribution of the active book",
        "What was the monthly collection rate over the last 12 months?",
    ],
    "Forbearance": [
        "Which forbearance measure types have the highest re-default rate?",
        "List loans restructured more than twice that are non-performing again",
    ],
    "Collateral & legal": [
        "How many collaterals have stale valuations (older than 3 years), by type?",
        "What is the average LTV of non-performing mortgages by region?",
        "How much have we recovered from completed auctions per year?",
    ],
}

KPI_SQL = """
SELECT
    COUNT(*)                                                                  AS active_loans,
    SUM(outstanding_principal_eur + accrued_interest_eur)                     AS gbv,
    100.0 * SUM(CASE WHEN is_npe = 1 THEN outstanding_principal_eur + accrued_interest_eur ELSE 0 END)
          / SUM(outstanding_principal_eur + accrued_interest_eur)            AS npe_ratio,
    100.0 * SUM(CASE WHEN is_npe = 1 THEN provision_eur ELSE 0 END)
          / SUM(CASE WHEN is_npe = 1 THEN outstanding_principal_eur + accrued_interest_eur END) AS npe_coverage,
    SUM(is_forborne)                                                          AS forborne_loans
FROM loans
WHERE loan_status = 'Active'
"""


# --------------------------------------------------------------------------
# resources
# --------------------------------------------------------------------------
@st.cache_resource
def get_executor(db_path: str) -> SafeExecutor:
    return SafeExecutor(Path(db_path))


@st.cache_resource
def get_engine(model: str, db_path: str, max_retries: int) -> TextToSQL:
    return TextToSQL(OpenAIClient(model), get_executor(db_path), max_retries=max_retries)


@st.cache_data
def load_kpis(db_path: str) -> dict:
    return get_executor(db_path).run_trusted(KPI_SQL).iloc[0].to_dict()


# --------------------------------------------------------------------------
# rendering helpers
# --------------------------------------------------------------------------
def eur_m(value: float) -> str:
    return f"€{value / 1e6:,.1f}m"


def ask_later(question: str) -> None:
    st.session_state.pending = question


def render_chart(spec: ChartSpec, df: pd.DataFrame) -> None:
    orders = {spec.x: list(spec.category_order)} if spec.category_order else None
    if spec.kind == "line":
        fig = px.line(df, x=spec.x, y=spec.y, color=spec.color, markers=True)
    elif spec.kind == "pie":
        fig = px.pie(df, names=spec.x, values=spec.y, hole=0.45)
    else:
        fig = px.bar(df, x=spec.x, y=spec.y, color=spec.color, category_orders=orders)
    fig.update_layout(margin=dict(l=10, r=10, t=30, b=10), height=380, legend_title_text="")
    st.plotly_chart(fig)


def render_answer(ans: Answer, key: str, show_follow_ups: bool) -> None:
    if ans.error:
        st.error(ans.error)
    elif not ans.answerable:
        st.info(ans.explanation)
    else:
        if ans.answer_text:
            st.markdown(ans.answer_text)
        if ans.explanation:
            st.caption(f"🔎 {ans.explanation}")
        if ans.chart is not None:
            render_chart(ans.chart, ans.df)
        st.dataframe(ans.df.style.format(precision=2, thousands=","), hide_index=True)
        if ans.truncated:
            st.caption(f"Showing the first {len(ans.df):,} rows.")
        st.download_button(
            "Download CSV",
            ans.df.to_csv(index=False).encode("utf-8"),
            file_name=f"loan_copilot_{key}.csv",
            mime="text/csv",
            key=f"dl_{key}",
        )

    if ans.attempts:
        with st.expander("SQL & details"):
            if ans.sql:
                st.code(ans.sql, language="sql")
            failed = [a for a in ans.attempts if a.error]
            if failed:
                st.markdown(f"**Self-correction:** {len(failed)} failed attempt(s) were fixed automatically.")
                for a in failed:
                    st.code(f"-- rejected: {a.error}\n{a.sql}", language="sql")
            st.caption(
                f"{ans.latency_ms / 1000:.1f}s · {ans.usage.calls} LLM call(s) · {ans.usage.total_tokens:,} tokens"
            )

    if show_follow_ups and ans.follow_ups:
        st.markdown("**Ask next:**")
        cols = st.columns(len(ans.follow_ups))
        for i, (col, q) in enumerate(zip(cols, ans.follow_ups)):
            col.button(q, key=f"fu_{key}_{i}", on_click=ask_later, args=(q,))


# --------------------------------------------------------------------------
# page
# --------------------------------------------------------------------------
def main() -> None:
    settings = Settings.from_env()
    db = str(settings.db_path)
    st.session_state.setdefault("history", [])

    with st.sidebar:
        st.markdown("## 🏦 Loan Portfolio AI Copilot")
        st.caption(
            "Ask questions about an NPL loan portfolio in plain English. The copilot writes SQL, "
            "runs it in a read-only sandbox, and explains the result."
        )
        summarize = st.toggle("Plain-English summary", value=True, help="Second LLM call that explains the result.")
        if st.button("🗑️ Clear conversation"):
            st.session_state.history = []
            st.rerun()
        st.divider()
        st.markdown("**Try asking**")
        for group, questions in EXAMPLES.items():
            with st.expander(group, expanded=group == "Risk & IFRS 9"):
                for i, q in enumerate(questions):
                    st.button(q, key=f"ex_{group}_{i}", on_click=ask_later, args=(q,))
        st.divider()
        st.caption(f"Model: `{settings.openai_model}` · Data: synthetic, as of {SNAPSHOT_DATE:%d %b %Y}")

    st.title("Loan Portfolio AI Copilot")
    st.caption(f"Synthetic non-performing loan book · reporting date {SNAPSHOT_DATE:%d %B %Y} · all amounts in EUR")

    if not Path(db).exists():
        st.error("No database found. Generate the synthetic portfolio first:")
        st.code("python -m copilot.data.generate", language="bash")
        st.stop()

    k = load_kpis(db)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Active loans", f"{int(k['active_loans']):,}")
    c2.metric("Gross book value", eur_m(k["gbv"]))
    c3.metric("NPE ratio", f"{k['npe_ratio']:.1f}%")
    c4.metric("NPE coverage", f"{k['npe_coverage']:.1f}%")
    c5.metric("Forborne loans", f"{int(k['forborne_loans']):,}")

    if not os.getenv("OPENAI_API_KEY"):
        st.warning("Set `OPENAI_API_KEY` in a `.env` file in the project folder (see `.env.example`) and restart.")
        st.stop()

    engine = get_engine(settings.openai_model, db, settings.max_retries)
    history: list[Answer] = st.session_state.history

    typed = st.chat_input("Ask about the portfolio, e.g. “What is the NPE ratio by product?”")
    question = typed or st.session_state.pop("pending", None)

    for i, past in enumerate(history):
        with st.chat_message("user"):
            st.markdown(past.question)
        with st.chat_message("assistant"):
            render_answer(past, key=str(i), show_follow_ups=not question and i == len(history) - 1)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking, writing SQL and running it…"):
                ans = engine.ask(question, history=history, summarize=summarize)
            history.append(ans)
            render_answer(ans, key=str(len(history) - 1), show_follow_ups=True)


main()
