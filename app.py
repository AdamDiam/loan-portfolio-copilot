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

from copilot.assistant import Copilot, Turn  # noqa: E402
from copilot.charts import ChartSpec  # noqa: E402
from copilot.config import SNAPSHOT_DATE, Settings  # noqa: E402
from copilot.llm import OpenAIClient, OpenAIEmbedder  # noqa: E402
from copilot.rag.answerer import DocAnswer, DocQA  # noqa: E402
from copilot.rag.retriever import HybridRetriever  # noqa: E402
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
    "📚 Regulation (EBA / ECB)": [
        "When does a forborne exposure stop being classified as forborne?",
        "When can a defaulted exposure return to non-defaulted status?",
        "How often should collateral of non-performing exposures be revalued?",
    ],
    "📊+📚 Data + regulation": [
        "How many of our active loans are forborne, and how long must the probation period last according to the guidelines?",
        "What is our NPE ratio, and what do the EBA guidelines require from banks above the NPE strategy threshold?",
        "How many NPE collaterals have valuations older than one year, and what do the guidelines say about revaluation frequency?",
    ],
}

ROUTE_LABELS = {"data": "📊 Portfolio data", "docs": "📚 Regulation", "both": "📊+📚 Data + regulation"}

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
def get_retriever(docs_db: str, embedding_model: str) -> HybridRetriever | None:
    if not Path(docs_db).exists():
        return None
    return HybridRetriever(Path(docs_db), OpenAIEmbedder(embedding_model))


@st.cache_resource
def get_copilot(model: str, db_path: str, max_retries: int, docs_db: str, embedding_model: str) -> Copilot:
    llm = OpenAIClient(model)
    retriever = get_retriever(docs_db, embedding_model)
    return Copilot(
        llm,
        TextToSQL(llm, get_executor(db_path), max_retries=max_retries),
        DocQA(llm, retriever) if retriever else None,
    )


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


def render_data(ans: Answer, key: str, show_follow_ups: bool) -> None:
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


def render_docs(ans: DocAnswer) -> None:
    if ans.error:
        st.error(ans.error)
        return
    if not ans.answerable:
        st.info(ans.answer_text or "The indexed regulation does not cover this question.")
    else:
        st.markdown(ans.answer_text)
    if ans.citations:
        with st.expander(f"Sources ({len(ans.citations)})", expanded=False):
            for c in ans.citations:
                section = f" · {c.section}" if c.section else ""
                st.markdown(f"**[{c.number}] {c.doc_id}, page {c.page}**{section}  \n*{c.title}*")
                st.caption(c.snippet + ("…" if len(c.snippet) >= 400 else ""))
    st.caption(f"{ans.latency_ms / 1000:.1f}s · {ans.usage.total_tokens:,} tokens · {len(ans.retrieved)} passages retrieved")


def render_turn(turn: Turn, key: str, show_follow_ups: bool) -> None:
    st.caption(f"Answered from: **{ROUTE_LABELS.get(turn.route, turn.route)}**")
    if turn.route == "both":
        if turn.synthesis:
            st.markdown(turn.synthesis)
            st.divider()
        if turn.data is not None:
            st.markdown("##### 📊 Portfolio data")
            render_data(turn.data, key, show_follow_ups=False)
        if turn.docs is not None:
            st.markdown("##### 📚 What the regulation says")
            render_docs(turn.docs)
    elif turn.docs is not None:
        render_docs(turn.docs)
    elif turn.data is not None:
        render_data(turn.data, key, show_follow_ups=show_follow_ups)


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
            "Ask about an NPL loan portfolio, or about the EBA / ECB rules that govern it, in plain English. "
            "Data questions become read-only SQL; regulatory questions are answered from the official texts, with page citations."
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
        retriever = get_retriever(str(settings.docs_db_path), settings.embedding_model) if os.getenv("OPENAI_API_KEY") else None
        if retriever:
            st.caption(f"📚 Regulation index: {len(retriever.documents)} documents · {retriever.n_chunks:,} passages")
        else:
            st.caption("📚 Regulation index not built: run `python scripts/download_docs.py` then `python -m copilot.rag.ingest`.")
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

    copilot = get_copilot(
        settings.openai_model, db, settings.max_retries, str(settings.docs_db_path), settings.embedding_model
    )
    history: list[Turn] = st.session_state.history

    typed = st.chat_input("Ask about the portfolio or the regulation, e.g. “When does a forborne exposure exit probation?”")
    question = typed or st.session_state.pop("pending", None)

    for i, past in enumerate(history):
        with st.chat_message("user"):
            st.markdown(past.question)
        with st.chat_message("assistant"):
            render_turn(past, key=str(i), show_follow_ups=not question and i == len(history) - 1)

    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Thinking: choosing the source, querying the data and/or reading the regulation…"):
                turn = copilot.ask(question, history=history, summarize=summarize)
            history.append(turn)
            render_turn(turn, key=str(len(history) - 1), show_follow_ups=True)


main()
