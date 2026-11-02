"""Data-quality checks for the loan portfolio.

Each check is a SQL query that must return 0 offending rows (hard checks)
or is reported as a warning (soft checks: real-world data issues that a
servicer would monitor, e.g. stale collateral valuations).

Usage:
    python -m copilot.data.quality                 # checks data/portfolio.db
    python -m copilot.data.quality --db other.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

from copilot.config import Settings


@dataclass(frozen=True)
class Check:
    name: str
    sql: str  # returns COUNT(*) of offending rows
    severity: str = "error"  # "error" fails the run, "warning" is informational


CHECKS: list[Check] = [
    # --- referential integrity ---
    Check("loans -> borrowers FK", "SELECT COUNT(*) FROM loans l LEFT JOIN borrowers b USING (borrower_id) WHERE b.borrower_id IS NULL"),
    Check("collaterals -> loans FK", "SELECT COUNT(*) FROM collaterals c LEFT JOIN loans l USING (loan_id) WHERE l.loan_id IS NULL"),
    Check("payments -> loans FK", "SELECT COUNT(*) FROM payments p LEFT JOIN loans l USING (loan_id) WHERE l.loan_id IS NULL"),
    Check("forbearance -> loans FK", "SELECT COUNT(*) FROM forbearance_measures f LEFT JOIN loans l USING (loan_id) WHERE l.loan_id IS NULL"),
    Check("legal actions -> loans FK", "SELECT COUNT(*) FROM legal_actions a LEFT JOIN loans l USING (loan_id) WHERE l.loan_id IS NULL"),
    # --- lifecycle consistency ---
    Check(
        "active loans have DPD, bucket and stage",
        "SELECT COUNT(*) FROM loans WHERE loan_status = 'Active' AND (days_past_due IS NULL OR dpd_bucket IS NULL OR ifrs9_stage IS NULL)",
    ),
    Check(
        "non-active loans carry no DPD/stage/balance",
        """SELECT COUNT(*) FROM loans WHERE loan_status <> 'Active'
           AND (days_past_due IS NOT NULL OR ifrs9_stage IS NOT NULL OR outstanding_principal_eur <> 0 OR is_npe = 1)""",
    ),
    Check("closed loans have closed_date", "SELECT COUNT(*) FROM loans WHERE loan_status = 'Closed' AND closed_date IS NULL"),
    Check(
        "written-off loans have write-off date & amount",
        "SELECT COUNT(*) FROM loans WHERE loan_status = 'Written-off' AND (write_off_date IS NULL OR write_off_amount_eur IS NULL)",
    ),
    Check("origination before maturity", "SELECT COUNT(*) FROM loans WHERE maturity_date IS NOT NULL AND maturity_date <= origination_date"),
    Check("default after origination", "SELECT COUNT(*) FROM loans WHERE default_date IS NOT NULL AND default_date < origination_date"),
    # --- risk-classification consistency (EBA / IFRS 9 rules) ---
    Check(
        "NPE <=> DPD > 90 or unlikely-to-pay",
        """SELECT COUNT(*) FROM loans WHERE loan_status = 'Active'
           AND is_npe <> CASE WHEN days_past_due > 90 OR unlikely_to_pay_flag = 1 THEN 1 ELSE 0 END""",
    ),
    Check("stage 3 <=> NPE", "SELECT COUNT(*) FROM loans WHERE loan_status = 'Active' AND (ifrs9_stage = 3) <> (is_npe = 1)"),
    Check(
        "performing DPD > 30 or forborne => stage 2",
        "SELECT COUNT(*) FROM loans WHERE loan_status = 'Active' AND is_npe = 0 AND (days_past_due > 30 OR is_forborne = 1) AND ifrs9_stage <> 2",
    ),
    Check("NPE loans have default_date", "SELECT COUNT(*) FROM loans WHERE is_npe = 1 AND default_date IS NULL"),
    Check(
        "DPD bucket matches DPD",
        """SELECT COUNT(*) FROM loans WHERE loan_status = 'Active' AND dpd_bucket <> CASE
             WHEN days_past_due = 0 THEN 'Current' WHEN days_past_due <= 30 THEN '1-30'
             WHEN days_past_due <= 60 THEN '31-60' WHEN days_past_due <= 90 THEN '61-90'
             WHEN days_past_due <= 180 THEN '91-180' WHEN days_past_due <= 360 THEN '181-360'
             WHEN days_past_due <= 720 THEN '361-720' ELSE '720+' END""",
    ),
    Check(
        "forborne flag <=> measure active or in 24m probation",
        """SELECT COUNT(*) FROM loans l WHERE l.loan_status = 'Active' AND l.is_forborne <> EXISTS (
             SELECT 1 FROM forbearance_measures f WHERE f.loan_id = l.loan_id
             AND f.end_date >= date((SELECT value FROM portfolio_metadata WHERE key = 'snapshot_date'), '-730 days'))""",
    ),
    Check("provisions do not exceed GBV", "SELECT COUNT(*) FROM loans WHERE provision_eur > outstanding_principal_eur + accrued_interest_eur + 0.01"),
    # --- payments vs arrears ---
    Check("amount paid <= amount due", "SELECT COUNT(*) FROM payments WHERE amount_paid_eur > amount_due_eur + 0.01"),
    Check("payment date only when something was paid", "SELECT COUNT(*) FROM payments WHERE (amount_paid_eur > 0) <> (payment_date IS NOT NULL)"),
    Check(
        "current loans paid their latest instalment",
        """SELECT COUNT(*) FROM loans l JOIN payments p ON p.loan_id = l.loan_id
           WHERE l.loan_status = 'Active' AND l.days_past_due = 0
             AND p.due_date = (SELECT MAX(due_date) FROM payments) AND p.amount_paid_eur = 0""",
    ),
    # --- forbearance / legal ---
    Check("measure start before end", "SELECT COUNT(*) FROM forbearance_measures WHERE start_date > end_date"),
    Check(
        "'Active' measures end after snapshot",
        "SELECT COUNT(*) FROM forbearance_measures WHERE status = 'Active' AND end_date <= (SELECT value FROM portfolio_metadata WHERE key = 'snapshot_date')",
    ),
    Check("recoveries only on completed actions", "SELECT COUNT(*) FROM legal_actions WHERE recovered_amount_eur IS NOT NULL AND status <> 'Completed'"),
    # --- soft checks: things a servicer would actually monitor ---
    Check(
        "collateral valuations older than 3 years",
        """SELECT COUNT(*) FROM collaterals WHERE valuation_date <
           date((SELECT value FROM portfolio_metadata WHERE key = 'snapshot_date'), '-3 years')""",
        severity="warning",
    ),
    Check(
        "NPE loans with no collateral and no legal action",
        """SELECT COUNT(*) FROM loans l WHERE l.is_npe = 1
           AND NOT EXISTS (SELECT 1 FROM collaterals c WHERE c.loan_id = l.loan_id)
           AND NOT EXISTS (SELECT 1 FROM legal_actions a WHERE a.loan_id = l.loan_id)""",
        severity="warning",
    ),
]


def run_checks(db_path: Path) -> list[tuple[Check, int]]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return [(check, int(con.execute(check.sql).fetchone()[0])) for check in CHECKS]
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run data-quality checks on the portfolio database.")
    parser.add_argument("--db", type=Path, default=Settings.from_env().db_path)
    args = parser.parse_args()
    if not args.db.exists():
        sys.exit(f"Database not found: {args.db}. Run `python -m copilot.data.generate` first.")

    results = run_checks(args.db)
    failures = 0
    print(f"Data-quality report for {args.db}\n")
    for check, n in results:
        if n == 0:
            mark = "PASS"
        elif check.severity == "warning":
            mark = "WARN"
        else:
            mark = "FAIL"
            failures += 1
        print(f"  [{mark}] {check.name:<55} {n:>8,} rows")
    print(f"\n{len(results) - failures}/{len(results)} checks passed" + (" (warnings are informational)" if failures == 0 else ""))
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
