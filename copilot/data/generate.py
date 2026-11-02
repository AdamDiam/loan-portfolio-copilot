"""Synthetic NPL loan-portfolio generator.

Produces an internally consistent, reproducible (seeded) loan book of a
servicer managing mostly legacy, partly non-performing exposures:

    borrowers -> loans -> collaterals / payments / forbearance / legal actions

Consistency rules that the data honours (and that `quality.py` verifies):
  * DPD, DPD bucket, IFRS 9 stage and NPE status agree with each other
  * NPE  <=>  DPD > 90 OR unlikely-to-pay flag
  * Stage 3 <=> NPE; stage 2 for DPD > 30, forborne or SICR
  * forborne flag  <=>  a forbearance measure active or ended < 24m ago (probation)
  * recent missed instalments in `payments` line up with DPD
  * only Active loans carry DPD / stage / balances; closed and written-off
    loans carry close / write-off dates instead

Usage:
    python -m copilot.data.generate            # 20,000 loans -> data/portfolio.db
    python -m copilot.data.generate --loans 5000 --seed 7 --out data/small.db
"""
from __future__ import annotations

import argparse
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from copilot.config import DATA_DIR, SCHEMA_PATH, SNAPSHOT_DATE

EPOCH = date(1970, 1, 1)
SNAP = (SNAPSHOT_DATE - EPOCH).days  # snapshot as integer day number

REGIONS = {
    "Attica": 0.38,
    "Central Macedonia": 0.18,
    "Thessaly": 0.07,
    "Crete": 0.06,
    "Western Greece": 0.06,
    "Peloponnese": 0.05,
    "Eastern Macedonia and Thrace": 0.05,
    "Central Greece": 0.05,
    "Epirus": 0.03,
    "South Aegean": 0.03,
    "North Aegean": 0.02,
    "Ionian Islands": 0.02,
}
SECTORS = {
    "Retail Trade": 0.20,
    "Manufacturing": 0.14,
    "Construction": 0.13,
    "Tourism and Hospitality": 0.17,
    "Agriculture": 0.09,
    "Transport and Logistics": 0.08,
    "Real Estate Activities": 0.07,
    "Professional Services": 0.12,
}
EMPLOYMENT = {"Employed": 0.58, "Self-employed": 0.17, "Unemployed": 0.12, "Retired": 0.13}

PRODUCTS_BY_TYPE = {
    "Individual": {"Mortgage": 0.45, "Consumer Loan": 0.35, "Credit Card": 0.20},
    "SME": {"SME Term Loan": 0.60, "Credit Line": 0.40},
    "Corporate": {"Corporate Loan": 0.70, "Credit Line": 0.30},
}
# product -> (median amount, lognormal sigma, term months (lo, hi) or None, rate % (lo, hi), p_floating)
PRODUCT_SPEC = {
    "Mortgage": (90_000, 0.50, (240, 360), (2.0, 5.0), 0.65),
    "Consumer Loan": (12_000, 0.60, (36, 96), (7.0, 14.0), 0.0),
    "Credit Card": (4_000, 0.50, None, (14.0, 20.0), 0.0),
    "SME Term Loan": (120_000, 0.80, (36, 120), (4.0, 9.0), 0.60),
    "Credit Line": (60_000, 0.70, None, (5.0, 10.0), 0.70),
    "Corporate Loan": (1_200_000, 0.90, (60, 180), (3.0, 7.0), 0.70),
}
REVOLVING = {"Credit Card", "Credit Line"}

FORBEARANCE_TYPES = {
    "Payment Holiday": 0.18,
    "Term Extension": 0.27,
    "Interest Rate Reduction": 0.20,
    "Interest-only Period": 0.15,
    "Split and Balloon": 0.12,
    "Partial Debt Write-down": 0.08,
}

DPD_BUCKETS = [  # (upper bound inclusive, label)
    (0, "Current"),
    (30, "1-30"),
    (60, "31-60"),
    (90, "61-90"),
    (180, "91-180"),
    (360, "181-360"),
    (720, "361-720"),
    (10**9, "720+"),
]


# ----------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------
def _choice(rng: np.random.Generator, options: dict, size: int) -> np.ndarray:
    keys = list(options)
    p = np.array([options[k] for k in keys], dtype=float)
    return rng.choice(np.array(keys, dtype=object), size=size, p=p / p.sum())


def _iso(day_numbers) -> np.ndarray:
    """Integer day numbers -> ISO date strings (None for NaN)."""
    out = np.empty(len(day_numbers), dtype=object)
    for i, d in enumerate(day_numbers):
        out[i] = None if d is None or (isinstance(d, float) and np.isnan(d)) else (
            EPOCH + timedelta(days=int(d))
        ).isoformat()
    return out


def dpd_bucket(dpd: float | None) -> str | None:
    if dpd is None or (isinstance(dpd, float) and np.isnan(dpd)):
        return None
    for upper, label in DPD_BUCKETS:
        if dpd <= upper:
            return label
    return "720+"


def _add_months(day_number: int, months: int) -> int:
    d = EPOCH + timedelta(days=int(day_number))
    y, m = divmod(d.month - 1 + months, 12)
    year, month = d.year + y, m + 1
    day = min(d.day, 28)
    return (date(year, month, day) - EPOCH).days


# ----------------------------------------------------------------------------
# generation
# ----------------------------------------------------------------------------
def generate(n_loans: int = 20_000, seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)

    # ---------------- borrowers ----------------
    n_b = max(1, int(n_loans * 0.8))
    b_type = _choice(rng, {"Individual": 0.78, "SME": 0.18, "Corporate": 0.04}, n_b)
    b_region = _choice(rng, REGIONS, n_b)
    is_ind = b_type == "Individual"
    b_sector = np.where(is_ind, None, _choice(rng, SECTORS, n_b))
    b_employment = np.where(is_ind, _choice(rng, EMPLOYMENT, n_b), None)

    # ---------------- loans: static terms ----------------
    borrower_idx = np.concatenate([np.arange(n_b), rng.integers(0, n_b, n_loans - n_b)])
    rng.shuffle(borrower_idx)
    l_btype = b_type[borrower_idx]

    product = np.empty(n_loans, dtype=object)
    for t, opts in PRODUCTS_BY_TYPE.items():
        mask = l_btype == t
        product[mask] = _choice(rng, opts, int(mask.sum()))

    # vintages skew to the pre-crisis / crisis years typical of NPL books
    years = np.arange(2004, 2024)
    w = np.array([3, 5, 7, 8, 9, 8, 6, 5, 3, 2, 2, 2, 2, 3, 3, 3, 4, 4, 4, 4], dtype=float)
    orig_year = rng.choice(years, size=n_loans, p=w / w.sum())
    orig_day = np.array([(date(int(y), 1, 1) - EPOCH).days for y in orig_year]) + rng.integers(0, 365, n_loans)

    amount = np.zeros(n_loans)
    rate = np.zeros(n_loans)
    term = np.full(n_loans, np.nan)
    floating = np.zeros(n_loans, dtype=bool)
    for p, (median, sigma, term_rng, rate_rng, p_float) in PRODUCT_SPEC.items():
        m = product == p
        k = int(m.sum())
        if not k:
            continue
        amount[m] = np.round(np.exp(np.log(median) + sigma * rng.standard_normal(k)), -2).clip(500)
        rate[m] = np.round(rng.uniform(*rate_rng, k), 2)
        floating[m] = rng.random(k) < p_float
        if term_rng:
            term[m] = rng.integers(term_rng[0], term_rng[1] + 1, k)
    maturity_day = np.where(np.isnan(term), np.nan, orig_day + np.nan_to_num(term) * 30.44)
    matured = ~np.isnan(maturity_day) & (maturity_day < SNAP)

    # ---------------- lifecycle status ----------------
    u = rng.random(n_loans)
    status = np.where(
        matured,
        np.where(u < 0.75, "Closed", np.where(u < 0.90, "Written-off", "Active")),
        np.where(u < 0.05, "Closed", np.where(u < 0.11, "Written-off", "Active")),
    ).astype(object)
    active = status == "Active"
    closed = status == "Closed"
    wo = status == "Written-off"

    # ---------------- credit risk: NPE / DPD ----------------
    b_emp = b_employment[borrower_idx]
    b_sec = b_sector[borrower_idx]
    p_npe = np.full(n_loans, 0.33)
    p_npe *= np.where(b_emp == "Unemployed", 1.55, np.where(b_emp == "Retired", 0.9, 1.0))
    p_npe *= np.select(
        [b_sec == "Construction", b_sec == "Real Estate Activities", b_sec == "Tourism and Hospitality", b_sec == "Agriculture"],
        [1.35, 1.2, 1.15, 1.1],
        1.0,
    )
    p_npe *= np.select([product == "Consumer Loan", product == "Credit Card", product == "Mortgage"], [1.15, 1.2, 0.95], 1.0)
    p_npe *= np.where((orig_year >= 2006) & (orig_year <= 2010), 1.2, 1.0)
    p_npe = np.where(matured, 1.0, p_npe.clip(0, 0.95))  # matured-but-active => in default

    npe = active & (rng.random(n_loans) < p_npe)
    age_days = SNAP - orig_day

    dpd = np.full(n_loans, np.nan)
    # performing (or early arrears)
    perf = active & ~npe
    k = int(perf.sum())
    bucket = rng.choice(4, size=k, p=[0.70, 0.14, 0.09, 0.07])
    lo = np.array([0, 1, 31, 61])[bucket]
    hi = np.array([0, 30, 60, 90])[bucket]
    dpd[perf] = rng.integers(lo, hi + 1)
    # non-performing by arrears
    k = int(npe.sum())
    bucket = rng.choice(4, size=k, p=[0.15, 0.17, 0.20, 0.48])
    lo = np.array([91, 181, 361, 721])[bucket]
    hi = np.array([180, 360, 720, 4000])[bucket]
    dpd[npe] = np.minimum(rng.integers(lo, hi + 1), np.maximum(age_days[npe] - 30, 91))

    # unlikely-to-pay: NPE despite DPD <= 90
    utp = perf & (rng.random(n_loans) < 0.05)
    npe = npe | utp

    # ---------------- dates for defaults / closures / write-offs ----------------
    default_day = np.full(n_loans, np.nan)
    arrears_npe = npe & ~utp
    default_day[arrears_npe] = SNAP - (dpd[arrears_npe] - 90)
    default_day[utp] = SNAP - rng.integers(30, 400, int(utp.sum()))

    closed_day = np.full(n_loans, np.nan)
    early_close = closed & ~matured
    closed_day[closed & matured] = maturity_day[closed & matured]
    closed_day[early_close] = orig_day[early_close] + rng.uniform(0.15, 1.0, int(early_close.sum())) * (
        SNAP - orig_day[early_close] - 1
    )

    wo_day = np.full(n_loans, np.nan)
    k = int(wo.sum())
    earliest_wo = np.minimum(orig_day[wo] + 3 * 365, SNAP - 30)
    wo_day[wo] = earliest_wo + rng.uniform(0, 1, k) * (SNAP - earliest_wo)
    default_day[wo] = np.maximum(orig_day[wo] + 180, wo_day[wo] - rng.integers(180, 1500, k))
    default_day[wo] = np.minimum(default_day[wo], wo_day[wo] - 1)
    default_day = np.where(np.isnan(default_day), np.nan, np.minimum(default_day, SNAP))
    default_day = np.where(np.isnan(default_day), np.nan, np.maximum(default_day, orig_day + 1))

    # ---------------- forbearance measures ----------------
    p_forb = np.select(
        [active & npe, active & (np.nan_to_num(dpd) > 30), active, wo, closed],
        [0.36, 0.25, 0.10, 0.30, 0.06],
        0.0,
    )
    has_forb = rng.random(n_loans) < p_forb
    measure_rows = []
    m_types = list(FORBEARANCE_TYPES)
    m_p = np.array(list(FORBEARANCE_TYPES.values()))
    m_p = m_p / m_p.sum()
    end_limit = np.where(closed, closed_day, np.where(wo, wo_day, SNAP + 3 * 365))
    for i in np.flatnonzero(has_forb):
        n_m = int(rng.choice([1, 2, 3, 4], p=[0.62, 0.24, 0.10, 0.04]))
        latest_start = min(SNAP, end_limit[i]) - 60
        start = orig_day[i] + 365 + rng.integers(0, 3 * 365)
        if start > latest_start:
            start = orig_day[i] + rng.uniform(0.3, 0.9) * max(latest_start - orig_day[i], 1)
        loan_measures = []
        for _ in range(n_m):
            if start > latest_start:
                break
            duration = int(rng.integers(6, 37)) * 30
            end = min(start + duration, end_limit[i])
            loan_measures.append([int(start), int(end)])
            start = end + rng.integers(30, 365)
        for j, (s, e) in enumerate(loan_measures):
            last = j == len(loan_measures) - 1
            if e > SNAP:
                st = "Active"
            elif not last:
                st = "Failed" if rng.random() < 0.6 else "Completed"
            elif closed[i]:
                st = "Completed"
            elif wo[i] or npe[i]:
                st = "Failed" if rng.random() < 0.7 else "Completed"
            else:
                st = "Completed" if rng.random() < 0.8 else "Failed"
            measure_rows.append((i, rng.choice(m_types, p=m_p), s, e, st))

    forb = pd.DataFrame(measure_rows, columns=["loan_idx", "measure_type", "start", "end", "status"])
    # forborne = measure still running or ended within the 24-month probation period
    recent = forb[forb["end"] >= SNAP - 730]
    is_forborne = np.zeros(n_loans, dtype=bool)
    is_forborne[recent["loan_idx"].unique()] = True
    is_forborne &= active
    active_measure = np.zeros(n_loans, dtype=bool)
    active_measure[forb.loc[forb["status"] == "Active", "loan_idx"].unique()] = True

    # ---------------- IFRS 9 stage ----------------
    sicr = rng.random(n_loans) < 0.08
    stage = np.full(n_loans, np.nan)
    stage[active] = np.where(
        npe[active], 3, np.where((dpd[active] > 30) | is_forborne[active] | sicr[active], 2, 1)
    )

    # ---------------- balances ----------------
    outstanding = np.zeros(n_loans)
    revolving = np.isin(product, list(REVOLVING))
    amort = active & ~revolving
    stop_day = np.where(npe & ~np.isnan(default_day), default_day, SNAP)
    elapsed = ((stop_day - orig_day) / (np.nan_to_num(term, nan=1) * 30.44)).clip(0, 1)
    remaining = (1 - elapsed**1.5).clip(0.03, 1) * rng.uniform(0.95, 1.03, n_loans)
    outstanding[amort] = amount[amort] * remaining[amort]
    matured_active = active & matured
    outstanding[matured_active] = amount[matured_active] * rng.uniform(0.05, 0.4, int(matured_active.sum()))
    rev_active = active & revolving
    util = np.where(npe, rng.uniform(0.85, 1.10, n_loans), rng.uniform(0.25, 0.95, n_loans))
    outstanding[rev_active] = amount[rev_active] * util[rev_active]
    outstanding = np.round(outstanding, 2)

    accrued = np.zeros(n_loans)
    days_for_interest = np.where(npe, np.minimum(np.nan_to_num(dpd), 1095), np.nan_to_num(dpd))
    accrued[active] = (
        outstanding[active] * rate[active] / 100 * days_for_interest[active] / 365 * rng.uniform(0.6, 1.0, int(active.sum()))
    )
    accrued = np.round(accrued, 2)
    gbv = outstanding + accrued

    write_off_amount = np.full(n_loans, np.nan)
    write_off_amount[wo] = np.round(amount[wo] * rng.uniform(0.3, 0.95, int(wo.sum())), 2)

    # ---------------- collaterals ----------------
    hpi = np.select([orig_year <= 2008, orig_year <= 2013, orig_year <= 2018, orig_year <= 2021], [0.85, 0.90, 1.25, 1.20], 1.05)
    coll_rows = []
    secured_prob = np.select(
        [product == "Mortgage", product == "SME Term Loan", product == "Corporate Loan", product == "Credit Line", product == "Consumer Loan"],
        [1.0, 0.70, 0.85, 0.40, 0.03],
        0.0,
    )
    has_coll = (~closed) & (rng.random(n_loans) < secured_prob)
    l_region = b_region[borrower_idx]
    region_names = np.array(list(REGIONS), dtype=object)
    for i in np.flatnonzero(has_coll):
        p = product[i]
        if p == "Mortgage":
            items = ["Residential Real Estate"] + (["Land"] if rng.random() < 0.05 else [])
            total = amount[i] / rng.uniform(0.55, 0.90) * hpi[i] * rng.uniform(0.85, 1.15)
        elif p == "Consumer Loan":
            items = ["Cash Deposit"]
            total = amount[i] * rng.uniform(0.2, 0.6)
        else:
            n_items = int(rng.choice([1, 2, 3], p=[0.65, 0.27, 0.08]))
            items = list(
                rng.choice(
                    ["Commercial Real Estate", "Land", "Equipment", "Cash Deposit", "Residential Real Estate"],
                    size=n_items,
                    p=[0.45, 0.17, 0.18, 0.08, 0.12],
                )
            )
            total = amount[i] * rng.uniform(0.5, 1.4) * hpi[i]
        shares = rng.dirichlet(np.ones(len(items))) if len(items) > 1 else np.array([1.0])
        for item, share in zip(items, shares):
            value = total * share
            if item == "Equipment":
                value *= rng.uniform(0.2, 0.5)  # depreciated
            stale = rng.random() < 0.2
            val_day = SNAP - (rng.integers(3 * 365 + 1, 8 * 365) if stale else rng.integers(0, 730))
            val_day = max(val_day, orig_day[i])
            reg = l_region[i] if rng.random() < 0.9 else rng.choice(region_names)
            lien = int(rng.choice([1, 2, 3], p=[0.88, 0.10, 0.02]))
            coll_rows.append((i, item, reg, round(max(value, 500.0), 2), val_day, lien))
    coll = pd.DataFrame(coll_rows, columns=["loan_idx", "collateral_type", "region", "valuation_eur", "val_day", "lien_rank"])
    coll_total = np.zeros(n_loans)
    np.add.at(coll_total, coll["loan_idx"].to_numpy(dtype=int), coll["valuation_eur"].to_numpy())

    # ---------------- provisions (loan-loss allowance) ----------------
    coverage = np.zeros(n_loans)
    s1, s2, s3 = active & (stage == 1), active & (stage == 2), active & (stage == 3)
    coverage[s1] = rng.uniform(0.003, 0.012, int(s1.sum()))
    coverage[s2] = rng.uniform(0.03, 0.10, int(s2.sum()))
    unsecured_lgd = rng.uniform(0.55, 0.90, n_loans)
    secured_lgd = (np.maximum(0, gbv - 0.6 * coll_total) / np.where(gbv > 0, gbv, 1)).clip(0.15, 0.90)
    lgd = np.where(coll_total > 0, secured_lgd, unsecured_lgd)
    years_in_default = np.where(np.isnan(default_day), 0, (SNAP - default_day) / 365)
    coverage[s3] = (lgd[s3] + np.minimum(0.05 * years_in_default[s3], 0.25)).clip(0.15, 0.98)
    provision = np.round(gbv * coverage, 2)

    # ---------------- servicing strategy ----------------
    strategy = np.full(n_loans, None, dtype=object)
    dpd0 = np.nan_to_num(dpd)
    strategy[active & ~npe & (dpd0 == 0)] = "Standard Servicing"
    strategy[active & ~npe & (dpd0 > 0)] = "Early Collections"
    secured = coll_total > 0
    g1 = active & npe & (dpd0 <= 360)
    strategy[g1] = _choice(rng, {"Amicable Settlement": 0.45, "Restructuring": 0.45, "Legal Enforcement": 0.10}, int(g1.sum()))
    g2 = active & npe & (dpd0 > 360) & secured
    strategy[g2] = _choice(
        rng,
        {"Legal Enforcement": 0.40, "Collateral Liquidation": 0.30, "Restructuring": 0.15, "Loan Sale Candidate": 0.15},
        int(g2.sum()),
    )
    g3 = active & npe & (dpd0 > 360) & ~secured
    strategy[g3] = _choice(rng, {"Amicable Settlement": 0.30, "Legal Enforcement": 0.30, "Loan Sale Candidate": 0.40}, int(g3.sum()))
    strategy[active & npe & active_measure] = "Restructuring"

    # ---------------- payments (last 24 monthly instalments, active loans) ----------------
    due_days = np.array([(date(SNAPSHOT_DATE.year, SNAPSHOT_DATE.month, 10) - EPOCH).days])
    due_days = np.array([_add_months(int(due_days[0]), -k) for k in range(23, -1, -1)])  # oldest -> newest
    pay_idx = np.flatnonzero(active)
    r = rate[pay_idx] / 1200
    n_term = np.nan_to_num(term[pay_idx], nan=120)
    with np.errstate(divide="ignore", invalid="ignore"):
        annuity = amount[pay_idx] * r / (1 - (1 + r) ** (-n_term))
    instalment = np.where(revolving[pay_idx], np.maximum(25.0, 0.03 * outstanding[pay_idx]), annuity)
    instalment = np.round(np.nan_to_num(instalment, nan=50.0), 2)
    n_pay = len(pay_idx)
    missed = np.minimum(np.ceil(np.nan_to_num(dpd[pay_idx]) / 30), 24).astype(int)
    month_pos = np.arange(24)[None, :]
    in_arrears = month_pos >= (24 - missed[:, None])
    paid_ratio = np.where(rng.random((n_pay, 24)) < 0.95, 1.0, rng.uniform(0.3, 0.9, (n_pay, 24)))
    arrears_ratio = np.where(rng.random((n_pay, 24)) < 0.8, 0.0, rng.uniform(0.1, 0.5, (n_pay, 24)))
    paid_ratio = np.where(in_arrears, arrears_ratio, paid_ratio)
    # long-term defaulters sometimes make small goodwill payments under amicable talks
    long_default = (np.nan_to_num(dpd[pay_idx]) > 720)[:, None]
    goodwill = (rng.random((n_pay, 1)) < 0.15) & (rng.random((n_pay, 24)) < 0.25)
    paid_ratio = np.where(long_default & goodwill, rng.uniform(0.05, 0.3, (n_pay, 24)), np.where(long_default, 0.0, paid_ratio))
    # utp loans pay on time (by definition DPD is low)
    due_amt = np.repeat(instalment[:, None], 24, axis=1)
    paid_amt = np.round(due_amt * paid_ratio, 2)
    started = due_days[None, :] > orig_day[pay_idx][:, None]
    pay_lag = rng.integers(0, 13, (n_pay, 24))
    payments = pd.DataFrame(
        {
            "loan_idx": np.repeat(pay_idx, 24),
            "due_day": np.tile(due_days, n_pay),
            "amount_due_eur": due_amt.ravel(),
            "amount_paid_eur": paid_amt.ravel(),
            "pay_day": np.where(paid_amt > 0, due_days[None, :] + pay_lag, np.nan).ravel(),
            "started": started.ravel(),
        }
    )
    payments = payments[payments["started"]].drop(columns="started")
    payments["pay_day"] = np.where(payments["pay_day"] > SNAP, SNAP, payments["pay_day"])

    # ---------------- legal actions ----------------
    legal_rows = []
    legal_candidates = (active & npe & np.isin(strategy, ["Legal Enforcement", "Collateral Liquidation"])) | (
        active & npe & (dpd0 > 360) & (rng.random(n_loans) < 0.30)
    ) | (wo & (rng.random(n_loans) < 0.40))
    l_is_ind = l_btype == "Individual"
    for i in np.flatnonzero(legal_candidates):
        if secured[i]:
            if l_is_ind[i] and rng.random() < 0.25:
                steps = ["Payment Order", "Household Insolvency Filing (Law 3869/2010)"]
            else:
                steps = ["Payment Order", "Property Seizure", "Auction Scheduled", "Auction Completed"]
        else:
            steps = ["Payment Order", "Bank Account Garnishment"]
            if not l_is_ind[i] and rng.random() < 0.15:
                steps.append("Bankruptcy Filing")
        n_steps = int(rng.integers(1, len(steps) + 1))
        end_day = wo_day[i] if wo[i] else SNAP
        start_day = (default_day[i] if not np.isnan(default_day[i]) else orig_day[i]) + rng.integers(60, 400)
        if start_day >= end_day:
            start_day = end_day - rng.integers(30, 120)
        gaps = np.sort(rng.uniform(0, 1, n_steps)) * max(end_day - start_day, n_steps)
        for j in range(n_steps):
            step = steps[j]
            filed = int(start_day + gaps[j])
            last = j == n_steps - 1
            if not last:
                st = "Completed"
            elif step == "Auction Completed":
                st = "Completed"
            else:
                st = str(rng.choice(["Pending", "In Progress", "Suspended"], p=[0.30, 0.55, 0.15]))
            recovered = None
            if step == "Auction Completed":
                recovered = round(coll_total[i] * rng.uniform(0.40, 0.75), 2)
            elif step == "Bank Account Garnishment" and st == "Completed":
                recovered = round(gbv[i] * rng.uniform(0.01, 0.08), 2)
            costs = round(float(rng.uniform(300, 5000)), 2)
            legal_rows.append((i, step, filed, st, costs, recovered))
    legal = pd.DataFrame(legal_rows, columns=["loan_idx", "action_type", "filed_day", "status", "legal_costs_eur", "recovered_amount_eur"])

    # ---------------- assemble frames ----------------
    loan_ids = np.array([f"L{i + 1:07d}" for i in range(n_loans)], dtype=object)
    borrower_ids = np.array([f"B{i + 1:06d}" for i in range(n_b)], dtype=object)

    first_loan_day = pd.Series(orig_day).groupby(borrower_idx).min().reindex(range(n_b)).to_numpy()
    customer_since = first_loan_day - rng.integers(0, 5 * 365, n_b)

    borrowers = pd.DataFrame(
        {
            "borrower_id": borrower_ids,
            "borrower_type": b_type,
            "region": b_region,
            "industry_sector": b_sector,
            "employment_status": b_employment,
            "customer_since": _iso(customer_since),
        }
    )

    dpd_out = np.where(active, dpd, np.nan)
    loans = pd.DataFrame(
        {
            "loan_id": loan_ids,
            "borrower_id": borrower_ids[borrower_idx],
            "product_type": product,
            "origination_date": _iso(orig_day),
            "maturity_date": _iso(maturity_day),
            "original_amount_eur": amount,
            "interest_rate_pct": rate,
            "rate_type": np.where(floating, "Floating", "Fixed"),
            "loan_status": status,
            "outstanding_principal_eur": outstanding,
            "accrued_interest_eur": accrued,
            "days_past_due": [None if np.isnan(x) else int(x) for x in dpd_out],
            "dpd_bucket": [dpd_bucket(x) for x in dpd_out],
            "ifrs9_stage": [None if np.isnan(x) else int(x) for x in stage],
            "is_npe": (active & npe).astype(int),
            "unlikely_to_pay_flag": (active & utp).astype(int),
            "is_forborne": is_forborne.astype(int),
            "default_date": _iso(np.where(active & npe | wo, default_day, np.nan)),
            "provision_eur": provision,
            "servicing_strategy": strategy,
            "closed_date": _iso(closed_day),
            "write_off_date": _iso(wo_day),
            "write_off_amount_eur": [None if np.isnan(x) else float(x) for x in write_off_amount],
        }
    )

    collaterals = pd.DataFrame(
        {
            "collateral_id": [f"C{i + 1:07d}" for i in range(len(coll))],
            "loan_id": loan_ids[coll["loan_idx"].to_numpy(dtype=int)],
            "collateral_type": coll["collateral_type"].to_numpy(),
            "region": coll["region"].to_numpy(),
            "valuation_eur": coll["valuation_eur"].to_numpy(),
            "valuation_date": _iso(coll["val_day"].to_numpy()),
            "lien_rank": coll["lien_rank"].to_numpy(dtype=int),
        }
    )

    payments_df = pd.DataFrame(
        {
            "payment_id": np.arange(1, len(payments) + 1),
            "loan_id": loan_ids[payments["loan_idx"].to_numpy(dtype=int)],
            "due_date": _iso(payments["due_day"].to_numpy()),
            "amount_due_eur": payments["amount_due_eur"].to_numpy(),
            "amount_paid_eur": payments["amount_paid_eur"].to_numpy(),
            "payment_date": _iso(payments["pay_day"].to_numpy()),
        }
    )

    forbearance = pd.DataFrame(
        {
            "measure_id": [f"F{i + 1:07d}" for i in range(len(forb))],
            "loan_id": loan_ids[forb["loan_idx"].to_numpy(dtype=int)],
            "measure_type": forb["measure_type"].to_numpy(),
            "start_date": _iso(forb["start"].to_numpy()),
            "end_date": _iso(forb["end"].to_numpy()),
            "status": forb["status"].to_numpy(),
        }
    )

    legal_actions = pd.DataFrame(
        {
            "action_id": [f"A{i + 1:07d}" for i in range(len(legal))],
            "loan_id": loan_ids[legal["loan_idx"].to_numpy(dtype=int)],
            "action_type": legal["action_type"].to_numpy(),
            "filed_date": _iso(legal["filed_day"].to_numpy()),
            "status": legal["status"].to_numpy(),
            "legal_costs_eur": legal["legal_costs_eur"].to_numpy(),
            "recovered_amount_eur": legal["recovered_amount_eur"].to_numpy(),
        }
    )

    metadata = pd.DataFrame(
        {
            "key": ["snapshot_date", "currency", "generator_seed", "n_loans", "data_origin"],
            "value": [SNAPSHOT_DATE.isoformat(), "EUR", str(seed), str(n_loans), "synthetic"],
        }
    )

    return {
        "portfolio_metadata": metadata,
        "borrowers": borrowers,
        "loans": loans,
        "collaterals": collaterals,
        "payments": payments_df,
        "forbearance_measures": forbearance,
        "legal_actions": legal_actions,
    }


# ----------------------------------------------------------------------------
# persistence
# ----------------------------------------------------------------------------
def _to_python(value):
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    if isinstance(value, np.generic):
        value = value.item()
        if isinstance(value, float) and np.isnan(value):
            return None
    return value


def write_sqlite(frames: dict[str, pd.DataFrame], db_path: Path) -> None:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    con = sqlite3.connect(db_path)
    try:
        con.executescript(SCHEMA_PATH.read_text())
        for table, df in frames.items():  # dict order respects FK dependencies
            cols = list(df.columns)
            placeholders = ", ".join("?" for _ in cols)
            rows = ([_to_python(v) for v in row] for row in df.itertuples(index=False, name=None))
            con.executemany(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({placeholders})", rows)
        con.commit()
        con.execute("ANALYZE")
        con.commit()
    finally:
        con.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the synthetic NPL loan portfolio.")
    parser.add_argument("--loans", type=int, default=20_000, help="number of loans (default 20,000)")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    parser.add_argument("--out", type=Path, default=DATA_DIR / "portfolio.db", help="SQLite output path")
    args = parser.parse_args()

    t0 = time.perf_counter()
    frames = generate(args.loans, args.seed)
    write_sqlite(frames, args.out)
    print(f"Wrote {args.out} in {time.perf_counter() - t0:.1f}s")
    for name, df in frames.items():
        print(f"  {name:<22} {len(df):>10,} rows")


if __name__ == "__main__":
    main()
