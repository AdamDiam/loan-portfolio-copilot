-- =====================================================================
-- Loan Portfolio AI Copilot — synthetic NPL servicing data model
-- Engine: SQLite. Dates are ISO-8601 TEXT ('YYYY-MM-DD').
-- All data is synthetic; structure is loosely inspired by the public
-- EBA NPL data templates (borrower / loan / collateral / enforcement).
-- =====================================================================

PRAGMA foreign_keys = ON;

CREATE TABLE portfolio_metadata (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE borrowers (
    borrower_id        TEXT PRIMARY KEY,
    borrower_type      TEXT NOT NULL CHECK (borrower_type IN ('Individual', 'SME', 'Corporate')),
    region             TEXT NOT NULL,
    industry_sector    TEXT,              -- NULL for individuals
    employment_status  TEXT,              -- NULL for companies
    customer_since     TEXT NOT NULL
);

CREATE TABLE loans (
    loan_id                    TEXT PRIMARY KEY,
    borrower_id                TEXT NOT NULL REFERENCES borrowers (borrower_id),
    product_type               TEXT NOT NULL,
    origination_date           TEXT NOT NULL,
    maturity_date              TEXT,              -- NULL for revolving products
    original_amount_eur        REAL NOT NULL CHECK (original_amount_eur > 0),
    interest_rate_pct          REAL NOT NULL CHECK (interest_rate_pct >= 0),
    rate_type                  TEXT NOT NULL CHECK (rate_type IN ('Fixed', 'Floating')),
    loan_status                TEXT NOT NULL CHECK (loan_status IN ('Active', 'Closed', 'Written-off')),
    outstanding_principal_eur  REAL NOT NULL DEFAULT 0 CHECK (outstanding_principal_eur >= 0),
    accrued_interest_eur       REAL NOT NULL DEFAULT 0 CHECK (accrued_interest_eur >= 0),
    days_past_due              INTEGER CHECK (days_past_due >= 0),   -- NULL unless Active
    dpd_bucket                 TEXT,                                 -- NULL unless Active
    ifrs9_stage                INTEGER CHECK (ifrs9_stage IN (1, 2, 3)),  -- NULL unless Active
    is_npe                     INTEGER NOT NULL DEFAULT 0 CHECK (is_npe IN (0, 1)),
    unlikely_to_pay_flag       INTEGER NOT NULL DEFAULT 0 CHECK (unlikely_to_pay_flag IN (0, 1)),
    is_forborne                INTEGER NOT NULL DEFAULT 0 CHECK (is_forborne IN (0, 1)),
    default_date               TEXT,
    provision_eur              REAL NOT NULL DEFAULT 0 CHECK (provision_eur >= 0),
    servicing_strategy         TEXT,
    closed_date                TEXT,
    write_off_date             TEXT,
    write_off_amount_eur       REAL
);

CREATE TABLE collaterals (
    collateral_id    TEXT PRIMARY KEY,
    loan_id          TEXT NOT NULL REFERENCES loans (loan_id),
    collateral_type  TEXT NOT NULL,
    region           TEXT NOT NULL,
    valuation_eur    REAL NOT NULL CHECK (valuation_eur > 0),
    valuation_date   TEXT NOT NULL,
    lien_rank        INTEGER NOT NULL CHECK (lien_rank >= 1)
);

CREATE TABLE payments (
    payment_id       INTEGER PRIMARY KEY,
    loan_id          TEXT NOT NULL REFERENCES loans (loan_id),
    due_date         TEXT NOT NULL,
    amount_due_eur   REAL NOT NULL CHECK (amount_due_eur >= 0),
    amount_paid_eur  REAL NOT NULL CHECK (amount_paid_eur >= 0),
    payment_date     TEXT              -- NULL when nothing was paid
);

CREATE TABLE forbearance_measures (
    measure_id    TEXT PRIMARY KEY,
    loan_id       TEXT NOT NULL REFERENCES loans (loan_id),
    measure_type  TEXT NOT NULL,
    start_date    TEXT NOT NULL,
    end_date      TEXT NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('Active', 'Completed', 'Failed'))
);

CREATE TABLE legal_actions (
    action_id             TEXT PRIMARY KEY,
    loan_id               TEXT NOT NULL REFERENCES loans (loan_id),
    action_type           TEXT NOT NULL,
    filed_date            TEXT NOT NULL,
    status                TEXT NOT NULL CHECK (status IN ('Pending', 'In Progress', 'Completed', 'Suspended')),
    legal_costs_eur       REAL NOT NULL CHECK (legal_costs_eur >= 0),
    recovered_amount_eur  REAL
);

CREATE INDEX ix_loans_borrower       ON loans (borrower_id);
CREATE INDEX ix_loans_status_npe     ON loans (loan_status, is_npe);
CREATE INDEX ix_collaterals_loan     ON collaterals (loan_id);
CREATE INDEX ix_payments_loan_due    ON payments (loan_id, due_date);
CREATE INDEX ix_forbearance_loan     ON forbearance_measures (loan_id);
CREATE INDEX ix_legal_loan           ON legal_actions (loan_id);

-- Convenience view: one row per loan with borrower attributes and the
-- most-used derived metrics (GBV, collateral, LTV, forbearance count).
CREATE VIEW v_loan_overview AS
SELECT
    l.*,
    b.borrower_type,
    b.region,
    b.industry_sector,
    b.employment_status,
    l.outstanding_principal_eur + l.accrued_interest_eur AS gross_book_value_eur,
    COALESCE(c.total_collateral_eur, 0)                  AS total_collateral_eur,
    CASE WHEN c.total_collateral_eur > 0
         THEN ROUND((l.outstanding_principal_eur + l.accrued_interest_eur) / c.total_collateral_eur, 4)
    END                                                  AS ltv_ratio,
    COALESCE(f.forbearance_count, 0)                     AS forbearance_count
FROM loans l
JOIN borrowers b ON b.borrower_id = l.borrower_id
LEFT JOIN (
    SELECT loan_id, SUM(valuation_eur) AS total_collateral_eur
    FROM collaterals GROUP BY loan_id
) c ON c.loan_id = l.loan_id
LEFT JOIN (
    SELECT loan_id, COUNT(*) AS forbearance_count
    FROM forbearance_measures GROUP BY loan_id
) f ON f.loan_id = l.loan_id;
