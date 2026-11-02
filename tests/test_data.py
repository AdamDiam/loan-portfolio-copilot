"""The synthetic portfolio must satisfy every hard data-quality rule."""
import pytest

from copilot.data.generate import dpd_bucket, generate
from copilot.data.quality import run_checks


def test_all_hard_quality_checks_pass(small_db):
    failures = [(c.name, n) for c, n in run_checks(small_db) if c.severity == "error" and n > 0]
    assert failures == []


def test_generation_is_deterministic():
    a = generate(n_loans=500, seed=1)["loans"]
    b = generate(n_loans=500, seed=1)["loans"]
    assert a.equals(b)


def test_portfolio_has_a_realistic_mix():
    loans = generate(n_loans=3000, seed=3)["loans"]
    active = loans[loans.loan_status == "Active"]
    npe_share = active.is_npe.mean()
    assert 0.25 < npe_share < 0.65, "an NPL servicer book should be heavily non-performing"
    assert set(active.ifrs9_stage.dropna().astype(int)) == {1, 2, 3}
    assert {"Active", "Closed", "Written-off"} == set(loans.loan_status)


@pytest.mark.parametrize(
    "dpd, label",
    [(0, "Current"), (1, "1-30"), (30, "1-30"), (90, "61-90"), (91, "91-180"), (721, "720+"), (None, None)],
)
def test_dpd_bucket_boundaries(dpd, label):
    assert dpd_bucket(dpd) == label
