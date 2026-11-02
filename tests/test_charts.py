import pandas as pd

from copilot.charts import choose_chart


def test_valid_suggestion_is_used():
    df = pd.DataFrame({"region": ["A", "B"], "gbv": [1.0, 2.0]})
    spec = choose_chart(df, {"type": "bar", "x": "region", "y": "gbv"})
    assert spec.kind == "bar" and spec.x == "region"


def test_invalid_suggestion_falls_back_to_heuristic():
    df = pd.DataFrame({"month": ["2026-01", "2026-02", "2026-03"], "rate_pct": [60.0, 58.0, 57.0]})
    spec = choose_chart(df, {"type": "bar", "x": "missing", "y": "rate_pct"})
    assert spec.kind == "line" and spec.x == "month"


def test_single_value_is_not_charted():
    assert choose_chart(pd.DataFrame({"npe_ratio_pct": [49.1]}), {"type": "bar"}) is None


def test_dpd_buckets_keep_natural_order():
    df = pd.DataFrame({"dpd_bucket": ["720+", "Current", "1-30"], "n": [3, 1, 2]})
    spec = choose_chart(df, {"type": "bar", "x": "dpd_bucket", "y": "n"})
    assert spec.category_order == ("Current", "1-30", "720+")
