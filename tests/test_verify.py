import numpy as np
import pandas as pd

from themescan import synthetic, verify
from themescan.config import load_config


def _df():
    return pd.DataFrame({
        "score": [90, 80, 40, 10], "surp": [20.0, 10.0, 0.0, np.nan], "beat": [1.0, 0.9, 0.5, np.nan],
        "surp_chg": [5.0, 1.0, -2.0, np.nan], "accel": 0.0, "stage": "Leading", "ext": 0.0, "rsi": 50.0,
    }, index=["a", "b", "c", "d"])


def test_conviction_blends_earnings_and_price():
    out = verify.add_conviction(_df())
    assert out.loc["a", "conviction"] > out.loc["b", "conviction"] > out.loc["c", "conviction"]
    assert out.loc["d", "conviction"] == 10  # no earnings -> price score only
    assert not out.loc["d", "earn_covered"]


def test_price_only_theme_cannot_be_strong():
    df = _df()
    df.loc["d", "score"] = 99
    out = verify.add_conviction(df)
    assert out.loc["d", "conviction"] >= 80 and out.loc["d", "verdict"] == "Positive"


def test_noisy_earnings_themes_are_masked():
    cfg = load_config("themes.yaml")
    assert cfg.themes["crypto"].use_earnings is False
    df = _df().rename(index={"a": "crypto"})
    out = verify.mask_noisy_earnings(df, cfg.themes)
    assert out.loc["crypto", ["surp", "beat", "surp_chg"]].isna().all()


def test_checklist_statuses_and_summary():
    row = verify.add_conviction(_df()).loc["a"]
    checks = verify.build_checks(row, dict(fwd_pe=-5.0, eps_g=10.0, upside=20.0, rev_bal=0.5), (0, 50), 0.7, 0.0, 0.2)
    by = {c["name"]: c["status"] for c in checks}
    assert by["EPS surprise in top half of themes"] == "pass"
    assert by["Valuation not stretched vs growth"] == "na"  # negative P/E is not a pass
    assert by["Analyst upgrades net positive (60d)"] == "na"
    assert verify.summarize(checks)["validated"]["total"] == 4


def test_full_scan_demo_runs():
    cfg = load_config("themes.yaml")
    close, vol = synthetic.make(cfg)
    df, note = verify.full_scan(cfg, close, vol, fundamentals=False)
    assert {"conviction", "verdict", "earn_covered"} <= set(df.columns) and df.conviction.between(0, 100).all()
