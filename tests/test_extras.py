import numpy as np
import pandas as pd

from themescan import chains as ch
from themescan import clusters as cl
from themescan import drivers as dr
from themescan import weekly


def test_driver_tailwind_uses_fixed_signs():
    z = pd.DataFrame({"real_yield": [1.0], "dollar": [-2.0], "wti": [0.5]}, index=pd.to_datetime(["2026-01-02"]))
    assert dr.tailwind(z, "g:gold") == np.mean([-1.0, 2.0])  # rising real yield hurts, falling dollar helps
    assert dr.tailwind(z, "b:oil") == 0.5
    assert np.isnan(dr.tailwind(z, "b:defense"))  # no mapped driver


def test_driver_series_are_lagged_for_publication():
    assert dr.SERIES["copper"][2] >= 30 and dr.SERIES["real_yield"][2] >= 1


def test_fuel_gauge_skips_off_cycle_rows():
    idx = pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-08-31", "2026-09-30"])
    c = pd.DataFrame({"A": [10, 12, 14, 16, np.nan, 20], "B": [5, 6, 7, 8, np.nan, 10], "O": [np.nan, np.nan, np.nan, np.nan, 30, np.nan]}, index=idx)
    f = cl.fuel_from_frame(c)
    assert f["companies"] == ["A", "B"] and abs(f["yoy"] - 1.0) < 1e-9 and abs(f["qoq"] - (30 / 24 - 1)) < 1e-9


def test_chain_yoy_matches_year_ago_quarter():
    s = pd.Series([100.0, 110, 120, 130, 150], index=pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"]))
    yoy, qoq, last = ch._yoy(s)
    assert abs(yoy - 0.5) < 1e-9 and abs(qoq - (150 / 130 - 1)) < 1e-9


def test_chain_reading_flags_fundamentals_ahead_of_price():
    res = dict(fuel=dict(latest=100.0, companies=["A"], yoy=0.8, qoq=0.1),
               suppliers={"power": dict(rev_yoy=0.2, gm_chg=0.0, verdict="Avoid", score=30.0, rev_qoq=0.0)})
    assert any("fundamentals ahead of price" in l for l in ch.reading(res))


def test_weekly_changes_and_compact():
    now = pd.DataFrame({"conviction": [80.0, 30.0, 70.0], "verdict": ["Strong", "Avoid", "Positive"]}, index=["a", "b", "c"])
    prev = pd.DataFrame({"conviction": [60.0, 50.0, 72.0], "verdict": ["Neutral", "Neutral", "Positive"], "date": pd.Timestamp("2026-10-23")}, index=["a", "b", "c"])
    lines = weekly.changes_section("themes.yaml", now, prev)
    text = "\n".join(lines)
    assert "a (Neutral -> Strong)" in text and "b (Neutral -> Avoid)" in text
    assert "Strong" in weekly.compact({"Picks": lines})
