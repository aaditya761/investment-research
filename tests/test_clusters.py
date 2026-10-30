import numpy as np
import pandas as pd

from themescan import clusters as cl


def test_cluster_groups_correlated_themes_together():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2024-01-05", periods=70, freq="W-FRI")
    a, b = rng.normal(0, 0.03, 70), rng.normal(0, 0.03, 70)
    X = pd.DataFrame({f"a{i}": a + rng.normal(0, 0.005, 70) for i in range(3)} | {f"b{i}": b + rng.normal(0, 0.005, 70) for i in range(3)}, index=idx)
    lab = cl.cluster_at(X, idx[-1])
    assert lab["a0"] == lab["a1"] == lab["a2"] and lab["b0"] == lab["b1"] and lab["a0"] != lab["b0"]


def test_hot_requires_size_strength_and_breadth():
    sc = pd.Series({**{f"t{i}": 50.0 for i in range(20)}, "a": 95.0, "b": 90.0, "c": 85.0})
    assert cl.is_hot(["a", "b", "c"], sc)["hot"]
    assert not cl.is_hot(["a", "b"], sc)["hot"]  # too small
    assert not cl.is_hot(["t0", "t1", "t2"], sc)["hot"]  # not strong


def test_fuel_gauge_is_like_for_like():
    idx = pd.to_datetime(["2025-09-30", "2025-12-31", "2026-03-31", "2026-06-30", "2026-09-30"])
    c = pd.DataFrame({"A": [10, 12, 14, 16, 20], "B": [5, 6, 7, 8, 10], "C": [4, 4, 5, 5, np.nan]}, index=idx)
    f = cl.fuel_from_frame(c)
    assert f["companies"] == ["A", "B"] and abs(f["yoy"] - 1.0) < 1e-9 and abs(f["qoq"] - (30 / 24 - 1)) < 1e-9
