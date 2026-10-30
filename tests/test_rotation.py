import numpy as np
import pandas as pd

from themescan import rotation as ro
from themescan import synthetic
from themescan.config import load_config


def test_weekdays_drops_weekend_rows():
    idx = pd.date_range("2024-01-01", periods=14, freq="D")
    c = pd.DataFrame({"a": range(14)}, index=idx)
    c2, v2 = ro.weekdays(c, c)
    assert (c2.index.dayofweek < 5).all() and len(c2) == 10


def test_episodes_found_after_drop_with_gap():
    idx = pd.bdate_range("2022-01-03", periods=400)
    r = pd.Series(0.0005, index=idx)
    r.iloc[200:210] = -0.02  # ~ -19% drop
    eps = ro.episodes(r, drop=-0.10)
    assert len(eps) == 1 and eps[0] >= idx[200]


def test_current_state_flags_defensive_winner():
    cfg = load_config("themes.yaml")
    close, vol = synthetic.make(cfg, n_days=1700)
    close, vol = ro.weekdays(close, vol)
    themes = ro.merge_universes({"g": cfg})
    now, info = ro.current_state(close, vol, themes, cfg.benchmark, source=["g:memory", "g:ai_semis"])
    assert "g:memory" not in now.index and {"resid", "flow", "beta", "both"} <= set(now.columns)
    assert -1 <= now.flow.dropna().min() and now.flow.dropna().max() <= 1
