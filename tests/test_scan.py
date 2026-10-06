import warnings

import pandas as pd
import pytest

from themescan import backtest, scoring, synthetic
from themescan.config import load_config

warnings.filterwarnings("ignore")


@pytest.fixture(scope="module")
def world():
    cfg = load_config("themes.yaml")
    close, vol = synthetic.make(cfg, hot={"gold": 60, "korea": 45})
    return cfg, close, vol


def test_planted_runs_rank_top_and_get_flagged(world):
    cfg, close, vol = world
    df, _ = scoring.scan(close, vol, close[cfg.benchmark], cfg.themes)
    assert {"gold", "korea"} <= set(df.index[:6])
    assert df.loc["gold", "stage"] in ("Early trend", "Basing", "Extended")
    assert df.loc["gold", "score"] > df.score.median()


def test_no_lookahead(world):
    """Scoring as of row N must not change if later data is altered."""
    cfg, close, vol = world
    n = len(close) - 100
    a, _ = scoring.score_asof(close, vol, close[cfg.benchmark], cfg.themes, n)
    c2 = close.copy()
    c2.iloc[n:] *= 3
    b, _ = scoring.score_asof(c2, vol, c2[cfg.benchmark], cfg.themes, n)
    pd.testing.assert_frame_equal(a.drop(columns="leaders"), b.drop(columns="leaders"))


def test_walk_forward_shape(world):
    cfg, close, vol = world
    h = backtest.walk_forward(close, vol, close[cfg.benchmark], cfg.themes, start="2026-01-01", step=20)
    assert {"theme", "stage", "prepping", "fwd_excess"} <= set(h.columns)
    assert h.theme.nunique() == len(cfg.themes)
    assert not backtest.summarize(h).empty
