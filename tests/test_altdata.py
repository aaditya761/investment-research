import numpy as np
import pandas as pd

from themescan import altdata
from themescan.config import Theme


def test_news_features_use_only_past_data():
    idx = pd.date_range("2022-01-01", periods=900, freq="D")
    s = pd.Series(np.random.default_rng(0).uniform(1, 2, len(idx)), index=idx)
    cut = idx[700]
    a = altdata.news_features(s, pd.DatetimeIndex([cut]))
    s2 = s.copy()
    s2.loc[idx[701]:] *= 100  # future spike must not change the as-of value
    b = altdata.news_features(s2, pd.DatetimeIndex([cut]))
    pd.testing.assert_frame_equal(a, b)


def test_analyst_features_net_upgrades():
    acts = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-10 14:30", "2023-01-12 09:00", "2023-01-15 10:00"]),
        "action": ["up", "up", "down"], "ticker": ["AAA", "AAA", "BBB"],
    })
    themes = {"t": Theme("t", "T", [], ["AAA", "BBB"])}
    f = altdata.analyst_features(acts, themes, pd.DatetimeIndex(["2023-01-20"]))
    assert f.net_upgr.iloc[0] == (2 - 1) / 2  # time-of-day must not break the daily alignment


def test_earnings_features_no_lookahead():
    earn = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-10", "2023-01-12", "2023-01-15", "2023-03-01"]),
        "surprise": [10.0, 20.0, -5.0, 99.0], "ticker": ["A", "B", "C", "A"],
    })
    themes = {"t": Theme("t", "T", [], ["A", "B", "C"])}
    f = altdata.earnings_features(earn, themes, pd.DatetimeIndex(["2023-01-20"]))
    assert f.surp.iloc[0] == 10.0 and f.beat.iloc[0] == 2 / 3  # the March report must not leak in
