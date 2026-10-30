"""Does news add information? Point-in-time tests on the headline archive (monthly), same train/test discipline as the rest.

Features are computed only from headlines dated inside each month and used from that month's END, with forward returns measured after it.
"""
import numpy as np
import pandas as pd

from . import news as nw


def month_end_alpha(hist: pd.DataFrame, horizon: int = 126) -> pd.DataFrame:
    """Forward alpha (vs same-date average theme) at each month end from a backtest CSV."""
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    h["month"] = h.date.dt.to_period("M").astype(str)
    last = h.sort_values("date").groupby(["theme", "month"]).last().reset_index()
    col = f"fwd_excess_{horizon}"
    last = last.dropna(subset=[col])
    last["alpha"] = last[col] - last.groupby("month")[col].transform("mean")
    return last[["theme", "month", "alpha", "score"]]


def build(h_news: pd.DataFrame, hist: pd.DataFrame, horizon=126) -> pd.DataFrame:
    f = nw.monthly_features(h_news)
    f = f[f.n >= 15].copy()  # months with too few headlines are noise
    f = f.sort_values(["theme", "month"])
    g = f.groupby("theme")
    f["tone_chg3"] = f.tone - g.tone.transform(lambda s: s.shift(1).rolling(3).mean())
    f["tone_z"] = (f.tone - g.tone.transform(lambda s: s.shift(1).rolling(12, min_periods=6).mean())) / g.tone.transform(lambda s: s.shift(1).rolling(12, min_periods=6).std())
    f["net"] = f.pos - f.neg
    a = month_end_alpha(hist, horizon)
    d = f.merge(a, on=["theme", "month"])
    d["score_z"] = d.groupby("month").score.transform(lambda s: (s - s.mean()) / s.std())
    d["tone_rank"] = d.groupby("month").tone.rank(pct=True)
    d["div"] = d.groupby("month").tone_chg3.rank(pct=True) - d.groupby("month").score.rank(pct=True)  # news improving more than price
    return d


def ic_table(d: pd.DataFrame, feats=("tone", "tone_chg3", "tone_z", "net", "pos", "neg", "div", "score"), split="2024-01") -> pd.DataFrame:
    rows = []
    for per, x in (("all", d), ("early", d[d.month < split]), ("late", d[d.month >= split])):
        for ft in feats:
            s = x.dropna(subset=[ft, "alpha"]).groupby("month").apply(lambda g: g[ft].rank().corr(g.alpha.rank()) if g[ft].nunique() > 2 and len(g) > 5 else np.nan, include_groups=False).dropna()
            if len(s) >= 6:
                rows.append(dict(period=per, feature=ft, months=len(s), ic=s.mean(), t=s.mean() / (s.std() / np.sqrt(len(s))) if s.std() > 0 else np.nan, pos_months=int((s > 0).sum())))
    return pd.DataFrame(rows).round(3)


def quintiles(d: pd.DataFrame, ft: str) -> dict:
    x = d.dropna(subset=[ft, "alpha"])
    q = np.ceil(x.groupby("month")[ft].rank(pct=True, method="first") * 5).clip(1, 5)
    return (x.groupby(q).alpha.mean() * 100).round(2).to_dict()
