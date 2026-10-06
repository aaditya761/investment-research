"""Aggregate ticker signals into theme scores and classify the stage of each theme."""
import numpy as np
import pandas as pd

from .config import Theme
from .signals import ticker_metrics

# Relative weights of the score components (percentile ranks across themes).
WEIGHTS = dict(rs=0.30, trend=0.15, breadth=0.25, volume=0.10, coil=0.10, high=0.10)
ACCEL_BARS = 20  # score change measured over ~4 weeks
EXT_Z, EXT_RSI = 2.0, 78


def aggregate(tm: pd.DataFrame, themes: dict[str, Theme]) -> pd.DataFrame:
    rows = {}
    for key, th in themes.items():
        sub = tm.reindex([t for t in th.tickers if t in tm.index]).dropna(how="all")
        if len(sub) < 2:
            continue
        coil = ((1 - sub.sqz) * sub.above50).mean()  # compressed recently AND now above 50d
        rows[key] = dict(
            n=len(sub), rs=sub.rs.median(), r3m=sub.r3m.median(), d200=sub.d200.median(),
            slope200=sub.slope200.median(), breadth50=sub.above50.mean(), breadth200=sub.above200.mean(),
            at_high=sub.at_high.mean(), nh=sub.near_high.median(), volr=sub.volr.median(),
            coil=coil, ext=sub.ext.median(), rsi=sub.rsi.median(),
            leaders=",".join(sub.rs.sort_values(ascending=False).index[:3]),
        )
    return pd.DataFrame.from_dict(rows, orient="index")


def _pct(s: pd.Series) -> pd.Series:
    return s.rank(pct=True).fillna(0.5)


def score(df: pd.DataFrame) -> pd.Series:
    comp = dict(
        rs=_pct(df.rs),
        trend=(_pct(df.d200) + _pct(df.slope200)) / 2,
        breadth=(_pct(df.breadth50) + _pct(df.breadth200) + _pct(df.at_high)) / 3,
        volume=_pct(df.volr),
        coil=_pct(df.coil),
        high=_pct(df.nh),
    )
    raw = sum(WEIGHTS[k] * v for k, v in comp.items()) / sum(WEIGHTS.values())
    penalty = ((df.ext.fillna(0) - EXT_Z) / 2).clip(0, 1) * 15  # blow-off penalty, up to 15 points
    return (100 * raw - penalty).clip(0, 100)


def score_asof(close, volume, bench, themes, end: int | None = None):
    """Score all themes using only data up to row `end` (exclusive). Returns (theme_df, ticker_df)."""
    end = len(close) if end is None else end
    tm = ticker_metrics(close.iloc[:end], volume.iloc[:end], bench.iloc[:end])
    df = aggregate(tm, themes)
    df["score"] = score(df)
    return df, tm


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ext = (df.ext > EXT_Z) | (df.rsi > EXT_RSI)
    hi = df.score >= 55
    early = (df.score >= 60) & (df.breadth200 >= 0.5) & (df.d200 > 0)
    basing = (df.score >= 40) & (df.accel > 0) & (df.nh > -0.20)
    df["stage"] = np.select(
        [ext & hi, hi & (df.accel <= -8), early, basing],
        ["Extended", "Rolling over", "Early trend", "Basing"],
        default="Dormant",
    )
    df["prepping"] = df.stage.isin(["Basing", "Early trend"]) & (df.accel > 0)
    return df


def scan(close, volume, bench, themes, end: int | None = None):
    """Full scan: current scores, 4-week score acceleration, stage, prepping flag."""
    end = len(close) if end is None else end
    now, tm = score_asof(close, volume, bench, themes, end)
    prev, _ = score_asof(close, volume, bench, themes, end - ACCEL_BARS)
    now["score_prev"] = prev.score.reindex(now.index)
    now["accel"] = (now.score - now.score_prev).fillna(0)
    return classify(now).sort_values("score", ascending=False), tm
