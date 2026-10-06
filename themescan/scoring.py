"""Aggregate ticker signals into theme scores and classify the stage of each theme."""
import numpy as np
import pandas as pd

from .config import Theme
from .signals import ticker_metrics

# Score = weighted percentile ranks across themes. Weights chosen by walk-forward research (see README):
# the rising 200d trend was the only signal with a consistent edge across universes and periods;
# breadth, volume, squeeze and near-high added nothing, so they are reported but not scored.
WEIGHTS = dict(slope=0.5, rs=0.3, d200=0.2)
ACCEL_BARS = 20  # score change measured over ~4 weeks (descriptive only)
EXT_Z, EXT_RSI = 2.0, 78  # "extended" label thresholds (descriptive only)
LEAD, LAG = 60, 30  # score cut-offs for the Leading / Lagging stages


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
    comp = dict(slope=_pct(df.slope200), rs=_pct(df.rs), d200=_pct(df.d200))
    return 100 * sum(WEIGHTS[k] * v for k, v in comp.items()) / sum(WEIGHTS.values())


def score_asof(close, volume, bench, themes, end: int | None = None):
    """Score all themes using only data up to row `end` (exclusive). Returns (theme_df, ticker_df)."""
    end = len(close) if end is None else end
    tm = ticker_metrics(close.iloc[:end], volume.iloc[:end], bench.iloc[:end])
    df = aggregate(tm, themes)
    df["score"] = score(df)
    return df, tm


def classify(df: pd.DataFrame) -> pd.DataFrame:
    """Stages. `leading` (score >= LEAD) is the validated flag; Improving/Extended are descriptive labels."""
    df = df.copy()
    ext = (df.ext > EXT_Z) | (df.rsi > EXT_RSI)
    lead = df.score >= LEAD
    df["stage"] = np.select(
        [lead & ext, lead, df.score <= LAG, (df.score >= 40) & (df.accel > 0)],
        ["Extended", "Leading", "Lagging", "Improving"],
        default="Neutral",
    )
    df["leading"] = lead
    return df


def scan(close, volume, bench, themes, end: int | None = None):
    """Full scan: current scores, 4-week score acceleration, stage, prepping flag."""
    end = len(close) if end is None else end
    now, tm = score_asof(close, volume, bench, themes, end)
    prev, _ = score_asof(close, volume, bench, themes, end - ACCEL_BARS)
    now["score_prev"] = prev.score.reindex(now.index)
    now["accel"] = (now.score - now.score_prev).fillna(0)
    return classify(now).sort_values("score", ascending=False), tm
