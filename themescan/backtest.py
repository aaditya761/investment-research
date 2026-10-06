"""Walk-forward backtest: how early did the scanner flag known runs, and do its flags carry edge?"""
import numpy as np
import pandas as pd

from .scoring import ACCEL_BARS, classify, score_asof

FWD = 63  # forward horizon in bars (~3 months)


def walk_forward(close, volume, bench, themes, start: str, end: str | None = None, step: int = 5, fwd: int = FWD):
    """Weekly scan history. One row per (date, theme) with stage, prepping flag and forward excess return."""
    lo = close.index.searchsorted(pd.Timestamp(start))
    hi = len(close) if end is None else close.index.searchsorted(pd.Timestamp(end))
    pts = list(range(max(lo, 260 + ACCEL_BARS), hi, step))
    cache: dict[int, pd.DataFrame] = {}

    def at(i):
        if i not in cache:
            cache[i] = score_asof(close, volume, bench, themes, i)[0]
        return cache[i]

    rows = []
    for i in pts:
        now = at(i).copy()
        now["accel"] = (now.score - at(i - ACCEL_BARS).score.reindex(now.index)).fillna(0)
        now = classify(now)
        for key in now.index:
            fe = np.nan
            if i - 1 + fwd < len(close):
                cols = [t for t in themes[key].tickers if t in close]
                r = close[cols].iloc[i - 1 + fwd] / close[cols].iloc[i - 1] - 1
                b = bench.iloc[i - 1 + fwd] / bench.iloc[i - 1] - 1
                fe = r.median() - b
            rows.append((close.index[i - 1], key, now.score[key], now.accel[key], now.stage[key], bool(now.prepping[key]), fe))
    return pd.DataFrame(rows, columns=["date", "theme", "score", "accel", "stage", "prepping", "fwd_excess"])


def summarize(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.dropna(subset=["fwd_excess"])
    g = h.groupby(h.prepping.map({True: "PREPPING", False: "other"}).rename("flag")).fwd_excess
    by_flag = pd.DataFrame({"n": g.size(), "mean_excess": g.mean(), "median_excess": g.median(), "hit_rate": g.apply(lambda x: (x > 0).mean())})
    g2 = h.groupby("stage").fwd_excess
    by_stage = pd.DataFrame({"n": g2.size(), "mean_excess": g2.mean(), "median_excess": g2.median(), "hit_rate": g2.apply(lambda x: (x > 0).mean())})
    return pd.concat({"by flag": by_flag, "by stage": by_stage})


def event_lead_times(hist: pd.DataFrame, events: list[dict], lookback_weeks: int = 26, lookahead_weeks: int = 13):
    """For each known run: first prepping flag in [start - lookback, start + lookahead]; lead>0 means flagged before the run."""
    out = []
    for ev in events:
        s = pd.Timestamp(ev["start"])
        h = hist[(hist.theme == ev["theme"]) & hist.prepping]
        win = h[(h.date >= s - pd.Timedelta(weeks=lookback_weeks)) & (h.date <= s + pd.Timedelta(weeks=lookahead_weeks))]
        if win.empty:
            out.append(dict(event=ev.get("label", ev["theme"]), run_start=s.date(), first_flag=None, lead_weeks=None))
        else:
            f = win.date.min()
            out.append(dict(event=ev.get("label", ev["theme"]), run_start=s.date(), first_flag=f.date(), lead_weeks=round((s - f).days / 7, 1)))
    return pd.DataFrame(out)
