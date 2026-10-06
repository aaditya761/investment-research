"""Walk-forward backtest: how early did the scanner flag known runs, and do its flags carry edge?"""
import numpy as np
import pandas as pd

from .scoring import ACCEL_BARS, classify, score_asof

FWD = 63  # forward horizon in bars (~3 months)


FWDS = (21, 63, 126)


def walk_forward(close, volume, bench, themes, start: str, end: str | None = None, step: int = 5, fwd: int = FWD):
    """Weekly scan history. One row per (date, theme): raw signals, stage, leading flag, forward excess returns.

    `fwd_excess` is the `fwd`-bar forward return of the theme basket (median) minus the benchmark;
    `fwd_excess_{h}` holds the same for each horizon in FWDS.
    """
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
        prev = at(i - ACCEL_BARS).reindex(now.index)
        now["accel"] = (now.score - prev.score).fillna(0)
        now["rs_chg"] = (now.rs - prev.rs).fillna(0)
        now["breadth_chg"] = (now.breadth200 - prev.breadth200).fillna(0)
        now = classify(now).drop(columns="leaders")
        fw = {}
        for h in sorted(set(FWDS) | {fwd}):
            if i - 1 + h < len(close):
                b = bench.iloc[i - 1 + h] / bench.iloc[i - 1] - 1
                r = close.iloc[i - 1 + h] / close.iloc[i - 1] - 1
                fw[h] = {k: r[[t for t in themes[k].tickers if t in close]].median() - b for k in now.index}
        for key, rec in now.iterrows():
            row = {"date": close.index[i - 1], "theme": key, **rec.to_dict()}
            for h in sorted(set(FWDS) | {fwd}):
                row[f"fwd_excess_{h}"] = fw[h][key] if h in fw else np.nan
            row["fwd_excess"] = row[f"fwd_excess_{fwd}"]
            rows.append(row)
    return pd.DataFrame(rows)


def summarize(hist: pd.DataFrame) -> pd.DataFrame:
    h = hist.dropna(subset=["fwd_excess"])
    g = h.groupby(h.leading.map({True: "LEADING", False: "other"}).rename("flag")).fwd_excess
    by_flag = pd.DataFrame({"n": g.size(), "mean_excess": g.mean(), "median_excess": g.median(), "hit_rate": g.apply(lambda x: (x > 0).mean())})
    g2 = h.groupby("stage").fwd_excess
    by_stage = pd.DataFrame({"n": g2.size(), "mean_excess": g2.mean(), "median_excess": g2.median(), "hit_rate": g2.apply(lambda x: (x > 0).mean())})
    return pd.concat({"by flag": by_flag, "by stage": by_stage})


def event_lead_times(hist: pd.DataFrame, events: list[dict], lookback_weeks: int = 26, lookahead_weeks: int = 13):
    """For each known run: first leading flag in [start - lookback, start + lookahead]; lead>0 means flagged before the run."""
    out = []
    for ev in events:
        s = pd.Timestamp(ev["start"])
        h = hist[(hist.theme == ev["theme"]) & hist.leading]
        win = h[(h.date >= s - pd.Timedelta(weeks=lookback_weeks)) & (h.date <= s + pd.Timedelta(weeks=lookahead_weeks))]
        if win.empty:
            out.append(dict(event=ev.get("label", ev["theme"]), run_start=s.date(), first_flag=None, lead_weeks=None))
        else:
            f = win.date.min()
            out.append(dict(event=ev.get("label", ev["theme"]), run_start=s.date(), first_flag=f.date(), lead_weeks=round((s - f).days / 7, 1)))
    return pd.DataFrame(out)
