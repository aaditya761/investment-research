"""Mega-theme detector: find groups of themes that move together AND are strong, and track how mature the trade is.

How it works
  1. Weekly excess returns (vs world) of every theme -> correlation over 52 weeks -> hierarchical clustering. Themes that are
     really one trade (AI chips + hyperscalers + memory + power + cybersecurity ...) end up in the same cluster.
  2. A cluster is HOT when it has >= 3 members, average price score >= 65 and at least half its members in the top quartile.
  3. Members are classed Leading (price score >= 60), Mixed, or Lagging (< 40).
  4. History: how long the group has been hot, and a quarterly timeline for a seed theme.
  5. A 'fuel gauge' (hyperscaler capex) for the AI trade: the spending that funds the downstream legs.

What was tested (2020-2026, global + broad ETF universes): participating members of hot clusters beat other themes by about +2-3% per 6 months
in both train and test periods; a 'laggard catch-up' gap signal did NOT work (it flipped sign out of sample). So this explains the trade and
shows who is participating; it does not reliably name the next leg.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

from . import rotation as ro
from . import scoring

HOT_AVG, HOT_TOP_FRAC, MIN_SIZE = 65, 0.5, 3
CACHE = Path("data/cluster_scores.csv")


def weekly_excess(close_wd: pd.DataFrame, themes: dict, bench="ACWI") -> pd.DataFrame:
    R = ro.theme_returns(close_wd, themes)
    W = (1 + R.fillna(0)).resample("W-FRI").prod() - 1
    W = W.where(R.resample("W-FRI").count() > 2)
    B = (1 + close_wd[bench].pct_change().fillna(0)).resample("W-FRI").prod() - 1
    return W.sub(B, axis=0)


def cluster_at(X: pd.DataFrame, date, lookback=52, cut=0.75) -> pd.Series:
    x = X.loc[:date].tail(lookback).dropna(axis=1, thresh=int(lookback * 0.8))
    if x.shape[1] < 4:
        return pd.Series(dtype=int)
    c = x.corr().fillna(0).values
    d = 1 - c
    np.fill_diagonal(d, 0)
    d = (d + d.T) / 2
    return pd.Series(fcluster(linkage(squareform(d, checks=False), "average"), t=cut, criterion="distance"), index=x.columns)


def is_hot(members: list, sc: pd.Series) -> dict:
    m = [t for t in members if t in sc.index]
    if len(m) < MIN_SIZE:
        return dict(hot=False, size=len(m), n_top=0, avg=np.nan)
    top = sc.rank(pct=True) >= 0.75
    n_top, avg = int(top[m].sum()), float(sc[m].mean())
    return dict(hot=avg >= HOT_AVG and n_top / len(m) >= HOT_TOP_FRAC, size=len(m), n_top=n_top, avg=avg)


def weekly_scores(cfgs: dict, panels: dict, n_weeks=156, cache=CACHE, fridays=None) -> pd.DataFrame:
    """Price score per theme per week (prefixed keys), cached incrementally. panels: prefix -> (close, vol) in the scan's calendar."""
    old = pd.read_csv(cache, index_col=0, parse_dates=True) if Path(cache).exists() else pd.DataFrame()
    close0 = next(iter(panels.values()))[0]
    fr = pd.date_range(end=close0.index[-1], periods=n_weeks, freq="W-FRI") if fridays is None else fridays
    todo = [d for d in fr if old.empty or d not in old.index]  # also backfills earlier weeks
    rows = {}
    for d in todo:
        rec = {}
        for p, cfg in cfgs.items():
            close, vol = panels[p]
            i = close.index.searchsorted(d, side="right")
            if i < 300:
                continue
            df, _ = scoring.score_asof(close, vol, close[cfg.benchmark], cfg.themes, i)
            rec.update({f"{p}:{k}": v for k, v in df.score.items()})
        rows[d] = rec
    new = pd.DataFrame.from_dict(rows, orient="index")
    out = pd.concat([old, new]).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    Path(cache).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache)
    return out


def current_clusters(X, scores: pd.DataFrame, asof=None):
    asof = asof or scores.index.max()
    sc = scores.loc[asof].dropna()
    cl = cluster_at(X, asof)
    out = []
    for cid, g in cl.groupby(cl):
        m = list(g.index)
        h = is_hot(m, sc)
        if h["size"] < MIN_SIZE:
            continue
        age = 0
        for w in scores.index[::-1]:
            hw = is_hot(m, scores.loc[w].dropna())
            if not hw["hot"]:
                break
            age += 1
        members = sorted([(t, float(sc[t])) for t in m if t in sc.index], key=lambda x: -x[1])
        out.append(dict(id=int(cid), **h, hot_weeks=age, members=members))
    return sorted(out, key=lambda c: (c["hot"], c["n_top"], c["avg"]), reverse=True), asof


def timeline(X, scores, seed, every=13):
    rows = []
    for d in scores.index[::every]:
        cl = cluster_at(X, d)
        if seed not in cl.index:
            continue
        m = list(cl.index[cl == cl[seed]])
        h = is_hot(m, scores.loc[d].dropna())
        rows.append(dict(date=d, **h, members=", ".join(t.split(":")[1] for t in m[:8])))
    return pd.DataFrame(rows)


def first_alert(X, scores, seed, min_size=4, top_frac=0.6, avg=70):
    """First week the seed's cluster had >= min_size members, >= top_frac of them in the top quartile and average score >= avg."""
    for d in scores.index:
        cl = cluster_at(X, d)
        if seed not in cl.index:
            continue
        m = list(cl.index[cl == cl[seed]])
        h = is_hot(m, scores.loc[d].dropna())
        if h["size"] >= min_size and h["n_top"] / max(h["size"], 1) >= top_frac and h["avg"] >= avg:
            return d, h
    return None, None


def fuel_from_frame(c: pd.DataFrame) -> dict:
    """c: index = quarter-end, columns = companies, values = capex in $B (NaN when not reported). Like-for-like totals over the
    companies that reported the latest quarter; prior / year-ago periods are the nearest earlier rows where all of them reported."""
    c = c.sort_index()
    last = c.index[-1]
    cols = [t for t in c.columns if pd.notna(c.at[last, t])]
    out = dict(quarter=str(last.date()), companies=cols, latest=float(c.loc[last, cols].sum()))
    full = c[cols].dropna().index
    prev = [d for d in full if (last - d).days >= 60]
    if prev and (last - prev[-1]).days <= 120:
        out["qoq"] = float(c.loc[last, cols].sum() / c.loc[prev[-1], cols].sum() - 1)
    yoy = [d for d in full if 330 <= (last - d).days <= 400]
    if yoy:
        out["yoy"] = float(c.loc[last, cols].sum() / c.loc[yoy[-1], cols].sum() - 1)
    return out


def hyperscaler_capex(tickers=("MSFT", "GOOGL", "META", "AMZN")) -> pd.DataFrame:
    import yfinance as yf

    cols = {}
    for t in tickers:
        try:
            cf = yf.Ticker(t).quarterly_cashflow
            if cf is not None and "Capital Expenditure" in cf.index:
                s = (-cf.loc["Capital Expenditure"].dropna() / 1e9).sort_index()
                s.index = pd.to_datetime(s.index)
                cols[t] = s
        except Exception:
            pass
    return pd.DataFrame(cols).sort_index()
