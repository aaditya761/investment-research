"""Macro-driver tailwind: how exposed is each theme to recent moves in the dollar, yields, oil, copper, credit...

For every week t, using only data up to t:
  1. beta_i = ridge regression of the theme's weekly excess return on weekly factor changes over the trailing 104 weeks
  2. z_i    = standardised 13-week change in factor i
  3. tailwind = sum_i beta_i * z_i
"""
import numpy as np
import pandas as pd

from .data import load_prices

# name -> (yahoo tickers, kind). kind: "log" = log return of level, "diff" = difference, "ratio" = log of a/b, "mean_log" = mean of log changes
FACTORS = {
    "usd": (["DX-Y.NYB"], "log"),
    "y10": (["^TNX"], "diff"),
    "curve": (["^TNX", "^IRX"], "spread"),
    "vix": (["^VIX"], "log"),
    "oil": (["CL=F"], "log"),
    "copper": (["HG=F"], "log"),
    "gold": (["GC=F"], "log"),
    "credit": (["HYG", "IEF"], "ratio"),
    "usdjpy": (["JPY=X"], "log"),
    "em_fx": (["CNY=X", "INR=X", "BRL=X", "KRW=X"], "mean_log"),
}
BETA_WEEKS, MOM_WEEKS, RIDGE = 104, 13, 50.0


def macro_tickers() -> list[str]:
    return sorted({t for ts, _ in FACTORS.values() for t in ts})


def load_macro(cache_dir="data/cache", offline=False, start="2017-01-01") -> pd.DataFrame:
    close, _, missing = load_prices(macro_tickers(), start=start, cache_dir=cache_dir, offline=offline)
    if missing:
        print("macro tickers missing:", missing)
    return close


def factor_changes(macro: pd.DataFrame) -> pd.DataFrame:
    """Weekly (Friday) change of each factor."""
    w = macro.resample("W-FRI").last().ffill(limit=2)
    out = {}
    for name, (ts, kind) in FACTORS.items():
        if not all(t in w for t in ts):
            continue
        if kind == "log":
            out[name] = np.log(w[ts[0]]).diff()
        elif kind == "diff":
            out[name] = w[ts[0]].diff()
        elif kind == "spread":
            out[name] = (w[ts[0]] - w[ts[1]]).diff()
        elif kind == "ratio":
            out[name] = np.log(w[ts[0]] / w[ts[1]]).diff()
        elif kind == "mean_log":
            out[name] = np.log(w[ts]).diff().mean(axis=1)
    return pd.DataFrame(out)


def theme_weekly_excess(close: pd.DataFrame, bench: pd.Series, themes: dict) -> pd.DataFrame:
    w = close.resample("W-FRI").last()
    r = w.pct_change(fill_method=None)
    b = bench.resample("W-FRI").last().pct_change(fill_method=None)
    out = {}
    for k, th in themes.items():
        cols = [t for t in th.tickers if t in r]
        if len(cols) >= 2:
            out[k] = r[cols].median(axis=1) - b
    return pd.DataFrame(out)


def tailwinds(close, bench, themes, macro: pd.DataFrame, last_n: int | None = None) -> pd.DataFrame:
    """Long frame (theme, date, macro_tw, plus per-factor contributions) indexed by week-end date."""
    fc = factor_changes(macro)
    ex = theme_weekly_excess(close, bench, themes)
    idx = ex.index.intersection(fc.index)
    fc, ex = fc.loc[idx], ex.loc[idx]
    mom = fc.rolling(MOM_WEEKS).sum()
    z = mom / mom.rolling(156, min_periods=52).std()
    sd = fc.rolling(BETA_WEEKS, min_periods=52).std()
    rows = []
    first = BETA_WEEKS if last_n is None else max(BETA_WEEKS, len(idx) - last_n)
    for i in range(first, len(idx)):
        win = slice(i - BETA_WEEKS + 1, i + 1)
        X = fc.iloc[win] / sd.iloc[i]
        zi = z.iloc[i]
        ok = X.notna().all() & zi.notna()
        if ok.sum() < 4:
            continue
        Xv = X.loc[:, ok].to_numpy()
        A = Xv.T @ Xv + RIDGE * np.eye(Xv.shape[1])
        for k in ex.columns:
            y = ex[k].iloc[win].to_numpy()
            m = ~np.isnan(y)
            if m.sum() < 52:
                continue
            beta = np.linalg.solve(Xv[m].T @ Xv[m] + RIDGE * np.eye(Xv.shape[1]), Xv[m].T @ y[m])
            contrib = beta * zi[ok].to_numpy()
            rows.append({"theme": k, "date": idx[i], "macro_tw": contrib.sum(), **dict(zip(X.columns[ok], contrib))})
    return pd.DataFrame(rows)
