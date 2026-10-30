"""Where is money going when a leading complex sells off?

Money flows are not directly observable, so three indirect, price/volume-based lenses are combined:
  1. Residual strength: recent excess return NOT explained by the theme's exposure (beta) to the source complex.
  2. Volume-signed pressure: share of recent dollar volume that traded on up days (net buying proxy) and volume vs normal.
  3. History: forward performance of each theme after past selloffs of the same source complex (event study).
A destination that shows up in all three is a stronger candidate than one that shows up in one.
"""
import numpy as np
import pandas as pd

# Source complex = what is being sold. Keys are "g:" (themes.yaml) / "b:" (themes_broad.yaml).
AI_SOURCE = ["g:ai_semis", "g:memory", "g:taiwan", "g:korea", "g:hyperscalers", "g:power_ai", "b:semis", "b:cloud", "b:china_internet", "b:robotics_ai"]

GROUPS = {
    "US sectors": ["us_tech", "us_financials", "us_energy", "us_health", "us_industrials", "us_materials", "us_utilities", "us_staples", "us_discretionary", "us_comm", "us_realestate"],
    "Developed countries": ["canada", "australia", "uk", "france", "germany", "switzerland", "sweden", "spain", "italy", "netherlands", "singapore", "hong_kong", "japan", "intl_dev", "europe"],
    "Emerging countries": ["korea", "taiwan", "india", "china", "brazil", "mexico", "south_africa", "turkey", "indonesia", "thailand", "vietnam", "malaysia", "philippines", "poland", "chile", "argentina", "saudi", "israel", "emerging", "frontier"],
    "Factors / size": ["value", "growth", "momentum_factor", "quality", "low_vol", "small_caps", "mid_caps", "dividend", "equal_weight"],
    "Commodities / metals": ["gold", "gold_miners", "silver", "copper", "oil", "nat_gas", "agriculture", "broad_commodities", "uranium", "lithium_batt", "coal_old_energy"],
    "Rates / credit / FX": ["long_treasuries", "intermediate_treasuries", "tips", "high_yield", "inv_grade", "em_bonds", "dollar", "yen", "euro"],
}
_GROUP_OF = {f"b:{k}": g for g, ks in GROUPS.items() for k in ks}


def weekdays(close: pd.DataFrame, vol: pd.DataFrame):
    """Trading-day panel: drop weekend rows (24/7 assets like crypto otherwise force a 7-day calendar onto everything)."""
    keep = close.index.dayofweek < 5
    return close[keep], vol.reindex(close.index)[keep]


def merge_universes(cfgs: dict) -> dict:
    """cfgs: prefix -> Config. Returns {prefixed_key: Theme}."""
    out = {}
    for p, cfg in cfgs.items():
        for k, th in cfg.themes.items():
            out[f"{p}:{k}"] = th
    return out


def theme_returns(close: pd.DataFrame, themes: dict) -> pd.DataFrame:
    r = close.pct_change(fill_method=None)
    cols = {}
    for k, th in themes.items():
        c = [t for t in th.tickers if t in r]
        if len(c) >= 2:
            s = r[c].median(axis=1)
            cols[k] = s.where(r[c].notna().sum(axis=1) >= min(2, len(c)))
    return pd.DataFrame(cols)


def source_factor(R: pd.DataFrame, source: list[str]) -> pd.Series:
    return R[[s for s in source if s in R]].mean(axis=1)


def _cum(r: pd.Series, n: int) -> float:
    return float((1 + r.tail(n).fillna(0)).prod() - 1)


def current_state(close, vol, themes, bench: str, source=AI_SOURCE, window=20, end=None):
    if end is not None:
        close, vol = close.iloc[:end], vol.iloc[:end]
    R = theme_returns(close, themes)
    F, B = source_factor(R, source), close[bench].pct_change()
    idx = (1 + F.fillna(0)).cumprod()
    dd = float(idx.iloc[-1] / idx.tail(60).max() - 1)
    src_cum = _cum(F, window) - _cum(B, window)
    dv = close * vol.reindex(close.index)
    sign = np.sign(close.diff())
    rows = {}
    for k, th in themes.items():
        if k in source or k not in R:
            continue
        y = (R[k] - B).dropna().tail(250)
        x = (F - B).reindex(y.index)
        beta = float(np.cov(y, x)[0, 1] / x.var()) if len(y) > 120 and x.var() > 0 else np.nan
        excess = _cum(R[k], window) - _cum(B, window)
        resid = excess - (beta if beta == beta else 0) * src_cum
        cols = [t for t in th.tickers if t in close and (vol[t].tail(90) > 0).mean() > 0.8]
        flow = ratio = np.nan
        if cols:
            w = dv[cols].tail(window)
            flow = float(((sign[cols].tail(window) * w).sum() / w.sum()).median())
            ratio = float((dv[cols].tail(window).mean() / dv[cols].tail(90).mean()).median())
        corr = float(R[k].tail(60).corr(F.tail(60)))
        rows[k] = dict(group=_GROUP_OF.get(k, "Hand-picked themes (stocks)" if k.startswith("g:") else "Other ETFs"), ret=_cum(R[k], window), excess=excess,
                       beta=beta, resid=resid, flow=flow, vol_ratio=ratio, corr_ai=corr)
    df = pd.DataFrame.from_dict(rows, orient="index")
    df["resid_rank"] = df.resid.rank(pct=True)
    df["flow_rank"] = df.flow.rank(pct=True)
    df["both"] = (df.resid_rank >= 0.75) & (df.flow > 0)
    return df.sort_values("resid", ascending=False), dict(source_return=_cum(F, window), source_excess=src_cum, drawdown=dd, high_date=str(idx.tail(60).idxmax().date()))


def episodes(F: pd.Series, drop=-0.10, lookback=60, gap=60) -> list[pd.Timestamp]:
    """Start dates of selloffs: first day the source index is `drop` below its trailing high, at least `gap` bars after the previous trigger."""
    idx = (1 + F.fillna(0)).cumprod()
    dd = idx / idx.rolling(lookback).max() - 1
    out, last = [], -10**9
    for i, (d, v) in enumerate(dd.items()):
        if v <= drop and i - last > gap and i >= lookback:
            out.append(d)
            last = i
    return out


def event_study(close, themes, bench, source=AI_SOURCE, drop=-0.10, horizons=(20, 60)):
    R = theme_returns(close, themes)
    F, B = source_factor(R, source), close[bench].pct_change()
    eps = episodes(F, drop)
    L = (1 + R.fillna(0)).cumprod().where(R.notna().cummax())
    LB = (1 + B.fillna(0)).cumprod()
    rows = []
    for hz in horizons:
        fwd = L.shift(-hz) / L - 1 - (LB.shift(-hz) / LB - 1).values[:, None]  # excess vs benchmark, all dates
        uncond = fwd.mean()
        for d in eps:
            if d not in fwd.index or fwd.loc[d].isna().all():
                continue
            for k in fwd.columns:
                v = fwd.at[d, k]
                if v == v and k not in source:
                    rows.append(dict(h=hz, date=d, theme=k, excess=v, edge=v - uncond[k]))
    ev = pd.DataFrame(rows)
    return ev, eps


def summarize_events(ev: pd.DataFrame, hz: int, min_n=4) -> pd.DataFrame:
    x = ev[ev.h == hz]
    g = x.groupby("theme").agg(n=("excess", "size"), mean_excess=("excess", "mean"), mean_edge=("edge", "mean"), hit=("excess", lambda s: (s > 0).mean()))
    return g[g.n >= min_n].sort_values("mean_edge", ascending=False)


def combine(now: pd.DataFrame, ev: pd.DataFrame, hz=60) -> pd.DataFrame:
    h = summarize_events(ev, hz)
    out = now.join(h[["n", "mean_edge", "hit"]].rename(columns={"n": "episodes", "mean_edge": "hist_edge", "hit": "hist_hit"}))
    out["hist_good"] = (out.hist_edge > 0) & (out.hist_hit >= 0.6)
    out["signals"] = (out.resid_rank >= 0.75).astype(int) + (out.flow > 0).astype(int) + out.hist_good.fillna(False).astype(int)
    return out.sort_values(["signals", "resid"], ascending=False)


def validate_residual(close, vol, themes, bench, eps, source=AI_SOURCE, horizons=(20, 60)):
    """At each selloff date, does last-20-day residual strength / net buying predict continued relative strength?"""
    rows = []
    for d in eps:
        i = close.index.get_loc(d) + 1
        if i < 300:
            continue
        now, _ = current_state(close, vol, themes, bench, source, 20, end=i)
        for hz in horizons:
            if i - 1 + hz >= len(close):
                continue
            b = close[bench].iloc[i - 1 + hz] / close[bench].iloc[i - 1] - 1
            for k, th in themes.items():
                if k not in now.index:
                    continue
                cols = [t for t in th.tickers if t in close]
                f = (close[cols].iloc[i - 1 + hz] / close[cols].iloc[i - 1] - 1).median() - b
                rows.append(dict(date=d, h=hz, theme=k, resid=now.at[k, "resid"], flow=now.at[k, "flow"], excess=now.at[k, "excess"], fwd=f))
    return pd.DataFrame(rows)


CROSS_ASSET = ["b:gold", "b:gold_miners", "b:long_treasuries", "b:intermediate_treasuries", "b:dollar", "b:yen", "b:high_yield", "b:crypto", "b:broad_commodities", "g:india", "g:europe", "g:japan", "g:brazil"]


def analyze(close, vol, themes, bench="ACWI", source=AI_SOURCE, drop=-0.10, window=20):
    """Everything the rotation view needs. `close`/`vol` should already be weekday-only."""
    now, info = current_state(close, vol, themes, bench, source, window)
    R = theme_returns(close, themes)
    F = source_factor(R, source)
    eps = episodes(F, drop)
    ev, _ = event_study(close, themes, bench, source, drop)
    table = combine(now, ev, 60)
    val = validate_residual(close, vol, themes, bench, eps, source)
    ic = {}
    for hz in (20, 60):
        x = val[val.h == hz].dropna(subset=["resid", "fwd"])
        r = x.groupby("date").apply(lambda g: g.resid.rank().corr(g.fwd.rank()), include_groups=False).dropna()
        f = x.dropna(subset=["flow"]).groupby("date").apply(lambda g: g.flow.rank().corr(g.fwd.rank()), include_groups=False).dropna()
        ic[hz] = dict(resid_ic=float(r.mean()), resid_pos=int((r > 0).sum()), flow_ic=float(f.mean()), flow_pos=int((f > 0).sum()), n=len(r))
    groups = now.groupby("group").agg(n=("resid", "size"), mean_resid=("resid", "mean"), mean_ret=("ret", "mean"), pct_positive_resid=("resid", lambda s: (s > 0).mean())).sort_values("mean_resid", ascending=False)
    info = dict(info, episodes=[str(e.date()) for e in eps], validation=ic, window=window, drop=drop)
    return dict(info=info, table=table, groups=groups)


def label(key: str, themes: dict) -> str:
    return ("[stocks] " if key.startswith("g:") else "[ETFs] ") + themes[key].label if key in themes else key
