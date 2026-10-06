"""Evaluate which signals predict forward theme returns, and search for a better 'leading' rule.

Works on the CSV written by `backtest --csv`. Returns are measured as *alpha*: forward excess return
of a theme minus the average theme on the same date, so a rising tide does not make every flag look good.
"""
import itertools

import numpy as np
import pandas as pd

SIGNALS = ["score", "accel", "rs", "rs_chg", "d200", "slope200", "breadth50", "breadth200", "breadth_chg",
           "at_high", "nh", "volr", "coil", "ext", "rsi"]
STEP_BARS = 5  # rows are weekly


def prepare(hist: pd.DataFrame, horizon: int = 63) -> pd.DataFrame:
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    col = f"fwd_excess_{horizon}"
    h = h.dropna(subset=[col]).copy()
    h["alpha"] = h[col] - h.groupby("date")[col].transform("mean")
    return h


def ic_table(hist: pd.DataFrame, horizons=(21, 63, 126)) -> pd.DataFrame:
    """Mean cross-sectional Spearman IC of each signal vs forward alpha, with an overlap-adjusted t-stat."""
    out = {}
    for hz in horizons:
        h = prepare(hist, hz)
        eff = max(hz / STEP_BARS, 1)  # overlapping windows: shrink effective sample size
        for s in SIGNALS:
            ics = h.groupby("date").apply(
                lambda g: g[s].rank().corr(g["alpha"].rank()) if g[s].nunique() > 2 else np.nan, include_groups=False
            ).dropna()
            t = ics.mean() / (ics.std() / np.sqrt(len(ics) / eff)) if len(ics) > 5 and ics.std() > 0 else np.nan
            out[(s, hz)] = (ics.mean(), t)
    df = pd.DataFrame(out, index=["ic", "t"]).T.unstack()
    df.columns = [f"{a}_{b}d" for a, b in df.columns]
    return df.round(3)


def eval_mask(h: pd.DataFrame, mask: pd.Series, hz_eff: float = 63 / STEP_BARS) -> dict:
    y = h.loc[mask, "alpha"]
    if len(y) < 30:
        return dict(n=len(y), mean=np.nan, hit=np.nan, t=np.nan)
    t = y.mean() / (y.std() / np.sqrt(len(y) / hz_eff))
    return dict(n=len(y), mean=y.mean(), hit=(y > 0).mean(), t=t)


def current_rule(h):
    return h.leading.astype(bool)


def make_rule(score_min, accel_min, ext_max, nh_min, b200_min, rs_chg_min):
    def rule(h):
        return ((h.score >= score_min) & (h.accel >= accel_min) & (h.ext.fillna(0) <= ext_max)
                & (h.nh >= nh_min) & (h.breadth200 >= b200_min) & (h.rs_chg >= rs_chg_min))
    return rule


GRID = dict(
    score_min=[40, 55, 70], accel_min=[0, 8, 15], ext_max=[1.5, 99], nh_min=[-0.05, -0.10, -0.25],
    b200_min=[0.0, 0.4], rs_chg_min=[-9, 0.0, 0.03],
)


def grid_search(train: pd.DataFrame, min_frac: float = 0.03, top: int = 8) -> pd.DataFrame:
    rows = []
    for combo in itertools.product(*GRID.values()):
        p = dict(zip(GRID, combo))
        r = eval_mask(train, make_rule(**p)(train))
        if r["n"] >= max(100, min_frac * len(train)) and r["t"] == r["t"]:
            rows.append({**p, **r})
    return pd.DataFrame(rows).sort_values("t", ascending=False).head(top).reset_index(drop=True)


def report(hist: pd.DataFrame, split: str = "2023-01-01", horizon: int = 63, validate: pd.DataFrame | None = None) -> str:
    h = prepare(hist, horizon)
    train, test = h[h.date < split], h[h.date >= split]
    lines = [f"obs: {len(h)} | train {len(train)} (<{split}) | test {len(test)} | alpha = {horizon}d fwd excess minus same-date theme mean\n"]
    lines += ["== Signal IC (Spearman vs forward alpha; |t|>2 meaningful) ==", ic_table(hist).to_string(), ""]
    base = lambda d: f"n={len(d)} mean_alpha={d.alpha.mean():+.4f}"
    lines += [f"Base rates: train {base(train)} | test {base(test)}", ""]
    cur = {k: eval_mask(d, current_rule(d)) for k, d in (("train", train), ("test", test))}
    lines += ["== Current leading rule ==", pd.DataFrame(cur).T.round(3).to_string(), ""]
    gs = grid_search(train)
    lines += ["== Top rules on TRAIN (ranked by overlap-adjusted t) with their TEST result =="]
    res = []
    for _, r in gs.iterrows():
        p = {k: r[k] for k in GRID}
        te = eval_mask(test, make_rule(**p)(test))
        res.append({**p, "tr_n": r.n, "tr_mean": r["mean"], "tr_t": r.t, "te_n": te["n"], "te_mean": te["mean"], "te_hit": te["hit"], "te_t": te["t"]})
    res = pd.DataFrame(res)
    lines += [res.round(3).to_string(), ""]
    if validate is not None and len(res):
        v = prepare(validate, horizon)
        p = {k: res.iloc[0][k] for k in GRID}
        lines += ["== Best-train rule on a DIFFERENT universe (full period) ==",
                  pd.DataFrame({"current": eval_mask(v, current_rule(v)), "best_train": eval_mask(v, make_rule(**p)(v)),
                                "all": eval_mask(v, pd.Series(True, index=v.index))}).T.round(3).to_string()]
    return "\n".join(lines)


def blend_report(hist: pd.DataFrame, earn_feat: pd.DataFrame, weights=(0, 0.3, 0.5, 0.7, 1.0), split="2023-01-01", horizon=63) -> str:
    """How does mixing the earnings composite into the price score change out-of-sample ranking power?

    Per weight: rank-IC (and overlap-adjusted t) plus top-quintile alpha and top-minus-bottom spread, train vs test.
    """
    d = prepare(hist, horizon).merge(earn_feat, on=["theme", "date"], how="left")
    pc = lambda c: d.groupby("date")[c].rank(pct=True)
    e = (0.5 * pc("surp") + 0.3 * pc("beat") + 0.2 * pc("surp_chg").fillna(0.5)).where(d.surp.notna())
    rows = []
    for w in weights:
        d["c"] = np.where(e.notna(), w * e + (1 - w) * d.score / 100, d.score / 100) if w < 1 else e
        d["rk"] = d.groupby("date").c.rank(pct=True, method="first")
        for per, x in (("train", d[d.date < split]), ("test", d[d.date >= split])):
            x = x[x.c.notna()]
            ic = x.groupby("date").apply(lambda g: g.c.rank().corr(g.alpha.rank()) if g.c.nunique() > 2 else np.nan, include_groups=False).dropna()
            top, bot = x[x.rk >= 0.8], x[x.rk <= 0.2]
            sp = (top.groupby("date").alpha.mean() - bot.groupby("date").alpha.mean()).dropna()
            rows.append(dict(w_earn=w, period=per, ic=ic.mean(), t=ic.mean() / (ic.std() / np.sqrt(len(ic) / 12.6)),
                             top20_alpha_pct=100 * top.alpha.mean(), top_minus_bottom_pct=100 * sp.mean()))
    return pd.DataFrame(rows).round(3).to_string(index=False)
