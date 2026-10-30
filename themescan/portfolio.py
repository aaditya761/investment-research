"""Portfolio-style backtest of the ranking signals, against trivial baselines, with a permutation test.

Every `every` weekly rows we hold the top-N themes (equal weight) for ~21 bars. Returns are the baskets' forward
excess return vs the benchmark, so results are 'excess vs ACWI-style benchmark' before costs, then net of costs.
"""
import numpy as np
import pandas as pd

from . import verify

HORIZON = "fwd_excess_21"
SIGNALS = {
    "conviction (50% price + 50% earnings)": "conviction",
    "price score only": "score",
    "200d slope only (plain trend)": "slope200",
    "relative strength only (plain momentum)": "rs",
    "earnings composite only": "earn",
}


def earn_features_for(hist: pd.DataFrame, cfg, cache: str | None = None) -> pd.DataFrame:
    """Earnings features at each history date (cached to CSV: this is the slow step)."""
    from pathlib import Path

    from . import altdata

    if cache and Path(cache).exists():
        return pd.read_csv(cache, parse_dates=["date"])
    stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t})
    themes = {k: t for k, t in cfg.themes.items() if t.use_earnings}
    ef = altdata.earnings_features(altdata.fetch_earnings(stocks, cache_only=True), themes, pd.DatetimeIndex(sorted(pd.to_datetime(hist["date"]).unique())))
    if ef.empty:  # ETF-only universes have no stocks, hence no earnings
        ef = pd.DataFrame(columns=["theme", "date", "surp", "beat", "surp_chg", "n_rep"])
        ef["date"] = pd.to_datetime(ef["date"])
    if cache:
        ef.to_csv(cache, index=False)
    return ef


def build_panel(hist: pd.DataFrame, earn_feat: pd.DataFrame) -> pd.DataFrame:
    h = hist.copy()
    h["date"] = pd.to_datetime(h["date"])
    d = h.merge(earn_feat, on=["theme", "date"], how="left")
    parts = []
    for _, g in d.groupby("date"):
        g = g.set_index("theme")
        g.index.name = None
        parts.append(verify.add_conviction(g[[c for c in g.columns if c not in ("conviction", "verdict", "earn", "earn_covered")]]).assign(date=g["date"]))
    out = pd.concat(parts).rename_axis("theme").reset_index()
    return out


def run_strategy(panel, signal, n=5, every=4, offset=0, cost_bps=10.0):
    dates = sorted(panel.date.unique())[offset::every]
    prev, rows = set(), []
    for d in dates:
        g = panel[(panel.date == d)].dropna(subset=[signal, HORIZON])
        if len(g) < n + 2:
            continue
        picks = list(g.sort_values(signal, ascending=False).theme[:n])
        gross = g[g.theme.isin(picks)][HORIZON].mean()
        turnover = 1.0 if not prev else len(set(picks) - prev) / n
        rows.append(dict(date=d, gross=gross, net=gross - 2 * turnover * cost_bps / 1e4, turnover=turnover, universe_avg=g[HORIZON].mean(), picks=",".join(picks)))
        prev = set(picks)
    return pd.DataFrame(rows)


def stats(r: pd.Series, per_year=12) -> dict:
    r = r.dropna()
    if len(r) < 6:
        return dict(n=len(r))
    eq = (1 + r).cumprod()
    return dict(n=len(r), mean_pct=100 * r.mean(), ann_pct=100 * r.mean() * per_year, ir=r.mean() / r.std() * np.sqrt(per_year) if r.std() > 0 else np.nan,
                hit=100 * (r > 0).mean(), max_dd_pct=100 * (eq / eq.cummax() - 1).min(), cum_pct=100 * (eq.iloc[-1] - 1))


def permutation_p(panel, picks_df, n=5, every=4, offset=0, draws=5000, seed=0) -> tuple[float, float]:
    """P(random top-N selection on the same dates does at least as well) and the random-selection mean."""
    rng = np.random.default_rng(seed)
    per_date = []
    for d in picks_df.date:
        g = panel[panel.date == d].dropna(subset=[HORIZON])
        per_date.append(g[HORIZON].to_numpy())
    actual = picks_df.gross.mean()
    sims = np.empty(draws)
    for i in range(draws):
        sims[i] = np.mean([rng.choice(v, size=min(n, len(v)), replace=False).mean() for v in per_date])
    return float((sims >= actual).mean()), float(sims.mean())


def report(panel, n=5, every=4, cost_bps=10.0, split="2023-01-01") -> str:
    lines = [f"Top-{n} themes, equal weight, rebalanced every {every} weeks, {cost_bps:.0f} bps per one-way trade. Returns = excess vs benchmark per ~month.\n"]
    rows = []
    for name, col in SIGNALS.items():
        if col not in panel or panel[col].notna().sum() == 0:
            continue
        for per, lo, hi in (("all", None, None), ("train", None, split), ("test", split, None)):
            res = []
            for off in range(every):  # average over start offsets so no single calendar is cherry-picked
                r = run_strategy(panel, col, n, every, off, cost_bps)
                if lo:
                    r = r[r.date >= lo]
                if hi:
                    r = r[r.date < hi]
                res.append(stats(r.net))
            ok = [x for x in res if "ir" in x]
            if ok:
                rows.append({"strategy": name, "period": per, **{k: np.mean([x[k] for x in ok]) for k in ok[0]}, "turnover%": 100 * np.mean([run_strategy(panel, col, n, every, o, cost_bps).turnover.mean() for o in range(every)])})
    base = []
    for off in range(every):
        r = run_strategy(panel, "score", n, every, off, 0)
        base.append(r.universe_avg)
    ew = pd.concat(base)
    lines.append("Baseline: equal-weight ALL themes (excess vs benchmark): mean %.2f%% / month\n" % (100 * ew.mean()))
    t = pd.DataFrame(rows)
    lines.append(t.round(2).to_string(index=False))
    r0 = run_strategy(panel, "conviction", n, every, 0, 0)
    p, sim = permutation_p(panel, r0, n, every)
    lines.append(f"\nPermutation test (conviction top-{n} vs {5000} random top-{n} picks on the same dates, gross):")
    lines.append(f"  actual mean {100 * r0.gross.mean():+.2f}%/month vs random {100 * sim:+.2f}%/month -> p = {p:.3f} (smaller = less likely luck)")
    return "\n".join(lines)
