"""Forward test: log today's verdicts, score them once enough bars have passed. No hindsight possible.

`log` appends one row per theme to logs/forward_log.csv (committed to git so the history is auditable).
`score` later measures each logged theme's forward return vs the benchmark and vs the average theme, grouped by verdict.
"""
from pathlib import Path

import numpy as np
import pandas as pd

LOG = Path("logs/forward_log.csv")
COLS = ["universe", "asof", "theme", "conviction", "verdict", "score", "earn", "news", "stage", "price_only", "logged_at"]
HORIZONS = (21, 63)


def write_log(universe: str, df: pd.DataFrame, asof: pd.Timestamp) -> int:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame(columns=COLS)
    if ((old["universe"] == universe) & (old["asof"] == str(asof.date()))).any():
        return 0
    new = pd.DataFrame({
        "universe": universe, "asof": str(asof.date()), "theme": df.index, "conviction": df.conviction.round(2).values, "verdict": df.verdict.values,
        "score": df.score.round(2).values, "earn": df.earn.round(2).values, "news": (df.news.round(3).values if "news" in df else np.nan), "stage": df.stage.values, "price_only": (~df.earn_covered).values,
        "logged_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
    })
    pd.concat([old, new])[COLS].to_csv(LOG, index=False)
    return len(new)


def score_log(panels: dict) -> str:
    """panels: universe -> (cfg, close). Returns a text report of matured forward results."""
    if not LOG.exists():
        return "No forward log yet. Run `themescan log` to start it."
    log = pd.read_csv(LOG, parse_dates=["asof"])
    rows, pending = [], []
    for (u, asof), g in log.groupby(["universe", "asof"]):
        if u not in panels:
            continue
        cfg, close = panels[u]
        i = close.index.searchsorted(asof, side="right") - 1
        for hz in HORIZONS:
            if i + hz >= len(close):
                pending.append((u, asof.date(), hz, i + hz - len(close) + 1))
                continue
            b = close[cfg.benchmark].iloc[i + hz] / close[cfg.benchmark].iloc[i] - 1
            ex = {}
            for k in g.theme:
                cols = [t for t in cfg.themes[k].tickers if t in close]
                ex[k] = (close[cols].iloc[i + hz] / close[cols].iloc[i] - 1).median() - b
            gg = g.assign(h=hz, excess=g.theme.map(ex))
            gg["alpha"] = gg.excess - gg.excess.mean()
            rows.append(gg)
    out = []
    if rows:
        r = pd.concat(rows)
        for hz, x in r.groupby("h"):
            t = x.groupby("verdict").agg(n=("theme", "size"), mean_excess=("excess", lambda s: 100 * s.mean()), mean_alpha=("alpha", lambda s: 100 * s.mean()),
                                         hit=("alpha", lambda s: 100 * (s > 0).mean())).reindex(["Strong", "Positive", "Neutral", "Avoid"]).round(2)
            out.append(f"== {hz}-bar forward results by verdict (excess vs benchmark; alpha vs average theme; hit = beat average theme) ==\n{t.to_string()}\n")
    else:
        out.append("No logged picks have matured yet.")
    if pending:
        p = pd.DataFrame(pending, columns=["universe", "asof", "horizon", "bars_to_go"]).groupby(["universe", "horizon"]).agg(entries=("asof", "size"), first=("asof", "min"), max_bars_to_go=("bars_to_go", "max"))
        out.append("Pending (not enough price history yet):\n" + p.to_string())
    out.append(f"\nLog: {len(log)} rows, {log.groupby(['universe', 'asof']).ngroups} snapshots, {log["asof"].min().date()} -> {log["asof"].max().date()}")
    return "\n".join(out)
