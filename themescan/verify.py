"""Conviction score and the evidence checklist.

Two backtested components get a weight: the price score and the earnings composite (EPS surprise, beat rate and
surprise trend). Walk-forward tests picked a 50/50 blend (earnings is then the largest single component; pure earnings
failed on India out of sample). Everything else is a labelled check, shown but never added to the score.
"""
import numpy as np
import pandas as pd

from . import altdata, scoring

EARN_W = 0.5
TIERS = [(80, "Strong"), (65, "Positive"), (35, "Neutral"), (-1, "Avoid")]  # thresholds validated: monotonic alpha by tier


def earn_composite(df: pd.DataFrame) -> pd.Series:
    pc = lambda c: df[c].rank(pct=True)
    e = 0.5 * pc("surp") + 0.3 * pc("beat") + 0.2 * pc("surp_chg").fillna(0.5)
    return e.where(df.surp.notna())


def add_conviction(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "surp" not in df:
        df["surp"] = df["beat"] = df["surp_chg"] = np.nan
    e = earn_composite(df)
    p = df.score / 100
    df["earn"] = e * 100
    df["earn_covered"] = e.notna()
    df["conviction"] = 100 * np.where(e.notna(), EARN_W * e + (1 - EARN_W) * p, p)
    df["verdict"] = df.conviction.map(lambda c: next(name for cut, name in TIERS if c >= cut))
    # Without earnings corroboration a theme can't be "Strong": cap price-only themes at "Positive".
    df.loc[(df.verdict == "Strong") & ~df.earn_covered, "verdict"] = "Positive"
    return df.sort_values("conviction", ascending=False)


def mask_noisy_earnings(df: pd.DataFrame, themes: dict) -> pd.DataFrame:
    df = df.copy()
    for k, th in themes.items():
        if not th.use_earnings and k in df.index:
            df.loc[k, ["surp", "beat", "surp_chg"]] = np.nan
    return df


def full_scan(cfg, close, vol, end=None, fundamentals=True):
    """Price scan + earnings columns (cache only) + conviction. Returns (df, note)."""
    from pathlib import Path

    df, _ = scoring.scan(close, vol, close[cfg.benchmark], cfg.themes, end)
    end = len(close) if end is None else end
    note = None
    if fundamentals:
        d = Path("data/alt/earnings")
        stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t})
        if d.exists() and any(d.iterdir()):
            earn = altdata.fetch_earnings(stocks, cache_only=True)
            if len(earn):
                df = altdata.add_earnings_columns(df, earn, cfg.themes, close.index[end - 1])
        else:
            note = "Earnings data not downloaded yet, so conviction is price-only. Run 'Download data' (or `altfetch --skip-news`)."
    df = add_conviction(mask_noisy_earnings(df, cfg.themes))
    if note is None and not df.earn_covered.all():
        note = f"No earnings coverage for: {', '.join(df.index[~df.earn_covered])} (conviction = price score for these)."
    return df, note


def _chk(name, status, detail, tier):
    return dict(name=name, status=status, detail=detail, tier=tier)


V, U, N = "validated", "live only, not backtestable", "tested: no edge"


def build_checks(row: pd.Series, snap: dict | None, tone: tuple | None, macro_rank, upgr, wiki_z) -> list[dict]:
    """Checklist for one theme. `row` is a full_scan row; the rest are optional live / context inputs."""
    c = []
    c.append(_chk("Trend & relative strength", "pass" if row.score >= scoring.LEAD else "fail",
                  f"price score {row.score:.0f} (needs {scoring.LEAD}); stage {row.stage}", V))
    if row.earn_covered:
        c.append(_chk("EPS surprise in top half of themes", "pass" if row.earn >= 50 else "fail",
                      f"median surprise {row.surp:.1f}% over 120d, earnings composite {row.earn:.0f}/100", V))
        c.append(_chk("Beat rate >= 70%", "pass" if row.beat >= 0.7 else "fail", f"{row.beat * 100:.0f}% of reports beat", V))
        c.append(_chk("Surprise improving vs prior 120d", "na" if pd.isna(row.surp_chg) else ("pass" if row.surp_chg > 0 else "fail"),
                      "n/a" if pd.isna(row.surp_chg) else f"change {row.surp_chg:+.1f} pts", V))
    else:
        c.append(_chk("Earnings data", "na", "excluded or no coverage (e.g. crypto EPS is mark-to-market noise)", V))
    ext = (row.ext > scoring.EXT_Z) or (row.rsi > scoring.EXT_RSI)
    c.append(_chk("Not already stretched", "fail" if ext else "pass", f"{row.ext:.1f} std above 200d, RSI {row.rsi:.0f} (filtering on this hurt backtests)", N))
    if snap:
        c.append(_chk("Estimate revisions net up (FY EPS, 30d)", "na" if snap["rev_bal"] != snap["rev_bal"] else ("pass" if snap["rev_bal"] > 0 else "fail"),
                      "n/a" if snap["rev_bal"] != snap["rev_bal"] else f"balance {snap['rev_bal']:+.2f} (-1..+1)", U))
        pe, eg = snap["fwd_pe"], snap["eps_g"]
        ok = pe == pe and (pe <= 30 or (eg == eg and eg >= pe))
        c.append(_chk("Valuation not stretched vs growth", "na" if (pe != pe or pe <= 0) else ("pass" if ok else "fail"),
                      "n/a (no positive forward earnings)" if (pe != pe or pe <= 0) else (f"forward P/E {pe:.1f}, EPS growth {eg:.0f}%" if eg == eg else f"forward P/E {pe:.1f}"), U))
        up = snap["upside"]
        c.append(_chk("Upside to analyst targets", "na" if up != up else ("pass" if up > 0 else "fail"), "n/a" if up != up else f"median {up:+.0f}%", U))
    if tone:
        c.append(_chk("Recent headline tone not negative", "pass" if tone[0] >= 0 else "fail", f"{tone[1]} stories in 14d, crude tone {tone[0]:+d}", U))
    if macro_rank is not None:
        c.append(_chk("Macro tailwind in top half", "pass" if macro_rank > 0.5 else "fail", f"rank {macro_rank * 100:.0f}th percentile of themes", N))
    if upgr is not None and upgr == upgr:
        c.append(_chk("Analyst upgrades net positive (60d)", "na" if abs(upgr) < 1e-9 else ("pass" if upgr > 0 else "fail"), f"{upgr:+.2f} per covered stock", N))
    if wiki_z is not None and wiki_z == wiki_z:
        c.append(_chk("Public attention not crowded", "pass" if wiki_z < 1.5 else "fail", f"Wikipedia views z = {wiki_z:.1f}", N))
    return c


def summarize(checks: list[dict]) -> dict:
    out = {}
    for tier in (V, U, N):
        t = [c for c in checks if c["tier"] == tier and c["status"] != "na"]
        out[tier] = dict(passed=sum(c["status"] == "pass" for c in t), total=len(t))
    return out


def gather_live(cfg, close, theme: str, end: int, row: pd.Series, live=True, universe_df: pd.DataFrame | None = None):
    """Fetch the non-scored inputs for one theme (live snapshot, headlines, cached macro / analyst / wiki)."""
    from pathlib import Path

    import yaml

    from . import macro

    th = cfg.themes[theme]
    asof = close.index[end - 1]
    snap = tone = macro_rank = upgr = wiki_z = None
    stocks = [t for t in th.stocks if not t.startswith("^") and "-USD" not in t]
    if live:
        try:
            s = altdata.snapshot(stocks)
            s = s.reindex(stocks).dropna(how="all")
            if len(s) >= 2:
                snap = dict(fwd_pe=s.fwd_pe.median(), eps_g=s.eps_g.median() * 100, upside=s.upside.median() * 100, rev_bal=s.rev_bal.median())
        except Exception:
            pass
        try:
            q = yaml.safe_load(Path("news_queries.yaml").read_text()) if Path("news_queries.yaml").exists() else {}
            items, total = altdata.headlines(q.get(theme, th.label), n=8)
            if items:
                tone = (sum(i["tone"] for i in items), total)
        except Exception:
            pass
    try:
        m = macro.load_macro(offline=True)
        tw = macro.tailwinds(close.iloc[:end], close[cfg.benchmark].iloc[:end], cfg.themes, m, last_n=2)
        last = tw[tw.date == tw.date.max()].set_index("theme").macro_tw
        if theme in last.index:
            macro_rank = last.rank(pct=True)[theme]
    except Exception:
        pass
    try:
        acts = altdata.fetch_analyst_actions(stocks, cache_only=True)
        if len(acts):
            f = altdata.analyst_features(acts, {theme: th}, pd.DatetimeIndex([asof]))
            upgr = f.net_upgr.iloc[0]
    except Exception:
        pass
    wf = Path("data/alt/wiki") / f"{theme}.csv"
    if wf.exists():
        try:
            wiki_z = altdata.news_features(pd.read_csv(wf, index_col=0, parse_dates=True).iloc[:, 0], pd.DatetimeIndex([asof])).news_z.iloc[0]
        except Exception:
            pass
    return build_checks(row, snap, tone, macro_rank, upgr, wiki_z)
