import argparse
import warnings

import pandas as pd

from . import backtest as bt
from . import scoring, synthetic
from .config import load_config
from .data import load_prices

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 30)


def _load(args, cfg):
    if args.demo:
        print("[demo mode: synthetic prices, not real data]\n")
        return synthetic.make(cfg)
    close, vol, missing = load_prices(cfg.all_tickers, start=args.start, cache_dir=args.cache, refresh=args.refresh, offline=args.offline)
    if missing:
        print(f"note: no data for {len(missing)} tickers: {', '.join(missing)}\n")
    return close, vol


def _fmt(df):
    t = pd.DataFrame(index=df.index)
    t["conv"], t["verdict"] = df.conviction.round(0), df.verdict
    t["price"], t["earn"] = df.score.round(0), df.earn.round(0)
    t["4w_chg"], t["stage"] = df.accel.round(0), df.stage
    t["rel_str%"] = (df.rs * 100).round(1)
    t["3m%"] = (df.r3m * 100).round(0)
    t[">200d"] = (df.breadth200 * 100).round(0)
    t["surp%"] = df.surp.round(1)
    t["beat%"] = (df.beat * 100).round(0)
    t["leaders"] = df.leaders
    return t


def cmd_scan(args):
    from . import verify

    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    end = len(close) if not args.asof else close.index.searchsorted(pd.Timestamp(args.asof), side="right")
    df, note = verify.full_scan(cfg, close, vol, end, fundamentals=not args.no_fundamentals and not args.demo)
    _, tm = scoring.score_asof(close, vol, close[cfg.benchmark], cfg.themes, end)
    print(f"As of {close.index[end - 1].date()} | benchmark {cfg.benchmark} | {len(df)} themes")
    print("conviction = 50% price (200d trend slope, relative strength, distance above 200d) + 50% earnings (EPS surprise, beat rate, surprise trend)")
    print("Strong >= 80 | Positive 65-80 | Neutral 35-65 | Avoid < 35.  Tiers were monotonic in walk-forward tests on global and India themes.")
    if note:
        print(f"note: {note}")
    top, avoid = df[df.verdict.isin(["Strong", "Positive"])], df[df.verdict == "Avoid"]
    print("\n== STRONG / POSITIVE ==")
    print(_fmt(top).to_string() if len(top) else "none")
    print("\n== AVOID ==")
    print(_fmt(avoid).to_string() if len(avoid) else "none")
    print("\n== ALL THEMES ==")
    print(_fmt(df).to_string())
    print("\nNext: `themescan verify <theme>` runs the full evidence checklist (valuation, revisions, analysts, headlines, macro).")
    if args.detail:
        print(f"\n== {args.detail}: ticker detail ==")
        t = tm.reindex(cfg.themes[args.detail].tickers).dropna(how="all")
        print(t[["r1m", "r3m", "r6m", "rs", "d50", "d200", "near_high", "volr", "rsi", "ext"]].round(2).sort_values("rs", ascending=False).to_string())
    if args.csv:
        df.to_csv(args.csv)


def cmd_verify(args):
    from . import verify

    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    df, note = verify.full_scan(cfg, close, vol)
    if note:
        print(f"note: {note}\n")
    sym = {"pass": "PASS", "fail": "FAIL", "na": " -- "}
    for k in args.theme:
        if k not in cfg.themes:
            print(f"unknown theme {k!r}; choose from: {', '.join(cfg.themes)}")
            continue
        r = df.loc[k]
        checks = verify.gather_live(cfg, close, k, len(close), r)
        sm = verify.summarize(checks)
        print(f"## {cfg.themes[k].label} ({k}): conviction {r.conviction:.0f} -> {r.verdict}  [price {r.score:.0f}, earnings {r.earn:.0f}]" if r.earn_covered else f"## {cfg.themes[k].label} ({k}): conviction {r.conviction:.0f} -> {r.verdict}  [price-only]")
        for c in checks:
            print(f"  [{sym[c['status']]}] {c['name']:44s} {c['detail']}   ({c['tier']})")
        print("  " + " | ".join(f"{t}: {v['passed']}/{v['total']}" for t, v in sm.items()) + "\n")


def cmd_backtest(args):
    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    hist = bt.walk_forward(close, vol, close[cfg.benchmark], cfg.themes, args.bt_start, args.bt_end, step=args.step)
    print(f"Walk-forward {hist.date.min().date()} -> {hist.date.max().date()}, forward window {bt.FWD} bars, excess vs {cfg.benchmark}\n")
    print(bt.summarize(hist).round(3).to_string())
    print("\n== Known runs: lead time of first LEADING flag ==")
    print(bt.event_lead_times(hist, cfg.events).to_string(index=False))
    if args.csv:
        hist.to_csv(args.csv, index=False)


def cmd_research(args):
    from . import research

    hist = pd.read_csv(args.hist)
    val = pd.read_csv(args.validate) if args.validate else None
    print(research.report(hist, split=args.split, horizon=args.horizon, validate=val))
    if args.blend:
        from . import altdata

        cfg = load_config(args.themes)
        stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t})
        dates = pd.DatetimeIndex(sorted(pd.to_datetime(hist["date"]).unique()))
        ef = altdata.earnings_features(altdata.fetch_earnings(stocks, cache_only=True), {k: t for k, t in cfg.themes.items() if t.use_earnings}, dates)
        print("\n== Blending earnings into the price score (w_earn = weight on the earnings composite) ==")
        print(research.blend_report(hist, ef, split=args.split, horizon=args.horizon))


def cmd_altfetch(args):
    import yaml

    from . import altdata

    cfg = load_config(args.themes)
    q = yaml.safe_load(open(args.queries))
    if not args.skip_news:
        n = altdata.fetch_news({k: v for k, v in q.items() if k in cfg.themes}, refresh=args.refresh)
        print(f"news: {len(n)} themes")
    if args.skip_analyst:
        return
    stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t})
    e = altdata.fetch_earnings(stocks, refresh=args.refresh)
    print(f"earnings: {e.ticker.nunique()} tickers, {len(e)} reports")
    a = altdata.fetch_analyst_actions(stocks, refresh=args.refresh)
    print(f"analyst: {a.ticker.nunique()} tickers, {len(a)} actions")


def cmd_headlines(args):
    import yaml

    from . import altdata

    cfg = load_config(args.themes)
    queries = yaml.safe_load(open(args.queries))
    keys = args.theme
    if not keys:
        close, vol = _load(args, cfg)
        df, _ = scoring.scan(close, vol, close[cfg.benchmark], cfg.themes)
        keys = list(df[df.leading].index)
    print("Live headlines (catalyst check only: not backtestable; tone is a crude keyword count)\n")
    for k in keys:
        items, total = altdata.headlines(queries.get(k, cfg.themes[k].label), n=args.n)
        tone = sum(i["tone"] for i in items)
        print(f"## {cfg.themes[k].label} ({k}): {total} stories in 14d, tone {tone:+d} over top {len(items)}")
        for i in items:
            print(f"  {i['date']}  {i['title'][:110]}")
        print()


def cmd_fundamentals(args):
    from . import altdata

    cfg = load_config(args.themes)
    keys = args.theme or list(cfg.themes)
    stocks = sorted({t for k in keys for t in cfg.themes[k].stocks if not t.startswith("^") and "-USD" not in t})
    snap = altdata.snapshot(stocks)
    rows = {}
    for k in keys:
        sub = snap.reindex([t for t in cfg.themes[k].stocks if t in snap.index]).dropna(how="all")
        if len(sub) >= 2:
            rows[k] = dict(n=len(sub), fwd_pe=sub.fwd_pe.median(), rev_growth=sub.rev_g.median() * 100, eps_growth=sub.eps_g.median() * 100,
                           upside_to_target=sub.upside.median() * 100, rev_balance=sub.rev_bal.median())
    print("Live snapshot, theme medians (not backtested): rev_balance = (upward - downward FY EPS revisions, 30d) / total, -1..+1\n")
    print(pd.DataFrame.from_dict(rows, orient="index").round(1).sort_values("rev_balance", ascending=False).to_string())


def cmd_ui(args):
    from . import ui

    ui.serve(args.port, not args.no_browser)


def main(argv=None):
    warnings.filterwarnings("ignore")
    p = argparse.ArgumentParser(prog="themescan")
    p.add_argument("--themes", default="themes.yaml")
    p.add_argument("--demo", action="store_true", help="use synthetic prices (no network)")
    p.add_argument("--offline", action="store_true", help="use only cached prices")
    p.add_argument("--refresh", action="store_true", help="force re-download")
    p.add_argument("--start", default="2018-01-01", help="price history start")
    p.add_argument("--cache", default="data/cache")
    p.add_argument("--csv")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scan", help="rank themes today (or --asof)")
    s.add_argument("--asof")
    s.add_argument("--detail", help="theme key to show ticker-level detail")
    s.add_argument("--no-fundamentals", action="store_true", help="skip earnings-surprise columns")
    s.set_defaults(fn=cmd_scan)
    b = sub.add_parser("backtest", help="walk-forward test + lead time on known runs")
    b.add_argument("--bt-start", default="2020-01-01")
    b.add_argument("--bt-end")
    b.add_argument("--step", type=int, default=5, help="bars between scans")
    b.set_defaults(fn=cmd_backtest)
    f = sub.add_parser("altfetch", help="download news-attention and analyst-action history into data/alt/")
    f.add_argument("--queries", default="news_queries.yaml")
    f.add_argument("--skip-news", action="store_true")
    f.add_argument("--skip-analyst", action="store_true")
    f.set_defaults(fn=cmd_altfetch)
    u = sub.add_parser("ui", help="open the point-and-click web UI")
    u.add_argument("--port", type=int, default=8765)
    u.add_argument("--no-browser", action="store_true")
    u.set_defaults(fn=cmd_ui)
    vf = sub.add_parser("verify", help="full evidence checklist for one or more themes")
    vf.add_argument("theme", nargs="+")
    vf.set_defaults(fn=cmd_verify)
    fu = sub.add_parser("fundamentals", help="live valuation/growth/estimate-revision snapshot per theme")
    fu.add_argument("--theme", nargs="*")
    fu.set_defaults(fn=cmd_fundamentals)
    hd = sub.add_parser("headlines", help="live news headlines for leading themes (catalyst check)")
    hd.add_argument("--theme", nargs="*", help="theme keys (default: all currently Leading)")
    hd.add_argument("--queries", default="news_queries.yaml")
    hd.add_argument("--n", type=int, default=5)
    hd.set_defaults(fn=cmd_headlines)
    r = sub.add_parser("research", help="signal IC + rule search on a backtest CSV")
    r.add_argument("hist", help="CSV from `backtest --csv`")
    r.add_argument("--validate", help="CSV from another universe for out-of-universe check")
    r.add_argument("--split", default="2023-01-01")
    r.add_argument("--horizon", type=int, default=63)
    r.add_argument("--blend", action="store_true", help="also test blending earnings into the score (use --themes for the matching universe)")
    r.set_defaults(fn=cmd_research)
    args = p.parse_args(argv)
    args.fn(args)
