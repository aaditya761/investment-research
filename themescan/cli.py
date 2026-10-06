import argparse
import warnings
from pathlib import Path

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
    t["score"], t["4w_chg"], t["stage"] = df.score.round(0), df.accel.round(0), df.stage
    t["lead"] = df.leading.map({True: "YES", False: ""})
    t["rel_str%"] = (df.rs * 100).round(1)
    t["3m%"] = (df.r3m * 100).round(0)
    t[">200d"] = (df.breadth200 * 100).round(0)
    t["ext_z"] = df.ext.round(1)
    t["vol_x"] = df.volr.round(2)
    if "surp" in df:
        t["surp%"] = df.surp.round(1)
        t["beat%"] = (df.beat * 100).round(0)
        t["confirmed"] = df.confirmed.map({True: "YES", False: ""})
    t["leaders"] = df.leaders
    return t


def cmd_scan(args):
    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    end = len(close) if not args.asof else close.index.searchsorted(pd.Timestamp(args.asof), side="right")
    df, tm = scoring.scan(close, vol, close[cfg.benchmark], cfg.themes, end)
    earn_dir = Path("data/alt/earnings")
    if earn_dir.exists() and not args.no_fundamentals:
        from . import altdata

        stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t})
        df = altdata.add_earnings_columns(df, altdata.fetch_earnings(stocks), cfg.themes, close.index[end - 1])
    print(f"As of {close.index[end - 1].date()} | benchmark {cfg.benchmark} | {len(df)} themes\n")
    lead, lag = df[df.leading], df[df.stage == "Lagging"]
    if "confirmed" in df:
        print("confirmed = Leading AND top-half EPS surprise over the last 120d (best-validated combination)\n")
    print(f"== LEADING (score >= {scoring.LEAD}: rising 200d trend + relative strength; 'Extended' = already stretched) ==")
    print(_fmt(lead).to_string() if len(lead) else "none")
    print(f"\n== LAGGING (score <= {scoring.LAG}: the most reliable signal in the backtest is to avoid these) ==")
    print(_fmt(lag).to_string() if len(lag) else "none")
    print("\n== ALL THEMES ==")
    print(_fmt(df).to_string())
    if args.detail:
        print(f"\n== {args.detail}: ticker detail ==")
        t = tm.reindex(cfg.themes[args.detail].tickers).dropna(how="all")
        print(t[["r1m", "r3m", "r6m", "rs", "d50", "d200", "near_high", "volr", "rsi", "ext"]].round(2).sort_values("rs", ascending=False).to_string())
    if args.csv:
        df.to_csv(args.csv)


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
    r.set_defaults(fn=cmd_research)
    args = p.parse_args(argv)
    args.fn(args)
