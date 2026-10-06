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
    t["score"], t["4w_chg"], t["stage"] = df.score.round(0), df.accel.round(0), df.stage
    t["lead"] = df.leading.map({True: "YES", False: ""})
    t["rel_str%"] = (df.rs * 100).round(1)
    t["3m%"] = (df.r3m * 100).round(0)
    t[">200d"] = (df.breadth200 * 100).round(0)
    t["ext_z"] = df.ext.round(1)
    t["vol_x"] = df.volr.round(2)
    t["leaders"] = df.leaders
    return t


def cmd_scan(args):
    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    end = len(close) if not args.asof else close.index.searchsorted(pd.Timestamp(args.asof), side="right")
    df, tm = scoring.scan(close, vol, close[cfg.benchmark], cfg.themes, end)
    print(f"As of {close.index[end - 1].date()} | benchmark {cfg.benchmark} | {len(df)} themes\n")
    lead, lag = df[df.leading], df[df.stage == "Lagging"]
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
    s.set_defaults(fn=cmd_scan)
    b = sub.add_parser("backtest", help="walk-forward test + lead time on known runs")
    b.add_argument("--bt-start", default="2020-01-01")
    b.add_argument("--bt-end")
    b.add_argument("--step", type=int, default=5, help="bars between scans")
    b.set_defaults(fn=cmd_backtest)
    r = sub.add_parser("research", help="signal IC + rule search on a backtest CSV")
    r.add_argument("hist", help="CSV from `backtest --csv`")
    r.add_argument("--validate", help="CSV from another universe for out-of-universe check")
    r.add_argument("--split", default="2023-01-01")
    r.add_argument("--horizon", type=int, default=63)
    r.set_defaults(fn=cmd_research)
    args = p.parse_args(argv)
    args.fn(args)
