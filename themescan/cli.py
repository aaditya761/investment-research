import argparse
from pathlib import Path
import warnings

import numpy as np
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
    t["news"] = df.news.round(2)
    t["surp%"] = df.surp.round(1)
    t["beat%"] = (df.beat * 100).round(0)
    t["leaders"] = df.leaders
    return t


def cmd_scan(args):
    from . import verify

    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    end = len(close) if not args.asof else close.index.searchsorted(pd.Timestamp(args.asof), side="right")
    df, note = verify.full_scan(cfg, close, vol, end, fundamentals=not args.no_fundamentals and not args.demo,
                                news="off" if (args.asof or args.demo or args.no_news) else ("cache" if args.offline else "fetch"))
    _, tm = scoring.score_asof(close, vol, close[cfg.benchmark], cfg.themes, end)
    print(f"As of {close.index[end - 1].date()} | benchmark {cfg.benchmark} | {len(df)} themes")
    print("conviction = price (trend slope, relative strength, distance above 200d) and earnings (EPS surprise, beat rate, surprise trend) blended 50/50, then 20% weight on 30-day news tone (net: 40% price / 40% earnings / 20% news)")
    print("Strong >= 80 | Positive 65-80 | Neutral 35-65 | Avoid < 35.  Tiers were monotonic in walk-forward tests on global and India themes.")
    if note:
        print(f"note: {note}")
    print("Use as a tilt (avoid the Avoid list, favour Strong/Positive across a diversified set); a concentrated top-5 rotation did not hold up on a broad universe.")
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
    df, note = verify.full_scan(cfg, close, vol, news="cache" if args.offline else "fetch")
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


def cmd_portfolio(args):
    from . import portfolio

    cfg = load_config(args.themes)
    hist = pd.read_csv(args.hist)
    ef = portfolio.earn_features_for(hist, cfg, cache=f"data/earn_feat_{Path(args.hist).stem}.csv")
    panel = portfolio.build_panel(hist, ef)
    print(portfolio.report(panel, n=args.n, every=args.every, cost_bps=args.cost))


def cmd_log(args):
    from . import forward, verify

    for u in args.universes:
        cfg = load_config(u)
        close, vol = _load(args, cfg)
        df, _ = verify.full_scan(cfg, close, vol, news="cache" if args.offline else "fetch")
        n = forward.write_log(Path(u).name, df, close.index[-1])
        print(f"{u}: {'logged ' + str(n) + ' themes' if n else 'already logged'} for {close.index[-1].date()}  "
              f"(Strong {int((df.verdict == 'Strong').sum())}, Positive {int((df.verdict == 'Positive').sum())}, Avoid {int((df.verdict == 'Avoid').sum())})")
    print(f"log file: {forward.LOG}  (commit it to keep an auditable history)")


def cmd_scorelog(args):
    from . import forward

    panels = {}
    if forward.LOG.exists():
        for u in pd.read_csv(forward.LOG).universe.unique():
            cfg = load_config(u)
            close, _ = _load(args, cfg)
            panels[u] = (cfg, close)
    print(forward.score_log(panels))


def cmd_rotation(args):
    from . import rotation as ro

    cfgs = {"g": load_config("themes.yaml"), "b": load_config("themes_broad.yaml")}
    tick = sorted(set(cfgs["g"].all_tickers) | set(cfgs["b"].all_tickers))
    close, vol, _ = load_prices(tick, start="2018-01-01", cache_dir=args.cache, refresh=args.refresh, offline=args.offline)
    close, vol = ro.weekdays(close, vol)
    themes = ro.merge_universes(cfgs)
    src = args.source or ro.AI_SOURCE
    out = ro.analyze(close, vol, themes, "ACWI", src, drop=-abs(args.drop) / 100, window=args.window)
    i, t = out["info"], out["table"]
    print(f"As of {close.index[-1].date()} | source complex: {len(src)} themes ({', '.join(src[:6])}{'...' if len(src) > 6 else ''})")
    print(f"Source: {i['source_return'] * 100:+.1f}% over {i['window']} trading days ({i['source_excess'] * 100:+.1f}% vs world), {i['drawdown'] * 100:.1f}% below its 60-day high of {i['high_date']}\n")
    print(f"== WHERE IT IS GOING NOW (last {i['window']} trading days) ==")
    print("resid = excess return beyond what the theme's beta to the source explains; flow = share of dollar volume on up days minus down days (net buying proxy);")
    print("hist = average forward 60d edge after past selloffs of this source (episodes = number of past selloffs).  signals = resid top-quartile + net buying + historically reliable.\n")
    c = t.copy()
    c.insert(0, "theme", [ro.label(k, themes) for k in c.index])
    show = c[["theme", "group", "ret", "resid", "flow", "beta", "corr_ai", "hist_edge", "hist_hit", "signals"]].head(args.top).copy()
    for col in ("ret", "resid", "hist_edge"):
        show[col] = (show[col] * 100).round(1)
    show["hist_hit"] = (show.hist_hit * 100).round(0)
    show[["flow", "beta", "corr_ai"]] = show[["flow", "beta", "corr_ai"]].round(2)
    print(show.rename(columns={"ret": "ret%", "resid": "resid%", "hist_edge": "hist_edge%", "hist_hit": "hist_hit%"}).to_string(index=False))
    print("\n== BY ASSET GROUP ==")
    g = out["groups"].copy()
    g[["mean_resid", "mean_ret"]] = (g[["mean_resid", "mean_ret"]] * 100).round(1)
    g["pct_positive_resid"] = (g.pct_positive_resid * 100).round(0)
    print(g.to_string())
    print("\n== CROSS-ASSET CHECK (where 'safe' money would go) ==")
    ca = c[c.index.isin(ro.CROSS_ASSET)][["theme", "ret", "resid", "flow"]].copy()
    ca[["ret", "resid"]] = (ca[["ret", "resid"]] * 100).round(1)
    ca["flow"] = ca.flow.round(2)
    print(ca.sort_values("resid", ascending=False).rename(columns={"ret": "ret%", "resid": "resid%"}).to_string(index=False))
    v = i["validation"]
    print(f"\n== CAN THIS PREDICT WHERE MONEY GOES NEXT? (tested on {len(i['episodes'])} past selloffs of this source since 2018) ==")
    for hz in (20, 60):
        x = v[hz]
        print(f"  {hz}d ahead: residual-strength rank-correlation {x['resid_ic']:+.3f} (positive in {x['resid_pos']}/{x['n']} episodes); net-buying {x['flow_ic']:+.3f} ({x['flow_pos']}/{x['n']})")
    print("  Reading: values near 0 and ~half of episodes positive mean today's rotation winners did NOT reliably keep winning. This shows where money is going now, not that it will continue.")
    print("  Past selloff dates: " + ", ".join(i["episodes"]))


def cmd_clusters(args):
    from . import clusters as cl
    from . import rotation as ro

    cfgs = {"g": load_config("themes.yaml"), "b": load_config("themes_broad.yaml")}
    panels = {}
    for p, cfg in cfgs.items():
        close, vol, _ = load_prices(cfg.all_tickers, start="2018-01-01", cache_dir=args.cache, offline=args.offline)
        panels[p] = (close, vol)
    allt = sorted(set(cfgs["g"].all_tickers) | set(cfgs["b"].all_tickers))
    close, vol, _ = load_prices(allt, start="2018-01-01", cache_dir=args.cache, offline=args.offline)
    close_wd, _ = ro.weekdays(close, vol)
    themes = ro.merge_universes(cfgs)
    X = cl.weekly_excess(close_wd, themes)
    print("computing weekly price scores (first run takes a few minutes; cached afterwards)...", flush=True)
    scores = cl.weekly_scores(cfgs, panels, n_weeks=args.weeks)
    found, asof = cl.current_clusters(X, scores)
    print(f"\nAs of {asof.date()}: themes that move together AND are strong are one 'trade'. HOT = >= {cl.MIN_SIZE} members, avg price score >= {cl.HOT_AVG}, >= half in the top quartile.\n")
    for c in found[: args.top]:
        tag = "HOT " if c["hot"] else "    "
        print(f"{tag}cluster of {c['size']}: {c['n_top']} in top quartile, avg price score {c['avg']:.0f}" + (f", hot for {c['hot_weeks']} weeks" if c["hot"] else ""))
        print("      " + "  ".join(f"{ro.label(t, themes).split('] ')[1]}({sc:.0f})" for t, sc in c["members"][:12]) + "\n")
    seed = args.seed
    if seed in themes:
        print(f"== Lifecycle of the cluster containing {seed} (quarterly) ==")
        tl = cl.timeline(X, scores, seed)
        tl["avg"] = tl.avg.round(0)
        print(tl[["date", "size", "n_top", "avg", "hot", "members"]].to_string(index=False))
        d, h = cl.first_alert(X, scores, seed)
        print(f"\nFirst 'mega-theme alert' (>=4 members, >=60% in top quartile, avg score >=70): {d.date() if d is not None else 'none in window'}" + (f" with {h['size']} members" if d is not None else ""))
    if args.seed == "g:ai_semis":
        try:
            fuel = cl.fuel_from_frame(cl.hyperscaler_capex())
            print(f"\n== AI fuel gauge: hyperscaler capex ({', '.join(fuel['companies'])}; like-for-like) ==")
            print(f"  quarter ending {fuel['quarter']}: ${fuel['latest']:.0f}B" + (f", {fuel['yoy'] * 100:+.0f}% vs a year earlier" if 'yoy' in fuel else "") + (f", {fuel['qoq'] * 100:+.0f}% vs prior quarter" if 'qoq' in fuel else ""))
            print("  The downstream legs (chips, memory, power, cybersecurity) are paid for by this spending. Rising capex while prices fall = fundamentals not (yet) confirming the selloff; a capex cut would be the real warning.")
        except Exception as e:
            print(f"(capex data unavailable: {e})")
    print("\nEvidence (2020-2026 tests): participating members of hot clusters beat other themes by ~2-3% per 6 months in both train and test periods;")
    print("a 'laggard catch-up' signal did NOT hold out of sample, so this does not reliably name the next leg.")


def cmd_news(args):
    import yaml

    from . import news as nw
    from . import verify

    cfg = load_config(args.themes)
    q = yaml.safe_load(open(args.queries))
    keys = args.theme or [k for k in cfg.themes if k in q]
    close, vol = _load(args, cfg)
    df, _ = verify.full_scan(cfg, close, vol)
    nw.refresh_tone30(q, [k for k in cfg.themes if k in q], fetch=True)
    arch = nw.load_history()
    print(f"Live news, last {args.days} days vs the {args.days} days before (headline tone from a finance word list; Google News feed). Tone is -1..+1.\n")
    rows = nw.dashboard(q, keys, arch if len(arch) else None, args.days)
    out = []
    for k, f in rows.items():
        sc = df.score.get(k, np.nan) if hasattr(df, "score") else np.nan
        out.append(dict(theme=k, verdict=df.verdict.get(k, ""), price=round(sc), tone=round(f["tone"], 2), tone_chg=round(f["tone_chg"], 2), pos_pct=round(100 * f["pos"]), neg_pct=round(100 * f["neg"]),
                        catalysts=", ".join(f["catalysts"]), bursts=", ".join(f.get("bursts", [])), flag=nw.divergence(sc, f["tone_chg"], f["tone"])))
    t = pd.DataFrame(out).set_index("theme").sort_values("tone", ascending=False)
    pd.set_option("display.max_colwidth", 60)
    print(t.to_string())
    flagged = t[t.flag != ""]
    if len(flagged):
        print("\n== Price and news disagree (look closer; not a validated predictor) ==")
        print(flagged[["verdict", "price", "tone", "tone_chg", "flag", "catalysts"]].to_string())
    for k in (args.detail or []):
        print(f"\n## {k} headlines")
        for h in rows[k]["headlines"]:
            print(f"  {h['date']}  {h['tone']:+.2f}  {', '.join(h['catalysts'])[:34]:34s}  {h['title'][:100]}")


def cmd_narratives(args):
    from . import news as nw

    h = nw.load_history()
    if h.empty:
        print("No archive yet: run `python -m themescan newsarchive` first (about 45 min, resumable).")
        return
    months = sorted(h.month.unique())
    month = args.month or months[-1]
    print(f"Archive: {len(h)} headlines, {h.theme.nunique()} themes, {months[0]} -> {months[-1]}\n")
    print(f"== Terms bursting in {month} across themes (rate vs prior 6 months; seen in >= 3 different themes) ==")
    b = nw.term_bursts(h, month)
    print(b.round(1).to_string(index=False) if len(b) else "not enough history")
    print("\n== Narrative spread: share of headlines matching, and number of themes where it exceeds 15% ==")
    for name, pat in [("AI", r"\b(ai|a\.i\.|artificial intelligence|generative|chatgpt|openai|llm|data cent(er|re)s?|hbm|gpu)\b")] + [(a.split("=")[0], a.split("=", 1)[1]) for a in (args.pattern or [])]:
        sp = nw.narrative_spread(h, pat)
        print(f"\n[{name}]")
        print(sp.iloc[::args.every].assign(share_all=lambda d: (d.share_all * 100).round(1)).to_string())


def cmd_newsarchive(args):
    import yaml

    from . import news as nw

    q = yaml.safe_load(open(args.queries))
    nw.fetch_history({k: v for k, v in q.items() if not args.theme or k in args.theme}, start=args.start, log=lambda m: print(m, flush=True))
    print("DONE")


def cmd_chains(args):
    from . import chains as ch
    from . import verify

    cfg = load_config(args.themes)
    close, vol = _load(args, cfg)
    df, _ = verify.full_scan(cfg, close, vol, news="cache")
    allc = ch.load_chains(args.chains)
    for name in (args.chain or list(allc)):
        c = allc[name]
        print(f"\n== {c['label']} ==")
        res = ch.analyze(c, df.verdict, df.score)
        for line in ch.reading(res):
            print("  " + line)
        print("  Supplier detail (latest quarter vs year ago):")
        for k, s in res["suppliers"].items():
            d = s["stocks"].assign(rev_yoy=lambda x: (x.rev_yoy * 100).round(0), rev_qoq=lambda x: (x.rev_qoq * 100).round(0), gm=lambda x: (x.gm * 100).round(0), gm_chg=lambda x: (x.gm_chg * 100).round(1))
            print("   " + k + ": " + "; ".join(f"{r.ticker} rev {r.rev_yoy:+.0f}% yoy, margin {r.gm:.0f}% ({r.gm_chg:+.1f}pp)" for r in d.itertuples()))
        if res["intensity"]:
            print("  Capex as % of revenue (latest quarter): " + ", ".join(f"{t} {v * 100:.0f}%" for t, v in res["intensity"].items()))
    print("\nLive fundamentals only (Yahoo keeps ~5 quarters), so this cannot be backtested: it is evidence to read next to the score, not part of it.")


def cmd_calendar(args):
    from . import altdata
    from . import chains as ch

    cfg = load_config(args.themes)
    stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t and "." not in t})
    cal = altdata.fetch_calendar(stocks, days_ahead=args.days)
    cal = cal[cal.date <= pd.Timestamp.today().normalize() + pd.Timedelta(days=args.days)]
    key = {t: "spender" for c in ch.load_chains(args.chains).values() for t in c["spenders"]}
    for c in ch.load_chains(args.chains).values():
        for th, ts in c["suppliers"].items():
            key.update({t: f"supplier ({th})" for t in ts})
    rows = []
    for k, th in cfg.themes.items():
        sub = cal[cal.ticker.isin(th.stocks)]
        for r in sub.itertuples():
            rows.append(dict(date=r.date.date(), theme=k, ticker=r.ticker, eps_est=r.estimate, note=key.get(r.ticker, "")))
    t = pd.DataFrame(rows).sort_values(["date", "theme"]) if rows else pd.DataFrame()
    print(f"Earnings in the next {args.days} days for theme stocks (spenders and suppliers of tracked chains flagged):\n")
    print(t.to_string(index=False) if len(t) else "none found")


def cmd_weekly(args):
    import datetime as dt

    import yaml

    from . import altdata, forward, verify, weekly
    from . import chains as ch
    from . import news as nw
    from . import rotation as ro

    secs: dict[str, list[str]] = {}
    qf = Path(args.queries)
    q = yaml.safe_load(qf.read_text()) if qf.exists() else {}
    today = None
    dfs = {}
    for u in args.universes:
        try:
            cfg = load_config(u)
            close, vol = _load(args, cfg)
            df, note = verify.full_scan(cfg, close, vol, news="cache" if args.offline else "fetch", queries=q)
            asof = close.index[-1]
            today = today or asof
            prev = weekly.previous_snapshot(Path(u).name, asof)
            secs[f"Picks: {Path(u).stem}"] = weekly.changes_section(Path(u).name, df, prev) + ([f"_{note}_"] if note else [])
            forward.write_log(Path(u).name, df, asof)
            dfs[u] = (cfg, close, vol, df)
        except Exception as e:
            secs[f"Picks: {u}"] = [f"_failed: {type(e).__name__}: {e}_"]
    if "themes.yaml" in dfs:
        cfg, close, vol, df = dfs["themes.yaml"]
        try:
            cfgs = {"g": cfg, "b": load_config("themes_broad.yaml")}
            tick = sorted(set(cfgs["g"].all_tickers) | set(cfgs["b"].all_tickers))
            c2, v2, _ = load_prices(tick, start="2018-01-01", cache_dir=args.cache, offline=args.offline)
            c2, v2 = ro.weekdays(c2, v2)
            now, info = ro.current_state(c2, v2, ro.merge_universes(cfgs), "ACWI")
            themes = ro.merge_universes(cfgs)
            top = now.head(5)
            secs["Rotation"] = [f"AI source: {info['source_return'] * 100:+.1f}% over 20 trading days ({info['source_excess'] * 100:+.1f}% vs world), {info['drawdown'] * 100:.1f}% below its 60-day high.",
                                "Where it is going (residual strength + net buying): " + ", ".join(f"{ro.label(k, themes).split('] ')[1]} ({r.resid * 100:+.0f}%)" for k, r in top.iterrows()),
                                "_Past selloffs: this did not reliably predict persistence._"]
        except Exception as e:
            secs["Rotation"] = [f"_failed: {type(e).__name__}: {e}_"]
        try:
            keys = [k for k in cfg.themes if k in q]
            rows = nw.dashboard(q, keys, None, 14)
            flags = [f"{cfg.themes[k].label}: {nw.divergence(float(df.score.get(k, np.nan)), f['tone_chg'], f['tone'])} (tone {f['tone']:+.2f})" for k, f in rows.items() if nw.divergence(float(df.score.get(k, np.nan)), f["tone_chg"], f["tone"])]
            secs["News"] = ["Price and news disagree: " + ("; ".join(flags) if flags else "none this week")]
        except Exception as e:
            secs["News"] = [f"_failed: {type(e).__name__}: {e}_"]
        try:
            allc = ch.load_chains(args.chains)
            lines = []
            for name, c in allc.items():
                res = ch.analyze(c, df.verdict, df.score)
                lines += [f"{c['label']}:"] + ["- " + l for l in ch.reading(res)]
            secs["Demand chains"] = lines
        except Exception as e:
            secs["Demand chains"] = [f"_failed: {type(e).__name__}: {e}_"]
        try:
            stocks = sorted({t for th in cfg.themes.values() for t in th.stocks if not t.startswith("^") and "-USD" not in t and "." not in t})
            cal = altdata.fetch_calendar(stocks, days_ahead=7)
            cal = cal[cal.date <= pd.Timestamp.today().normalize() + pd.Timedelta(days=7)].sort_values("date")
            watch = set(df[df.verdict.isin(["Strong", "Positive"])].index)
            rows = [f"{r.date.date()} {r.ticker}" + (" *" if any(r.ticker in cfg.themes[k].stocks for k in watch) else "") for r in cal.itertuples()]
            secs["Earnings in the next 7 days"] = ["(* = a Strong/Positive theme) " + (", ".join(rows) if rows else "none")]
        except Exception as e:
            secs["Earnings in the next 7 days"] = [f"_failed: {type(e).__name__}: {e}_"]
    try:
        panels = {}
        for u in args.universes:
            if u in dfs:
                panels[Path(u).name] = (dfs[u][0], dfs[u][1])
        secs["Forward test"] = ["```", forward.score_log(panels), "```"]
    except Exception as e:
        secs["Forward test"] = [f"_failed: {e}_"]
    day = (today or pd.Timestamp.today()).date()
    md = [f"# Themescan weekly report: {day}", "", "_Research screen, not investment advice. Backtests show a modest, regime-dependent edge; the forward test below is the honest scoreboard._", ""]
    for name, lines in secs.items():
        md += [f"## {name}", ""] + lines + [""]
    weekly.OUT.mkdir(exist_ok=True)
    path = weekly.OUT / f"weekly_{day}.md"
    path.write_text("\n".join(md))
    print("\n".join(md))
    print(f"\nsaved: {path}")
    topic = args.ntfy or __import__("os").environ.get("NTFY_TOPIC")
    if topic:
        ok = weekly.push_ntfy(topic, f"Themescan {day}", weekly.compact(secs))
        print(f"push to ntfy topic '{topic}': {'sent' if ok else 'failed'}")


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
    s.add_argument("--no-news", action="store_true", help="leave news tone out of conviction")
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
    ro_ = sub.add_parser("rotation", help="where is money going when a complex (default: AI) sells off?")
    ro_.add_argument("--source", nargs="+", help="source theme keys, e.g. g:ai_semis g:memory b:semis (default: the AI complex)")
    ro_.add_argument("--window", type=int, default=20, help="trading days for the current-rotation lens")
    ro_.add_argument("--drop", type=float, default=10.0, help="selloff threshold in percent below the 60-day high, for the historical study")
    ro_.add_argument("--top", type=int, default=20)
    ro_.set_defaults(fn=cmd_rotation)
    nw_ = sub.add_parser("news", help="live news dashboard: tone, tone change, catalysts, bursting terms per theme")
    nw_.add_argument("--theme", nargs="*")
    nw_.add_argument("--detail", nargs="*", help="theme keys whose headlines to print")
    nw_.add_argument("--days", type=int, default=14)
    nw_.add_argument("--queries", default="news_queries.yaml")
    nw_.set_defaults(fn=cmd_news)
    na = sub.add_parser("narratives", help="emerging terms across themes and the spread of a narrative (e.g. AI) over time")
    na.add_argument("--month", help="YYYY-MM (default: latest in archive)")
    na.add_argument("--pattern", nargs="*", help='extra narratives as NAME=REGEX')
    na.add_argument("--every", type=int, default=3, help="print every Nth month")
    na.set_defaults(fn=cmd_narratives)
    ar = sub.add_parser("newsarchive", help="download monthly headline history per theme (resumable, ~45 min for all)")
    ar.add_argument("--theme", nargs="*")
    ar.add_argument("--start", default="2021-01-01")
    ar.add_argument("--queries", default="news_queries.yaml")
    ar.set_defaults(fn=cmd_newsarchive)
    chn = sub.add_parser("chains", help="demand-chain tracker: spender capex -> supplier revenue/margins -> price")
    chn.add_argument("--chain", nargs="*")
    chn.add_argument("--chains", default="chains.yaml")
    chn.set_defaults(fn=cmd_chains)
    cal = sub.add_parser("calendar", help="upcoming earnings for theme stocks (catalyst calendar)")
    cal.add_argument("--days", type=int, default=21)
    cal.add_argument("--chains", default="chains.yaml")
    cal.set_defaults(fn=cmd_calendar)
    wk = sub.add_parser("weekly", help="weekly report: logs picks, shows what changed, writes reports/weekly_DATE.md (optional ntfy push)")
    wk.add_argument("--universes", nargs="+", default=["themes.yaml", "themes_india.yaml"])
    wk.add_argument("--queries", default="news_queries.yaml")
    wk.add_argument("--chains", default="chains.yaml")
    wk.add_argument("--ntfy", help="ntfy.sh topic to push a short summary to (or set NTFY_TOPIC). Anyone who knows the topic can read it: pick a long random name.")
    wk.set_defaults(fn=cmd_weekly)
    cx = sub.add_parser("clusters", help="mega-theme detector: groups of correlated, strong themes + lifecycle + AI fuel gauge")
    cx.add_argument("--seed", default="g:ai_semis", help="theme whose cluster lifecycle to show")
    cx.add_argument("--weeks", type=int, default=260, help="weeks of history to score (cached)")
    cx.add_argument("--top", type=int, default=8)
    cx.set_defaults(fn=cmd_clusters)
    lg = sub.add_parser("log", help="forward test: record today's verdicts to logs/forward_log.csv")
    lg.add_argument("--universes", nargs="+", default=["themes.yaml", "themes_india.yaml"])
    lg.set_defaults(fn=cmd_log)
    sl = sub.add_parser("scorelog", help="forward test: score logged picks that have matured")
    sl.set_defaults(fn=cmd_scorelog)
    pf = sub.add_parser("portfolio", help="top-N rotation backtest vs baselines + permutation test")
    pf.add_argument("hist", help="CSV from `backtest --csv`")
    pf.add_argument("--n", type=int, default=5)
    pf.add_argument("--every", type=int, default=4, help="weeks between rebalances")
    pf.add_argument("--cost", type=float, default=10.0, help="bps per one-way trade")
    pf.set_defaults(fn=cmd_portfolio)
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
