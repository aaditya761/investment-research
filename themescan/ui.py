"""Local web UI: `python -m themescan ui`. Standard library only; binds to 127.0.0.1.

Quick things (scan, ticker detail, headlines, one-theme fundamentals) run in-process and return JSON.
Slow things (downloads, backtest, research, all-theme fundamentals) run the normal CLI as a background
job whose output the page polls, so the UI and the CLI can never disagree.
"""
import json
import re
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import altdata, rotation, scoring, synthetic, verify
from . import news as newsmod
from .config import load_config
from .data import load_prices

ROOT = Path.cwd()
PAGE = Path(__file__).with_name("ui.html")
DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_PANELS: dict = {}  # (universe, demo, offline) -> (time, cfg, close, volume, missing)
_JOBS: dict = {}
_LOCK = threading.Lock()


# ---------- helpers ----------
def universes() -> list[str]:
    return sorted(p.name for p in ROOT.glob("themes*.yaml"))


def hist_files() -> list[str]:
    return sorted(p.name for p in (ROOT / "data").glob("hist_*.csv")) if (ROOT / "data").exists() else []


def _universe(name: str) -> str:
    if name not in universes():
        raise ValueError(f"unknown universe {name!r}")
    return name


def _clean(v):
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else round(float(v), 4)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    return v


def _records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in r.items()} for r in df.reset_index().to_dict("records")]


def panel(p: dict):
    key = (_universe(p.get("universe", "themes.yaml")), bool(p.get("demo")), bool(p.get("offline")))
    hit = _PANELS.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1:]
    cfg = load_config(ROOT / key[0])
    if key[1]:
        close, vol = synthetic.make(cfg)
        missing = []
    else:
        close, vol, missing = load_prices(cfg.all_tickers, start="2018-01-01", cache_dir=ROOT / "data/cache", offline=key[2])
    _PANELS[key] = (time.time(), cfg, close, vol, missing)
    return cfg, close, vol, missing


def _stocks(cfg, keys=None):
    keys = keys or list(cfg.themes)
    return sorted({t for k in keys for t in cfg.themes[k].stocks if not t.startswith("^") and "-USD" not in t})


# ---------- quick endpoints ----------
def api_config(_p):
    out = {}
    for u in universes():
        cfg = load_config(ROOT / u)
        out[u] = dict(benchmark=cfg.benchmark, themes={k: t.label for k, t in cfg.themes.items()})
    return dict(universes=out, hist=hist_files(), demo_available=True)


def _scan(p):
    cfg, close, vol, missing = panel(p)
    asof = p.get("asof") or ""
    if asof and not DATE.match(asof):
        raise ValueError("bad date")
    end = close.index.searchsorted(pd.Timestamp(asof), side="right") if asof else len(close)
    if end < 300:
        raise ValueError("not enough price history before that date")
    df, note = verify.full_scan(cfg, close, vol, end, fundamentals=bool(p.get("fundamentals", True)) and not p.get("demo"), news="off" if (p.get("demo") or p.get("asof")) else "cache")
    return cfg, close, end, df, note, missing


def api_scan(p):
    cfg, close, end, df, note, missing = _scan(p)
    df["label"] = [cfg.themes[k].label for k in df.index]
    cols = ["label", "n", "conviction", "verdict", "score", "earn", "earn_covered", "accel", "stage", "leading", "rs", "r3m", "d200",
            "breadth200", "ext", "rsi", "volr", "surp", "beat", "surp_chg", "news", "news_rank", "leaders"]
    return dict(asof=str(close.index[end - 1].date()), benchmark=cfg.benchmark, missing=missing, note=note, rows=_records(df[cols]))


def api_verify(p):
    cfg, close, end, df, note, _ = _scan(p)
    k = p["theme"]
    if k not in cfg.themes:
        raise ValueError("unknown theme")
    checks = verify.gather_live(cfg, close, k, end, df.loc[k], live=not p.get("demo"))
    return dict(theme=k, checks=checks, summary=verify.summarize(checks), conviction=_clean(df.loc[k].conviction), verdict=df.loc[k].verdict)


def api_status(p):
    """What data is on disk for the selected universe, how complete and how old: drives the pipeline cards."""
    import json as _json

    u = _universe(p.get("universe", "themes.yaml"))
    cfg = load_config(ROOT / u)
    now = time.time()

    def age(paths):
        m = [x.stat().st_mtime for x in paths]
        return round((now - max(m)) / 3600, 1) if m else None

    cache = ROOT / "data/cache"
    files = [cache / f"{t.replace('^', '_')}.csv" for t in cfg.all_tickers]
    have = [f for f in files if f.exists()]
    bench = cache / f"{cfg.benchmark.replace('^', '_')}.csv"
    last_bar = str(pd.read_csv(bench, usecols=[0]).iloc[-1, 0])[:10] if bench.exists() else None
    prices = dict(last_bar=last_bar, have=len(have), need=len(files), age_h=age(have))
    stocks = _stocks(cfg)
    ef = [ROOT / "data/alt/earnings" / f"{t.replace('^', '_')}.csv" for t in stocks]
    eh = [f for f in ef if f.exists() and f.stat().st_size > 5]
    earnings = dict(have=len(eh), need=len(stocks), age_h=age(eh))
    qf = ROOT / "news_queries.yaml"
    q = yaml.safe_load(qf.read_text()) if qf.exists() else {}
    keys = [k for k in cfg.themes if k in q]
    nf = ROOT / "data/news_live.json"
    news = dict(have=0, need=len(keys), age_h=None)
    if nf.exists() and keys:
        d = _json.loads(nf.read_text())
        ok = [k for k in keys if d.get(k, {}).get("tone") is not None]
        news = dict(have=len(ok), need=len(keys), age_h=round((now - min(d[k]["t"] for k in ok)) / 3600, 1) if ok else None)
    arch = ROOT / "data/news_hist"
    af = [f for f in arch.glob("*.csv")] if arch.exists() else []
    fl = ROOT / "logs/forward_log.csv"
    snaps, last_log = 0, None
    if fl.exists():
        lg = pd.read_csv(fl)
        snaps, last_log = int(lg.groupby(["universe", "asof"]).ngroups), str(lg["asof"].max())
    return dict(universe=u, prices=prices, earnings=earnings, news=news,
                extras=dict(archive_themes=len(af), fred=len(list((ROOT / "data/fred").glob("*.csv"))) if (ROOT / "data/fred").exists() else 0,
                            forward_snapshots=snaps, forward_last=last_log, backtests=len(hist_files())))


_ROT: dict = {}
_NEWS: dict = {}
_FLOW: dict = {}


def api_flows(p):
    window = int(p.get("window", 20))
    if window not in (5, 20, 60):
        raise ValueError("window must be 5, 20 or 60")
    key = (bool(p.get("offline")), window)
    hit = _FLOW.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    cfgs = {"g": load_config(ROOT / "themes.yaml"), "b": load_config(ROOT / "themes_broad.yaml")}
    close, vol, _ = load_prices(sorted(set(cfgs["g"].all_tickers) | set(cfgs["b"].all_tickers)), start="2018-01-01", cache_dir=ROOT / "data/cache", offline=key[0])
    close, vol = rotation.weekdays(close, vol)
    themes = rotation.merge_universes(cfgs)
    df = rotation.flow_map(close, vol, themes, "ACWI", window=window, recent=max(3, window // 4))
    df["label_name"] = [rotation.label(k, themes) for k in df.index]
    df["short"] = [rotation.label(k, themes).split("] ", 1)[-1] for k in df.index]
    cols = ["label_name", "short", "group", "score", "score_recent", "label", "turning", "ret", "excess", "excess_recent", "flow", "flow_recent", "vol_ratio", "n"]
    res = dict(asof=str(close.index[-1].date()), window=window, rows=_records(df[cols]), groups=_records(rotation.flow_groups(df)))
    _FLOW[key] = (time.time(), res)
    return res
AI_RX = r"\b(ai|a\.i\.|artificial intelligence|generative|chatgpt|openai|llm|data cent(er|re)s?|hbm|gpu)\b"


def api_news(p):
    key = (p.get("universe", "themes.yaml"), bool(p.get("offline")))
    hit = _NEWS.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    cfg, close, vol, _ = panel(dict(p, demo=False))
    qf = ROOT / "news_queries.yaml"
    q = yaml.safe_load(qf.read_text()) if qf.exists() else {}
    arch = newsmod.load_history()
    keys = [k for k in cfg.themes if k in q]
    newsmod.refresh_tone30(q, keys, fetch=True)
    df, _n = verify.full_scan(cfg, close, vol, news="cache")
    rows = newsmod.dashboard(q, keys, arch if len(arch) else None, 14)
    out = []
    for k, f in rows.items():
        sc = float(df.score.get(k, float("nan")))
        out.append(dict(index=k, label=cfg.themes[k].label, verdict=df.verdict.get(k, ""), price=_clean(sc), tone=_clean(f["tone"]), tone_chg=_clean(f["tone_chg"]), pos=_clean(f["pos"]), neg=_clean(f["neg"]),
                        n=f["n"], catalysts=list(f["catalysts"]), bursts=f.get("bursts", []), flag=newsmod.divergence(sc, f["tone_chg"], f["tone"]),
                        headlines=[{k2: (_clean(v) if not isinstance(v, (list, str)) else v) for k2, v in h.items()} for h in f["headlines"]]))
    narr = None
    if len(arch):
        months = sorted(arch.month.unique())
        b = newsmod.term_bursts(arch, months[-1])
        sp = newsmod.narrative_spread(arch, AI_RX)
        narr = dict(month=months[-1], bursts=_records(b) if len(b) else [], ai_spread=[dict(month=m, share=_clean(r.share_all), themes_over=int(r.themes_over_15pct), themes=int(r.themes)) for m, r in sp.iloc[::3].iterrows()],
                    archive=dict(headlines=len(arch), themes=int(arch.theme.nunique()), first=months[0], last=months[-1]))
    res = dict(asof=str(close.index[-1].date()), rows=sorted(out, key=lambda r: -(r["tone"] if r["tone"] is not None else -9)), narratives=narr)
    _NEWS[key] = (time.time(), res)
    return res


def api_rotation(p):
    key = (bool(p.get("offline")), tuple(p.get("source") or ()), float(p.get("drop", 10)), int(p.get("window", 20)))
    hit = _ROT.get(key)
    if hit and time.time() - hit[0] < 600:
        return hit[1]
    cfgs = {"g": load_config(ROOT / "themes.yaml"), "b": load_config(ROOT / "themes_broad.yaml")}
    close, vol, _ = load_prices(sorted(set(cfgs["g"].all_tickers) | set(cfgs["b"].all_tickers)), start="2018-01-01", cache_dir=ROOT / "data/cache", offline=key[0])
    close, vol = rotation.weekdays(close, vol)
    themes = rotation.merge_universes(cfgs)
    src = [k for k in (p.get("source") or rotation.AI_SOURCE) if k in themes]
    out = rotation.analyze(close, vol, themes, "ACWI", src, drop=-abs(key[2]) / 100, window=key[3])
    t = out["table"].copy()
    t["label"] = [rotation.label(k, themes) for k in t.index]
    cols = ["label", "group", "ret", "excess", "resid", "flow", "vol_ratio", "beta", "corr_ai", "episodes", "hist_edge", "hist_hit", "signals"]
    ca = t[t.index.isin(rotation.CROSS_ASSET)]
    res = dict(asof=str(close.index[-1].date()), info=out["info"], rows=_records(t[cols]), groups=_records(out["groups"]), cross=_records(ca[cols]),
               themes={k: v.label for k, v in themes.items()})
    _ROT[key] = (time.time(), res)
    return res


def api_detail(p):
    cfg, close, vol, _ = panel(p)
    k = p["theme"]
    if k not in cfg.themes:
        raise ValueError("unknown theme")
    asof = p.get("asof") or ""
    end = close.index.searchsorted(pd.Timestamp(asof), side="right") if asof and DATE.match(asof) else len(close)
    _, tm = scoring.score_asof(close, vol, close[cfg.benchmark], cfg.themes, end)
    th = cfg.themes[k]
    t = tm.reindex(th.tickers).dropna(how="all").sort_values("rs", ascending=False)
    t["kind"] = ["ETF/index" if x in th.etfs else "stock" for x in t.index]
    return dict(theme=k, label=th.label, rows=_records(t[["kind", "r1m", "r3m", "r6m", "rs", "d200", "near_high", "volr", "rsi", "ext"]]))


def api_headlines(p):
    cfg = load_config(ROOT / _universe(p.get("universe", "themes.yaml")))
    k = p["theme"]
    if k not in cfg.themes:
        raise ValueError("unknown theme")
    q = yaml.safe_load((ROOT / "news_queries.yaml").read_text()) if (ROOT / "news_queries.yaml").exists() else {}
    items, total = altdata.headlines(q.get(k, cfg.themes[k].label), n=int(p.get("n", 8)))
    return dict(theme=k, label=cfg.themes[k].label, total=total, items=items)


def api_fundamentals(p):
    cfg = load_config(ROOT / _universe(p.get("universe", "themes.yaml")))
    k = p["theme"]
    if k not in cfg.themes:
        raise ValueError("unknown theme")
    stocks = _stocks(cfg, [k])
    snap = altdata.snapshot(stocks)
    snap.index.name = None
    snap["upside"] = snap.upside * 100
    snap["rev_g"] = snap.rev_g * 100
    snap["eps_g"] = snap.eps_g * 100
    return dict(theme=k, label=cfg.themes[k].label, rows=_records(snap))


# ---------- background jobs (the normal CLI) ----------
def build_argv(cmd: str, p: dict) -> list[str]:
    u = _universe(p.get("universe", "themes.yaml"))
    glob = ["--themes", u]
    cfg = load_config(ROOT / u)
    themes = [t for t in p.get("themes", []) if t in cfg.themes]
    if cmd == "altfetch":
        a = glob + (["--refresh"] if p.get("refresh") else []) + ["altfetch"]  # --refresh is a global option: it precedes the subcommand
        if not p.get("news"):
            a.append("--skip-news")
        return a
    if cmd == "fundamentals":
        return glob + ["fundamentals"] + (["--theme", *themes] if themes else [])
    if cmd == "headlines":
        return glob + (["--offline"] if p.get("offline") else []) + ["headlines", "--n", str(int(p.get("n", 5)))] + (["--theme", *themes] if themes else [])
    if cmd == "backtest":
        start, step = p.get("start", "2019-06-01"), int(p.get("step", 5))
        if not DATE.match(start) or not 1 <= step <= 60:
            raise ValueError("bad backtest parameters")
        out = f"data/hist_{u.removesuffix('.yaml').removeprefix('themes').strip('_') or 'global'}.csv"
        return glob + ["--csv", out, "backtest", "--bt-start", start, "--step", str(step)]
    if cmd == "research":
        h, v = p.get("hist"), p.get("validate")
        if h not in hist_files() or (v and v not in hist_files()):
            raise ValueError("pick a backtest result from the list (run a backtest first)")
        return ["research", f"data/{h}"] + (["--validate", f"data/{v}"] if v else [])
    if cmd == "prices":
        return glob + ["--refresh", "scan", "--no-news", "--no-fundamentals"]
    if cmd == "earnings":
        return glob + (["--refresh"] if p.get("refresh") else []) + ["altfetch", "--skip-news", "--skip-analyst"]
    if cmd == "newstone":
        return ["newstone"]
    if cmd == "chains":
        return glob + (["--offline"] if p.get("offline") else []) + ["chains"]
    if cmd == "calendar":
        d = int(p.get("days", 21))
        if not 1 <= d <= 90:
            raise ValueError("days must be 1-90")
        return glob + ["calendar", "--days", str(d)]
    if cmd == "weekly":
        return (["--offline"] if p.get("offline") else []) + ["weekly"]
    if cmd == "newsarchive":
        return ["newsarchive"]
    if cmd == "narratives":
        m = p.get("month") or ""
        if m and not re.match(r"^\d{4}-\d{2}$", m):
            raise ValueError("bad month")
        return ["narratives"] + (["--month", m] if m else [])
    if cmd == "clusters":
        seed = p.get("seed", "g:ai_semis")
        if not re.match(r"^[gb]:[a-z_0-9]+$", seed):
            raise ValueError("bad seed theme")
        return (["--offline"] if p.get("offline") else []) + ["clusters", "--seed", seed]
    if cmd == "log":
        return (["--offline"] if p.get("offline") else []) + ["log"]
    if cmd == "scorelog":
        return (["--offline"] if p.get("offline") else []) + ["scorelog"]
    if cmd == "portfolio":
        h = p.get("hist")
        n, every, cost = int(p.get("n", 5)), int(p.get("every", 4)), float(p.get("cost", 10))
        if h not in hist_files() or not (1 <= n <= 20 and 1 <= every <= 26 and 0 <= cost <= 200):
            raise ValueError("pick a backtest result and sensible parameters")
        return glob + ["portfolio", f"data/{h}", "--n", str(n), "--every", str(every), "--cost", str(cost)]
    raise ValueError("unknown command")


def start_job(cmd: str, p: dict) -> dict:
    argv = build_argv(cmd, p)
    with _LOCK:
        if any(j["cmd"] == cmd and not j["done"] for j in _JOBS.values()):
            raise ValueError(f"a '{cmd}' job is already running")
        jid = uuid.uuid4().hex[:8]
        job = dict(id=jid, cmd=cmd, argv=argv, lines=[], done=False, code=None, started=time.time(), proc=None)
        _JOBS[jid] = job

    def run():
        env = dict(__import__("os").environ, PYTHONUNBUFFERED="1")
        proc = subprocess.Popen([sys.executable, "-m", "themescan", *argv], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        job["proc"] = proc
        for line in proc.stdout:
            if "Warning" in line or "HTTP Error 404" in line:
                continue
            job["lines"].append(line.rstrip("\n"))
        job["code"] = proc.wait()
        job["done"] = True
        _PANELS.clear(); _NEWS.clear(); _ROT.clear(); _FLOW.clear()  # data on disk changed: next scan must re-read it

    threading.Thread(target=run, daemon=True).start()
    return dict(id=jid, command="python -m themescan " + " ".join(argv))


def job_status(jid: str, since: int) -> dict:
    j = _JOBS.get(jid)
    if not j:
        raise ValueError("no such job")
    return dict(lines=j["lines"][since:], next=len(j["lines"]), done=j["done"], code=j["code"], cmd=j["cmd"], elapsed=round(time.time() - j["started"]))


def job_cancel(jid: str) -> dict:
    j = _JOBS.get(jid)
    if j and j["proc"] and not j["done"]:
        j["proc"].terminate()
    return dict(ok=True)


# ---------- HTTP ----------
ROUTES = {"/api/config": api_config, "/api/scan": api_scan, "/api/flows": api_flows, "/api/status": api_status, "/api/news": api_news, "/api/rotation": api_rotation, "/api/verify": api_verify, "/api/detail": api_detail, "/api/headlines": api_headlines, "/api/fundamentals": api_fundamentals}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _guard(self, fn):
        try:
            self._send(200, fn())
        except Exception as e:  # surface the message in the UI instead of a stack trace
            self._send(400, {"error": f"{type(e).__name__}: {e}"})

    def do_GET(self):
        path, _, qs = self.path.partition("?")
        if path == "/":
            return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
        if path == "/favicon.ico":
            return self._send(204, b"", "image/x-icon")
        if path == "/api/config":
            return self._guard(lambda: api_config({}))
        m = re.match(r"^/api/job/(\w+)$", path)
        if m:
            since = int(dict(x.split("=") for x in qs.split("&") if "=" in x).get("since", 0))
            return self._guard(lambda: job_status(m[1], since))
        self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.headers.get("Host", "").split(":")[0] not in ("127.0.0.1", "localhost"):
            return self._send(403, {"error": "forbidden"})  # blocks DNS-rebinding style requests
        n = int(self.headers.get("Content-Length") or 0)
        p = json.loads(self.rfile.read(n) or b"{}")
        path = self.path
        if path in ROUTES:
            return self._guard(lambda: ROUTES[path](p))
        m = re.match(r"^/api/run/(\w+)$", path)
        if m:
            return self._guard(lambda: start_job(m[1], p))
        m = re.match(r"^/api/job/(\w+)/cancel$", path)
        if m:
            return self._guard(lambda: job_cancel(m[1]))
        self._send(404, {"error": "not found"})


def serve(port=8765, open_browser=True):
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError:
        raise SystemExit(f"port {port} is already in use: close the other themescan window or run `python -m themescan ui --port {port + 1}`")
    url = f"http://127.0.0.1:{port}"
    print(f"themescan UI running at {url}  (Ctrl+C to stop)")
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
