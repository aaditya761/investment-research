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

from . import altdata, scoring, synthetic, verify
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
    df, note = verify.full_scan(cfg, close, vol, end, fundamentals=bool(p.get("fundamentals", True)) and not p.get("demo"))
    return cfg, close, end, df, note, missing


def api_scan(p):
    cfg, close, end, df, note, missing = _scan(p)
    df["label"] = [cfg.themes[k].label for k in df.index]
    cols = ["label", "n", "conviction", "verdict", "score", "earn", "earn_covered", "accel", "stage", "leading", "rs", "r3m", "d200",
            "breadth200", "ext", "rsi", "volr", "surp", "beat", "surp_chg", "leaders"]
    return dict(asof=str(close.index[end - 1].date()), benchmark=cfg.benchmark, missing=missing, note=note, rows=_records(df[cols]))


def api_verify(p):
    cfg, close, end, df, note, _ = _scan(p)
    k = p["theme"]
    if k not in cfg.themes:
        raise ValueError("unknown theme")
    checks = verify.gather_live(cfg, close, k, end, df.loc[k], live=not p.get("demo"))
    return dict(theme=k, checks=checks, summary=verify.summarize(checks), conviction=_clean(df.loc[k].conviction), verdict=df.loc[k].verdict)


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
        a = glob + ["altfetch"]
        if not p.get("news"):
            a.append("--skip-news")
        if p.get("refresh"):
            a.append("--refresh")
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
        env = {"PYTHONUNBUFFERED": "1", "PATH": __import__("os").environ.get("PATH", ""), "HOME": __import__("os").environ.get("HOME", "")}
        for k in ("HTTPS_PROXY", "HTTP_PROXY", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "NO_PROXY"):
            if k in __import__("os").environ:
                env[k] = __import__("os").environ[k]
        proc = subprocess.Popen([sys.executable, "-m", "themescan", *argv], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
        job["proc"] = proc
        for line in proc.stdout:
            if "Warning" in line or "HTTP Error 404" in line:
                continue
            job["lines"].append(line.rstrip("\n"))
        job["code"] = proc.wait()
        job["done"] = True

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
ROUTES = {"/api/config": api_config, "/api/scan": api_scan, "/api/verify": api_verify, "/api/detail": api_detail, "/api/headlines": api_headlines, "/api/fundamentals": api_fundamentals}


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
