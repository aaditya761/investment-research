"""News layer: historical + live headlines per theme, tone, narrative bursts, and cross-theme narrative spread.

Source: Google News RSS (free). Date-bounded queries (after:/before:) give a sampled historical archive back to ~2021 (patchy before that),
capped at ~100 headlines per query-window. Story COUNTS from this feed are unreliable (it clusters and samples), so features use the headlines'
text (tone, terms), not counts. Everything is cached under data/news_hist/ and is point-in-time (a headline is only used from its own date).
"""
import json
import re
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import requests

HIST = Path("data/news_hist")
UA = {"User-Agent": "Mozilla/5.0"}


def _fetch(q: str, after: str | None = None, before: str | None = None, tries=4):
    qq = q + (f" after:{after}" if after else "") + (f" before:{before}" if before else "")
    for a in range(tries):
        try:
            r = requests.get("https://news.google.com/rss/search", params=dict(q=qq, hl="en-US", gl="US", ceid="US:en"), headers=UA, timeout=30)
            if r.status_code == 200:
                out = []
                for it in ET.fromstring(r.content).findall(".//item"):
                    t = (it.findtext("title") or "").strip()
                    src = it.findtext("source") or ""
                    if src and t.endswith(" - " + src):
                        t = t[: -len(src) - 3]
                    out.append((t, pd.to_datetime(it.findtext("pubDate"), errors="coerce", utc=True), src))
                return out
            time.sleep(5 * (a + 1))
        except Exception:
            time.sleep(3 * (a + 1))
    return None


def fetch_history(queries: dict[str, str], start="2021-01-01", end=None, sleep=0.8, log=print, cache=HIST):
    """Monthly date-bounded headline archive per theme. Resumable: finished (theme, month) pairs are skipped."""
    cache.mkdir(parents=True, exist_ok=True)
    months = pd.date_range(start, end or pd.Timestamp.today().normalize(), freq="MS")
    for k, q in queries.items():
        f = cache / f"{k}.csv"
        have = pd.read_csv(f, parse_dates=["date"]) if f.exists() and f.stat().st_size > 10 else pd.DataFrame(columns=["month", "title", "date", "source"])
        done = set(have["month"].astype(str)) | (set(json.loads((cache / f"{k}.done").read_text())) if (cache / f"{k}.done").exists() else set())
        rows = []
        for m in months:
            key = m.strftime("%Y-%m")
            if key in done:
                continue
            nxt = (m + pd.offsets.MonthBegin(1)).strftime("%Y-%m-%d")
            res = _fetch(q, m.strftime("%Y-%m-%d"), nxt)
            time.sleep(sleep)
            if res is None:
                log(f"  {k} {key}: failed, will retry on next run")
                continue
            for t, d, s in res:
                rows.append(dict(month=key, title=t, date=d, source=s))
            done.add(key)
        new = pd.concat([have, pd.DataFrame(rows)], ignore_index=True).drop_duplicates(subset=["title", "month"]) if rows else have
        new.to_csv(f, index=False)
        (cache / f"{k}.done").write_text(json.dumps(sorted(done)))
        log(f"{k}: {len(new)} headlines, {len(done)} months done")


def load_history(keys=None, cache=HIST) -> pd.DataFrame:
    frames = []
    for f in sorted(cache.glob("*.csv")):
        if keys and f.stem not in keys:
            continue
        d = pd.read_csv(f, parse_dates=["date"])
        d["theme"] = f.stem
        frames.append(d)
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["month", "title", "date", "source", "theme"])
    out["date"] = pd.to_datetime(out["date"], utc=True, errors="coerce").dt.tz_localize(None)
    return out


# ---------------- tone: finance lexicon with negation ----------------
POS = {"surge", "surges", "surged", "soar", "soars", "soared", "rally", "rallies", "rallied", "jump", "jumps", "jumped", "boom", "booming", "beat", "beats", "record", "records",
       "upgrade", "upgrades", "upgraded", "breakout", "climb", "climbs", "gain", "gains", "gained", "strong", "stronger", "growth", "bullish", "outperform", "outperforms",
       "rallying", "surging", "soaring", "jumping", "climbing", "gaining", "rebounding", "booming", "rise", "rises", "rising", "rose", "higher", "highs", "raises", "raised", "optimism", "recovery", "rebound", "rebounds", "demand", "expands", "expansion", "wins", "win", "approval", "approved", "boost", "boosts", "buy", "accelerate", "accelerates"}
NEG = {"plunge", "plunges", "plunged", "slump", "slumps", "slumped", "fall", "falls", "fell", "crash", "crashes", "slip", "slips", "cut", "cuts", "downgrade", "downgrades", "downgraded",
       "miss", "misses", "missed", "probe", "selloff", "decline", "declines", "declined", "drop", "drops", "dropped", "weak", "weaker", "bearish", "underperform", "warns", "warning", "fears",
       "fear", "risk", "risks", "slowdown", "slows", "ban", "bans", "tariff", "tariffs", "sanctions", "lawsuit", "fraud", "bankruptcy", "layoffs", "default", "loss", "losses", "bubble", "tumble", "tumbles", "sink", "sinks", "falling", "slumping", "dropping", "plunging", "tumbling", "slipping", "sinking", "weakening", "lower", "lowers", "sell-off", "cutting", "dips", "dip", "lows"}
NEGATORS = {"no", "not", "never", "without", "fails", "failed", "unlikely"}
_tok = re.compile(r"[a-z][a-z'\-]+")


def tone(title: str) -> float:
    w = _tok.findall(title.lower())
    s = 0
    for i, x in enumerate(w):
        v = 1 if x in POS else -1 if x in NEG else 0
        if v and i > 0 and w[i - 1] in NEGATORS:
            v = -v
        s += v
    return float(np.tanh(s / 2))


DATEWORDS = set("january february march april may june july august september october november december jan feb mar apr jun jul aug sep sept oct nov dec monday tuesday wednesday thursday friday saturday sunday quarter third second first fourth week weekly daily year years month months".split())
STOP = set("the a an and or of to in on for with at by from as is are was were be been it its this that these those will would could can may might has have had not no new says say said after over amid into than more most up down out about how why what who when us we you your our their his her they them vs via per".split()) | DATEWORDS


GENERIC = set("price prices today stock stocks market markets shares share news week report percent says what why how".split())


def query_words(q: str) -> set:
    return set(_tok.findall(q.lower()))


def terms(title: str) -> set:
    w = [x for x in _tok.findall(title.lower()) if x not in STOP and len(x) > 2]
    return set(w) | {f"{a} {b}" for a, b in zip(w, w[1:])}


def monthly_features(h: pd.DataFrame) -> pd.DataFrame:
    """Per theme-month: mean tone, share positive/negative, headline count (diagnostic only)."""
    h = h.dropna(subset=["title"]).copy()
    h["tone"] = h.title.map(tone)
    g = h.groupby(["theme", "month"])
    out = g.agg(tone=("tone", "mean"), pos=("tone", lambda s: (s > 0).mean()), neg=("tone", lambda s: (s < 0).mean()), n=("tone", "size")).reset_index()
    return out


def term_bursts(h: pd.DataFrame, month: str, lookback=6, min_themes=3, min_df=8, top=25) -> pd.DataFrame:
    """Terms whose document frequency this month jumped vs the trailing months, and in how many different themes."""
    months = sorted(h.month.unique())
    if month not in months:
        return pd.DataFrame()
    i = months.index(month)
    prev = months[max(0, i - lookback) : i]
    if len(prev) < 3:
        return pd.DataFrame()
    def df_table(ms):
        sub = h[h.month.isin(ms)]
        n = max(len(sub), 1)
        c, th = Counter(), {}
        for t, k in zip(sub.title, sub.theme):
            for x in terms(t):
                c[x] += 1
                th.setdefault(x, set()).add(k)
        return c, th, n
    cur, cth, ncur = df_table([month])
    old, _, nold = df_table(prev)
    rows = []
    for x, c in cur.items():
        if c < min_df or len(cth[x]) < min_themes:
            continue
        rate_now, rate_old = c / ncur, (old.get(x, 0) + 1) / (nold + 1)
        rows.append(dict(term=x, count=c, themes=len(cth[x]), lift=rate_now / rate_old))
    return pd.DataFrame(rows).sort_values("lift", ascending=False).head(top) if rows else pd.DataFrame()


def narrative_spread(h: pd.DataFrame, pattern: str) -> pd.DataFrame:
    """For a regex narrative (e.g. AI terms): share of headlines matching, and in how many themes it exceeds 15%, per month."""
    rx = re.compile(pattern, re.I)
    h = h.dropna(subset=["title"]).copy()
    h["hit"] = h.title.map(lambda t: bool(rx.search(t)))
    t = h.groupby(["month", "theme"]).hit.mean().unstack()
    return pd.DataFrame({"share_all": h.groupby("month").hit.mean(), "themes_over_15pct": (t > 0.15).sum(axis=1), "themes": t.notna().sum(axis=1)})


# ---------------- catalyst tagging ----------------
CATALYSTS = {
    "earnings/guidance": r"\b(earnings|guidance|revenue|profit|forecast|outlook|quarter|results|eps)\b",
    "capex/demand": r"\b(capex|capital expenditure|spending|orders?|backlog|demand|data cent(er|re)s?|build-?out)\b",
    "supply/pricing": r"\b(shortage|supply|price hike|prices? (rise|jump|fall|drop)|inventory|contract prices?|hbm|dram|nand)\b",
    "policy/regulation": r"\b(regulat\w+|export (controls?|curbs?|ban)|antitrust|sanctions?|tariffs?|subsid\w+|ban|probe|lawsuit|approval|fda|sec)\b",
    "macro/rates": r"\b(fed|rate (cut|hike)|inflation|yields?|recession|jobs report|cpi|central bank|dollar)\b",
    "analyst action": r"\b(upgrade[sd]?|downgrade[sd]?|price target|initiates?|overweight|underweight|outperform|underperform)\b",
    "deals/capital": r"\b(acquire[sd]?|acquisition|merger|ipo|buyback|stake|invests?|investment|funding|raises \$)\b",
    "geopolitics": r"\b(war|china|taiwan|russia|ukraine|middle east|election|trump|sanctions?)\b",
}
_CRX = {k: re.compile(v, re.I) for k, v in CATALYSTS.items()}


def catalysts(title: str) -> list[str]:
    return [k for k, rx in _CRX.items() if rx.search(title)]


def window_headlines(query: str, end: pd.Timestamp, days: int) -> list:
    a, b = (end - pd.Timedelta(days=days)).strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    return _fetch(query, a, b) or []


def live_features(query: str, end: pd.Timestamp | None = None, days=14, archive: pd.DataFrame | None = None, theme: str | None = None) -> dict:
    """Last `days` of headlines vs the previous `days`: tone, tone change, catalyst mix, bursting terms vs the archive."""
    end = end or pd.Timestamp.today().normalize()
    cur = window_headlines(query, end, days)
    prev = window_headlines(query, end - pd.Timedelta(days=days), days)
    tc, tp = [tone(t) for t, _, _ in cur], [tone(t) for t, _, _ in prev]
    cat = Counter(c for t, _, _ in cur for c in catalysts(t))
    out = dict(n=len(cur), tone=float(np.mean(tc)) if tc else np.nan, tone_prev=float(np.mean(tp)) if tp else np.nan,
               pos=float(np.mean([x > 0 for x in tc])) if tc else np.nan, neg=float(np.mean([x < 0 for x in tc])) if tc else np.nan,
               catalysts=dict(cat.most_common(4)), headlines=[dict(title=t, date=str(d.date()) if pd.notna(d) else "", source=s, tone=tone(t), catalysts=catalysts(t)) for t, d, s in cur[:12]])
    out["tone_chg"] = out["tone"] - out["tone_prev"] if tc and tp else np.nan
    if archive is not None and theme is not None and len(cur):
        base = archive[archive.theme == theme]
        months = sorted(base.month.unique())[-6:]
        old = Counter(x for t in base[base.month.isin(months)].title for x in terms(t))
        nold = max(len(base[base.month.isin(months)]), 1)
        now = Counter(x for t, _, _ in cur for x in terms(t))
        skip = query_words(query) | GENERIC
        b = [(x, c, (c / len(cur)) / ((old.get(x, 0) + 1) / (nold + 1))) for x, c in now.items() if c >= 3 and " " in x and not set(x.split()) <= skip]
        out["bursts"] = [x for x, _, l in sorted(b, key=lambda r: -r[2])[:5]]
    return out


# ---------------- optional LLM classifier ----------------
def classify_llm(headlines: list[str], theme_label: str, model="claude-haiku-4-5-20251001"):
    """Optional: classify headlines with Claude. Needs `pip install anthropic` and ANTHROPIC_API_KEY. Returns a list of dicts
    {direction: bullish|bearish|neutral, catalyst: str, importance: 1-5} or None if unavailable. Not used unless called."""
    import os

    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic

        client = anthropic.Anthropic()
        prompt = (f"For an investor tracking the theme '{theme_label}', classify each numbered headline. Reply with ONLY a JSON list, one object per headline in order: "
                  '{"direction":"bullish|bearish|neutral","catalyst":"short phrase","importance":1-5}. Judge direction for the THEME, not the company.\n\n'
                  + "\n".join(f"{i + 1}. {h}" for i, h in enumerate(headlines)))
        msg = client.messages.create(model=model, max_tokens=1500, messages=[{"role": "user", "content": prompt}])
        return parse_llm_json(msg.content[0].text, len(headlines))
    except Exception:
        return None


def parse_llm_json(text: str, n: int):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        arr = json.loads(m.group(0))
    except Exception:
        return None
    out = []
    for x in arr[:n]:
        d = str(x.get("direction", "neutral")).lower()
        out.append(dict(direction=d if d in ("bullish", "bearish", "neutral") else "neutral", catalyst=str(x.get("catalyst", ""))[:60], importance=int(min(max(int(x.get("importance", 1)), 1), 5))))
    return out if len(out) == n else None


def dashboard(queries: dict, themes_keys: list[str], archive: pd.DataFrame | None, days=14, log=None) -> pd.DataFrame:
    """Live news features for many themes (2 requests each)."""
    rows = {}
    for k in themes_keys:
        if k not in queries:
            continue
        f = live_features(queries[k], days=days, archive=archive if archive is not None and len(archive) else None, theme=k)
        rows[k] = f
        time.sleep(0.3)
        if log:
            log(k)
    return rows


def divergence(price_score: float, tone_chg: float, tone: float) -> str:
    """Plain-language flag when price and news disagree. Not validated as a predictor: use as a prompt to look closer."""
    if price_score is None or tone is None or tone != tone:
        return ""
    if price_score >= 65 and (tone < -0.05 or (tone_chg == tone_chg and tone_chg < -0.2)):
        return "price strong, news souring"
    if price_score <= 40 and (tone > 0.1 or (tone_chg == tone_chg and tone_chg > 0.2)):
        return "price weak, news improving"
    return ""


LIVE_CACHE = Path("data/news_live.json")


def refresh_tone30(queries: dict, keys: list[str] | None = None, fetch=True, max_age_hours=6.0, cache=LIVE_CACHE, log=None) -> pd.Series:
    """Mean headline tone over the last 30 days per theme (the same monthly measure that was backtested). Cached; needs >= 15 headlines."""
    data = json.loads(cache.read_text()) if cache.exists() else {}
    now = time.time()
    keys = keys or list(queries)
    for k in keys:
        rec = data.get(k)
        fresh = rec and (now - rec["t"]) / 3600 < max_age_hours
        if fetch and not fresh and k in queries:
            res = window_headlines(queries[k], pd.Timestamp.today().normalize(), 30)
            tc = [tone(t) for t, _, _ in res]
            data[k] = dict(t=now, n=len(tc), tone=float(np.mean(tc)) if len(tc) >= 15 else None)
            time.sleep(0.3)
            if log:
                log(k)
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(data))
    return pd.Series({k: (data[k]["tone"] if k in data and data[k]["tone"] is not None else np.nan) for k in keys})
