"""Alternative data: news attention (GDELT) and analyst rating changes (Yahoo). Cached under data/alt/."""
import time
from pathlib import Path

import pandas as pd
import requests
import yaml

ALT = Path("data/alt")
GDELT = "https://api.gdeltproject.org/api/v2/doc/doc"


def fetch_news(queries: dict[str, str], start="20190101000000", end=None, refresh=False) -> dict[str, pd.Series]:
    """Daily GDELT news volume (% of global coverage) per theme. GDELT allows ~1 request / 5s."""
    (ALT / "news").mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y%m%d000000")
    out = {}
    for key, q in queries.items():
        f = ALT / "news" / f"{key}.csv"
        if f.exists() and not refresh:
            out[key] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
            continue
        for attempt in range(8):
            time.sleep(8 + 12 * attempt)  # back off: GDELT throttles harder than its stated 1 req / 5s
            try:
                r = requests.get(GDELT, params=dict(query=q, mode="timelinevol", format="json", startdatetime=start, enddatetime=end), timeout=90)
                if r.status_code == 200 and r.text.strip().startswith("{"):
                    d = r.json()["timeline"][0]["data"]
                    s = pd.Series({pd.to_datetime(x["date"]).tz_localize(None).normalize(): x["value"] for x in d}, name=key)
                    s.to_frame().to_csv(f)
                    out[key] = s
                    break
                print(f"  {key}: {r.status_code} {r.text[:80]!r}")
            except Exception as e:
                print(f"  {key}: {e}")
    return out


def fetch_analyst_actions(tickers: list[str], refresh=False) -> pd.DataFrame:
    """Dated analyst up/downgrades per stock (yfinance). Returns long frame: date, ticker, action."""
    import yfinance as yf

    (ALT / "analyst").mkdir(parents=True, exist_ok=True)
    frames = []
    for t in tickers:
        f = ALT / "analyst" / f"{t.replace('^', '_')}.csv"
        if f.exists() and not refresh:
            df = pd.read_csv(f, parse_dates=["date"]) if f.stat().st_size > 5 else None
        else:
            df = None
            try:
                u = yf.Ticker(t).upgrades_downgrades
                if u is not None and len(u):
                    df = u.reset_index().rename(columns={u.index.name or "GradeDate": "date", "Action": "action"})
                    df = df.rename(columns={df.columns[0]: "date"})[["date", "action"]]
                    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
            except Exception:
                pass
            (df if df is not None else pd.DataFrame(columns=["date", "action"])).to_csv(f, index=False)
            time.sleep(0.4)
        if df is not None and len(df):
            df["ticker"] = t
            frames.append(df)
    return pd.concat(frames) if frames else pd.DataFrame(columns=["date", "action", "ticker"])


def news_features(series: pd.Series, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """As-of features: attention level vs its own history, and its recent change. Uses only data up to each date."""
    s = series.asfreq("D").interpolate(limit=7)
    m14, m90 = s.rolling(14).mean(), s.rolling(90).mean()
    base, sd = s.rolling(365).mean(), s.rolling(365).std()
    f = pd.DataFrame({
        "news_z": (m14 - base) / sd,                       # attention spike vs last year
        "news_chg": m14 / m90 - 1,                         # attention rising / falling
        "news_pct": m14.rolling(730, min_periods=365).rank(pct=True),  # percentile of attention within 2y
    })
    return f.reindex(dates, method="ffill")


def analyst_features(actions: pd.DataFrame, themes: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Net upgrades per covered stock over trailing 60 days, per theme, plus its 90-day change."""
    a = actions[actions.action.isin(["up", "down"])].copy()
    a["date"] = pd.to_datetime(a["date"]).dt.normalize()
    a["v"] = a.action.map({"up": 1, "down": -1})
    rows = {}
    for key, th in themes.items():
        sub = a[a.ticker.isin(th.tickers)]
        n = max(sub.ticker.nunique(), 1)
        daily = sub.groupby("date").v.sum().reindex(pd.date_range(dates.min() - pd.Timedelta(days=200), dates.max())).fillna(0)
        net60 = daily.rolling(60).sum() / n
        rows[key] = pd.DataFrame({"net_upgr": net60.reindex(dates), "upgr_chg": (net60 - net60.shift(90)).reindex(dates), "covered": n})
    return pd.concat(rows, names=["theme", "date"]).reset_index()


def fetch_wiki(articles: dict[str, str], start="20190101", end=None, refresh=False) -> dict[str, pd.Series]:
    """Daily human pageviews of a Wikipedia article per theme."""
    (ALT / "wiki").mkdir(parents=True, exist_ok=True)
    end = end or pd.Timestamp.today().strftime("%Y%m%d")
    out = {}
    for key, art in articles.items():
        f = ALT / "wiki" / f"{key}.csv"
        if f.exists() and not refresh:
            out[key] = pd.read_csv(f, index_col=0, parse_dates=True).iloc[:, 0]
            continue
        url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/user/{art}/daily/{start}/{end}"
        for _ in range(3):
            try:
                r = requests.get(url, headers={"User-Agent": "themescan-research/0.1"}, timeout=60)
                if r.status_code == 200:
                    items = r.json()["items"]
                    s = pd.Series({pd.to_datetime(x["timestamp"][:8]): x["views"] for x in items}, name=key)
                    s.to_frame().to_csv(f)
                    out[key] = s
                    break
                print(f"  {key}: {r.status_code}")
                if r.status_code == 404:
                    break
            except Exception as e:
                print(f"  {key}: {e}")
            time.sleep(5)
        time.sleep(1.5)
    return out


_POS = ("surge", "rally", "record", "soar", "jump", "boom", "beat", "upgrade", "breakout", "climb", "gain")
_NEG = ("plunge", "slump", "fall", "crash", "slip", "cut", "downgrade", "miss", "probe", "selloff", "decline", "drop")


def headlines(query: str, n: int = 6, days: int = 14) -> tuple[list[dict], int]:
    """Recent headlines from Google News RSS (live only; there is no history, so this cannot be backtested)."""
    import xml.etree.ElementTree as ET

    try:
        r = requests.get("https://news.google.com/rss/search", params=dict(q=f"{query} when:{days}d", hl="en-US", gl="US", ceid="US:en"),
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        items = ET.fromstring(r.content).findall(".//item")
    except Exception:
        return [], 0
    out = []
    for it in items[:n * 3]:
        title = (it.findtext("title") or "").strip()
        low = title.lower()
        out.append(dict(title=title, date=(it.findtext("pubDate") or "")[5:16], source=it.findtext("source") or "",
                        tone=sum(w in low for w in _POS) - sum(w in low for w in _NEG)))
    return out[:n], len(items)
