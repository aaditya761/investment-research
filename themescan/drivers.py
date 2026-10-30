"""Theme-specific macro drivers from FRED (free, long history), with fixed economic signs (not fitted to returns).

Each theme maps to the few drivers that economically matter for it (gold <- real yields and the dollar, homebuilders <- mortgage rates,
energy <- oil, small caps / banks <- credit spreads and financial conditions ...). For each driver: the 63-trading-day change, standardised by its own
3-year history, times the sign that helps the theme. A theme's tailwind is the average over its drivers. Point-in-time: each series is lagged by its
typical publication delay before use.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CACHE = Path("data/fred")
# id -> (label, publication lag in days, how to change: 'diff' for rate-like, 'log' for price-like)
SERIES = {
    "real_yield": ("DFII10", "10y real yield", 1, "diff"),
    "dollar": ("DTWEXBGS", "trade-weighted dollar", 7, "log"),
    "hy_spread": ("BAMLH0A0HYM2", "high-yield credit spread", 1, "diff"),
    "wti": ("DCOILWTICO", "WTI oil", 3, "log"),
    "gas": ("DHHNGSP", "Henry Hub gas", 3, "log"),
    "mortgage": ("MORTGAGE30US", "30y mortgage rate", 3, "diff"),
    "nfci": ("NFCI", "financial conditions (lower = easier)", 6, "diff"),
    "fed_bs": ("WALCL", "Fed balance sheet", 3, "log"),
    "copper": ("PCOPPUSDM", "copper price", 45, "log"),
    "curve": ("T10Y2Y", "10y-2y curve", 1, "diff"),
}

# sign: +1 means a RISING driver is a tailwind for the theme. Keys are theme names without the g:/b: prefix.
_RATE_SENSITIVE = {"real_yield": -1}
MAP = {
    "gold": {"real_yield": -1, "dollar": -1}, "gold_miners": {"real_yield": -1, "dollar": -1}, "silver": {"real_yield": -1, "dollar": -1},
    "long_treasuries": {"real_yield": -1}, "intermediate_treasuries": {"real_yield": -1}, "tips": {"real_yield": -1},
    "oil": {"wti": 1}, "us_energy": {"wti": 1}, "energy": {"wti": 1}, "coal_old_energy": {"wti": 1, "gas": 1},
    "nat_gas": {"gas": 1},
    "homebuilders": {"mortgage": -1}, "us_realestate": {"mortgage": -1}, "reits": {"mortgage": -1}, "reits_mortgage": {"mortgage": -1},
    "small_caps": {"hy_spread": -1, "nfci": -1}, "mid_caps": {"hy_spread": -1, "nfci": -1}, "us_financials": {"curve": 1, "hy_spread": -1}, "financials": {"curve": 1, "hy_spread": -1},
    "regional_banks": {"curve": 1, "hy_spread": -1}, "high_yield": {"hy_spread": -1},
    "copper": {"copper": 1, "dollar": -1}, "copper_metals": {"copper": 1, "dollar": -1}, "us_materials": {"copper": 1, "dollar": -1}, "broad_commodities": {"dollar": -1, "wti": 1},
    "us_utilities": _RATE_SENSITIVE, "dividend": _RATE_SENSITIVE, "low_vol": _RATE_SENSITIVE,
    "growth": _RATE_SENSITIVE, "us_tech": _RATE_SENSITIVE, "ark_innovation": _RATE_SENSITIVE, "cloud": _RATE_SENSITIVE, "semis": _RATE_SENSITIVE, "cybersecurity": _RATE_SENSITIVE,
    "crypto": {"fed_bs": 1, "dollar": -1}, "uranium": {"real_yield": -1}, "lithium_batt": {"real_yield": -1}, "clean_energy": {"real_yield": -1}, "solar": {"real_yield": -1},
}
for k in ["brazil", "mexico", "south_africa", "turkey", "indonesia", "thailand", "vietnam", "malaysia", "philippines", "poland", "chile", "argentina", "saudi", "emerging", "frontier",
          "india", "china", "korea", "taiwan", "em_bonds", "latam_ex_brazil", "china_hk"]:
    MAP[k] = {"dollar": -1, "hy_spread": -1}


def fetch(series_ids=None, refresh_hours=20) -> pd.DataFrame:
    """Daily frame of the series (forward-filled, lagged by publication delay)."""
    import time

    CACHE.mkdir(parents=True, exist_ok=True)
    cols = {}
    for name, (sid, _lbl, lag, _how) in SERIES.items():
        if series_ids and name not in series_ids:
            continue
        f = CACHE / f"{sid}.csv"
        if not f.exists() or (time.time() - f.stat().st_mtime) / 3600 > refresh_hours:
            try:
                r = requests.get(f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}", timeout=40)
                if r.status_code == 200 and len(r.text) > 100:
                    f.write_text(r.text)
            except Exception:
                pass
        if not f.exists():
            continue
        d = pd.read_csv(f)
        d.columns = ["date", "v"]
        d["date"] = pd.to_datetime(d["date"]) + pd.Timedelta(days=lag)  # not usable before it was published
        d["v"] = pd.to_numeric(d["v"], errors="coerce")
        cols[name] = d.dropna().set_index("date")["v"]
    return pd.DataFrame(cols).sort_index().ffill()


def driver_z(fred: pd.DataFrame, days=63) -> pd.DataFrame:
    """Standardised change of each driver over `days` rows of a business-day grid (z vs trailing ~3 years)."""
    g = fred.resample("B").last().ffill()
    out = {}
    for name, (_sid, _lbl, _lag, how) in SERIES.items():
        if name not in g:
            continue
        ch = np.log(g[name]).diff(days) if how == "log" else g[name].diff(days)
        out[name] = (ch - ch.rolling(756, min_periods=250).mean()) / ch.rolling(756, min_periods=250).std()
    return pd.DataFrame(out)


def tailwind(z: pd.DataFrame, key: str, date=None) -> float:
    m = MAP.get(key.split(":")[-1])
    if not m:
        return np.nan
    row = z.loc[:date].iloc[-1] if date is not None else z.iloc[-1]
    v = [s * row[d] for d, s in m.items() if d in row and pd.notna(row[d])]
    return float(np.mean(v)) if v else np.nan


def tailwind_panel(z: pd.DataFrame, keys: list[str], dates) -> pd.DataFrame:
    rows = []
    for k in keys:
        m = MAP.get(k.split(":")[-1])
        if not m:
            continue
        s = sum(sg * z[d] for d, sg in m.items() if d in z) / sum(1 for d in m if d in z)
        rows.append(pd.DataFrame({"theme": k, "date": dates, "tailwind": s.reindex(dates, method="ffill").values}))
    return pd.concat(rows) if rows else pd.DataFrame(columns=["theme", "date", "tailwind"])
