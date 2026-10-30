"""Demand-chain tracker: follow the money from the spenders (customers) to the suppliers.

For a chain such as the AI build-out:  FUEL = spender capex growth;  DELIVERY = suppliers' revenue growth and gross-margin change;
PRICE = what each supplier theme's price/conviction says. Disagreements (fuel rising while price falls, or price rising while deliveries slow)
are the interesting cases. Data: Yahoo quarterly statements (about 5 quarters, so one year-over-year point per company). Live only: not backtestable.
"""
import numpy as np
import pandas as pd
import yaml

from . import clusters as cl


def _yoy(s: pd.Series):
    """Latest value vs the value ~4 quarters earlier (matched within 25 days). Returns (yoy, qoq, latest_date)."""
    s = s.dropna().sort_index()
    if len(s) < 2:
        return np.nan, np.nan, None
    last = s.index[-1]
    prior = s.index[(abs((s.index - (last - pd.DateOffset(years=1))).days) <= 25)]
    yoy = s.iloc[-1] / s[prior[0]] - 1 if len(prior) and s[prior[0]] > 0 else np.nan
    qoq = s.iloc[-1] / s.iloc[-2] - 1 if s.iloc[-2] > 0 else np.nan
    return float(yoy), float(qoq), last


def quarterly(t: str):
    import yfinance as yf

    tk = yf.Ticker(t)
    out = {}
    try:
        inc = tk.quarterly_income_stmt
        if inc is not None and "Total Revenue" in inc.index:
            rev = inc.loc["Total Revenue"].dropna().sort_index().astype(float)
            rev.index = pd.to_datetime(rev.index)
            out["revenue"] = rev
            if "Gross Profit" in inc.index:
                gm = (inc.loc["Gross Profit"] / inc.loc["Total Revenue"]).dropna().sort_index().astype(float)
                gm.index = pd.to_datetime(gm.index)
                out["gm"] = gm
    except Exception:
        pass
    try:
        cf = tk.quarterly_cashflow
        if cf is not None and "Capital Expenditure" in cf.index:
            cx = (-cf.loc["Capital Expenditure"].dropna()).sort_index().astype(float)
            cx.index = pd.to_datetime(cx.index)
            out["capex"] = cx
    except Exception:
        pass
    return out


def analyze(chain: dict, verdicts: pd.Series | None = None, scores: pd.Series | None = None) -> dict:
    spend = {t: quarterly(t) for t in chain["spenders"]}
    cap = pd.DataFrame({t: d["capex"] / 1e9 for t, d in spend.items() if "capex" in d}).sort_index()
    fuel = cl.fuel_from_frame(cap) if len(cap) >= 2 else {}
    # capex intensity: latest quarter capex / revenue, like-for-like companies
    inten = {}
    for t, d in spend.items():
        if "capex" in d and "revenue" in d:
            c, r = d["capex"].dropna(), d["revenue"].dropna()
            both = c.index.intersection(r.index)
            if len(both):
                inten[t] = float(c[both[-1]] / r[both[-1]])
    sup = {}
    for theme, tickers in chain["suppliers"].items():
        rows = []
        for t in tickers:
            q = quarterly(t)
            if "revenue" in q:
                yoy, qoq, last = _yoy(q["revenue"])
                gm_chg = np.nan
                if "gm" in q:
                    g = q["gm"].dropna()
                    prior = g.index[(abs((g.index - (g.index[-1] - pd.DateOffset(years=1))).days) <= 25)] if len(g) else []
                    gm_chg = float(g.iloc[-1] - g[prior[0]]) if len(prior) else np.nan
                rows.append(dict(ticker=t, rev_yoy=yoy, rev_qoq=qoq, gm=float(q["gm"].iloc[-1]) if "gm" in q and len(q["gm"]) else np.nan, gm_chg=gm_chg))
        d = pd.DataFrame(rows)
        sup[theme] = dict(stocks=d, rev_yoy=float(d.rev_yoy.median()) if len(d) else np.nan, rev_qoq=float(d.rev_qoq.median()) if len(d) else np.nan,
                          gm_chg=float(d.gm_chg.median()) if len(d) else np.nan, verdict=(verdicts.get(theme, "") if verdicts is not None else ""),
                          score=(float(scores.get(theme, np.nan)) if scores is not None else np.nan))
    return dict(fuel=fuel, intensity=inten, suppliers=sup)


def reading(res: dict) -> list[str]:
    """Plain-language lines about agreement / disagreement between fuel, deliveries and price."""
    out = []
    f = res["fuel"]
    fuel_up = f.get("yoy", np.nan) > 0.2 and f.get("qoq", 0) > 0
    if f:
        out.append(f"Fuel: spender capex ${f['latest']:.0f}B ({', '.join(f['companies'])}) {f.get('yoy', np.nan) * 100:+.0f}% YoY, {f.get('qoq', np.nan) * 100:+.0f}% QoQ -> " + ("still accelerating" if fuel_up else "NOT accelerating: the real warning sign" if f.get("qoq", 0) < 0 else "growing"))
    for k, s in res["suppliers"].items():
        bits = []
        if s["rev_yoy"] == s["rev_yoy"]:
            bits.append(f"revenue {s['rev_yoy'] * 100:+.0f}% YoY")
        if s["gm_chg"] == s["gm_chg"]:
            bits.append(f"gross margin {s['gm_chg'] * 100:+.1f}pp")
        flag = ""
        if fuel_up and s["score"] == s["score"] and s["score"] < 45:
            flag = "  <- fuel rising but price weak (fundamentals ahead of price)"
        elif not fuel_up and f and s["score"] == s["score"] and s["score"] > 70:
            flag = "  <- price strong while fuel is not accelerating (risk)"
        out.append(f"{k}: " + ", ".join(bits) + f"; price score {s['score']:.0f}, verdict {s['verdict']}" + flag)
    return out


def load_chains(path="chains.yaml") -> dict:
    return yaml.safe_load(open(path))
