"""Per-ticker signals, computed as of the last row of the data passed in."""
import numpy as np
import pandas as pd

HORIZONS = (21, 63, 126)  # ~1, 3, 6 months of trading days
MIN_BARS = 210
TAIL = 470  # enough for 200d MA + 252d dispersion window


def rsi(s: pd.Series, n: int = 14) -> float:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean().iloc[-1]
    return 100.0 if dn == 0 else 100 - 100 / (1 + up / dn)


def ticker_metrics(close: pd.DataFrame, volume: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    close, volume = close.tail(TAIL), volume.tail(TAIL)
    bench = bench.reindex(close.index).ffill()
    rows = {}
    for t in close.columns:
        s = close[t].dropna()
        if len(s) < MIN_BARS or s.iloc[-1] != s.iloc[-1]:
            continue
        b = bench.reindex(s.index).ffill()
        last = s.iloc[-1]
        ret = {h: last / s.iloc[-1 - h] - 1 for h in HORIZONS}
        bret = {h: b.iloc[-1] / b.iloc[-1 - h] - 1 for h in HORIZONS}
        rs = float(np.mean([ret[h] - bret[h] for h in HORIZONS]))

        sma50, sma200 = s.rolling(50).mean(), s.rolling(200).mean()
        d50, d200 = last / sma50.iloc[-1] - 1, last / sma200.iloc[-1] - 1
        slope200 = sma200.iloc[-1] / sma200.iloc[-21] - 1
        dev = (s / sma200 - 1).dropna().tail(252)
        ext = d200 / dev.std() if len(dev) > 60 and dev.std() > 0 else np.nan
        near_high = last / s.tail(252).max() - 1

        # Bollinger bandwidth: how compressed was the stock recently, and is it expanding now?
        m20 = s.rolling(20).mean()
        bw = (4 * s.rolling(20).std() / m20).dropna().tail(252)
        bw_min = bw.tail(30).min()
        squeeze = float((bw < bw_min).mean())  # 0 = tightest in a year; low means recently coiled

        volr = np.nan
        if t in volume:
            dv = (volume[t].reindex(s.index).replace(0, np.nan) * s).dropna()
            if len(dv) >= 90 and dv.tail(90).mean() > 0:
                volr = dv.tail(20).mean() / dv.tail(90).mean()

        rows[t] = dict(
            r1m=ret[21], r3m=ret[63], r6m=ret[126], rs=rs, d50=d50, d200=d200, slope200=slope200,
            above50=float(d50 > 0), above200=float(d200 > 0), near_high=near_high,
            at_high=float(near_high > -0.05), sqz=squeeze, volr=volr, rsi=rsi(s), ext=ext,
        )
    return pd.DataFrame.from_dict(rows, orient="index")
