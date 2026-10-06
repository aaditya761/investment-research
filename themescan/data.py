"""Price loading with a per-ticker CSV cache. Yahoo Finance via yfinance."""
from datetime import date, timedelta
from pathlib import Path

import pandas as pd


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, index_col=0, parse_dates=True)


def _split_download(raw: pd.DataFrame, tickers: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    if raw is None or raw.empty:
        return out
    if isinstance(raw.columns, pd.MultiIndex):
        for t in tickers:
            for lvl in (0, 1):
                if t in raw.columns.get_level_values(lvl):
                    df = raw.xs(t, axis=1, level=lvl)
                    break
            else:
                continue
            if "Close" in df and df["Close"].notna().any():
                out[t] = df[["Close", "Volume"]].dropna(subset=["Close"])
    elif len(tickers) == 1 and "Close" in raw:
        out[tickers[0]] = raw[["Close", "Volume"]].dropna(subset=["Close"])
    return out


def load_prices(
    tickers: list[str],
    start: str = "2018-01-01",
    cache_dir: str | Path = "data/cache",
    refresh: bool = False,
    offline: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Return (close, volume, missing). Close is adjusted; holidays are forward-filled up to 5 bars."""
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    frames: dict[str, pd.DataFrame] = {}
    stale: list[str] = []
    fresh_after = pd.Timestamp(date.today() - timedelta(days=4))
    for t in tickers:
        df = _read_cache(cache / f"{t.replace('^', '_')}.csv")
        if df is not None and not df.empty:
            frames[t] = df
        if offline:
            continue
        if refresh or df is None or df.empty or df.index[-1] < fresh_after or df.index[0] > pd.Timestamp(start) + timedelta(days=30):
            stale.append(t)

    if stale:
        import yfinance as yf

        for i in range(0, len(stale), 40):
            chunk = stale[i : i + 40]
            try:
                raw = yf.download(chunk, start=start, auto_adjust=True, progress=False, group_by="ticker", threads=True)
            except Exception as e:  # network / rate limit: fall back to whatever is cached
                print(f"warning: download failed for {len(chunk)} tickers: {e}")
                continue
            for t, df in _split_download(raw, chunk).items():
                df.index = pd.to_datetime(df.index).tz_localize(None)
                df.to_csv(cache / f"{t.replace('^', '_')}.csv")
                frames[t] = df

    missing = [t for t in tickers if t not in frames]
    if not frames:
        raise RuntimeError("no price data available (network blocked and cache empty?)")
    close = pd.DataFrame({t: f["Close"] for t, f in frames.items()}).sort_index()
    vol = pd.DataFrame({t: f["Volume"] for t, f in frames.items()}).reindex(close.index)
    close = close.loc[start:].ffill(limit=5)
    return close, vol.loc[close.index], missing
