"""Synthetic prices for offline demos and tests (no network). Plants known bull runs."""
import numpy as np
import pandas as pd

from .config import Config


def make(cfg: Config, n_days: int = 1700, seed: int = 7, hot: dict[str, int] | None = None, end="2026-09-30"):
    """`hot` maps theme -> bars-before-end at which a bull run starts. Default plants three runs."""
    rng = np.random.default_rng(seed)
    hot = hot if hot is not None else {"memory": 70, "gold": 90, "japan": 400}
    idx = pd.bdate_range(end=end, periods=n_days)
    mkt = rng.normal(0.0003, 0.008, n_days)
    out = {cfg.benchmark: 100 * np.exp(np.cumsum(mkt))}
    vols = {cfg.benchmark: np.full(n_days, 1e6)}
    for key, th in cfg.themes.items():
        drift = np.zeros(n_days)
        if key in hot:
            start = n_days - hot[key]
            drift[start:] = 0.0020 if hot[key] < 200 else 0.0030
            if hot[key] > 200:  # an old run that has since blown off and stalled
                drift[start + 150 :] = -0.0005
        factor = rng.normal(0, 0.010, n_days) + drift
        for t in th.tickers:
            if t in out:
                continue
            ret = 0.8 * mkt + 1.0 * factor + rng.normal(0, 0.008, n_days)
            out[t] = 50 * np.exp(np.cumsum(ret))
            v = rng.lognormal(13, 0.2, n_days)
            if key in hot and hot[key] < 200:
                v[n_days - hot[key] :] *= 1.6
            vols[t] = v
    return pd.DataFrame(out, index=idx), pd.DataFrame(vols, index=idx)
