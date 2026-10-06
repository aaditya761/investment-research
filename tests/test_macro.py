import numpy as np
import pandas as pd

from themescan import macro, synthetic
from themescan.config import load_config


def test_tailwinds_shape_and_no_nan_theme_rows():
    cfg = load_config("themes.yaml")
    close, _ = synthetic.make(cfg, n_days=1500)
    rng = np.random.default_rng(1)
    idx = close.index
    cols = macro.macro_tickers()
    m = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.005, (len(idx), len(cols))), axis=0)), index=idx, columns=cols)
    tw = macro.tailwinds(close, close[cfg.benchmark], cfg.themes, m)
    assert {"theme", "date", "macro_tw", "usd", "oil"} <= set(tw.columns)
    assert tw.macro_tw.notna().all() and tw.theme.nunique() > 20
