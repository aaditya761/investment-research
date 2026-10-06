import warnings

from themescan import scoring, synthetic
from themescan.config import load_config

warnings.filterwarnings("ignore")


def test_india_universe_scans():
    cfg = load_config("themes_india.yaml")
    assert cfg.benchmark == "^NSEI" and len(cfg.themes) >= 20
    close, vol = synthetic.make(cfg)
    df, _ = scoring.scan(close, vol, close[cfg.benchmark], cfg.themes)
    assert len(df) == len(cfg.themes)
    assert df.score.between(0, 100).all()
