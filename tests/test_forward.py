import numpy as np
import pandas as pd

from themescan import forward, synthetic, verify
from themescan.config import load_config


def test_log_is_idempotent_and_scores_matured_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(forward, "LOG", tmp_path / "log.csv")
    cfg = load_config("themes.yaml")
    close, vol = synthetic.make(cfg, n_days=1500)
    asof_i = len(close) - 80  # 79 bars of "future" exist, so the 21 and 63 bar horizons have matured
    df, _ = verify.full_scan(cfg, close, vol, asof_i, fundamentals=False)
    asof = close.index[asof_i - 1]
    assert forward.write_log("themes.yaml", df, asof) == len(df)
    assert forward.write_log("themes.yaml", df, asof) == 0  # second run for the same date adds nothing
    out = forward.score_log({"themes.yaml": (cfg, close)})
    assert "21-bar forward results" in out and "63-bar forward results" in out and "Pending" not in out


def test_unmatured_entries_are_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(forward, "LOG", tmp_path / "log.csv")
    cfg = load_config("themes.yaml")
    close, vol = synthetic.make(cfg, n_days=1500)
    df, _ = verify.full_scan(cfg, close, vol, fundamentals=False)
    forward.write_log("themes.yaml", df, close.index[-1])
    out = forward.score_log({"themes.yaml": (cfg, close)})
    assert "No logged picks have matured" in out and "Pending" in out
