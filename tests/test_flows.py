import numpy as np

from themescan import rotation as ro
from themescan import synthetic
from themescan.config import load_config


def _panel():
    cfg = load_config("themes.yaml")
    close, vol = synthetic.make(cfg, n_days=900)
    close, vol = ro.weekdays(close, vol)
    return cfg, close, vol, ro.merge_universes({"g": cfg})


def test_flow_map_covers_themes_and_labels():
    cfg, close, vol, themes = _panel()
    df = ro.flow_map(close, vol, themes, cfg.benchmark, window=20, recent=5)
    assert len(df) >= 25 and df.score.between(0, 100).all()
    assert set(df.label) <= set(ro.LABELS)
    assert df.loc[df.score.idxmax(), "label"] == "Strong buying" and df.loc[df.score.idxmin(), "label"] == "Heavy selling"
    assert df.flow.dropna().between(-1, 1).all()


def test_flow_map_window_changes_the_answer_and_groups_summarise():
    cfg, close, vol, themes = _panel()
    a = ro.flow_map(close, vol, themes, cfg.benchmark, window=60, recent=15)
    b = ro.flow_map(close, vol, themes, cfg.benchmark, window=5, recent=3)
    assert not np.allclose(a.score.sort_index().values, b.score.sort_index().values)
    g = ro.flow_groups(a)
    assert abs(g.n.sum() - len(a)) < 1e-9 and g.buying.between(0, 1).all()
