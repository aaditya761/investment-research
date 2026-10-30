import numpy as np
import pandas as pd

from themescan import portfolio


def _panel(seed=0, dates=60, themes=12, edge=0.0):
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.date_range("2022-01-07", periods=dates, freq="W-FRI"):
        sig = rng.normal(size=themes)
        ret = edge * sig + rng.normal(0, 0.05, themes)
        for i in range(themes):
            rows.append(dict(date=d, theme=f"t{i}", conviction=sig[i], score=sig[i], fwd_excess_21=ret[i]))
    return pd.DataFrame(rows)


def test_run_strategy_picks_top_n_and_costs_reduce_returns():
    p = _panel()
    free, paid = portfolio.run_strategy(p, "conviction", 3, 1, 0, 0), portfolio.run_strategy(p, "conviction", 3, 1, 0, 50)
    assert (free.picks.str.count(",") == 2).all()
    assert paid.net.mean() < free.net.mean()


def test_real_edge_beats_permutation_and_noise_does_not():
    good, noise = _panel(edge=0.05), _panel(seed=1, edge=0.0)
    pg = portfolio.permutation_p(good, portfolio.run_strategy(good, "conviction", 3, 1, 0, 0), 3, draws=300)[0]
    pn = portfolio.permutation_p(noise, portfolio.run_strategy(noise, "conviction", 3, 1, 0, 0), 3, draws=300)[0]
    assert pg < 0.05 < pn
