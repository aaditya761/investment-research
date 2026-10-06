# themescan

Ranks global themes (crypto, gold, East Asia, hyperscalers, memory, power, ...) by how likely they are to be
prepping for a bull run. Each theme is a basket of ETF proxies plus individual stocks defined in `themes.yaml`.

## Run

    pip install -r requirements.txt
    python -m themescan scan                       # today's ranking (downloads + caches prices in data/cache)
    python -m themescan scan --detail memory       # ticker-level view of one theme
    python -m themescan scan --asof 2025-01-15     # what did it look like then?
    python -m themescan backtest --bt-start 2020-01-01
    python -m themescan --demo scan                # synthetic data, no network
    pytest

## How it scores (0-100, percentile-ranked across themes)

| Component | Weight | Meaning |
|---|---|---|
| relative strength | 30% | median basket return vs benchmark (1/3/6 month) |
| breadth | 25% | share of members above 50d/200d MA and within 5% of 52w high |
| trend | 15% | distance above 200d MA and its slope |
| volume | 10% | 20d vs 90d dollar volume |
| coil | 10% | recent Bollinger-band compression while now above the 50d |
| near-high | 10% | closeness to 52-week high |

An overextension penalty (up to 15 points) applies when price is >2 std above its 200d MA.
`4w_chg` is the score change over 20 bars. Stages: **Dormant, Basing, Early trend, Extended, Rolling over**.
**PREPPING** = Basing or Early trend with a rising score.

## Caveats

- Scores are *relative* across your universe; a broad bear market still produces a "top" theme.
- Tune thresholds (`scoring.py`) and event dates (`themes.yaml`) against real history with `backtest`;
  judge the flags by false positives as well as hits. The `--demo` backtest is synthetic and meaningless.
- Research screen only, not investment advice.
