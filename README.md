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
    python -m themescan --themes themes_india.yaml scan   # India sectors vs Nifty 50
    pytest

## How it scores (0-100, percentile-ranked across themes)

Score = 50% slope of the 200d MA, 30% relative strength vs benchmark (1/3/6 month), 20% distance above the 200d MA.
Breadth, volume, Bollinger squeeze, near-high and acceleration are still *reported* but not scored: in the
walk-forward research they had no predictive power (below).

Stages: **Leading** (score >= 60), **Extended** (leading but stretched: >2 std above the 200d or RSI > 78), **Improving**
(40-60 and rising), **Neutral**, **Lagging** (score <= 30). The `lead` flag = Leading or Extended.

## Validation (`python -m themescan research data/hist_global.csv --validate data/hist_india.csv`)

Walk-forward 2019-2026, weekly, 31 global themes (incl. 8 "control" themes that never had a bull run) and 25 India
sectors. Alpha = 63-day forward return minus the same-date average theme. Train < 2023, test >= 2023.

- The original design (breadth, volume, squeeze, near-high, acceleration, "prepping" flag) had **no edge**: flagged
  themes had ~0 train alpha and -0.8% out of sample. Its "flagged N weeks before the run" lead times were an artefact
  of flagging ~28% of all theme-weeks.
- Only slow trend signals carried information: slope of the 200d MA (top quintile +2.2% vs bottom -0.8% per quarter
  globally; positive in 6 of 8 years in both universes), then relative strength and distance above the 200d.
- Current `lead` flag: +1.0% (train) / +1.2% (test) alpha per quarter, hit rate ~47-48% (the edge comes from a few big
  winners). The `Lagging` side is the most consistent: -0.7% to -1.9% in every period and universe.
- **Nothing is statistically significant** (overlap-adjusted t-stats 1.0-1.8). Treat the ranking as a descriptive
  trend screen and an avoid-list, not as a predictor of the next bull run.
- Extension filters and acceleration made results *worse*; a grid search over rule thresholds found nothing that
  held out of sample.

## Caveats

- Scores are *relative* across your universe; a broad bear market still produces a "top" theme.
- Tune thresholds (`scoring.py`) and event dates (`themes.yaml`) against real history with `backtest`;
  judge the flags by false positives as well as hits. The `--demo` backtest is synthetic and meaningless.
- Research screen only, not investment advice.
