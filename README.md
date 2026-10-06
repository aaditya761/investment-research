# themescan

Ranks global themes (crypto, gold, East Asia, hyperscalers, memory, power, ...) by how likely they are to be
prepping for a bull run. Each theme is a basket of ETF proxies plus individual stocks defined in `themes.yaml`.

## Point-and-click UI

    python -m themescan ui            # opens http://127.0.0.1:8765 (add --port N to change, --no-browser to skip opening)

Scan tab: pick a universe (global / India), tick *Cached prices only* for instant reruns, press **Scan**. Click any column to sort,
use the filter pills (Leading / Confirmed / Improving / Lagging), and click a theme for its tickers, live headlines and a valuation
snapshot. *Data & jobs* tab has a button for each long-running command (download data, fundamentals, headlines, backtest, research)
with a live console. It runs on your machine only (bound to 127.0.0.1, no extra dependencies).

## Commands (CLI equivalents)

Global options go **before** the subcommand: `--themes FILE`, `--demo`, `--offline`, `--refresh`, `--start DATE`, `--cache DIR`, `--csv OUT`.

    pip install -r requirements.txt
    pytest                                                     # 8 tests, no network needed

    # 1. Rank themes (downloads + caches prices in data/cache on first run)
    python -m themescan scan                                   # global themes (themes.yaml)
    python -m themescan scan --detail memory                   # ticker-level view of one theme
    python -m themescan scan --asof 2025-01-15                 # what it looked like on a past date
    python -m themescan scan --csv out.csv                     # save the table
    python -m themescan --themes themes_india.yaml scan        # India sectors vs Nifty 50
    python -m themescan --demo scan                            # synthetic data, no network

    # 2. Fundamentals and news (context for the scan)
    python -m themescan altfetch                               # download earnings, analyst and news history (slow: news is rate-limited)
    python -m themescan altfetch --skip-news                   # earnings + analyst only (fast, enough for `confirmed`)
    python -m themescan fundamentals --theme memory japan      # live P/E, growth, estimate revisions
    python -m themescan headlines                              # live headlines for every Leading theme
    python -m themescan headlines --theme brazil --n 8         # headlines for chosen themes

    # 3. Validate / re-tune (slow: a few minutes each)
    python -m themescan --csv data/hist_global.csv backtest --bt-start 2019-06-01 --step 5
    python -m themescan --themes themes_india.yaml --csv data/hist_india.csv backtest --bt-start 2019-06-01 --step 5
    python -m themescan research data/hist_global.csv --validate data/hist_india.csv

Typical routine: `scan` (add `--detail` for any theme you care about) -> `headlines` for the Leading ones -> `fundamentals` for valuation.
Re-run `backtest` + `research` after changing themes or scoring.

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

## Looking for *early* signals (what was tried)

Goal: be early in a run rather than late. Each idea was tested the same way (alpha vs same-date average theme,
train < 2023 / test >= 2023, overlap-adjusted t-stats). **None produced a robust edge:**

| Idea | Source | Result |
|---|---|---|
| Trend age (bars since crossing above 200d), early vs late stage | prices | sign flips between train and test |
| Fresh-breakout / not-extended filters, accel, squeeze, volume | prices | no edge, filters made it worse |
| Analyst upgrades/downgrades (level and 90d change) | Yahoo, 96 stocks | IC ~0; upgrades trail price |
| Wikipedia pageview attention (level / spike / change) | Wikimedia, 26 themes | IC t < 1.2; crowded-vs-quiet flips sign |
| GDELT news volume (spike / change) | GDELT, 9 themes only (rate-limited) | IC t < 0.6; low power |
| Macro-driver tailwind (rolling betas to dollar, yields, curve, VIX, oil, copper, gold, credit, FX x 13w factor move) | Yahoo | IC -0.02, non-monotonic quintiles: no edge |
| **Earnings surprise** (median EPS surprise % of reports in the last 120d) | Yahoo, 120 stocks | **IC +0.08, t 2.9; top-fifth minus bottom-fifth positive in 8 of 8 years; survives dropping any one theme.** Weak on India sectors (t 1.1, flips in test). |

The one validated improvement: among Leading themes, those also in the top half by EPS surprise had +1.7% (train) / +2.7% (test)
quarterly alpha vs -0.5% / -0.5% for Leading themes with weak surprises. `scan` shows this as `confirmed`. Caveats: strongest in
memory / energy / crypto-adjacent baskets (the 2019-26 winners, so survivorship flatters it); crypto EPS is mark-to-market noise;
surprises are only visible after reports, so this confirms a run rather than predicting it.

Also available: `fundamentals` prints a live (non-backtestable) snapshot: forward P/E, growth, upside to analyst targets and
the balance of FY EPS estimate revisions.

Tooling kept for further work: `altfetch` (download alt data into `data/alt/`), `research` (IC + rule search),
and `headlines` (live Google News headlines for Leading themes as a *manual catalyst check*, not backtestable).

## Caveats

- Scores are *relative* across your universe; a broad bear market still produces a "top" theme.
- Tune thresholds (`scoring.py`) and event dates (`themes.yaml`) against real history with `backtest`;
  judge the flags by false positives as well as hits. The `--demo` backtest is synthetic and meaningless.
- Research screen only, not investment advice.
