# themescan

Ranks global themes (crypto, gold, East Asia, hyperscalers, memory, power, ...) by how likely they are to be
prepping for a bull run. Each theme is a basket of ETF proxies plus individual stocks defined in `themes.yaml`.

## Point-and-click UI

    python -m themescan ui            # opens http://127.0.0.1:8765 (add --port N to change, --no-browser to skip opening)

**Dashboard** is the product. A pipeline at the top shows the three inputs that change a theme's score (① prices 40%, ② earnings 40%, ③ news tone 20%) feeding Conviction, then the Verdict. Each input card shows how complete and
how fresh its data is and has an **Update** button with a spinner; when the update finishes the table re-scans automatically and a banner plus a Δ column show which themes changed. Click any theme for its tickers, evidence
checklist, headlines and valuation. **News** and **Rotation** tabs and **Tools** (context tools, validation and tracking jobs) never change the scores. A bar at the bottom of the page shows every running job with live output.
It runs on your machine only (bound to 127.0.0.1, no extra dependencies).

## Quickstart (local)

Needs Python 3.10+ (3.11+ recommended) and internet access (Yahoo Finance, Google News, FRED). No API keys.

    git clone -b claude/epic-volta-ksrkce https://github.com/aaditya761/investment-research.git
    cd investment-research
    python3 -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python -m themescan ui                                    # opens http://127.0.0.1:8765

Run everything from the repo root (config files and the `data/` cache use relative paths). First time in the UI (Dashboard):
1. Press **Scan** (downloads prices, about 1-2 minutes, then cached).
2. Press **Update** on the **Earnings** card, then on the **News tone** card. Each finishes with an automatic re-scan and a banner listing what changed.
3. Optional (Tools tab): **Download headline archive** (about 45 minutes, resumable) to enable the News tab's narrative radar.
4. Every week: Tools > **Weekly report** (or `python -m themescan weekly`), then commit `logs/forward_log.csv`.

Schedule the weekly run (Mac/Linux cron, Fridays after the US close, adjust the path):

    30 22 * * 5  cd /path/to/investment-research && .venv/bin/python -m themescan weekly >> reports/cron.log 2>&1

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
    python -m themescan altfetch --skip-news                   # earnings + analyst only (fast; needed for the earnings half of conviction)
    python -m themescan verify memory japan                    # full evidence checklist for chosen themes
    python -m themescan fundamentals --theme memory japan      # live P/E, growth, estimate revisions
    python -m themescan headlines                              # live headlines for every Leading theme
    python -m themescan headlines --theme brazil --n 8         # headlines for chosen themes

    # 3. Validate / re-tune (slow: a few minutes each)
    python -m themescan --csv data/hist_global.csv backtest --bt-start 2019-06-01 --step 5
    python -m themescan --themes themes_india.yaml --csv data/hist_india.csv backtest --bt-start 2019-06-01 --step 5
    python -m themescan research data/hist_global.csv --validate data/hist_india.csv
    python -m themescan portfolio data/hist_global.csv         # top-N rotation vs baselines + random-picks test
    python -m themescan log                                    # forward test: record today's picks (run weekly, commit logs/)
    python -m themescan scorelog                               # score picks once 21 / 63 trading days have passed
    python -m themescan research data/hist_global.csv --blend  # test earnings weight (0%..100%) in the score

Typical routine: `scan` (add `--detail` for any theme you care about) -> `headlines` for the Leading ones -> `fundamentals` for valuation.
Re-run `backtest` + `research` after changing themes or scoring.

## How it scores

**Conviction (0-100) = 40% price + 40% earnings + 20% news.** Price and earnings are blended 50/50 first (earnings was the strongest signal in the walk-forward tests; earnings alone
failed out of sample on India), then 20% weight goes to the 30-day news-tone rank. News is the weakest-evidence component (see *News layer*).

- *Price score* (percentile-ranked across themes): 50% slope of the 200d MA, 30% relative strength vs benchmark (1/3/6 month), 20% distance above the 200d MA.
- *Earnings composite* (percentile-ranked): 50% median EPS surprise %, 30% share of beats, 20% change in surprise vs the prior 120 days. Themes whose EPS is
  accounting noise (`use_earnings: false`, e.g. crypto) or that lack coverage are price-only and cannot be "Strong".
- *Verdict:* **Strong** >= 80, **Positive** 65-80, **Neutral** 35-65, **Avoid** < 35. Average forward alpha rose with each tier in both periods on the global and India universes
  (Strong +3.9% to +5.1% a quarter globally, Avoid -1.4% to -1.6%).
- *Stage* (price only, descriptive): Leading (price score >= 60), Extended, Improving, Neutral, Lagging.

**Evidence checklist** (`verify` command, or click a theme in the UI): every other source is checked but **not scored**, each tagged by how much to trust it:
`validated` (backtested, in the score) / `live only` (valuation, estimate revisions, analyst targets, headline tone: cannot be backtested) /
`tested: no edge` (macro tailwind, analyst upgrades, Wikipedia attention, not-stretched filter: showed no predictive value, shown as context).

## Demand chains, drivers, calendar, weekly report

    python -m themescan chains                        # spender capex -> supplier revenue / gross margin -> price, for chains in chains.yaml (AI build-out included)
    python -m themescan calendar --days 21            # upcoming earnings for theme stocks; spenders and suppliers of tracked chains flagged
    python -m themescan weekly [--ntfy TOPIC]         # logs picks, shows what changed since last week, rotation, news flags, chains, next-7-day earnings -> reports/weekly_DATE.md
    (UI: Tools tab; the evidence checklist gained a theme-specific macro-driver line)

- **Chains** (live Yahoo statements, ~5 quarters, so not backtestable): for the AI chain the tracker shows hyperscaler capex +101% YoY / +12% QoQ, suppliers' revenue (semis +68%, memory +61%, cyber +26%, power +22% YoY) and margins, and flags
  disagreements with price (e.g. power: deliveries growing but price score 35). A capex *cut* by the spenders would be the real warning. Edit `chains.yaml` to track other chains.
- **Theme-specific macro drivers** (FRED, free, long history; fixed economic signs such as gold <- real yield and dollar, homebuilders <- mortgage rates, energy <- oil, small caps <- credit spreads): tested 2019-2026 on the global and broad
  universes, **no predictive value** (rank-correlation negative in 2021-23 and 2024+; positive only in the 2019-20 COVID period). Shown in the checklist as context, tagged "tested: no edge".
- **Cluster-capped selection** (at most 1-2 picks per correlated cluster): tested; it trims drawdowns somewhat on the broad ETF set (top-5 -55% -> -45%) but out of sample (2023+) the information ratio stays about zero, and
  it gives nothing on the global themes. **Not adopted.**
- **Weekly report**: fail-soft sections; the ntfy push is opt-in (anyone who knows your topic name can read it, so use a long random one). Run weekly and commit `logs/forward_log.csv`.

## News layer

    python -m themescan news                         # live: tone, tone change, catalysts, bursting phrases per theme; flags where price and news disagree (--detail THEME for headlines)
    python -m themescan newsarchive                  # one-off, ~45 min, resumable: monthly headline history per theme since 2021 (Google News date-bounded queries)
    python -m themescan narratives                   # phrases bursting across themes this month + how the AI narrative spread over time
    (UI: News tab; Tools > Headline archive)

Source is Google News RSS (free). Historical headlines are point-in-time and sampled (about 100 per theme-month; story *counts* are unreliable, so only text is used). Tone is a finance word list
with negation handling and catalyst tags (earnings, capex/demand, supply/pricing, policy, macro, analyst action, deals, geopolitics). An optional Claude classifier exists (`news.classify_llm`, needs
`ANTHROPIC_API_KEY`) but is not used by default and was not tested against real API output.

What the tests (31 themes, 2021-2026, monthly, 126-bar forward alpha) showed:
- Tone is only weakly correlated with price score (0.24), so it is not just a price proxy. Alone: rank-correlation +0.06 (t 2.1) overall, but about 0 in 2021-23 and +0.15 in 2024 onward.
- Adding it to conviction at 20% left 2021-23 about unchanged (rank-correlation 0.08 -> 0.09) and improved 2024+ (0.08 -> 0.10); higher weights hurt 2021-23. Verdict tiers stayed monotonic and sharpened
  (Strong +5.4% / +13.9%, Avoid -2.2% / -3.5% for 2021-23 / 2024+). Evidence is real but modest and regime-dependent.
- **News confirmed the AI trade; it did not lead it.** AI terms rose from about 1.5% of headlines (mid-2022) to 3.3% (Feb 2023) and 5-8% (2024), but within the individual themes they crossed 15% of headlines
  after price had already moved (cybersecurity Jan 2024, power and Taiwan Apr 2024, uranium Sep 2024).
- "News improving more than price" (divergence) and tone *change* showed no predictive value; the divergence flag is shown as a prompt to look closer only.

## Mega-themes: recognising "the AI trade" while it forms

    python -m themescan clusters                      # (UI: Tools > Mega-theme clusters); first run scores 5 years of weeks, then cached
    python -m themescan clusters --seed g:crypto      # lifecycle of another theme's cluster

Clusters themes by 52-week return correlation (themes that are really one trade end up together), flags a cluster **HOT** when it has >= 3 members, average price score >= 65 and
at least half its members in the top quartile, shows how long it has been hot, a quarterly lifecycle for a seed theme, and an AI "fuel gauge" (hyperscaler capex, like-for-like).
Retrospective on AI: the first alert (>= 4 members, >= 60% in the top quartile, avg score >= 70) fired on **2023-07-07**, about six months after AI semis began running and before the
memory / power / Taiwan-Korea legs had made most of their moves. Tests 2020-2026: participating members of hot clusters beat other themes by about +2-3% per 6 months in both train and test
periods; a "laggard catch-up" gap signal did **not** work out of sample. So it explains the trade and shows who is participating; it does not reliably name the next leg.

## Money flow: what is being bought and sold across ALL sectors

    python -m themescan flows [--window 5|20|60] [--top 15]        (UI: Money flow tab)

Covers 130+ themes (US sectors, countries, factors, commodities, bonds, currencies, thematic baskets). Each gets a **flow score** (0-100): half its return vs the world over the look-back, half net-buying pressure
(share of dollar volume traded on up days minus down days). Shows what is being bought, what is being sold, what is turning (last days very different from the window) and a by-asset-group view. Weekday prices only; India
sectors are not included (different benchmark and currency). **Tested over 133 themes since 2019: the score did not predict the next 20-60 days** (rank-correlation about 0, sign flips between periods), so it
describes what already happened. It is not used in the Dashboard score.

## Rotation: where does money go when a complex sells off?

    python -m themescan rotation                      # default source = the AI complex (AI semis, memory, Taiwan, Korea, hyperscalers, power, cloud, China internet...)
    python -m themescan rotation --source g:energy    # any other complex; --window N, --drop PERCENT, --top N
    (UI: Rotation tab)

Money flows are not directly observable, so three price/volume lenses are combined: **residual strength** (excess return beyond what the theme's beta to the source
explains), **net-buying pressure** (share of dollar volume on up days minus down days), and **history** (forward edge after 13 past selloffs of the same complex since 2018).
Tested on those 13 episodes, the first two lenses did **not** predict which themes kept outperforming (rank-correlation about -0.06 to 0, positive in about half the episodes),
so the output describes where money is going *now*, not where it will continue to go. Uses weekday-only prices.

## Known limitation: calendar-day windows in the scoring framework

The price panel used by `scan` / `backtest` includes weekend rows because 24/7 crypto tickers force a 7-day calendar, so windows labelled "200d", "63 bars", "21 bars"
are really ~200 / 63 / 21 *calendar* days (about 140 / 43 / 15 trading days). The backtests are internally consistent under that definition, but the labels are loose.
`rotation` already uses weekday-only data. Switching the main framework to trading days would require re-running and re-validating all results.

## Reality check: how to use it (portfolio tests, `portfolio` command)

Three universes: the hand-picked global themes, India sectors, and a **broad rule-based ETF universe** (`themes_broad.yaml`, 95 themes including faded
ones such as cannabis, ARK and metaverse) added to remove selection bias.

| Use | Global | India | Broad ETFs |
|---|---|---|---|
| Hold top-5 by conviction (excess vs benchmark, net of costs) | +12%/yr, IR 0.8 | +16%/yr, IR 1.0 | **~0%/yr, IR 0.01, max drawdown -55%; random picks do as well (p = 0.25)** |
| Drop the bottom 30-50% of themes vs holding all (test period) | +2.0 to +3.3%/yr | +2.8 to +2.9%/yr | +0.7 to +1.7%/yr |

- The hand-picked universes flatter any strategy: simply holding all of their themes already beat the benchmark (+3.7%/yr global, +10.8%/yr India) because
  they were chosen with hindsight. Treat the global/India top-5 numbers as upper bounds.
- The ranking is real but weak: on the broad set, forward alpha rises almost monotonically by score decile (-1.7% to +1.5% per quarter).
- A concentrated 5-theme rotation is **not supported**: picks are correlated (e.g. four oil/energy themes at once) and momentum reversals cause deep drawdowns.
  Diversifying helps (top-30: IR 0.19, drawdown -23%).
- Plain trend or momentum earns about as much as conviction (conviction is mainly better risk-adjusted on the global set; on India plain momentum was as good or better).
- **Supported use:** tilt away from Avoid themes and toward Strong/Positive across a diversified set. Not: bet the portfolio on the top few.
- **The honest verdict comes from the forward test** (`log` weekly, `scorelog` later): it records picks before the outcome is known.

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
quarterly alpha vs -0.5% / -0.5% for Leading themes with weak surprises. This is now the earnings half of **conviction**. Caveats: strongest in
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
