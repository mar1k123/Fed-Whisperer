# Fed Whisperer

**Fed Whisperer** is an empirical NLP and macro-econometric research suite that measures the hawkish/dovish tone of Federal Open Market Committee (FOMC) monetary policy statements and tests whether changes in statement tone predict Treasury yield and equity market reactions beyond the policy rate decision itself.

---

## Repository Structure

```text
fed-whisperer/
├── configs/
│   └── config.yaml                     # Scraper rate limits, FRED/yfinance tickers, embedding anchors
├── src/
│   └── fed_whisperer/
│       ├── __init__.py
│       ├── data/
│       │   ├── scraper.py              # Robots.txt-compliant FOMC scraper with local JSON cache
│       │   └── market.py               # FRED (DGS2, DGS10, DFEDTARU) & yfinance (SPY/^GSPC) loader
│       ├── scoring/
│       │   ├── dictionary.py           # Method A: Monetary policy lexicon with negation inversion
│       │   ├── embeddings.py           # Method B: Semantic projection onto Hawkish-Dovish anchor axis
│       │   ├── llm.py                  # Method C: Fixed-rubric Gemini scorer (temperature = 0.0)
│       │   └── agreement.py            # Pairwise Pearson r, Spearman rho, Cohen's kappa, sign match
│       └── analysis/
│           ├── diff.py                 # Sentence-level SequenceMatcher diff & edited-sentence delta
│           ├── event_study.py          # OLS regressions with HC1 robust standard errors & 95% CIs
│           └── robustness.py           # Window sensitivity (1d/2d/5d), Leave-One-Period-Out, Placebo
├── tests/
│   └── test_fed_whisperer.py           # Pytest suite for scorers, diff alignment, and OLS event study
├── app.py                              # Streamlit dashboard (Timeline, Diff Viewer, Event Study)
├── requirements.txt                    # Python 3.10+ dependencies
└── README.md
```

---

## Methodology

### 1. Data Collection & Caching
- **FOMC Statements (`src/fed_whisperer/data/scraper.py`)**: Parses `https://www.federalreserve.gov/robots.txt` via `urllib.robotparser.RobotFileParser`, enforces a configurable request delay (`rate_limit_seconds: 1.5`), extracts policy release links (`/newsevents/pressreleases/monetaryYYYYMMDDa.htm`), strips boilerplate header/media lines, segments sentences, and caches raw + parsed statements on disk.
- **Market Data (`src/fed_whisperer/data/market.py`)**: Queries daily Constant Maturity Treasury yields (`DGS2`, `DGS10`) and the Upper Target Federal Funds Rate (`DFEDTARU`) via `fredapi` (`FRED_API_KEY` from environment) alongside S&P 500 prices (`SPY` / `^GSPC`) via `yfinance`.

### 2. Three Tone Scoring Methods (`[-1.0, +1.0]`)
1. **Method A — Dictionary Approach (`dictionary.py`)**: Uses a curated monetary policy lexicon weighted by directional intensity (e.g., `"ongoing increases"`, `"additional policy firming"`, `"restrictive"` vs. `"accommodative"`, `"downside risks"`, `"transitory"`) with a 3-token preceding negation window (`"not"`, `"less"`, `"without"`).
2. **Method B — Embedding Axis Projection (`embeddings.py`)**: Embeds each sentence using `sentence-transformers` (`all-MiniLM-L6-v2`) and projects sentence vectors onto the unit direction vector separating canonical hawkish anchor sentences from canonical dovish anchor sentences:
   $$\hat{v}_{\text{axis}} = \frac{\bar{e}_{\text{hawk}} - \bar{e}_{\text{dove}}}{\|\bar{e}_{\text{hawk}} - \bar{e}_{\text{dove}}\|_2}, \quad s_i = \tanh(3.0 \cdot \langle e_i, \hat{v}_{\text{axis}} \rangle)$$
3. **Method C — Fixed-Rubric LLM Scoring (`llm.py`)**: Calls the Google Gemini API (`google-genai` SDK) with `temperature=0.0`, structured JSON schema output, and a deterministic 5-band monetary rubric decomposing overall tone into inflation stance, growth/labor stance, and forward-guidance stance.

### 3. Diff Mode (`analysis/diff.py`)
Because FOMC statements retain substantial boilerplate from meeting to meeting, level scores exhibit high autocorrelation. **Diff Mode** aligns sentences between statement $t-1$ and statement $t$ using `difflib.SequenceMatcher`, categorizing each sentence as `unchanged`, `modified`, `added`, or `removed`, and isolates the tone shift ($\Delta \text{Tone}_t$) driven specifically by textual revisions.

### 4. Event Study Specification (`analysis/event_study.py`)
For each FOMC release $t$ and window $w \in \{1, 2, 5\}$ trading days, we estimate via OLS with White's heteroskedasticity-consistent (`HC1`) standard errors:
$$\Delta Y_{t,w} = \alpha + \beta_1 \, \text{RateSurprise}_t + \beta_2 \, \Delta \text{Tone}_t + \varepsilon_{t,w}$$
where:
- $\Delta Y_{t,w}$ is the $w$-day change in the 2-Year Treasury yield (`DGS2`, bps), 10-Year Treasury yield (`DGS10`, bps), or S&P 500 return (%).
- $\text{RateSurprise}_t$ proxies the unexpected policy rate component (measured as the unexpected target rate change relative to pre-meeting short-rate pricing).
- $\Delta \text{Tone}_t = \text{Tone}_t - \text{Tone}_{t-1}$ is the sequential statement tone change.

---

## Honest Limitations & Econometric Caveats

1. **Small Sample Size ($N \approx 24\text{–}40$ meetings)**: The FOMC holds only 8 scheduled meetings per year. Over a 3-to-5-year sample, degrees of freedom are tight, resulting in wide 95% confidence intervals and elevated sensitivity to individual outlier meetings (such as March 2020 or June 2022).
2. **Simultaneous Information Confounders (SEP & Press Conference)**: At four meetings per year (March, June, September, December), the FOMC simultaneously releases the Summary of Economic Projections (the "dot plot"), and the Chair holds a live press conference 30 minutes after the 2:00 PM ET statement release. Daily market windows ($1\text{d}, 2\text{d}, 5\text{d}$) using end-of-day FRED series cannot separate the market reaction to the written statement from the reaction to the dot plot or the Chair's Q&A without high-frequency intraday tick data (e.g., 2:00 PM to 2:25 PM ET futures windows).
3. **Endogeneity & Pre-Meeting Communication**: Fed officials frequently preview policy shifts in speeches during the inter-meeting period. When a hawkish statement shift is already anticipated by bond traders, daily yield reactions at announcement may be muted or even reverse ("sell the rumor, buy the news").
4. **No Causal Claims**: Regression coefficients reported by this codebase reflect conditional associations within a finite sample of FOMC releases and must not be interpreted as structural causal effects.
