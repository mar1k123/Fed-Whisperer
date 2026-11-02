# Fed Whisperer

**Quantitative NLP Tone Measurement & Macro-Econometric Event Study of FOMC Monetary Policy Statements**

`fed-whisperer` is an end-to-end natural language processing and financial econometrics toolkit that quantifies the hawkish/dovish stance of Federal Open Market Committee (FOMC) press releases, isolates sentence-level revisions between consecutive meetings (**Diff Mode**), and tests whether textual tone shifts predict U.S. Treasury yield (`DGS2`, `DGS10`) and equity index (`^GSPC` / `SPY`) reactions beyond the policy rate decision itself.

---

## Table of Contents

1. [Project Overview & Architecture](#1-project-overview--architecture)
2. [Repository Structure](#2-repository-structure)
3. [Running as a Native PC Desktop App or Web Dashboard](#3-running-as-a-native-pc-desktop-app-or-web-dashboard)
4. [Data Pipeline: FOMC Scraper & Market Loaders](#4-data-pipeline-fomc-scraper--market-loaders)
5. [Three Tone Scoring Methodologies](#5-three-tone-scoring-methodologies)
6. [Inter-Method Agreement & Concordance](#6-inter-method-agreement--concordance)
7. [Sentence-Level Diff Mode](#7-sentence-level-diff-mode)
8. [Econometric Event Study & Robustness Suite](#8-econometric-event-study--robustness-suite)
9. [Empirical Results Summary](#9-empirical-results-summary)
10. [Honest Econometric Limitations & Identification Caveats](#10-honest-econometric-limitations--identification-caveats)
11. [Testing & Reproducibility](#11-testing--reproducibility)

---

## 1. Project Overview & Architecture

Central bank watchers scrutinize every word changed in an FOMC statement because policy rates often move in predictable 25 bps increments while forward guidance and risk assessments reshape the entire yield curve. **Fed Whisperer** formalizes this process through a reproducible pipeline:

```text
┌──────────────────────────────────────────────────────────────────────────────┐
│                         DATA INGESTION & LOCAL CACHE                         │
│  ┌────────────────────────────────────┐  ┌────────────────────────────────┐  │
│  │ Federal Reserve Website Scraper    │  │ Market Series Loader           │  │
│  │ • Respects robots.txt & rate limit │  │ • FRED: DGS2, DGS10, DFEDTARU  │  │
│  │ • Sentence segmentation & cleaning │  │ • yfinance: SPY / ^GSPC        │  │
│  └─────────────────┬──────────────────┘  └────────────────┬───────────────┘  │
└────────────────────┼──────────────────────────────────────┼──────────────────┘
                     ▼                                      │
┌────────────────────────────────────────────────────────┐  │
│             THREE-WAY NLP TONE SCORING [-1, +1]        │  │
│  ┌──────────────────────────────────────────────────┐  │  │
│  │ (A) Dictionary Lexicon + 3-Token Negation Window │  │  │
│  │ (B) Sentence-Transformer Semantic Axis Projection│  │  │
│  │ (C) Fixed-Rubric LLM Evaluation (Temperature = 0)│  │  │
│  └────────┬────────────────────────────────┬────────┘  │  │
│           ▼                                ▼           │  │
│  ┌──────────────────────┐        ┌──────────────────┐  │  │
│  │ Agreement Matrix     │        │ Diff Mode Engine │  │  │
│  │ • Pearson r          │        │ • SequenceMatcher│  │  │
│  │ • Spearman rho       │        │ • Word-level diff│  │  │
│  │ • 3-Class Cohen's κ  │        │ • Edited ΔTone   │  │  │
│  └──────────────────────┘        └─────────┬────────┘  │  │
└────────────────────────────────────────────┼───────────┘  │
                                             ▼              ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│                 ECONOMETRIC EVENT STUDY & ROBUSTNESS SUITE                   │
│  • OLS with White HC1 Robust SEs: ΔY_t = α + β₁·Surprise_t + β₂·ΔTone_t + ε_t│
│  • Window Sensitivity: 1-Day, 2-Day, and 5-Day Trading Windows               │
│  • Leave-One-Period-Out (LOPO): Drop 2021, 2022, 2023, or 2024–2025 regimes  │
│  • Placebo Test: 500 Monte Carlo random non-FOMC market window permutations  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Repository Structure

```text
fed-whisperer/
├── configs/
│   └── config.yaml                     # Scraper rate limits, FRED/yfinance tickers, embedding anchors, LOPO eras
├── desktop/
│   └── main.cjs                        # Electron native PC desktop wrapper
├── src/
│   ├── fed_whisperer/                  # Python 3.10+ Research Package
│   │   ├── __init__.py
│   │   ├── data/
│   │   │   ├── scraper.py              # Robots.txt-compliant FOMC scraper with disk caching
│   │   │   └── market.py               # FRED (fredapi) & S&P 500 (yfinance) event-window builder
│   │   ├── scoring/
│   │   │   ├── dictionary.py           # Method A: Monetary lexicon scorer with negation inversion
│   │   │   ├── embeddings.py           # Method B: Semantic projection onto Hawkish-Dovish anchor axis
│   │   │   ├── llm.py                  # Method C: Fixed-rubric LLM scorer (temperature = 0.0)
│   │   │   └── agreement.py            # Pairwise Pearson r, Spearman rho, Cohen's kappa, ΔTone sign match
│   │   └── analysis/
│   │       ├── diff.py                 # Sentence-level SequenceMatcher diff & revised-sentence ΔTone
│   │       ├── event_study.py          # Statsmodels OLS regression with HC1 standard errors & 95% CIs
│   │       └── robustness.py           # Window sensitivity (1d/2d/5d), Leave-One-Period-Out, Placebo test
│   ├── engine/                         # Interactive UI / Desktop TypeScript Analytics Engine
│   │   ├── nlp.ts                      # Real-time lexicon, semantic axis, word/sentence diff & agreement
│   │   ├── econometrics.ts             # Matrix OLS, White HC1 covariance, LOPO & Mulberry32 Monte Carlo
│   │   └── nlp.test.ts                 # Automated unit test suite
│   ├── data/
│   │   └── fomcCorpus.ts               # Historical FOMC corpus & market event dataset (2021–2025)
│   ├── components/
│   │   ├── TimelineView.tsx            # Interactive tone trajectory chart, method agreement & FOMC ledger
│   │   ├── DiffModeView.tsx            # Side-by-side & inline sentence diff viewer + live rubric scorer
│   │   ├── EventStudyView.tsx          # OLS regression tables, partial residual scatter & robustness suite
│   │   └── PythonPackageView.tsx       # Built-in code explorer for all Python & configuration files
│   ├── App.tsx                         # Main application shell
│   ├── main.tsx                        # Frontend entry point
│   └── index.css                       # Typography (Plus Jakarta Sans + JetBrains Mono tabular numerals)
├── tests/
│   └── test_fed_whisperer.py           # Pytest suite for Python package
├── app.py                              # Streamlit dashboard application
├── desktop_app.py                      # Native PC desktop window launcher (PyWebView / App-Mode)
├── server.ts                           # Express + Vite local backend server
├── requirements.txt                    # Python dependencies
├── package.json                        # Node / Desktop scripts & dependencies
└── README.md
```

---

## 3. Running as a Native PC Desktop App or Web Dashboard

You can run **Fed Whisperer** on your PC in three ways depending on your preferred workflow:

### Option A — Interactive Native PC Window (Python Launcher)
Launches the full interactive dashboard inside a dedicated desktop window (using `pywebview` if installed, or native frameless `--app` mode in Edge/Chrome):

```bash
pip install -r requirements.txt
# Optional for native OS WebView window: pip install pywebview
python desktop_app.py
```

### Option B — Electron Desktop App (Node.js)
Launches the full React + Express interactive menu in a standalone Electron desktop window:

```bash
npm install
npx electron desktop/main.cjs
```

### Option C — Local Web Server or Streamlit Dashboard
```bash
# Run the interactive TypeScript + Express dashboard on http://localhost:3000
npm install
npm run dev

# Or run the Python Streamlit dashboard on http://localhost:8501
streamlit run app.py
```

### Environment Variables (`.env`)
```ini
FRED_API_KEY="your_fred_api_key_here"
GEMINI_API_KEY="your_gemini_api_key_here"
```

---

## 4. Data Pipeline: FOMC Scraper & Market Loaders

1. **Robots.txt & Rate-Limited Scraping (`src/fed_whisperer/data/scraper.py`)**:
   - Downloads and parses `https://www.federalreserve.gov/robots.txt` using Python's standard `urllib.robotparser.RobotFileParser` before issuing any request.
   - Enforces a strict inter-request pause (`rate_limit_seconds: 1.5` in `configs/config.yaml`) via a monotonic clock throttle.
   - Extracts statement URLs matching `/newsevents/pressreleases/monetaryYYYYMMDDa.htm`, strips release time headers, voting tallies, and media inquiries boilerplate, segments sentences, and writes deterministic JSON cache files to `.cache/fed_whisperer/statements/`.
2. **Treasury Yields & Equity Prices (`src/fed_whisperer/data/market.py`)**:
   - Pulls daily 2-Year (`DGS2`) and 10-Year (`DGS10`) Constant Maturity Treasury yields plus the Federal Funds Target Range Upper Limit (`DFEDTARU`) from FRED via `fredapi`.
   - Pulls daily adjusted close prices for the S&P 500 (`SPY` / `^GSPC`) via `yfinance`.
   - Aligns each FOMC announcement date $t$ to the trading calendar and computes $1\text{-day}$, $2\text{-day}$, and $5\text{-day}$ changes relative to the prior trading close $t-1$.

---

## 5. Three Tone Scoring Methodologies

All three scoring engines map an FOMC statement (or individual sentence) onto a continuous hawkish/dovish index $S \in [-1.00, +1.00]$:

### (a) Dictionary Approach (`src/fed_whisperer/scoring/dictionary.py`)
- Matches domain-specific multi-word and single-word monetary phrases weighted by intensity:
  - **Hawkish examples**: `"ongoing increases"` ($+1.8$), `"additional policy firming"` ($+1.8$), `"unacceptably high"` ($+1.7$), `"restrictive"` ($+1.6$), `"inflationary pressures"` ($+1.5$), `"upside risks"` ($+1.4$), `"elevated"` ($+1.2$).
  - **Dovish examples**: `"lower the target range"` ($-1.9$), `"accommodative"` ($-1.6$), `"transitory"` ($-1.5$), `"downside risks"` ($-1.4$), `"progress on inflation"` ($-1.4$), `"eased"` ($-1.3$), `"slowed"` ($-1.2$).
- **Local Negation Inversion**: Inspects a 3-token window preceding each matched phrase (`not`, `no`, `never`, `less`, `without`, `hardly`). If negated (e.g., *"does not expect further increases"*), the directional weight is inverted and damped by $0.8\times$.
- **Statement Aggregation**:
  $$\text{Score}_{\text{dict}} = \frac{\sum w_{\text{hawk}} - \sum w_{\text{dove}}}{\sum w_{\text{hawk}} + \sum w_{\text{dove}} + \lambda}$$

### (b) Semantic Embedding Axis (`src/fed_whisperer/scoring/embeddings.py`)
- Encodes canonical hawkish anchor sentences $\mathcal{A}_H$ and canonical dovish anchor sentences $\mathcal{A}_D$ (configured in `configs/config.yaml`) using `sentence-transformers` (`all-MiniLM-L6-v2`).
- Constructs the unit semantic direction vector:
  $$\hat{v}_{\text{axis}} = \frac{\mu(\mathcal{A}_H) - \mu(\mathcal{A}_D)}{\|\mu(\mathcal{A}_H) - \mu(\mathcal{A}_D)\|_2}$$
- Projects each sentence embedding $e_i$ onto $\hat{v}_{\text{axis}}$ and squashes via hyperbolic tangent:
  $$s_i = \tanh\!\left(3.0 \cdot \langle e_i, \hat{v}_{\text{axis}} \rangle\right)$$

### (c) Fixed-Rubric LLM Scoring at Temperature 0 (`src/fed_whisperer/scoring/llm.py`)
- Evaluates each statement against a fixed 5-tier monetary policy rubric (`Strongly Dovish [-1.0, -0.6]`, `Moderately Dovish [-0.59, -0.2]`, `Neutral [-0.19, +0.19]`, `Moderately Hawkish [+0.2, +0.59]`, `Strongly Hawkish [+0.6, +1.0]`) at `temperature = 0.0` with structured JSON output decomposing overall tone into `inflation_stance`, `growth_labor_stance`, `forward_guidance_stance`, and a citation-backed `rationale`.

---

## 6. Inter-Method Agreement & Concordance

Across the $N = 27$ historical FOMC releases in the 2021–2025 corpus, the code computes four pairwise concordance metrics (`src/fed_whisperer/scoring/agreement.py`):

| Method Pair Comparison | Pearson Linear $r$ | Spearman Rank $\rho$ | 3-Class Cohen's $\kappa$ | $\Delta\text{Tone}$ Sign Agreement |
| :--- | :---: | :---: | :---: | :---: |
| **Dictionary (A) vs. Embedding Axis (B)** | `0.937` | `0.912` | `0.734` | `65.4%` |
| **Dictionary (A) vs. LLM Rubric (C)** | `0.876` | `0.894` | `0.729` | `73.1%` |
| **Embedding Axis (B) vs. LLM Rubric (C)** | `0.838` | `0.858` | `0.624` | `61.5%` |

*Note: All table values above are computed directly by `computeMethodAgreementMatrix` / `compute_method_agreement` on the corpus.*

---

## 7. Sentence-Level Diff Mode

Because FOMC statements repeat up to 70% of their text verbatim across consecutive meetings, raw level scores are heavily autocorrelated. **Diff Mode** (`src/fed_whisperer/analysis/diff.py`) aligns sentences between statement $t-1$ and statement $t$ using `difflib.SequenceMatcher`:

- **`unchanged`**: Exact or near-exact carry-over sentences ($\text{similarity} \ge 0.96$), assigned $\Delta s_i = 0$.
- **`modified`**: Revised sentences ($0.45 \le \text{similarity} < 0.96$), with word-level LCS token diffs highlighting deleted words and newly inserted qualifiers alongside $\Delta s_i = s_i^{(t)} - s_i^{(t-1)}$.
- **`added` / `removed`**: Newly inserted guidance sentences ($+s_i^{(t)}$) or dropped clauses ($-s_i^{(t-1)}$).

---

## 8. Econometric Event Study & Robustness Suite

### Baseline OLS Specification (`src/fed_whisperer/analysis/event_study.py`)
For each FOMC release $t \in \{1, \dots, N\}$ and event window $w \in \{1, 2, 5\}$ trading days, we estimate:

$$\Delta Y_{t,w} = \alpha + \beta_1 \, \text{RateSurprise}_t + \beta_2 \, \Delta \text{Tone}_t + \varepsilon_{t,w}$$

using Ordinary Least Squares with **White's `HC1` heteroskedasticity-consistent standard errors**.

### Three-Part Robustness Suite (`src/fed_whisperer/analysis/robustness.py`)
1. **Window Length Sensitivity ($1\text{d}, 2\text{d}, 5\text{d}$)**: Tests whether statement tone effects concentrate in the immediate 1-day post-release window or drift/dilute over 2-day and 5-day horizons as subsequent macroeconomic data releases arrive.
2. **Leave-One-Period-Out (`LOPO`)**: Re-estimates the regression four times, sequentially dropping:
   - *2021 Transitory Inflation Era*
   - *2022 Rapid Tightening Cycle*
   - *2023 High-for-Longer Peak*
   - *2024–2025 Calibration & Easing*
3. **Random-Date Placebo Test ($500$ Monte Carlo Permutations)**: Pairs the observed sequence of FOMC tone changes ($\Delta \text{Tone}_t$) with randomly sampled market windows to generate an empirical null distribution of $\hat{\beta}_2$ and compute a two-sided permutation $p$-value.

---

## 9. Empirical Results Summary

Generated directly from the $N = 26$ sequential FOMC release transitions ($\text{dof} = 23$) using Method C ($\Delta\text{Tone}_{\text{LLM}}$) and `HC1` robust standard errors:

| Target Market Outcome | Window | Rate Surprise $\hat{\beta}_1$ ($p$-val) | Tone Shift $\hat{\beta}_2$ | $\hat{\beta}_2$ 95% CI | $\hat{\beta}_2$ $p$-val | $R^2$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **2Y Treasury Yield (`DGS2`, bps)** | 1-Day | `+1.378` (`0.0012`) | `+5.476` | `[-8.04, +18.99]` | `0.4126` | `0.591` |
| **2Y Treasury Yield (`DGS2`, bps)** | 2-Day | `+2.104` (`0.0005`) | `+8.523` | `[-18.29, +35.33]` | `0.5191` | `0.486` |
| **2Y Treasury Yield (`DGS2`, bps)** | 5-Day | `+1.741` (`0.0256`) | `+17.258` | `[-18.52, +53.03]` | `0.3308` | `0.243` |

**Key Empirical Finding**: Once the short-end **Rate Surprise Proxy ($\beta_1$)** is controlled for ($t = +3.73, p = 0.0012$), the point estimate on **Statement Tone Shift ($\beta_2 = +5.48\text{ bps per unit }\Delta\text{Tone}$)** has the expected positive sign (a hawkish statement shift is associated with higher 2-year yields), but its 95% confidence interval (`[-8.04, +18.99]`) spans zero ($p = 0.4126$).

---

## 10. Honest Econometric Limitations & Identification Caveats

1. **Small Sample Size ($N = 26$ transitions, $\text{dof} = 23$)**:
   The FOMC holds only 8 scheduled meetings per year. Across a 3.5-year cycle, statistical power to detect secondary textual effects orthogonal to the rate surprise is inherently low, producing wide confidence intervals.
2. **Simultaneous Confounders at 2:00 PM & 2:30 PM ET**:
   At every March, June, September, and December meeting, the FOMC releases its **Summary of Economic Projections (SEP / Dot Plot)** at the exact same timestamp (2:00 PM ET) as the statement, followed 30 minutes later (2:30 PM ET) by the Chair's live press conference. Daily FRED yield changes (`DGS2`, `DGS10`) measure close-to-close moves and therefore bundle the statement, the dot plot, and the press conference Q&A together. For instance, on June 15, 2022, the written statement was strongly hawkish (+75 bps hike), yet 2-year yields fell 15 bps by market close following the press conference.
3. **Inter-Meeting Forward Guidance & Anticipation**:
   When FOMC participants pre-announce policy shifts in speeches prior to the blackout window, bond markets price the expected wording change ahead of the release date.
4. **No Claim of Causality**:
   All regression outputs in this repository measure conditional in-sample associations and must not be interpreted as structural causal effects.

---

## 11. Testing & Reproducibility

Run the automated test suites to verify all scoring, diffing, and econometric calculations:

```bash
# Run TypeScript NLP & Econometrics unit tests (Node built-in test runner)
npm test

# Run Python pytest suite
pytest -v
```
