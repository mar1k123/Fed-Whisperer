"""
Streamlit & Desktop Dashboard for Fed Whisperer:
FOMC Tone Timeline, Sentence Diff Viewer, OLS Event Study, and Robustness Suite.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.fed_whisperer.analysis.diff import compute_statement_diff
from src.fed_whisperer.analysis.event_study import run_event_study_regression
from src.fed_whisperer.analysis.robustness import (
    run_leave_one_period_out,
    run_placebo_test,
    run_window_robustness,
)
from src.fed_whisperer.data.scraper import split_sentences
from src.fed_whisperer.scoring.agreement import compute_method_agreement
from src.fed_whisperer.scoring.dictionary import DictionaryToneScorer
from src.fed_whisperer.scoring.embeddings import EmbeddingAxisScorer

st.set_page_config(
    page_title="Fed Whisperer — FOMC NLP Tone & Market Event Study",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom desktop-grade styling matching the React interface
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
        background-color: #f8fafc;
        color: #0f172a;
    }
    code, pre, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
        font-variant-numeric: tabular-nums;
    }
    .block-container {
        padding-top: 2rem;
        max-width: 1360px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

EMBEDDED_RELEASES: List[Dict[str, object]] = [
    {
        "date": "2021-04-28",
        "regime": "2021 Transitory Era",
        "target": "0.00%–0.25%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": -1.5,
        "dgs2_chg_1d_bps": -2.0,
        "dgs2_chg_2d_bps": -2.5,
        "dgs2_chg_5d_bps": -3.0,
        "dgs10_chg_1d_bps": -2.0,
        "dgs10_chg_2d_bps": 1.0,
        "dgs10_chg_5d_bps": -4.0,
        "sp500_ret_1d_pct": -0.08,
        "sp500_ret_2d_pct": 0.60,
        "sp500_ret_5d_pct": -0.45,
        "llm_score": -0.68,
        "text": (
            "Amid progress on vaccinations and strong policy support, indicators of economic activity and employment "
            "have strengthened. Inflation has risen, largely reflecting transitory factors. Overall financial conditions "
            "remain accommodative, in part reflecting policy measures to support the economy. The Committee decided to "
            "keep the target range for the federal funds rate at 0 to 1/4 percent."
        ),
    },
    {
        "date": "2021-09-22",
        "regime": "2021 Transitory Era",
        "target": "0.00%–0.25%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": 2.0,
        "dgs2_chg_1d_bps": 3.0,
        "dgs2_chg_2d_bps": 5.0,
        "dgs2_chg_5d_bps": 8.0,
        "dgs10_chg_1d_bps": -1.0,
        "dgs10_chg_2d_bps": 9.0,
        "dgs10_chg_5d_bps": 20.0,
        "sp500_ret_1d_pct": 0.95,
        "sp500_ret_2d_pct": 2.17,
        "sp500_ret_5d_pct": 0.12,
        "llm_score": -0.22,
        "text": (
            "Indicators of economic activity and employment have continued to strengthen. Inflation is elevated, "
            "largely reflecting transitory factors. If progress continues broadly as expected, the Committee judges "
            "that a moderation in the pace of asset purchases may soon be warranted."
        ),
    },
    {
        "date": "2021-12-15",
        "regime": "2021 Transitory Era",
        "target": "0.00%–0.25%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": 4.5,
        "dgs2_chg_1d_bps": 3.0,
        "dgs2_chg_2d_bps": -2.0,
        "dgs2_chg_5d_bps": 3.0,
        "dgs10_chg_1d_bps": 3.0,
        "dgs10_chg_2d_bps": -1.0,
        "dgs10_chg_5d_bps": 2.0,
        "sp500_ret_1d_pct": 1.63,
        "sp500_ret_2d_pct": 0.75,
        "sp500_ret_5d_pct": 1.02,
        "llm_score": 0.34,
        "text": (
            "Job gains have been solid in recent months, and the unemployment rate has declined substantially. "
            "Supply and demand imbalances related to the pandemic and the reopening of the economy have continued "
            "to contribute to elevated levels of inflation. The Committee decided to reduce the monthly pace of its "
            "net asset purchases at a faster pace."
        ),
    },
    {
        "date": "2022-03-16",
        "regime": "2022 Aggressive Hiking",
        "target": "0.25%–0.50%",
        "rate_chg_bps": 25,
        "rate_surprise_bps": 4.0,
        "dgs2_chg_1d_bps": 9.0,
        "dgs2_chg_2d_bps": 9.0,
        "dgs2_chg_5d_bps": 26.0,
        "dgs10_chg_1d_bps": 4.0,
        "dgs10_chg_2d_bps": 5.0,
        "dgs10_chg_5d_bps": 18.0,
        "sp500_ret_1d_pct": 2.24,
        "sp500_ret_2d_pct": 3.50,
        "sp500_ret_5d_pct": 4.56,
        "llm_score": 0.76,
        "text": (
            "Job gains have been strong in recent months, and the unemployment rate has declined substantially. "
            "Inflation remains elevated, reflecting supply and demand imbalances, higher energy prices, and broader "
            "price pressures. The Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 "
            "percent and anticipates that ongoing increases in the target range will be appropriate."
        ),
    },
    {
        "date": "2022-06-15",
        "regime": "2022 Aggressive Hiking",
        "target": "1.50%–1.75%",
        "rate_chg_bps": 75,
        "rate_surprise_bps": 6.0,
        "dgs2_chg_1d_bps": -15.0,
        "dgs2_chg_2d_bps": -27.0,
        "dgs2_chg_5d_bps": -37.0,
        "dgs10_chg_1d_bps": -15.0,
        "dgs10_chg_2d_bps": -18.0,
        "dgs10_chg_5d_bps": -32.0,
        "sp500_ret_1d_pct": 1.46,
        "sp500_ret_2d_pct": -1.84,
        "sp500_ret_5d_pct": 0.65,
        "llm_score": 0.91,
        "text": (
            "Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains "
            "elevated, reflecting supply and demand imbalances, higher food and energy prices, and broader price pressures. "
            "The Committee is highly attentive to inflation risks and is strongly committed to returning inflation to its "
            "2 percent objective. The Committee decided to raise the target range and anticipates that ongoing increases "
            "in the target range will be appropriate."
        ),
    },
    {
        "date": "2022-11-02",
        "regime": "2022 Aggressive Hiking",
        "target": "3.75%–4.00%",
        "rate_chg_bps": 75,
        "rate_surprise_bps": 5.0,
        "dgs2_chg_1d_bps": 7.0,
        "dgs2_chg_2d_bps": 17.0,
        "dgs2_chg_5d_bps": 9.0,
        "dgs10_chg_1d_bps": 5.0,
        "dgs10_chg_2d_bps": 7.0,
        "dgs10_chg_5d_bps": 7.0,
        "sp500_ret_1d_pct": -2.50,
        "sp500_ret_2d_pct": -3.53,
        "sp500_ret_5d_pct": -2.01,
        "llm_score": 0.62,
        "text": (
            "Recent indicators point to modest growth in spending and production. Job gains have been robust in recent "
            "months. Inflation remains elevated. The Committee anticipates that ongoing increases in the target range "
            "will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive to "
            "return inflation to 2 percent over time."
        ),
    },
    {
        "date": "2023-03-22",
        "regime": "2023 Peak Restrictive",
        "target": "4.75%–5.00%",
        "rate_chg_bps": 25,
        "rate_surprise_bps": -8.0,
        "dgs2_chg_1d_bps": -21.0,
        "dgs2_chg_2d_bps": -37.0,
        "dgs2_chg_5d_bps": -9.0,
        "dgs10_chg_1d_bps": -12.0,
        "dgs10_chg_2d_bps": -22.0,
        "dgs10_chg_5d_bps": -3.0,
        "sp500_ret_1d_pct": -1.65,
        "sp500_ret_2d_pct": -1.36,
        "sp500_ret_5d_pct": 0.61,
        "llm_score": 0.18,
        "text": (
            "Recent developments are likely to result in tighter credit conditions for households and businesses and to "
            "weigh on economic activity, hiring, and inflation. The Committee anticipates that some additional policy "
            "firming may be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive."
        ),
    },
    {
        "date": "2023-09-20",
        "regime": "2023 Peak Restrictive",
        "target": "5.25%–5.50%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": 5.5,
        "dgs2_chg_1d_bps": 6.0,
        "dgs2_chg_2d_bps": 5.0,
        "dgs2_chg_5d_bps": 4.0,
        "dgs10_chg_1d_bps": -2.0,
        "dgs10_chg_2d_bps": 12.0,
        "dgs10_chg_5d_bps": 24.0,
        "sp500_ret_1d_pct": -0.94,
        "sp500_ret_2d_pct": -2.56,
        "sp500_ret_5d_pct": -3.79,
        "llm_score": 0.28,
        "text": (
            "Recent indicators suggest that economic activity has been expanding at a solid pace. Job gains have slowed "
            "in recent months but remain strong. Inflation remains elevated. The Committee decided to maintain the "
            "target range at 5-1/4 to 5-1/2 percent and will continue to assess the extent of additional policy firming."
        ),
    },
    {
        "date": "2023-12-13",
        "regime": "2023 Peak Restrictive",
        "target": "5.25%–5.50%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": -9.5,
        "dgs2_chg_1d_bps": -27.0,
        "dgs2_chg_2d_bps": -34.0,
        "dgs2_chg_5d_bps": -39.0,
        "dgs10_chg_1d_bps": -17.0,
        "dgs10_chg_2d_bps": -28.0,
        "dgs10_chg_5d_bps": -35.0,
        "sp500_ret_1d_pct": 1.37,
        "sp500_ret_2d_pct": 1.63,
        "sp500_ret_5d_pct": 1.20,
        "llm_score": -0.14,
        "text": (
            "Recent indicators suggest that growth of economic activity has slowed from its strong pace in the third "
            "quarter. Job gains have moderated since earlier in the year, and inflation has eased over the past year "
            "but remains elevated. The Committee decided to maintain the target range at 5-1/4 to 5-1/2 percent."
        ),
    },
    {
        "date": "2024-07-31",
        "regime": "2024-2025 Calibration & Easing",
        "target": "5.25%–5.50%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": -4.5,
        "dgs2_chg_1d_bps": -10.0,
        "dgs2_chg_2d_bps": -20.0,
        "dgs2_chg_5d_bps": -36.0,
        "dgs10_chg_1d_bps": -10.0,
        "dgs10_chg_2d_bps": -17.0,
        "dgs10_chg_5d_bps": -25.0,
        "sp500_ret_1d_pct": 1.58,
        "sp500_ret_2d_pct": 0.19,
        "sp500_ret_5d_pct": -4.25,
        "llm_score": -0.26,
        "text": (
            "Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have "
            "moderated, and the unemployment rate has moved up but remains low. Inflation has eased over the past year "
            "but remains somewhat elevated. In recent months, there has been some further progress toward the "
            "Committee's 2 percent inflation objective."
        ),
    },
    {
        "date": "2024-09-18",
        "regime": "2024-2025 Calibration & Easing",
        "target": "4.75%–5.00%",
        "rate_chg_bps": -50,
        "rate_surprise_bps": -6.0,
        "dgs2_chg_1d_bps": 2.0,
        "dgs2_chg_2d_bps": -1.0,
        "dgs2_chg_5d_bps": -6.0,
        "dgs10_chg_1d_bps": 6.0,
        "dgs10_chg_2d_bps": 9.0,
        "dgs10_chg_5d_bps": 14.0,
        "sp500_ret_1d_pct": -0.29,
        "sp500_ret_2d_pct": 1.41,
        "sp500_ret_5d_pct": 1.85,
        "llm_score": -0.56,
        "text": (
            "Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have "
            "slowed, and the unemployment rate has moved up but remains low. Inflation has made further progress toward "
            "the Committee's 2 percent objective but remains somewhat elevated. In light of the progress on inflation "
            "and the balance of risks, the Committee decided to lower the target range for the federal funds rate by "
            "1/2 percentage point to 4-3/4 to 5 percent."
        ),
    },
    {
        "date": "2025-01-29",
        "regime": "2024-2025 Calibration & Easing",
        "target": "4.25%–4.50%",
        "rate_chg_bps": 0,
        "rate_surprise_bps": 2.5,
        "dgs2_chg_1d_bps": 3.0,
        "dgs2_chg_2d_bps": 1.0,
        "dgs2_chg_5d_bps": -2.0,
        "dgs10_chg_1d_bps": 1.0,
        "dgs10_chg_2d_bps": -1.0,
        "dgs10_chg_5d_bps": -11.0,
        "sp500_ret_1d_pct": -0.47,
        "sp500_ret_2d_pct": 0.05,
        "sp500_ret_5d_pct": 0.36,
        "llm_score": 0.14,
        "text": (
            "Recent indicators suggest that economic activity has continued to expand at a solid pace. The unemployment "
            "rate has stabilized at a low level in recent months, and labor market conditions remain solid. Inflation "
            "remains somewhat elevated. The Committee decided to maintain the target range for the federal funds rate "
            "at 4-1/4 to 4-1/2 percent."
        ),
    },
]


@st.cache_resource
def build_corpus_dataframe() -> tuple[pd.DataFrame, DictionaryToneScorer, EmbeddingAxisScorer]:
    dict_scorer = DictionaryToneScorer()
    embed_scorer = EmbeddingAxisScorer(use_transformer=False)

    rows = []
    for item in EMBEDDED_RELEASES:
        txt = str(item["text"])
        d_res = dict_scorer.score_text(txt)
        e_res = embed_scorer.score_text(txt)
        rows.append(
            {
                **item,
                "dict_score": d_res.score,
                "embed_score": e_res.score,
            }
        )

    df = pd.DataFrame(rows)
    df["dict_change"] = df["dict_score"].diff().fillna(0.0)
    df["embed_change"] = df["embed_score"].diff().fillna(0.0)
    df["llm_change"] = df["llm_score"].diff().fillna(0.0)
    return df, dict_scorer, embed_scorer


def main() -> None:
    df, dict_scorer, embed_scorer = build_corpus_dataframe()

    config_path = Path("configs/config.yaml")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}

    st.markdown("### Fed Whisperer")
    st.caption(
        "FOMC Monetary Policy NLP Tone Scorer (Dictionary · Semantic Embedding Axis · Fixed-Rubric LLM) "
        "& Treasury/Equity Event Study"
    )

    tab_timeline, tab_diff, tab_event, tab_robust = st.tabs(
        [
            "Tone Timeline & Agreement",
            "Statement Diff Mode",
            "Event Study (OLS HC1)",
            "Robustness & Placebo Suite",
        ]
    )

    with tab_timeline:
        st.subheader("01. Multi-Method Hawkish / Dovish Tone Trajectory")
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["dict_score"],
                mode="lines+markers",
                name="Method A: Dictionary Lexicon",
                line=dict(color="#0f172a", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["embed_score"],
                mode="lines+markers",
                name="Method B: Embedding Axis",
                line=dict(color="#1d4ed8", width=2.5),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df["llm_score"],
                mode="lines+markers",
                name="Method C: Fixed-Rubric LLM (T=0)",
                line=dict(color="#b45309", width=2.5),
            )
        )
        fig.add_hline(y=0.0, line_dash="dash", line_color="#94a3b8")
        fig.update_layout(
            height=380,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title="Tone Index [-1 Dovish, +1 Hawkish]", range=[-1.05, 1.05]),
            template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Inter-Method Agreement Matrix")
        agreement = compute_method_agreement(df)
        ag_rows = [
            {"Method Pair": k.replace("_", " "), **v} for k, v in agreement.items()
        ]
        st.dataframe(pd.DataFrame(ag_rows), use_container_width=True, hide_index=True)

        st.markdown("#### Cached FOMC Statement Ledger")
        st.dataframe(
            df[
                [
                    "date",
                    "regime",
                    "target",
                    "rate_chg_bps",
                    "rate_surprise_bps",
                    "dict_score",
                    "embed_score",
                    "llm_score",
                    "dgs2_chg_1d_bps",
                    "dgs10_chg_1d_bps",
                    "sp500_ret_1d_pct",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

    with tab_diff:
        st.subheader("02. Sentence-Level Statement Diff Mode")
        c1, c2 = st.columns(2)
        dates = df["date"].tolist()
        with c1:
            prev_date = st.selectbox("Baseline Release (t-1)", dates, index=max(0, len(dates) - 3))
        with c2:
            curr_date = st.selectbox("Comparison Release (t)", dates, index=max(0, len(dates) - 2))

        prev_row = df.loc[df["date"] == prev_date].iloc[0]
        curr_row = df.loc[df["date"] == curr_date].iloc[0]

        report = compute_statement_diff(
            prev_date=str(prev_date),
            prev_sentences=split_sentences(str(prev_row["text"])),
            curr_date=str(curr_date),
            curr_sentences=split_sentences(str(curr_row["text"])),
            sentence_scorer=lambda s: float(dict_scorer.score_sentence(s)["score"]),
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric(f"Baseline ({prev_date})", f"{report.overall_prev_score:+.3f}")
        k2.metric(f"Release ({curr_date})", f"{report.overall_curr_score:+.3f}")
        k3.metric("Full Statement ΔTone", f"{report.net_tone_delta:+.3f}")
        k4.metric("Edited Sentences ΔTone", f"{report.changed_sentences_delta:+.3f}")

        for item in report.items:
            badge = item.tag.upper()
            st.markdown(
                f"- **{badge}** (`ΔTone = {item.delta_score:+.3f}`, similarity `{item.similarity:.0%}`): "
                f"{item.curr_sentence or item.prev_sentence}"
            )

    with tab_event:
        st.subheader("03. Event Study OLS Regressions (White HC1 Robust SE)")
        reg_df = df.iloc[1:].copy()
        col_a, col_b = st.columns(2)
        with col_a:
            dep_choice = st.selectbox(
                "Target Market Variable",
                ["dgs2_chg_1d_bps", "dgs10_chg_1d_bps", "sp500_ret_1d_pct"],
            )
        with col_b:
            tone_choice = st.selectbox(
                "Tone Delta Regressor",
                ["llm_change", "dict_change", "embed_change"],
            )

        reg_df["tone_change"] = reg_df[tone_choice]
        res = run_event_study_regression(
            reg_df,
            dependent_col=dep_choice,
            rate_surprise_col="rate_surprise_bps",
            tone_change_col="tone_change",
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Observations (N)", str(res.n_obs))
        m2.metric("R-Squared", f"{res.r_squared:.3f}")
        m3.metric("Adj. R-Squared", f"{res.adj_r_squared:.3f}")
        m4.metric("F-Statistic", f"{res.f_stat:.2f}")

        coeff_rows = [
            {
                "Term": k,
                "Estimate (β)": v.estimate,
                "Robust SE (HC1)": v.std_err,
                "t-Stat": v.t_stat,
                "p-Value": v.p_value,
                "95% CI Lower": v.ci_lower,
                "95% CI Upper": v.ci_upper,
            }
            for k, v in res.coefficients.items()
        ]
        st.dataframe(pd.DataFrame(coeff_rows), use_container_width=True, hide_index=True)
        st.caption(res.small_sample_note)

    with tab_robust:
        st.subheader("04. Window Sensitivity, Leave-One-Period-Out & Placebo Dates")
        reg_df = df.iloc[1:].copy()
        reg_df["tone_change"] = reg_df["llm_change"]

        win_res = run_window_robustness(reg_df, asset_prefix="dgs2_chg", asset_suffix="d_bps")
        win_rows = [
            {
                "Window (Days)": w,
                "ΔTone β₂": r.coefficients["tone_change"].estimate,
                "Robust SE": r.coefficients["tone_change"].std_err,
                "p-Value": r.coefficients["tone_change"].p_value,
                "95% CI": f"[{r.coefficients['tone_change'].ci_lower:+.2f}, {r.coefficients['tone_change'].ci_upper:+.2f}]",
                "R²": r.r_squared,
            }
            for w, r in win_res.items()
        ]
        st.markdown("#### Window Length Sensitivity (1d vs. 2d vs. 5d DGS2 Yield Change)")
        st.dataframe(pd.DataFrame(win_rows), use_container_width=True, hide_index=True)

        regimes = config.get("robustness", {}).get(
            "leave_one_out_regimes",
            [
                {"name": "2021 Transitory Era", "start": "2021-01-01", "end": "2021-12-31"},
                {"name": "2022 Aggressive Hiking", "start": "2022-01-01", "end": "2022-12-31"},
                {"name": "2023 Peak Restrictive", "start": "2023-01-01", "end": "2023-12-31"},
                {"name": "2024-2025 Calibration & Easing", "start": "2024-01-01", "end": "2025-12-31"},
            ],
        )
        lopo = run_leave_one_period_out(reg_df, dependent_col="dgs2_chg_1d_bps", regimes=regimes)
        st.markdown("#### Leave-One-Period-Out (LOPO) Regime Stability")
        st.dataframe(pd.DataFrame([vars(x) for x in lopo]), use_container_width=True, hide_index=True)

        placebo = run_placebo_test(reg_df, dependent_col="dgs2_chg_1d_bps", n_simulations=500, seed=42)
        st.markdown(
            f"#### Random-Date Placebo Simulation (500 Iterations) — "
            f"Actual β₂ = `{placebo.actual_beta:+.3f}` vs. Placebo Mean = `{placebo.placebo_mean_beta:+.3f}` "
            f"(Empirical p = `{placebo.placebo_p_value_two_sided:.3f}`)"
        )


if __name__ == "__main__":
    main()
