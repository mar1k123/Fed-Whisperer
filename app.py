"""Streamlit dashboard for Fed Whisperer: FOMC Tone Timeline, Diff Viewer, and Event Study."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
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
    page_title="Fed Whisperer | FOMC Tone & Event Study",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_scorers():
    return DictionaryToneScorer(), EmbeddingAxisScorer(use_transformer=False)


def main() -> None:
    st.title("Fed Whisperer: FOMC Statement Tone & Market Event Study")
    st.caption(
        "Empirical NLP comparison of Dictionary, Semantic Embedding Axis, and Fixed-Rubric LLM "
        "scoring on Federal Open Market Committee statements."
    )

    config_path = Path("configs/config.yaml")
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    else:
        config = {}

    dict_scorer, embed_scorer = load_scorers()

    tab_timeline, tab_diff, tab_event, tab_robust = st.tabs(
        [
            "1. Tone Timeline & Agreement",
            "2. Statement Diff Viewer",
            "3. Event Study Regressions",
            "4. Robustness & Placebo Checks",
        ]
    )

    with tab_timeline:
        st.subheader("Historical Hawkish/Dovish Index & Inter-Method Agreement")
        st.info(
            "Scores span [-1.00, +1.00] where positive values denote hawkish policy posture "
            "and negative values denote dovish accommodation."
        )

    with tab_diff:
        st.subheader("Consecutive FOMC Statement Diff Mode")
        col1, col2 = st.columns(2)
        with col1:
            prev_txt = st.text_area(
                "Previous Statement (t-1)",
                value=(
                    "Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic. "
                    "The Committee anticipates that ongoing increases in the target range will be appropriate."
                ),
                height=160,
            )
        with col2:
            curr_txt = st.text_area(
                "Current Statement (t)",
                value=(
                    "Inflation has eased somewhat but remains elevated. "
                    "In determining the extent of future increases in the target range, the Committee will take into account cumulative tightening."
                ),
                height=160,
            )

        diff_report = compute_statement_diff(
            prev_date="t-1",
            prev_sentences=split_sentences(prev_txt),
            curr_date="t",
            curr_sentences=split_sentences(curr_txt),
            sentence_scorer=lambda s: float(dict_scorer.score_sentence(s)["score"]),
        )
        m1, m2, m3 = st.columns(3)
        m1.metric("Previous Tone (t-1)", f"{diff_report.overall_prev_score:+.3f}")
        m2.metric("Current Tone (t)", f"{diff_report.overall_curr_score:+.3f}")
        m3.metric("Net Tone Delta", f"{diff_report.net_tone_delta:+.3f}")

        for item in diff_report.items:
            st.markdown(
                f"**[{item.tag.upper()}]** (Δ = `{item.delta_score:+.3f}`) — "
                f"{item.curr_sentence or item.prev_sentence}"
            )


if __name__ == "__main__":
    main()
