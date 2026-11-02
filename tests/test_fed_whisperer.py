"""Pytest suite for Fed Whisperer tone scoring, diff mode, and econometric event study."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.fed_whisperer.analysis.diff import compute_statement_diff
from src.fed_whisperer.analysis.event_study import run_event_study_regression
from src.fed_whisperer.analysis.robustness import run_placebo_test
from src.fed_whisperer.data.scraper import split_sentences
from src.fed_whisperer.scoring.agreement import compute_method_agreement
from src.fed_whisperer.scoring.dictionary import DictionaryToneScorer
from src.fed_whisperer.scoring.embeddings import EmbeddingAxisScorer


def test_dictionary_hawkish_vs_dovish() -> None:
    scorer = DictionaryToneScorer()
    hawk_text = (
        "Inflation remains elevated and upside risks to prices warrant ongoing increases "
        "in the target range and additional policy firming."
    )
    dove_text = (
        "Economic activity has slowed and downside risks warrant a more accommodative stance "
        "and lowering the target range."
    )
    h_res = scorer.score_text(hawk_text)
    d_res = scorer.score_text(dove_text)
    assert h_res.score > 0.35
    assert d_res.score < -0.35


def test_embedding_axis_direction() -> None:
    scorer = EmbeddingAxisScorer(use_transformer=False)
    h_score = scorer.score_text(
        "Tight labor market and persistent inflationary pressures require restrictive policy firming."
    ).score
    d_score = scorer.score_text(
        "Inflation has eased substantially, allowing the Committee to lower the target rate."
    ).score
    assert h_score > d_score


def test_sentence_diff_mode() -> None:
    scorer = DictionaryToneScorer()
    prev = split_sentences(
        "Economic activity expanded at a solid pace. "
        "The Committee anticipates that ongoing increases in the target range will be appropriate."
    )
    curr = split_sentences(
        "Economic activity expanded at a solid pace. "
        "The Committee decided to lower the target range for the federal funds rate as inflation has eased."
    )
    report = compute_statement_diff(
        prev_date="2023-01-01",
        prev_sentences=prev,
        curr_date="2024-09-18",
        curr_sentences=curr,
        sentence_scorer=lambda s: float(scorer.score_sentence(s)["score"]),
    )
    assert report.net_tone_delta < -0.2
    assert any(item.tag in ("modified", "added", "removed") for item in report.items)


def test_event_study_and_placebo() -> None:
    rng = np.random.default_rng(123)
    n = 24
    tone_chg = rng.normal(0, 0.25, size=n)
    surprise = rng.normal(0, 4.0, size=n)
    dgs2 = 1.5 + 0.85 * surprise + 11.0 * tone_chg + rng.normal(0, 2.0, size=n)

    df = pd.DataFrame(
        {
            "date": pd.date_range("2022-01-01", periods=n, freq="45D").strftime("%Y-%m-%d"),
            "rate_surprise_bps": surprise,
            "tone_change": tone_chg,
            "dgs2_chg_1d_bps": dgs2,
        }
    )

    res = run_event_study_regression(df, dependent_col="dgs2_chg_1d_bps")
    assert res.n_obs == n
    assert res.coefficients["tone_change"].estimate > 0
    assert "Small-sample warning" in res.small_sample_note

    placebo = run_placebo_test(df, dependent_col="dgs2_chg_1d_bps", n_simulations=100, seed=42)
    assert len(placebo.placebo_betas) == 100
    assert 0.0 <= placebo.placebo_p_value_two_sided <= 1.0


def test_method_agreement_metrics() -> None:
    df = pd.DataFrame(
        {
            "dict_score": [-0.5, -0.3, 0.0, 0.4, 0.7],
            "embed_score": [-0.4, -0.25, 0.05, 0.35, 0.65],
            "llm_score": [-0.6, -0.2, 0.0, 0.5, 0.8],
        }
    )
    ag = compute_method_agreement(df)
    assert ag["dict_score_vs_embed_score"]["pearson_r"] > 0.9
