"""Robustness checks: window sensitivity, leave-one-period-out (LOPO), and random-date placebo tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd

from src.fed_whisperer.analysis.event_study import EventStudyResult, run_event_study_regression


@dataclass
class LeaveOnePeriodOutItem:
    """Tone beta sensitivity when omitting a specific macroeconomic regime."""

    omitted_regime: str
    n_obs: int
    tone_beta: float
    ci_lower: float
    ci_upper: float
    p_value: float
    r_squared: float


@dataclass
class PlaceboTestResult:
    """Monte Carlo placebo test assigning actual tone deltas to random non-FOMC trading dates."""

    actual_beta: float
    actual_t_stat: float
    placebo_mean_beta: float
    placebo_std_beta: float
    placebo_p_value_two_sided: float
    placebo_betas: List[float]


def run_window_robustness(
    df: pd.DataFrame,
    asset_prefix: str = "dgs2_chg",
    asset_suffix: str = "d_bps",
    windows: List[int] = [1, 2, 5],
    rate_surprise_col: str = "rate_surprise_bps",
    tone_change_col: str = "tone_change",
) -> Dict[int, EventStudyResult]:
    """Compare event study regression estimates across 1-day, 2-day, and 5-day windows."""
    results: Dict[int, EventStudyResult] = {}
    for w in windows:
        col = f"{asset_prefix}_{w}{asset_suffix}"
        if col in df.columns:
            results[w] = run_event_study_regression(
                df,
                dependent_col=col,
                rate_surprise_col=rate_surprise_col,
                tone_change_col=tone_change_col,
            )
    return results


def run_leave_one_period_out(
    df: pd.DataFrame,
    dependent_col: str,
    regimes: List[Dict[str, str]],
    rate_surprise_col: str = "rate_surprise_bps",
    tone_change_col: str = "tone_change",
) -> List[LeaveOnePeriodOutItem]:
    """Re-estimate the event study dropping one macroeconomic year/regime at a time."""
    outputs: List[LeaveOnePeriodOutItem] = []
    dates = pd.to_datetime(df["date"])

    for reg in regimes:
        start = pd.to_datetime(reg["start"])
        end = pd.to_datetime(reg["end"])
        mask = (dates < start) | (dates > end)
        sub = df.loc[mask].copy()
        if len(sub) < 6:
            continue

        res = run_event_study_regression(
            sub,
            dependent_col=dependent_col,
            rate_surprise_col=rate_surprise_col,
            tone_change_col=tone_change_col,
        )
        tc = res.coefficients[tone_change_col]
        outputs.append(
            LeaveOnePeriodOutItem(
                omitted_regime=reg["name"],
                n_obs=res.n_obs,
                tone_beta=tc.estimate,
                ci_lower=tc.ci_lower,
                ci_upper=tc.ci_upper,
                p_value=tc.p_value,
                r_squared=res.r_squared,
            )
        )
    return outputs


def run_placebo_test(
    df: pd.DataFrame,
    dependent_col: str,
    rate_surprise_col: str = "rate_surprise_bps",
    tone_change_col: str = "tone_change",
    n_simulations: int = 500,
    seed: int = 42,
    non_fomc_market_changes: np.ndarray | None = None,
) -> PlaceboTestResult:
    """
    Placebo test: Pair the observed sequence of FOMC tone changes with random non-FOMC market
    windows (or random permutations of market reactions) to construct a null distribution
    of beta_2 under no true announcement alignment.
    """
    actual_res = run_event_study_regression(
        df,
        dependent_col=dependent_col,
        rate_surprise_col=rate_surprise_col,
        tone_change_col=tone_change_col,
    )
    actual_beta = actual_res.coefficients[tone_change_col].estimate
    actual_t = actual_res.coefficients[tone_change_col].t_stat

    rng = np.random.default_rng(seed)
    clean = df[[dependent_col, rate_surprise_col, tone_change_col]].dropna()
    n_obs = len(clean)

    tone_vals = clean[tone_change_col].to_numpy(dtype=float)
    surprise_vals = clean[rate_surprise_col].to_numpy(dtype=float)

    if non_fomc_market_changes is not None and len(non_fomc_market_changes) >= n_obs:
        pool = np.asarray(non_fomc_market_changes, dtype=float)
    else:
        pool = clean[dependent_col].to_numpy(dtype=float)

    placebo_betas: List[float] = []
    X = np.column_stack([np.ones(n_obs), surprise_vals, tone_vals])
    xtx_inv_xt = np.linalg.pinv(X)

    for _ in range(n_simulations):
        replace_flag = len(pool) > n_obs
        sampled_y = rng.choice(pool, size=n_obs, replace=replace_flag)
        # Shuffle rate surprise too so pseudo-dates have independent background moves
        params = xtx_inv_xt @ sampled_y
        placebo_betas.append(round(float(params[2]), 4))

    arr = np.asarray(placebo_betas, dtype=float)
    exceed = np.mean(np.abs(arr) >= abs(actual_beta))

    return PlaceboTestResult(
        actual_beta=actual_beta,
        actual_t_stat=actual_t,
        placebo_mean_beta=round(float(np.mean(arr)), 4),
        placebo_std_beta=round(float(np.std(arr, ddof=1)), 4),
        placebo_p_value_two_sided=round(float(exceed), 4),
        placebo_betas=placebo_betas,
    )
