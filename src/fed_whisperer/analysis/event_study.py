"""Event study regressions testing whether FOMC tone changes predict market reactions beyond rate surprises."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import statsmodels.api as sm


@dataclass
class RegressionCoefficient:
    """Single regressor estimate with confidence interval and test statistics."""

    name: str
    estimate: float
    std_err: float
    t_stat: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class EventStudyResult:
    """OLS event study result for a target market outcome and window."""

    dependent_var: str
    tone_col: str
    n_obs: int
    r_squared: float
    adj_r_squared: float
    f_stat: float
    f_pvalue: float
    coefficients: Dict[str, RegressionCoefficient]
    small_sample_note: str


def run_event_study_regression(
    df: pd.DataFrame,
    dependent_col: str,
    rate_surprise_col: str = "rate_surprise_bps",
    tone_change_col: str = "tone_change",
    alpha: float = 0.05,
    hc_type: str = "HC1",
) -> EventStudyResult:
    """
    Estimate the event study specification:
        Delta_Market_t = beta_0 + beta_1 * RateSurprise_t + beta_2 * Delta_Tone_t + eps_t

    Uses heteroskedasticity-consistent standard errors (HC1) via statsmodels OLS.
    Explicitly reports sample size N and degrees of freedom caveats.
    """
    cols = [dependent_col, rate_surprise_col, tone_change_col]
    clean = df[cols].dropna().copy()
    n_obs = len(clean)
    if n_obs < 5:
        raise ValueError(f"Insufficient observations ({n_obs}) for OLS regression.")

    y = clean[dependent_col].astype(float)
    X = sm.add_constant(clean[[rate_surprise_col, tone_change_col]].astype(float), has_constant="add")

    model = sm.OLS(y, X)
    fitted = model.fit(cov_type=hc_type)
    conf = fitted.conf_int(alpha=alpha)

    coeffs: Dict[str, RegressionCoefficient] = {}
    for term in X.columns:
        coeffs[term] = RegressionCoefficient(
            name=term,
            estimate=round(float(fitted.params[term]), 4),
            std_err=round(float(fitted.bse[term]), 4),
            t_stat=round(float(fitted.tvalues[term]), 4),
            p_value=round(float(fitted.pvalues[term]), 4),
            ci_lower=round(float(conf.loc[term, 0]), 4),
            ci_upper=round(float(conf.loc[term, 1]), 4),
        )

    dof = n_obs - len(X.columns)
    note = (
        f"Small-sample warning: N={n_obs} FOMC releases (dof={dof}). "
        "Because the FOMC meets only 8 times per year, statistical power is limited, "
        "confidence intervals are wide, and contemporaneous press conferences or SEP dot-plot "
        "releases can confound statement-only tone effects."
    )

    return EventStudyResult(
        dependent_var=dependent_col,
        tone_col=tone_change_col,
        n_obs=n_obs,
        r_squared=round(float(fitted.rsquared), 4),
        adj_r_squared=round(float(fitted.rsquared_adj), 4),
        f_stat=round(float(fitted.fvalue) if fitted.fvalue is not None else 0.0, 4),
        f_pvalue=round(float(fitted.f_pvalue) if fitted.f_pvalue is not None else 1.0, 4),
        coefficients=coeffs,
        small_sample_note=note,
    )
