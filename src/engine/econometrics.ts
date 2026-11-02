/**
 * Pure Econometrics & Event Study Engine for Fed Whisperer.
 * Implements exact OLS estimation with HC1 robust standard errors, confidence intervals,
 * window length sensitivity (1d / 2d / 5d), Leave-One-Period-Out (LOPO), and Monte Carlo Placebo tests.
 */

export interface RegressionTerm {
  name: string;
  label: string;
  estimate: number;
  stdErr: number;
  tStat: number;
  pValue: number;
  ciLower: number;
  ciUpper: number;
}

export interface EventStudyRegressionOutput {
  dependentVar: string;
  dependentLabel: string;
  windowDays: 1 | 2 | 5;
  toneMethod: 'dict' | 'embed' | 'llm';
  nObs: number;
  dof: number;
  rSquared: number;
  adjRSquared: number;
  fStat: number;
  intercept: RegressionTerm;
  rateSurprise: RegressionTerm;
  toneChange: RegressionTerm;
  fittedPoints: {
    date: string;
    actualY: number;
    predictedY: number;
    toneChange: number;
    rateSurpriseBps: number;
    partialResidualY: number;
  }[];
  smallSampleWarning: string;
}

export interface LeaveOneOutResult {
  regimeName: string;
  startDate: string;
  endDate: string;
  nObs: number;
  toneBeta: number;
  stdErr: number;
  ciLower: number;
  ciUpper: number;
  pValue: number;
  rSquared: number;
}

export interface PlaceboSimulationOutput {
  actualBeta: number;
  actualTStat: number;
  placeboMeanBeta: number;
  placeboStdBeta: number;
  twoSidedPValue: number;
  iterations: number;
  histogramBins: { binCenter: number; count: number }[];
}

export interface EventObservation {
  date: string;
  regime: string;
  rateDecisionBps: number;
  rateSurpriseBps: number;
  dictToneChange: number;
  embedToneChange: number;
  llmToneChange: number;
  dgs2_1d_bps: number;
  dgs2_2d_bps: number;
  dgs2_5d_bps: number;
  dgs10_1d_bps: number;
  dgs10_2d_bps: number;
  dgs10_5d_bps: number;
  sp500_1d_pct: number;
  sp500_2d_pct: number;
  sp500_5d_pct: number;
}

/**
 * Invert a 3x3 symmetric matrix (X'X) using exact cofactor expansion.
 */
function invert3x3(m: number[][]): number[][] {
  const [
    [a, b, c],
    [d, e, f],
    [g, h, i],
  ] = m;

  const A = e * i - f * h;
  const B = -(d * i - f * g);
  const C = d * h - e * g;
  const D = -(b * i - c * h);
  const E = a * i - c * g;
  const F = -(a * h - b * g);
  const G = b * f - c * e;
  const H = -(a * f - c * d);
  const I = a * e - b * d;

  const det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) {
    // Ridge regularization fallback for near-collinear small samples
    const reg = m.map((row, rIdx) => row.map((v, cIdx) => (rIdx === cIdx ? v + 1e-6 : v)));
    return invert3x3(reg);
  }

  const invDet = 1.0 / det;
  return [
    [A * invDet, D * invDet, G * invDet],
    [B * invDet, E * invDet, H * invDet],
    [C * invDet, F * invDet, I * invDet],
  ];
}

/**
 * Two-sided p-value approximation for Student's t distribution with `dof` degrees of freedom.
 */
function studentTPValueTwoSided(tAbs: number, dof: number): number {
  if (!Number.isFinite(tAbs) || dof <= 0) return 1.0;
  // Transform Student-t to normal approximation with Cornish-Fisher finite-sample correction
  const z =
    tAbs *
    (1 - 1 / (4 * dof)) /
    Math.sqrt(1 + (tAbs * tAbs) / (2 * dof));
  // Abramowitz & Stegun normal tail approximation
  const p = 0.2316419;
  const b1 = 0.31938153;
  const b2 = -0.356563782;
  const b3 = 1.781477937;
  const b4 = -1.821255978;
  const b5 = 1.330274429;
  const k = 1.0 / (1.0 + p * z);
  const pdf = Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI);
  const tail = pdf * k * (b1 + k * (b2 + k * (b3 + k * (b4 + k * b5))));
  return Math.max(0.0001, Math.min(1.0, 2.0 * tail));
}

function studentTCritical95(dof: number): number {
  if (dof <= 5) return 2.571;
  if (dof <= 10) return 2.228;
  if (dof <= 15) return 2.131;
  if (dof <= 20) return 2.086;
  if (dof <= 25) return 2.06;
  if (dof <= 30) return 2.042;
  return 1.96;
}

export function getDependentColumn(
  obs: EventObservation,
  asset: 'dgs2' | 'dgs10' | 'sp500',
  windowDays: 1 | 2 | 5
): number {
  const key =
    asset === 'sp500'
      ? (`sp500_${windowDays}d_pct` as const)
      : (`${asset}_${windowDays}d_bps` as const);
  return obs[key];
}

export function getToneChangeValue(
  obs: EventObservation,
  method: 'dict' | 'embed' | 'llm'
): number {
  if (method === 'dict') return obs.dictToneChange;
  if (method === 'embed') return obs.embedToneChange;
  return obs.llmToneChange;
}

/**
 * Estimate OLS regression with White HC1 heteroskedasticity-consistent standard errors:
 *   Delta_Y_t = beta_0 + beta_1 * RateSurprise_t + beta_2 * Delta_Tone_t + eps_t
 */
export function runEventStudyOLS(
  observations: EventObservation[],
  asset: 'dgs2' | 'dgs10' | 'sp500' = 'dgs2',
  windowDays: 1 | 2 | 5 = 1,
  toneMethod: 'dict' | 'embed' | 'llm' = 'llm'
): EventStudyRegressionOutput {
  const n = observations.length;
  const k = 3;
  const dof = Math.max(1, n - k);

  const X: number[][] = [];
  const y: number[] = [];

  for (const obs of observations) {
    const x1 = obs.rateSurpriseBps;
    const x2 = getToneChangeValue(obs, toneMethod);
    const yVal = getDependentColumn(obs, asset, windowDays);
    X.push([1.0, x1, x2]);
    y.push(yVal);
  }

  // Compute X'X and X'y
  const XtX: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const Xty: number[] = [0, 0, 0];

  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      Xty[r] += X[i][r] * y[i];
      for (let c = 0; c < 3; c++) {
        XtX[r][c] += X[i][r] * X[i][c];
      }
    }
  }

  const XtXInv = invert3x3(XtX);
  const beta: number[] = [0, 0, 0];
  for (let r = 0; r < 3; r++) {
    beta[r] = XtXInv[r][0] * Xty[0] + XtXInv[r][1] * Xty[1] + XtXInv[r][2] * Xty[2];
  }

  // Residuals and R^2
  const yMean = y.reduce((a, b) => a + b, 0) / n;
  let rss = 0;
  let tss = 0;
  const residuals: number[] = [];

  for (let i = 0; i < n; i++) {
    const yHat = beta[0] + beta[1] * X[i][1] + beta[2] * X[i][2];
    const e = y[i] - yHat;
    residuals.push(e);
    rss += e * e;
    tss += (y[i] - yMean) * (y[i] - yMean);
  }

  const rSquared = tss > 0 ? Math.max(0, 1 - rss / tss) : 0;
  const adjRSquared = tss > 0 ? 1 - (1 - rSquared) * ((n - 1) / dof) : 0;
  const fStat =
    rss > 0 && rSquared > 0 ? ((rSquared / (k - 1)) / ((1 - rSquared) / dof)) : 0;

  // White HC1 covariance matrix: (n / (n - k)) * (X'X)^(-1) (sum e_i^2 x_i x_i') (X'X)^(-1)
  const meat: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  const hc1Scale = n / dof;
  for (let i = 0; i < n; i++) {
    const e2 = residuals[i] * residuals[i] * hc1Scale;
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        meat[r][c] += e2 * X[i][r] * X[i][c];
      }
    }
  }

  const temp: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      temp[r][c] = XtXInv[r][0] * meat[0][c] + XtXInv[r][1] * meat[1][c] + XtXInv[r][2] * meat[2][c];
    }
  }

  const covHC1: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      covHC1[r][c] = temp[r][0] * XtXInv[0][c] + temp[r][1] * XtXInv[1][c] + temp[r][2] * XtXInv[2][c];
    }
  }

  const tCrit = studentTCritical95(dof);
  const buildTerm = (idx: number, name: string, label: string): RegressionTerm => {
    const est = beta[idx];
    const se = Math.sqrt(Math.max(1e-10, covHC1[idx][idx]));
    const tStat = est / se;
    const pVal = studentTPValueTwoSided(Math.abs(tStat), dof);
    return {
      name,
      label,
      estimate: Number(est.toFixed(4)),
      stdErr: Number(se.toFixed(4)),
      tStat: Number(tStat.toFixed(3)),
      pValue: Number(pVal.toFixed(4)),
      ciLower: Number((est - tCrit * se).toFixed(4)),
      ciUpper: Number((est + tCrit * se).toFixed(4)),
    };
  };

  const assetLabels: Record<'dgs2' | 'dgs10' | 'sp500', string> = {
    dgs2: `2Y Treasury Yield Change (${windowDays}d, bps)`,
    dgs10: `10Y Treasury Yield Change (${windowDays}d, bps)`,
    sp500: `S&P 500 Index Return (${windowDays}d, %)`,
  };

  const fittedPoints = observations.map((obs, i) => {
    const x1 = obs.rateSurpriseBps;
    const x2 = getToneChangeValue(obs, toneMethod);
    const actualY = y[i];
    const predictedY = beta[0] + beta[1] * x1 + beta[2] * x2;
    // Partial residual controlling for rate surprise
    const partialResidualY = actualY - (beta[0] + beta[1] * x1);
    return {
      date: obs.date,
      actualY: Number(actualY.toFixed(3)),
      predictedY: Number(predictedY.toFixed(3)),
      toneChange: Number(x2.toFixed(3)),
      rateSurpriseBps: Number(x1.toFixed(2)),
      partialResidualY: Number(partialResidualY.toFixed(3)),
    };
  });

  return {
    dependentVar: `${asset}_${windowDays}d`,
    dependentLabel: assetLabels[asset],
    windowDays,
    toneMethod,
    nObs: n,
    dof,
    rSquared: Number(rSquared.toFixed(4)),
    adjRSquared: Number(adjRSquared.toFixed(4)),
    fStat: Number(fStat.toFixed(3)),
    intercept: buildTerm(0, 'const', 'Intercept (α)'),
    rateSurprise: buildTerm(1, 'rate_surprise_bps', 'Rate Surprise Proxy (β₁, per bps)'),
    toneChange: buildTerm(2, 'tone_change', 'Statement Tone Shift ΔTone (β₂)'),
    fittedPoints,
    smallSampleWarning: `Small-sample caution: N = ${n} scheduled FOMC releases (dof = ${dof}). Because the FOMC convenes only 8 times per year, confidence intervals remain wide and daily windows include press conference Q&A and SEP dot-plot releases.`,
  };
}

export const MACRO_REGIMES = [
  { name: '2021 Transitory Inflation Era', start: '2021-01-01', end: '2021-12-31' },
  { name: '2022 Rapid Tightening Cycle', start: '2022-01-01', end: '2022-12-31' },
  { name: '2023 High-for-Longer Peak', start: '2023-01-01', end: '2023-12-31' },
  { name: '2024–2025 Calibration & Easing', start: '2024-01-01', end: '2025-12-31' },
];

export function runLeaveOnePeriodOutAnalysis(
  observations: EventObservation[],
  asset: 'dgs2' | 'dgs10' | 'sp500',
  windowDays: 1 | 2 | 5,
  toneMethod: 'dict' | 'embed' | 'llm'
): LeaveOneOutResult[] {
  return MACRO_REGIMES.map((regime) => {
    const subset = observations.filter(
      (obs) => obs.date < regime.start || obs.date > regime.end
    );
    const res = runEventStudyOLS(subset, asset, windowDays, toneMethod);
    return {
      regimeName: regime.name,
      startDate: regime.start,
      endDate: regime.end,
      nObs: res.nObs,
      toneBeta: res.toneChange.estimate,
      stdErr: res.toneChange.stdErr,
      ciLower: res.toneChange.ciLower,
      ciUpper: res.toneChange.ciUpper,
      pValue: res.toneChange.pValue,
      rSquared: res.rSquared,
    };
  });
}

/**
 * Deterministic PRNG (Mulberry32) for reproducible Monte Carlo placebo random-date permutations.
 */
function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export function runPlaceboMonteCarlo(
  observations: EventObservation[],
  asset: 'dgs2' | 'dgs10' | 'sp500',
  windowDays: 1 | 2 | 5,
  toneMethod: 'dict' | 'embed' | 'llm',
  iterations = 500,
  seed = 42
): PlaceboSimulationOutput {
  const actual = runEventStudyOLS(observations, asset, windowDays, toneMethod);
  const actualBeta = actual.toneChange.estimate;
  const actualTStat = actual.toneChange.tStat;

  const rng = mulberry32(seed);
  const n = observations.length;
  const yPool = observations.map((o) => getDependentColumn(o, asset, windowDays));

  const placeboBetas: number[] = [];
  for (let it = 0; it < iterations; it++) {
    // Draw random non-aligned market windows (sampling with replacement + realistic non-FOMC day noise)
    const pseudoObs: EventObservation[] = observations.map((obs) => {
      const randIdx = Math.floor(rng() * n);
      const jitter = (rng() - 0.5) * 0.4;
      const sampledMove = yPool[randIdx] + jitter;
      return {
        ...obs,
        dgs2_1d_bps: sampledMove,
        dgs2_2d_bps: sampledMove,
        dgs2_5d_bps: sampledMove,
        dgs10_1d_bps: sampledMove,
        dgs10_2d_bps: sampledMove,
        dgs10_5d_bps: sampledMove,
        sp500_1d_pct: sampledMove,
        sp500_2d_pct: sampledMove,
        sp500_5d_pct: sampledMove,
      };
    });
    const simRes = runEventStudyOLS(pseudoObs, asset, windowDays, toneMethod);
    placeboBetas.push(simRes.toneChange.estimate);
  }

  const mean = placeboBetas.reduce((a, b) => a + b, 0) / iterations;
  const variance =
    placeboBetas.reduce((acc, v) => acc + (v - mean) * (v - mean), 0) / Math.max(1, iterations - 1);
  const std = Math.sqrt(variance);
  const extremeCount = placeboBetas.filter((b) => Math.abs(b) >= Math.abs(actualBeta)).length;
  const twoSidedPValue = Number((extremeCount / iterations).toFixed(4));

  // Build 15-bin histogram for the placebo distribution
  const minB = Math.min(...placeboBetas, actualBeta);
  const maxB = Math.max(...placeboBetas, actualBeta);
  const numBins = 15;
  const step = (maxB - minB) / numBins || 1;
  const bins = Array.from({ length: numBins }, (_, idx) => ({
    binCenter: Number((minB + (idx + 0.5) * step).toFixed(2)),
    count: 0,
  }));

  for (const b of placeboBetas) {
    const idx = Math.min(numBins - 1, Math.max(0, Math.floor((b - minB) / step)));
    bins[idx].count += 1;
  }

  return {
    actualBeta,
    actualTStat,
    placeboMeanBeta: Number(mean.toFixed(4)),
    placeboStdBeta: Number(std.toFixed(4)),
    twoSidedPValue,
    iterations,
    histogramBins: bins,
  };
}
