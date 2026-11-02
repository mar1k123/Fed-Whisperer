import React, { useMemo, useState } from 'react';
import {
  EventObservation,
  runEventStudyOLS,
  runLeaveOnePeriodOutAnalysis,
  runPlaceboMonteCarlo,
} from '../engine/econometrics';

interface EventStudyViewProps {
  observations: EventObservation[];
}

export const EventStudyView: React.FC<EventStudyViewProps> = ({ observations }) => {
  const [asset, setAsset] = useState<'dgs2' | 'dgs10' | 'sp500'>('dgs2');
  const [windowDays, setWindowDays] = useState<1 | 2 | 5>(1);
  const [toneMethod, setToneMethod] = useState<'dict' | 'embed' | 'llm'>('llm');

  const olsResult = useMemo(
    () => runEventStudyOLS(observations, asset, windowDays, toneMethod),
    [observations, asset, windowDays, toneMethod]
  );

  const windowSensitivity = useMemo(() => {
    return ([1, 2, 5] as const).map((w) =>
      runEventStudyOLS(observations, asset, w, toneMethod)
    );
  }, [observations, asset, toneMethod]);

  const lopoResults = useMemo(
    () => runLeaveOnePeriodOutAnalysis(observations, asset, windowDays, toneMethod),
    [observations, asset, windowDays, toneMethod]
  );

  const placeboResult = useMemo(
    () => runPlaceboMonteCarlo(observations, asset, windowDays, toneMethod, 500, 42),
    [observations, asset, windowDays, toneMethod]
  );

  // Scatter plot coordinates: X = Delta Tone, Y = Partial Residual Market Change (controlling for Rate Surprise)
  const scatterW = 540;
  const scatterH = 280;
  const padL = 52;
  const padR = 24;
  const padT = 24;
  const padB = 38;
  const plotW = scatterW - padL - padR;
  const plotH = scatterH - padT - padB;

  const xVals = olsResult.fittedPoints.map((p) => p.toneChange);
  const yVals = olsResult.fittedPoints.map((p) => p.partialResidualY);
  const minX = Math.min(-0.6, ...xVals);
  const maxX = Math.max(0.6, ...xVals);
  const minY = Math.min(-15, ...yVals) * 1.15;
  const maxY = Math.max(15, ...yVals) * 1.15;

  const scaleX = (x: number) => padL + ((x - minX) / (maxX - minX || 1)) * plotW;
  const scaleY = (y: number) => padT + ((maxY - y) / (maxY - minY || 1)) * plotH;

  const regLineY1 = olsResult.toneChange.estimate * minX;
  const regLineY2 = olsResult.toneChange.estimate * maxX;

  return (
    <div className="space-y-8">
      {/* Header & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-medium text-slate-500">
            03. Econometric Event Study & Robustness Suite · OLS with White HC1 Standard Errors
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">
            Market Reaction Regressions, Leave-One-Period-Out & Placebo Tests
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Target Asset Selector */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg" role="group" aria-label="Select target market asset">
            {(
              [
                { id: 'dgs2', label: 'DGS2 (2Y Yield)' },
                { id: 'dgs10', label: 'DGS10 (10Y Yield)' },
                { id: 'sp500', label: 'SPY (S&P 500)' },
              ] as const
            ).map((opt) => (
              <button
                key={opt.id}
                type="button"
                onClick={() => setAsset(opt.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  asset === opt.id ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                }`}
              >
                {opt.label}
              </button>
            ))}
          </div>

          {/* Window Selector */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg" role="group" aria-label="Select event window length">
            {([1, 2, 5] as const).map((w) => (
              <button
                key={w}
                type="button"
                onClick={() => setWindowDays(w)}
                className={`px-3 py-1.5 text-xs font-mono font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  windowDays === w ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                }`}
              >
                {w}-Day Window
              </button>
            ))}
          </div>

          {/* Tone Method Selector */}
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg" role="group" aria-label="Select tone scoring method">
            {(
              [
                { id: 'dict', label: 'Dictionary (A)' },
                { id: 'embed', label: 'Embedding (B)' },
                { id: 'llm', label: 'LLM Rubric (C)' },
              ] as const
            ).map((m) => (
              <button
                key={m.id}
                type="button"
                onClick={() => setToneMethod(m.id)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  toneMethod === m.id ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
                }`}
              >
                {m.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Regression Equation & Small Sample Caveat Banner */}
      <div className="bg-white border border-slate-200 rounded-xl p-6 space-y-4">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-100 pb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Baseline Event Study Specification: {olsResult.dependentLabel}
            </h2>
            <p className="text-xs font-mono text-slate-600 mt-1">
              ΔY_t = α + β₁ · RateSurprise_t (bps) + β₂ · ΔTone_t + ε_t
            </p>
          </div>

          <div className="flex items-center gap-6 text-xs font-mono tabular-nums text-slate-700">
            <span>N = {olsResult.nObs} releases</span>
            <span>·</span>
            <span>dof = {olsResult.dof}</span>
            <span>·</span>
            <span>R² = {olsResult.rSquared.toFixed(3)}</span>
            <span>·</span>
            <span>Adj. R² = {olsResult.adjRSquared.toFixed(3)}</span>
            <span>·</span>
            <span>F-stat = {olsResult.fStat.toFixed(2)}</span>
          </div>
        </div>

        {/* Coefficients Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold text-slate-600">
                <th className="py-2.5 pr-4">Regressor Term</th>
                <th className="py-2.5 px-4 text-right">Coefficient (β)</th>
                <th className="py-2.5 px-4 text-right">Robust SE (HC1)</th>
                <th className="py-2.5 px-4 text-right">t-Statistic</th>
                <th className="py-2.5 px-4 text-right">p-Value (Two-Sided)</th>
                <th className="py-2.5 pl-4 text-right">95% Confidence Interval</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm font-mono tabular-nums">
              {[olsResult.intercept, olsResult.rateSurprise, olsResult.toneChange].map((term) => (
                <tr key={term.name} className="hover:bg-slate-50">
                  <td className="py-3 pr-4 font-sans font-medium text-slate-900">{term.label}</td>
                  <td className="py-3 px-4 text-right font-semibold text-slate-900">
                    {term.estimate >= 0 ? `+${term.estimate.toFixed(3)}` : term.estimate.toFixed(3)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-600">{term.stdErr.toFixed(3)}</td>
                  <td className="py-3 px-4 text-right text-slate-800">
                    {term.tStat >= 0 ? `+${term.tStat.toFixed(2)}` : term.tStat.toFixed(2)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-800">{term.pValue.toFixed(4)}</td>
                  <td className="py-3 pl-4 text-right text-slate-700">
                    [{term.ciLower >= 0 ? `+${term.ciLower.toFixed(3)}` : term.ciLower.toFixed(3)},{' '}
                    {term.ciUpper >= 0 ? `+${term.ciUpper.toFixed(3)}` : term.ciUpper.toFixed(3)}]
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="pt-3 border-t border-slate-100 text-xs text-slate-600 leading-relaxed">
          <strong className="text-slate-900">Small Sample & Identification Discussion:</strong>{' '}
          {olsResult.smallSampleWarning}
        </div>
      </div>

      {/* Partial Regression Scatter Plot + Window Sensitivity */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 6 cols: Partial Residual Scatter Plot */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Partial Effect of ΔTone Controlling for Rate Surprise
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Vertical axis shows {olsResult.dependentLabel} orthogonalized against the policy rate surprise proxy (ΔY - α - β₁·RateSurprise). Slope equals β₂ ={' '}
              <span className="font-mono font-semibold text-slate-900">
                {olsResult.toneChange.estimate >= 0
                  ? `+${olsResult.toneChange.estimate.toFixed(2)}`
                  : olsResult.toneChange.estimate.toFixed(2)}
              </span>
              .
            </p>
          </div>

          <div className="my-4">
            <svg
              viewBox={`0 0 ${scatterW} ${scatterH}`}
              className="w-full h-auto"
              role="img"
              aria-label="Partial regression scatter plot of FOMC tone change versus market reaction"
            >
              {/* Y-Axis Grid & Numeric Ticks */}
              {[minY * 0.7, 0, maxY * 0.7].map((yTick, idx) => {
                const py = scaleY(yTick);
                return (
                  <g key={idx}>
                    <line
                      x1={padL}
                      y1={py}
                      x2={scatterW - padR}
                      y2={py}
                      stroke={yTick === 0 ? '#94a3b8' : '#e2e8f0'}
                      strokeDasharray={yTick === 0 ? 'none' : '4 4'}
                    />
                    <text
                      x={padL - 8}
                      y={py + 4}
                      textAnchor="end"
                      className="fill-slate-500 font-mono text-[10px]"
                    >
                      {yTick > 0 ? `+${yTick.toFixed(1)}` : yTick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* X-Axis Grid & Numeric Ticks */}
              {[-0.4, -0.2, 0.0, 0.2, 0.4].map((xTick) => {
                const px = scaleX(xTick);
                return (
                  <g key={xTick}>
                    <line
                      x1={px}
                      y1={padT}
                      x2={px}
                      y2={scatterH - padB}
                      stroke={xTick === 0 ? '#94a3b8' : '#f1f5f9'}
                      strokeDasharray={xTick === 0 ? '3 3' : 'none'}
                    />
                    <text
                      x={px}
                      y={scatterH - padB + 14}
                      textAnchor="middle"
                      className="fill-slate-500 font-mono text-[10px]"
                    >
                      {xTick > 0 ? `+${xTick.toFixed(1)}` : xTick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Fitted Slope Line */}
              <line
                x1={scaleX(minX)}
                y1={scaleY(regLineY1)}
                x2={scaleX(maxX)}
                y2={scaleY(regLineY2)}
                stroke="#1d4ed8"
                strokeWidth={2.25}
              />

              {/* Scatter Points */}
              {olsResult.fittedPoints.map((pt) => (
                <g key={pt.date}>
                  <circle
                    cx={scaleX(pt.toneChange)}
                    cy={scaleY(pt.partialResidualY)}
                    r={4.5}
                    fill="#0f172a"
                    fillOpacity={0.85}
                  />
                  <title>
                    {pt.date}: ΔTone = {pt.toneChange}, Residual ΔY = {pt.partialResidualY}
                  </title>
                </g>
              ))}

              {/* Axis Titles */}
              <text
                x={padL + plotW / 2}
                y={scatterH - 4}
                textAnchor="middle"
                className="fill-slate-600 font-mono text-[11px]"
              >
                Statement Tone Shift (ΔTone_t)
              </text>
            </svg>
          </div>

          <div className="text-xs text-slate-500 border-t border-slate-100 pt-3 font-mono tabular-nums">
            Slope β₂ = {olsResult.toneChange.estimate.toFixed(3)} · 95% CI [{olsResult.toneChange.ciLower.toFixed(2)},{' '}
            {olsResult.toneChange.ciUpper.toFixed(2)}] · p = {olsResult.toneChange.pValue.toFixed(4)}
          </div>
        </div>

        {/* Right 6 cols: Robustness 1 — Window Length Sensitivity & Placebo Histogram */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between space-y-6">
          <div>
            <h3 className="text-base font-semibold text-slate-900">
              Robustness I: Event Window Length Sensitivity (1d vs. 2d vs. 5d)
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Tests how quickly statement tone information is absorbed and whether multi-day macro news dilutes precision.
            </p>

            <div className="overflow-x-auto mt-3">
              <table className="w-full text-left border-collapse">
                <thead>
                  <tr className="border-b border-slate-200 text-xs font-semibold text-slate-600">
                    <th className="py-2 pr-3">Window</th>
                    <th className="py-2 px-3 text-right">ΔTone β₂</th>
                    <th className="py-2 px-3 text-right">95% CI</th>
                    <th className="py-2 px-3 text-right">p-Value</th>
                    <th className="py-2 pl-3 text-right">R²</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-xs font-mono tabular-nums">
                  {windowSensitivity.map((wRes) => (
                    <tr
                      key={wRes.windowDays}
                      className={wRes.windowDays === windowDays ? 'bg-slate-50 font-semibold' : ''}
                    >
                      <td className="py-2.5 pr-3 text-slate-900">{wRes.windowDays}-Trading Day</td>
                      <td className="py-2.5 px-3 text-right text-slate-900">
                        {wRes.toneChange.estimate >= 0
                          ? `+${wRes.toneChange.estimate.toFixed(3)}`
                          : wRes.toneChange.estimate.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-600">
                        [{wRes.toneChange.ciLower.toFixed(2)}, {wRes.toneChange.ciUpper.toFixed(2)}]
                      </td>
                      <td className="py-2.5 px-3 text-right text-slate-800">
                        {wRes.toneChange.pValue.toFixed(4)}
                      </td>
                      <td className="py-2.5 pl-3 text-right text-slate-800">
                        {wRes.rSquared.toFixed(3)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Robustness 2: Placebo Random Dates Test */}
          <div className="pt-4 border-t border-slate-200">
            <div className="flex items-center justify-between gap-2">
              <div>
                <h3 className="text-base font-semibold text-slate-900">
                  Robustness II: Random-Date Placebo Simulation ({placeboResult.iterations} Runs)
                </h3>
                <p className="text-xs text-slate-500 mt-0.5">
                  Null distribution of β₂ when pairing FOMC tone changes with randomly sampled non-aligned market windows.
                </p>
              </div>
              <div className="text-right font-mono tabular-nums">
                <div className="text-xs text-slate-500">Empirical p-value</div>
                <div className="text-sm font-bold text-slate-900">
                  p = {placeboResult.twoSidedPValue.toFixed(3)}
                </div>
              </div>
            </div>

            <div className="mt-3 flex items-end gap-1.5 h-24 pt-2 px-2 bg-slate-50 rounded-lg border border-slate-100">
              {placeboResult.histogramBins.map((bin, idx) => {
                const maxCount = Math.max(...placeboResult.histogramBins.map((b) => b.count), 1);
                const heightPct = Math.max(6, (bin.count / maxCount) * 100);
                const isNearActual =
                  Math.abs(bin.binCenter - placeboResult.actualBeta) <=
                  Math.abs(
                    (placeboResult.histogramBins[1]?.binCenter ?? 1) -
                      (placeboResult.histogramBins[0]?.binCenter ?? 0)
                  );
                return (
                  <div key={idx} className="flex-1 flex flex-col items-center justify-end h-full">
                    <div
                      style={{ height: `${heightPct}%` }}
                      className={`w-full rounded-t transition-all ${
                        isNearActual ? 'bg-blue-700' : 'bg-slate-300'
                      }`}
                      title={`Bin ${bin.binCenter}: ${bin.count} placebo runs`}
                    />
                  </div>
                );
              })}
            </div>
            <div className="flex items-center justify-between text-[11px] font-mono tabular-nums text-slate-500 mt-1.5">
              <span>Placebo Mean β₂: {placeboResult.placeboMeanBeta.toFixed(3)} (SD: {placeboResult.placeboStdBeta.toFixed(3)})</span>
              <span className="text-blue-700 font-semibold">
                Observed FOMC β₂: {placeboResult.actualBeta >= 0 ? `+${placeboResult.actualBeta.toFixed(3)}` : placeboResult.actualBeta.toFixed(3)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Robustness III: Leave-One-Period-Out Regime Stability */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <div className="mb-4">
          <h2 className="text-base font-semibold text-slate-900">
            Robustness III: Leave-One-Period-Out (LOPO) Macroeconomic Regime Stability
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Re-estimates the OLS event study omitting one entire policy regime at a time to test whether a single hiking or easing year drives the result.
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold text-slate-600">
                <th className="py-2.5 pr-4">Omitted Macroeconomic Sub-Period</th>
                <th className="py-2.5 px-4">Date Span Omitted</th>
                <th className="py-2.5 px-4 text-right">Remaining N</th>
                <th className="py-2.5 px-4 text-right">ΔTone Coefficient (β₂)</th>
                <th className="py-2.5 px-4 text-right">Robust SE</th>
                <th className="py-2.5 px-4 text-right">95% CI</th>
                <th className="py-2.5 px-4 text-right">p-Value</th>
                <th className="py-2.5 pl-4 text-right">R²</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs font-mono tabular-nums">
              {lopoResults.map((row) => (
                <tr key={row.regimeName} className="hover:bg-slate-50">
                  <td className="py-3 pr-4 font-sans font-medium text-slate-900">{row.regimeName}</td>
                  <td className="py-3 px-4 text-slate-500">
                    {row.startDate} to {row.endDate}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-800">{row.nObs}</td>
                  <td className="py-3 px-4 text-right font-semibold text-slate-900">
                    {row.toneBeta >= 0 ? `+${row.toneBeta.toFixed(3)}` : row.toneBeta.toFixed(3)}
                  </td>
                  <td className="py-3 px-4 text-right text-slate-600">{row.stdErr.toFixed(3)}</td>
                  <td className="py-3 px-4 text-right text-slate-600">
                    [{row.ciLower.toFixed(2)}, {row.ciUpper.toFixed(2)}]
                  </td>
                  <td className="py-3 px-4 text-right text-slate-800">{row.pValue.toFixed(4)}</td>
                  <td className="py-3 pl-4 text-right text-slate-800">{row.rSquared.toFixed(3)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
