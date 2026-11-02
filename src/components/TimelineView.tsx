import React, { useMemo, useState } from 'react';
import { EnrichedFOMCRelease } from '../data/fomcCorpus';
import { computeMethodAgreementMatrix } from '../engine/nlp';
import { ArrowUpRight, ArrowDownRight, Search } from 'lucide-react';

interface TimelineViewProps {
  corpus: EnrichedFOMCRelease[];
  selectedReleaseId: string;
  onSelectRelease: (id: string) => void;
  onOpenDiffForRelease: (currId: string, prevId: string) => void;
}

export const TimelineView: React.FC<TimelineViewProps> = ({
  corpus,
  selectedReleaseId,
  onSelectRelease,
  onOpenDiffForRelease,
}) => {
  const [regimeFilter, setRegimeFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState<string>('');
  const [visibleSeries, setVisibleSeries] = useState<{
    dict: boolean;
    embed: boolean;
    llm: boolean;
  }>({ dict: true, embed: true, llm: true });

  const regimes = useMemo(() => {
    const unique = Array.from(new Set(corpus.map((c) => c.regime)));
    return ['ALL', ...unique];
  }, [corpus]);

  const filteredCorpus = useMemo(() => {
    return corpus.filter((item) => {
      const matchesRegime = regimeFilter === 'ALL' || item.regime === regimeFilter;
      const matchesSearch =
        !searchQuery.trim() ||
        item.date.includes(searchQuery.trim()) ||
        item.statementText.toLowerCase().includes(searchQuery.toLowerCase()) ||
        item.precomputedLLM.rationale.toLowerCase().includes(searchQuery.toLowerCase());
      return matchesRegime && matchesSearch;
    });
  }, [corpus, regimeFilter, searchQuery]);

  const agreementMatrix = useMemo(() => computeMethodAgreementMatrix(corpus), [corpus]);

  const selectedRelease = useMemo(
    () => corpus.find((c) => c.id === selectedReleaseId) || corpus[corpus.length - 1],
    [corpus, selectedReleaseId]
  );

  const selectedIdx = useMemo(
    () => corpus.findIndex((c) => c.id === selectedRelease.id),
    [corpus, selectedRelease]
  );
  const prevRelease = selectedIdx > 0 ? corpus[selectedIdx - 1] : corpus[0];

  // SVG Chart dimensions
  const width = 920;
  const height = 300;
  const padLeft = 52;
  const padRight = 24;
  const padTop = 24;
  const padBottom = 42;
  const plotW = width - padLeft - padRight;
  const plotH = height - padTop - padBottom;

  const xForIndex = (i: number, total: number) =>
    total <= 1 ? padLeft + plotW / 2 : padLeft + (i / (total - 1)) * plotW;
  const yForScore = (score: number) => {
    const clamped = Math.max(-1, Math.min(1, score));
    return padTop + ((1 - clamped) / 2) * plotH;
  };

  const makePath = (vals: number[]) =>
    vals
      .map((v, idx) => `${idx === 0 ? 'M' : 'L'} ${xForIndex(idx, vals.length).toFixed(1)} ${yForScore(v).toFixed(1)}`)
      .join(' ');

  return (
    <div className="space-y-8">
      {/* Section Header & Filter Bar */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-medium text-slate-500">
            01. Multi-Method FOMC Sentiment Trajectory · {corpus[0]?.date} to {corpus[corpus.length - 1]?.date}
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1 text-balance">
            Hawkish / Dovish Tone Timeline & Method Agreement
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg" role="group" aria-label="Filter by monetary regime">
            {regimes.map((reg) => (
              <button
                key={reg}
                type="button"
                onClick={() => setRegimeFilter(reg)}
                className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                  regimeFilter === reg
                    ? 'bg-white text-slate-900 shadow-xs'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {reg === 'ALL' ? `All Regimes (N=${corpus.length})` : reg.replace(/^\d{4}(–\d{4})?\s+/, '')}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Main Chart + Selected Statement Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-stretch">
        {/* Left 8 cols: Interactive Trajectory Chart */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
          <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
            <div>
              <h2 className="text-base font-semibold text-slate-900">
                FOMC Tone Index Across Scoring Architectures
              </h2>
              <p className="text-xs text-slate-500 mt-0.5">
                Scale: -1.00 (Maximally Dovish) · 0.00 (Neutral) · +1.00 (Maximally Hawkish). Click any meeting node to inspect.
              </p>
            </div>

            <div className="flex items-center gap-2" role="group" aria-label="Toggle tone scoring methods">
              <button
                type="button"
                onClick={() => setVisibleSeries((s) => ({ ...s, dict: !s.dict }))}
                className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 ${
                  visibleSeries.dict
                    ? 'bg-slate-900 text-white border-slate-900'
                    : 'bg-white text-slate-500 border-slate-200'
                }`}
              >
                Method A: Dictionary
              </button>
              <button
                type="button"
                onClick={() => setVisibleSeries((s) => ({ ...s, embed: !s.embed }))}
                className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 ${
                  visibleSeries.embed
                    ? 'bg-blue-700 text-white border-blue-700'
                    : 'bg-white text-slate-500 border-slate-200'
                }`}
              >
                Method B: Embedding Axis
              </button>
              <button
                type="button"
                onClick={() => setVisibleSeries((s) => ({ ...s, llm: !s.llm }))}
                className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 ${
                  visibleSeries.llm
                    ? 'bg-amber-700 text-white border-amber-700'
                    : 'bg-white text-slate-500 border-slate-200'
                }`}
              >
                Method C: LLM Rubric (T=0)
              </button>
            </div>
          </div>

          <div className="w-full overflow-x-auto">
            <svg
              viewBox={`0 0 ${width} ${height}`}
              className="w-full h-auto select-none"
              role="img"
              aria-label="FOMC Hawkish-Dovish Tone Timeline Chart"
            >
              {/* Horizontal Reference Grid Lines */}
              {[-1.0, -0.5, 0.0, 0.5, 1.0].map((tick) => {
                const y = yForScore(tick);
                return (
                  <g key={tick}>
                    <line
                      x1={padLeft}
                      y1={y}
                      x2={width - padRight}
                      y2={y}
                      stroke={tick === 0 ? '#94a3b8' : '#e2e8f0'}
                      strokeDasharray={tick === 0 ? 'none' : '4 4'}
                      strokeWidth={tick === 0 ? 1.5 : 1}
                    />
                    <text
                      x={padLeft - 8}
                      y={y + 4}
                      textAnchor="end"
                      className="fill-slate-500 font-mono text-[11px]"
                    >
                      {tick > 0 ? `+${tick.toFixed(1)}` : tick.toFixed(1)}
                    </text>
                  </g>
                );
              })}

              {/* Zone Labels */}
              <text x={padLeft + 8} y={padTop + 14} className="fill-slate-400 text-[10px] font-medium">
                Hawkish Tightening Zone (+0.20 to +1.00)
              </text>
              <text x={padLeft + 8} y={padTop + plotH - 8} className="fill-slate-400 text-[10px] font-medium">
                Dovish Accommodation Zone (-0.20 to -1.00)
              </text>

              {/* Selected Meeting Vertical Highlight */}
              {selectedIdx >= 0 && (
                <line
                  x1={xForIndex(selectedIdx, corpus.length)}
                  y1={padTop}
                  x2={xForIndex(selectedIdx, corpus.length)}
                  y2={height - padBottom}
                  stroke="#0f172a"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                />
              )}

              {/* Trajectory Lines */}
              {visibleSeries.dict && (
                <path
                  d={makePath(corpus.map((c) => c.dictScore))}
                  fill="none"
                  stroke="#0f172a"
                  strokeWidth={2.25}
                />
              )}
              {visibleSeries.embed && (
                <path
                  d={makePath(corpus.map((c) => c.embedScore))}
                  fill="none"
                  stroke="#1d4ed8"
                  strokeWidth={2.25}
                />
              )}
              {visibleSeries.llm && (
                <path
                  d={makePath(corpus.map((c) => c.llmScore))}
                  fill="none"
                  stroke="#b45309"
                  strokeWidth={2.25}
                />
              )}

              {/* Interactive Meeting Nodes */}
              {corpus.map((item, idx) => {
                const cx = xForIndex(idx, corpus.length);
                const isSelected = item.id === selectedRelease.id;
                return (
                  <g
                    key={item.id}
                    className="cursor-pointer"
                    onClick={() => onSelectRelease(item.id)}
                  >
                    {visibleSeries.dict && (
                      <circle
                        cx={cx}
                        cy={yForScore(item.dictScore)}
                        r={isSelected ? 5.5 : 3.5}
                        fill={isSelected ? '#ffffff' : '#0f172a'}
                        stroke="#0f172a"
                        strokeWidth={2}
                      />
                    )}
                    {visibleSeries.embed && (
                      <circle
                        cx={cx}
                        cy={yForScore(item.embedScore)}
                        r={isSelected ? 5.5 : 3.5}
                        fill={isSelected ? '#ffffff' : '#1d4ed8'}
                        stroke="#1d4ed8"
                        strokeWidth={2}
                      />
                    )}
                    {visibleSeries.llm && (
                      <circle
                        cx={cx}
                        cy={yForScore(item.llmScore)}
                        r={isSelected ? 5.5 : 3.5}
                        fill={isSelected ? '#ffffff' : '#b45309'}
                        stroke="#b45309"
                        strokeWidth={2}
                      />
                    )}

                    {/* X-axis Date Labels (every 2nd date to prevent crowding) */}
                    {(idx % 2 === 0 || idx === corpus.length - 1) && (
                      <text
                        x={cx}
                        y={height - 14}
                        textAnchor="middle"
                        className={`font-mono text-[10px] ${
                          isSelected ? 'fill-slate-900 font-semibold' : 'fill-slate-500'
                        }`}
                      >
                        {item.date.slice(2, 7)}
                      </text>
                    )}
                  </g>
                );
              })}
            </svg>
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4 pt-3 border-t border-slate-100 text-xs text-slate-500">
            <span>
              Selected Release: <strong className="text-slate-900 font-mono">{selectedRelease.date}</strong> · Target Range:{' '}
              <strong className="text-slate-900 font-mono">{selectedRelease.targetRateRange}</strong> · Decision:{' '}
              <strong className="text-slate-900 font-mono">
                {selectedRelease.rateDecisionBps > 0
                  ? `+${selectedRelease.rateDecisionBps} bps`
                  : `${selectedRelease.rateDecisionBps} bps`}
              </strong>
            </span>
            <button
              type="button"
              onClick={() => onOpenDiffForRelease(selectedRelease.id, prevRelease.id)}
              className="text-xs font-semibold text-blue-700 hover:text-blue-900 transition-colors whitespace-nowrap shrink-0"
            >
              Inspect Sentence Diff vs. {prevRelease.date} →
            </button>
          </div>
        </div>

        {/* Right 4 cols: Selected Statement Breakdown */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-6 flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between gap-2 text-xs text-slate-500">
              <span>{selectedRelease.regime}</span>
              <span>·</span>
              <span className="font-mono">{selectedRelease.hasSEP ? 'Includes SEP Dot Plot' : 'Statement Only'}</span>
            </div>

            <h3 className="text-lg font-bold text-slate-900 mt-1">
              {selectedRelease.title}
            </h3>

            {/* 3-Method Comparison Strip */}
            <div className="grid grid-cols-3 gap-3 my-4 py-3 border-y border-slate-100">
              <div>
                <div className="text-[11px] text-slate-500">Dictionary (A)</div>
                <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                  {selectedRelease.dictScore >= 0 ? `+${selectedRelease.dictScore.toFixed(2)}` : selectedRelease.dictScore.toFixed(2)}
                </div>
                <div className="text-[11px] font-mono tabular-nums text-slate-500">
                  Δ {selectedRelease.dictToneChange >= 0 ? `+${selectedRelease.dictToneChange.toFixed(2)}` : selectedRelease.dictToneChange.toFixed(2)}
                </div>
              </div>

              <div>
                <div className="text-[11px] text-slate-500">Embedding (B)</div>
                <div className="text-lg font-bold font-mono tabular-nums text-blue-700 mt-0.5">
                  {selectedRelease.embedScore >= 0 ? `+${selectedRelease.embedScore.toFixed(2)}` : selectedRelease.embedScore.toFixed(2)}
                </div>
                <div className="text-[11px] font-mono tabular-nums text-slate-500">
                  Δ {selectedRelease.embedToneChange >= 0 ? `+${selectedRelease.embedToneChange.toFixed(2)}` : selectedRelease.embedToneChange.toFixed(2)}
                </div>
              </div>

              <div>
                <div className="text-[11px] text-slate-500">LLM Rubric (C)</div>
                <div className="text-lg font-bold font-mono tabular-nums text-amber-700 mt-0.5">
                  {selectedRelease.llmScore >= 0 ? `+${selectedRelease.llmScore.toFixed(2)}` : selectedRelease.llmScore.toFixed(2)}
                </div>
                <div className="text-[11px] font-mono tabular-nums text-slate-500">
                  Δ {selectedRelease.llmToneChange >= 0 ? `+${selectedRelease.llmToneChange.toFixed(2)}` : selectedRelease.llmToneChange.toFixed(2)}
                </div>
              </div>
            </div>

            <div className="space-y-3">
              <div>
                <div className="text-xs font-semibold text-slate-800">
                  Fixed-Rubric Rationale (Temperature = 0.0)
                </div>
                <p className="text-xs text-slate-600 leading-relaxed mt-1">
                  {selectedRelease.precomputedLLM.rationale}
                </p>
              </div>

              <div>
                <div className="text-xs font-semibold text-slate-800">
                  Statement Excerpt
                </div>
                <p className="text-xs text-slate-600 leading-relaxed mt-1 line-clamp-5">
                  {selectedRelease.statementText}
                </p>
              </div>

              <div className="pt-2 border-t border-slate-100 text-xs text-slate-500 space-y-1">
                <div>
                  Hawkish lexicon hits:{' '}
                  <span className="font-mono text-slate-800">
                    {selectedRelease.hawkishMatched.length > 0
                      ? selectedRelease.hawkishMatched.join(', ')
                      : 'none'}
                  </span>
                </div>
                <div>
                  Dovish lexicon hits:{' '}
                  <span className="font-mono text-slate-800">
                    {selectedRelease.dovishMatched.length > 0
                      ? selectedRelease.dovishMatched.join(', ')
                      : 'none'}
                  </span>
                </div>
              </div>
            </div>
          </div>

          <div className="mt-5 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
            <span className="text-slate-500 font-mono tabular-nums">
              1d DGS2: {selectedRelease.dgs2_1d_bps > 0 ? `+${selectedRelease.dgs2_1d_bps}` : selectedRelease.dgs2_1d_bps} bps · SPY:{' '}
              {selectedRelease.sp500_1d_pct > 0 ? `+${selectedRelease.sp500_1d_pct}%` : `${selectedRelease.sp500_1d_pct}%`}
            </span>
            <button
              type="button"
              onClick={() => onOpenDiffForRelease(selectedRelease.id, prevRelease.id)}
              className="px-3 py-1.5 bg-slate-900 text-white rounded-lg font-medium hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
            >
              Open in Diff Mode
            </button>
          </div>
        </div>
      </div>

      {/* Method Agreement Matrix */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-2 mb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Inter-Method Agreement & Rank Concordance (N = {corpus.length} Statements)
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Pairwise comparison of continuous tone levels (Pearson r, Spearman ρ), 3-regime classification (Cohen’s κ at ±0.12 threshold), and sequential ΔTone direction agreement.
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold text-slate-600">
                <th className="py-2.5 pr-4">Method Pair Comparison</th>
                <th className="py-2.5 px-4 text-right">Pearson Linear r</th>
                <th className="py-2.5 px-4 text-right">Spearman Rank ρ</th>
                <th className="py-2.5 px-4 text-right">3-Class Cohen’s κ</th>
                <th className="py-2.5 pl-4 text-right">ΔTone Sign Agreement</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-sm">
              {agreementMatrix.map((row) => (
                <tr key={row.pairLabel} className="hover:bg-slate-50/80">
                  <td className="py-3 pr-4 font-medium text-slate-900">{row.pairLabel}</td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-900">
                    {row.pearsonR.toFixed(3)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-900">
                    {row.spearmanRho.toFixed(3)}
                  </td>
                  <td className="py-3 px-4 text-right font-mono tabular-nums text-slate-900">
                    {row.cohenKappa.toFixed(3)}
                  </td>
                  <td className="py-3 pl-4 text-right font-mono tabular-nums text-slate-900">
                    {(row.deltaSignAgreement * 100).toFixed(1)}%
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Full Historical FOMC Statement Ledger */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Cached FOMC Monetary Policy Releases & Market Reactions
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Showing {filteredCorpus.length} of {corpus.length} releases. Select any row to inspect or launch sentence diff.
            </p>
          </div>

          <div className="relative w-full sm:w-72">
            <Search className="w-4 h-4 text-slate-400 absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
            <input
              type="search"
              aria-label="Search FOMC statements by date or phrase"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search date or policy phrase..."
              className="w-full pl-9 pr-3 py-1.5 text-xs bg-slate-50 border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-slate-900"
            />
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left border-collapse">
            <thead>
              <tr className="border-b border-slate-200 text-xs font-semibold text-slate-600">
                <th className="py-2.5 pr-3">Release Date</th>
                <th className="py-2.5 px-3">Target Range</th>
                <th className="py-2.5 px-3 text-right">Rate Δ (bps)</th>
                <th className="py-2.5 px-3 text-right">Surprise (bps)</th>
                <th className="py-2.5 px-3 text-right">Dict (A)</th>
                <th className="py-2.5 px-3 text-right">Embed (B)</th>
                <th className="py-2.5 px-3 text-right">LLM (C)</th>
                <th className="py-2.5 px-3 text-right">ΔTone (LLM)</th>
                <th className="py-2.5 px-3 text-right">DGS2 (1d)</th>
                <th className="py-2.5 px-3 text-right">DGS10 (1d)</th>
                <th className="py-2.5 pl-3 text-right">S&P 500 (1d)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 text-xs">
              {filteredCorpus.map((item) => {
                const isSelected = item.id === selectedRelease.id;
                return (
                  <tr
                    key={item.id}
                    onClick={() => onSelectRelease(item.id)}
                    className={`cursor-pointer transition-colors ${
                      isSelected ? 'bg-slate-100/90 font-medium' : 'hover:bg-slate-50'
                    }`}
                  >
                    <td className="py-2.5 pr-3 font-mono tabular-nums text-slate-900 whitespace-nowrap">
                      {item.date} {item.hasSEP ? '· SEP' : ''}
                    </td>
                    <td className="py-2.5 px-3 font-mono tabular-nums text-slate-600 whitespace-nowrap">
                      {item.targetRateRange}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-800">
                      {item.rateDecisionBps > 0 ? `+${item.rateDecisionBps}` : item.rateDecisionBps}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-600">
                      {item.rateSurpriseBps > 0 ? `+${item.rateSurpriseBps.toFixed(1)}` : item.rateSurpriseBps.toFixed(1)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-900">
                      {item.dictScore >= 0 ? `+${item.dictScore.toFixed(2)}` : item.dictScore.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-blue-700">
                      {item.embedScore >= 0 ? `+${item.embedScore.toFixed(2)}` : item.embedScore.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-amber-700">
                      {item.llmScore >= 0 ? `+${item.llmScore.toFixed(2)}` : item.llmScore.toFixed(2)}
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums">
                      <span
                        className={`inline-flex items-center gap-0.5 ${
                          item.llmToneChange > 0.05
                            ? 'text-rose-700'
                            : item.llmToneChange < -0.05
                            ? 'text-emerald-700'
                            : 'text-slate-500'
                        }`}
                      >
                        {item.llmToneChange > 0.05 && <ArrowUpRight className="w-3 h-3" />}
                        {item.llmToneChange < -0.05 && <ArrowDownRight className="w-3 h-3" />}
                        {item.llmToneChange >= 0 ? `+${item.llmToneChange.toFixed(2)}` : item.llmToneChange.toFixed(2)}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-800">
                      {item.dgs2_1d_bps > 0 ? `+${item.dgs2_1d_bps.toFixed(1)}` : item.dgs2_1d_bps.toFixed(1)} bps
                    </td>
                    <td className="py-2.5 px-3 text-right font-mono tabular-nums text-slate-800">
                      {item.dgs10_1d_bps > 0 ? `+${item.dgs10_1d_bps.toFixed(1)}` : item.dgs10_1d_bps.toFixed(1)} bps
                    </td>
                    <td className="py-2.5 pl-3 text-right font-mono tabular-nums text-slate-800">
                      {item.sp500_1d_pct > 0 ? `+${item.sp500_1d_pct.toFixed(2)}%` : `${item.sp500_1d_pct.toFixed(2)}%`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
