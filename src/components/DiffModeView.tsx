import React, { useMemo, useState } from 'react';
import { EnrichedFOMCRelease } from '../data/fomcCorpus';
import {
  computeStatementDiff,
  scoreTextDictionary,
  scoreTextEmbedding,
  evaluateDeterministicRubric,
  LLMRubricScore,
} from '../engine/nlp';

interface DiffModeViewProps {
  corpus: EnrichedFOMCRelease[];
  currReleaseId: string;
  prevReleaseId: string;
  onChangePair: (currId: string, prevId: string) => void;
}

export const DiffModeView: React.FC<DiffModeViewProps> = ({
  corpus,
  currReleaseId,
  prevReleaseId,
  onChangePair,
}) => {
  const [scorerMode, setScorerMode] = useState<'dictionary' | 'embedding'>('dictionary');
  const [filterTag, setFilterTag] = useState<'all' | 'changed'>('all');

  // Custom / Live Gemini Rubric Evaluator state
  const [customText, setCustomText] = useState<string>(
    'Recent indicators suggest that economic activity has continued to expand at a solid pace. Inflation has made further progress toward the Committee\'s 2 percent objective but remains somewhat elevated. In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point.'
  );
  const [liveRubricResult, setLiveRubricResult] = useState<LLMRubricScore | null>(() =>
    evaluateDeterministicRubric(
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. Inflation has made further progress toward the Committee\'s 2 percent objective but remains somewhat elevated. In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point.'
    )
  );
  const [isScoringLLM, setIsScoringLLM] = useState<boolean>(false);
  const [llmStatusNote, setLlmStatusNote] = useState<string>(
    'Ready · Evaluates statement across Dictionary (A), Embedding Axis (B), and Fixed-Rubric LLM (C).'
  );

  const currRelease = useMemo(
    () => corpus.find((c) => c.id === currReleaseId) || corpus[corpus.length - 1],
    [corpus, currReleaseId]
  );
  const prevRelease = useMemo(
    () => corpus.find((c) => c.id === prevReleaseId) || corpus[Math.max(0, corpus.length - 2)],
    [corpus, prevReleaseId]
  );

  const diffReport = useMemo(() => {
    return computeStatementDiff(
      prevRelease.date,
      prevRelease.statementText,
      currRelease.date,
      currRelease.statementText,
      scorerMode
    );
  }, [prevRelease, currRelease, scorerMode]);

  const visibleDiffItems = useMemo(() => {
    if (filterTag === 'changed') {
      return diffReport.items.filter((item) => item.tag !== 'unchanged');
    }
    return diffReport.items;
  }, [diffReport, filterTag]);

  const customDictScore = useMemo(() => scoreTextDictionary(customText), [customText]);
  const customEmbedScore = useMemo(() => scoreTextEmbedding(customText), [customText]);

  const handleRunLiveScorer = async (e: React.FormEvent) => {
    e.preventDefault();
    if (customText.trim().length < 20) {
      setLlmStatusNote('Validation error: Please enter at least 20 characters of FOMC policy prose.');
      return;
    }

    setIsScoringLLM(true);
    setLlmStatusNote('Querying server-side Gemini Rubric Scorer (temperature=0.0)...');

    try {
      const resp = await fetch('/api/score-llm', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: customText }),
      });

      if (resp.ok) {
        const data = (await resp.json()) as LLMRubricScore;
        setLiveRubricResult(data);
        setLlmStatusNote(`Evaluated via live ${data.model} API with fixed 5-point monetary rubric.`);
      } else {
        const fallback = evaluateDeterministicRubric(customText);
        setLiveRubricResult(fallback);
        setLlmStatusNote(
          'Evaluated via deterministic rubric engine (set GEMINI_API_KEY in Secrets for live Gemini API call).'
        );
      }
    } catch {
      const fallback = evaluateDeterministicRubric(customText);
      setLiveRubricResult(fallback);
      setLlmStatusNote('Evaluated via deterministic rubric engine.');
    } finally {
      setIsScoringLLM(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header & Controls */}
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-medium text-slate-500">
            02. Sentence-Level SequenceMatcher Alignment & Revision Attribution
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">
            FOMC Statement Diff Mode & Live Rubric Scorer
          </h1>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2">
            <label htmlFor="prev-release-select" className="text-xs font-medium text-slate-600">
              Baseline (t-1):
            </label>
            <select
              id="prev-release-select"
              value={prevRelease.id}
              onChange={(e) => onChangePair(currRelease.id, e.target.value)}
              className="px-2.5 py-1.5 text-xs font-mono bg-white border border-slate-200 rounded-lg text-slate-900"
            >
              {corpus.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.date} ({c.targetRateRange})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-2">
            <label htmlFor="curr-release-select" className="text-xs font-medium text-slate-600">
              Release (t):
            </label>
            <select
              id="curr-release-select"
              value={currRelease.id}
              onChange={(e) => {
                const idx = corpus.findIndex((c) => c.id === e.target.value);
                const autoPrev = idx > 0 ? corpus[idx - 1].id : prevRelease.id;
                onChangePair(e.target.value, autoPrev);
              }}
              className="px-2.5 py-1.5 text-xs font-mono bg-white border border-slate-200 rounded-lg text-slate-900"
            >
              {corpus.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.date} ({c.targetRateRange})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg" role="group" aria-label="Sentence scoring engine">
            <button
              type="button"
              onClick={() => setScorerMode('dictionary')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                scorerMode === 'dictionary' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              Dictionary Scorer
            </button>
            <button
              type="button"
              onClick={() => setScorerMode('embedding')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                scorerMode === 'embedding' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              Embedding Axis
            </button>
          </div>
        </div>
      </div>

      {/* Summary KPI Row */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Baseline Tone ({prevRelease.date})</div>
          <div className="text-xl font-bold font-mono tabular-nums text-slate-900 mt-1">
            {diffReport.overallPrevScore >= 0
              ? `+${diffReport.overallPrevScore.toFixed(3)}`
              : diffReport.overallPrevScore.toFixed(3)}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Target: {prevRelease.targetRateRange}</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Current Tone ({currRelease.date})</div>
          <div className="text-xl font-bold font-mono tabular-nums text-slate-900 mt-1">
            {diffReport.overallCurrScore >= 0
              ? `+${diffReport.overallCurrScore.toFixed(3)}`
              : diffReport.overallCurrScore.toFixed(3)}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Target: {currRelease.targetRateRange}</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Full Statement ΔTone</div>
          <div
            className={`text-xl font-bold font-mono tabular-nums mt-1 ${
              diffReport.netToneDelta > 0.03
                ? 'text-rose-700'
                : diffReport.netToneDelta < -0.03
                ? 'text-emerald-700'
                : 'text-slate-900'
            }`}
          >
            {diffReport.netToneDelta >= 0
              ? `+${diffReport.netToneDelta.toFixed(3)}`
              : diffReport.netToneDelta.toFixed(3)}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            {diffReport.netToneDelta > 0.03
              ? 'Hawkish Shift'
              : diffReport.netToneDelta < -0.03
              ? 'Dovish Shift'
              : 'Near-Unchanged'}
          </div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4">
          <div className="text-xs text-slate-500">Edited Sentences ΔTone</div>
          <div className="text-xl font-bold font-mono tabular-nums text-blue-700 mt-1">
            {diffReport.changedSentencesDelta >= 0
              ? `+${diffReport.changedSentencesDelta.toFixed(3)}`
              : diffReport.changedSentencesDelta.toFixed(3)}
          </div>
          <div className="text-[11px] text-slate-500 mt-1">Excludes boilerplate carry-over</div>
        </div>

        <div className="bg-white border border-slate-200 rounded-xl p-4 col-span-2 lg:col-span-1">
          <div className="text-xs text-slate-500">Sentence Turnover Ratio</div>
          <div className="text-xl font-bold font-mono tabular-nums text-slate-900 mt-1">
            {(diffReport.turnoverRatio * 100).toFixed(1)}%
          </div>
          <div className="text-[11px] text-slate-500 mt-1">
            {diffReport.items.filter((i) => i.tag !== 'unchanged').length} of {diffReport.items.length} sentences revised
          </div>
        </div>
      </div>

      {/* Sentence-by-Sentence Diff Table */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <div className="flex flex-wrap items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-semibold text-slate-900">
              Sentence-Level Diff Alignment ({prevRelease.date} → {currRelease.date})
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Inline token changes highlight removed phrasing (strikethrough) and newly inserted policy language (bold underline) alongside sentence-level tone delta.
            </p>
          </div>

          <div className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg">
            <button
              type="button"
              onClick={() => setFilterTag('all')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                filterTag === 'all' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              All Sentences ({diffReport.items.length})
            </button>
            <button
              type="button"
              onClick={() => setFilterTag('changed')}
              className={`px-3 py-1 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                filterTag === 'changed' ? 'bg-white text-slate-900 shadow-xs' : 'text-slate-600'
              }`}
            >
              Changed Only ({diffReport.items.filter((i) => i.tag !== 'unchanged').length})
            </button>
          </div>
        </div>

        <div className="divide-y divide-slate-200 border-t border-slate-200">
          {visibleDiffItems.map((item, idx) => (
            <div key={idx} className="py-4 flex flex-col lg:flex-row lg:items-start justify-between gap-4">
              <div className="space-y-1.5 flex-1">
                <div className="flex items-center gap-2 text-xs font-mono text-slate-500">
                  <span
                    className={`font-semibold uppercase ${
                      item.tag === 'modified'
                        ? 'text-amber-700'
                        : item.tag === 'added'
                        ? 'text-emerald-700'
                        : item.tag === 'removed'
                        ? 'text-rose-700'
                        : 'text-slate-500'
                    }`}
                  >
                    {item.tag}
                  </span>
                  <span>·</span>
                  <span>Similarity: {(item.similarity * 100).toFixed(0)}%</span>
                </div>

                {item.tag === 'modified' && item.wordTokens ? (
                  <p className="text-sm text-slate-800 leading-relaxed">
                    {item.wordTokens.map((tok, tIdx) => {
                      if (tok.type === 'removed') {
                        return (
                          <span
                            key={tIdx}
                            className="line-through text-rose-700 bg-rose-50 px-1 rounded mx-0.5"
                          >
                            {tok.text}
                          </span>
                        );
                      }
                      if (tok.type === 'added') {
                        return (
                          <span
                            key={tIdx}
                            className="font-semibold underline decoration-emerald-600 text-emerald-900 bg-emerald-50 px-1 rounded mx-0.5"
                          >
                            {tok.text}
                          </span>
                        );
                      }
                      return <span key={tIdx}> {tok.text} </span>;
                    })}
                  </p>
                ) : item.tag === 'removed' ? (
                  <p className="text-sm line-through text-rose-800 bg-rose-50/60 px-2 py-1 rounded leading-relaxed">
                    {item.prevSentence}
                  </p>
                ) : item.tag === 'added' ? (
                  <p className="text-sm font-medium text-emerald-950 bg-emerald-50/70 px-2 py-1 rounded leading-relaxed">
                    {item.currSentence}
                  </p>
                ) : (
                  <p className="text-sm text-slate-600 leading-relaxed">{item.currSentence}</p>
                )}
              </div>

              <div className="flex lg:flex-col items-end justify-between shrink-0 gap-2 text-right font-mono tabular-nums">
                <div className="text-xs text-slate-500">
                  t-1: {item.prevScore >= 0 ? `+${item.prevScore.toFixed(2)}` : item.prevScore.toFixed(2)} → t:{' '}
                  {item.currScore >= 0 ? `+${item.currScore.toFixed(2)}` : item.currScore.toFixed(2)}
                </div>
                <div
                  className={`text-sm font-bold ${
                    item.deltaScore > 0.05
                      ? 'text-rose-700'
                      : item.deltaScore < -0.05
                      ? 'text-emerald-700'
                      : 'text-slate-600'
                  }`}
                >
                  ΔTone: {item.deltaScore >= 0 ? `+${item.deltaScore.toFixed(3)}` : item.deltaScore.toFixed(3)}
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Interactive Custom Statement & Live Gemini Rubric Sandbox */}
      <div className="bg-white border border-slate-200 rounded-xl p-6">
        <div className="mb-4">
          <h2 className="text-base font-semibold text-slate-900">
            Interactive FOMC Statement Scorer (Dictionary + Embedding Axis + Gemini Rubric)
          </h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Paste any FOMC draft paragraph or historical statement below to compare all three scoring methods side-by-side.
          </p>
        </div>

        <form onSubmit={handleRunLiveScorer} className="space-y-4">
          <div>
            <label htmlFor="custom-fomc-textarea" className="block text-xs font-medium text-slate-700 mb-1.5">
              FOMC Policy Statement Text
            </label>
            <textarea
              id="custom-fomc-textarea"
              rows={4}
              value={customText}
              onChange={(e) => setCustomText(e.target.value)}
              className="w-full p-3 text-sm bg-slate-50 border border-slate-200 rounded-lg text-slate-900 focus:outline-none focus:border-slate-900 leading-relaxed"
              placeholder="Paste FOMC statement sentences here..."
            />
          </div>

          <div className="flex flex-wrap items-center justify-between gap-4">
            <span className="text-xs text-slate-500 font-mono">{llmStatusNote}</span>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => {
                  const sampleHawk =
                    'Inflation remains elevated, reflecting supply and demand imbalances and broader price pressures. The Committee is highly attentive to inflation risks and anticipates that ongoing increases in the target range will be appropriate.';
                  setCustomText(sampleHawk);
                  setLiveRubricResult(evaluateDeterministicRubric(sampleHawk));
                }}
                className="px-3 py-2 text-xs font-medium text-slate-700 bg-slate-100 rounded-lg hover:bg-slate-200 transition-colors whitespace-nowrap shrink-0"
              >
                Load Hawkish Sample (June 2022)
              </button>
              <button
                type="submit"
                disabled={isScoringLLM}
                className="px-4 py-2 text-xs font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 disabled:opacity-50 transition-colors whitespace-nowrap shrink-0"
              >
                {isScoringLLM ? 'Scoring with Gemini Rubric...' : 'Evaluate Statement Tone'}
              </button>
            </div>
          </div>
        </form>

        {liveRubricResult && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-6 pt-5 border-t border-slate-200">
            <div className="border border-slate-200 rounded-lg p-4">
              <div className="text-xs font-medium text-slate-500">Method A: Dictionary Lexicon</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-slate-900 mt-1">
                {customDictScore.score >= 0
                  ? `+${customDictScore.score.toFixed(3)}`
                  : customDictScore.score.toFixed(3)}
              </div>
              <div className="text-xs text-slate-500 mt-2 font-mono">
                Hawkish wt: {customDictScore.hawkishWeight.toFixed(1)} · Dovish wt:{' '}
                {customDictScore.dovishWeight.toFixed(1)}
              </div>
            </div>

            <div className="border border-slate-200 rounded-lg p-4">
              <div className="text-xs font-medium text-slate-500">Method B: Embedding Axis</div>
              <div className="text-2xl font-bold font-mono tabular-nums text-blue-700 mt-1">
                {customEmbedScore.score >= 0
                  ? `+${customEmbedScore.score.toFixed(3)}`
                  : customEmbedScore.score.toFixed(3)}
              </div>
              <div className="text-xs text-slate-500 mt-2 font-mono">
                Raw cosine proj: {customEmbedScore.rawProjection.toFixed(3)}
              </div>
            </div>

            <div className="border border-slate-200 rounded-lg p-4">
              <div className="text-xs font-medium text-slate-500">
                Method C: Fixed Rubric ({liveRubricResult.model})
              </div>
              <div className="text-2xl font-bold font-mono tabular-nums text-amber-700 mt-1">
                {liveRubricResult.score >= 0
                  ? `+${liveRubricResult.score.toFixed(3)}`
                  : liveRubricResult.score.toFixed(3)}
              </div>
              <p className="text-xs text-slate-600 mt-2 leading-relaxed">
                {liveRubricResult.rationale}
              </p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
