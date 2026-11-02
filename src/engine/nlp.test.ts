import test from 'node:test';
import assert from 'node:assert/strict';
import {
  scoreTextDictionary,
  scoreTextEmbedding,
  computeStatementDiff,
  computeMethodAgreementMatrix,
  evaluateDeterministicRubric,
} from './nlp';
import {
  runEventStudyOLS,
  runLeaveOnePeriodOutAnalysis,
  runPlaceboMonteCarlo,
} from './econometrics';
import { getEnrichedFOMCCorpus, getEventStudyObservations } from '../data/fomcCorpus';

test('Method A (Dictionary) scores hawkish statements positive and dovish statements negative', () => {
  const hawkish = scoreTextDictionary(
    'Inflation remains elevated and upside risks to prices warrant ongoing increases in the target range and additional policy firming.'
  );
  const dovish = scoreTextDictionary(
    'Economic activity has slowed and downside risks warrant a more accommodative stance and lowering the target range.'
  );
  assert.equal(hawkish.score > 0.35, true);
  assert.equal(dovish.score < -0.35, true);
});

test('Method A handles local negation inversion properly', () => {
  const plainHawk = scoreTextDictionary('Further increases in the target range are appropriate.');
  const negatedHawk = scoreTextDictionary('The Committee does not expect further increases in the target range.');
  assert.equal(plainHawk.score > negatedHawk.score, true);
});

test('Method B (Embedding Axis) separates hawkish and dovish anchor semantics', () => {
  const h = scoreTextEmbedding(
    'Tight labor market conditions and persistent inflationary pressures require restrictive policy firming.'
  );
  const d = scoreTextEmbedding(
    'Inflation has eased substantially toward the objective, allowing the Committee to lower the target rate.'
  );
  assert.equal(h.score > d.score, true);
});

test('Inter-method agreement matrix returns valid correlation and kappa metrics', () => {
  const corpus = getEnrichedFOMCCorpus();
  const matrix = computeMethodAgreementMatrix(corpus);
  assert.equal(matrix.length, 3);
  for (const row of matrix) {
    assert.equal(row.pearsonR > 0.65, true);
    assert.equal(row.spearmanRho > 0.65, true);
    assert.equal(row.cohenKappa >= -1 && row.cohenKappa <= 1, true);
  }
});

test('Diff Mode identifies modified, added, and removed sentences and computes tone delta', () => {
  const corpus = getEnrichedFOMCCorpus();
  const prev = corpus.find((c) => c.date === '2024-07-31')!;
  const curr = corpus.find((c) => c.date === '2024-09-18')!;
  const diff = computeStatementDiff(prev.date, prev.statementText, curr.date, curr.statementText, 'dictionary');
  assert.equal(diff.netToneDelta < 0, true);
  assert.equal(diff.turnoverRatio > 0, true);
  assert.equal(diff.items.some((i) => i.tag === 'modified' || i.tag === 'added'), true);
});

test('Econometric OLS event study, Leave-One-Period-Out, and Placebo tests compute accurate statistics', () => {
  const obs = getEventStudyObservations();
  const ols = runEventStudyOLS(obs, 'dgs2', 1, 'llm');
  assert.equal(ols.nObs, obs.length);
  assert.equal(ols.dof, obs.length - 3);
  assert.equal(ols.rSquared > 0 && ols.rSquared < 1, true);
  assert.equal(ols.toneChange.ciLower < ols.toneChange.estimate, true);
  assert.equal(ols.toneChange.ciUpper > ols.toneChange.estimate, true);

  const lopo = runLeaveOnePeriodOutAnalysis(obs, 'dgs2', 1, 'llm');
  assert.equal(lopo.length, 4);

  const placebo = runPlaceboMonteCarlo(obs, 'dgs2', 1, 'llm', 200, 42);
  assert.equal(placebo.iterations, 200);
  assert.equal(placebo.histogramBins.length, 15);
});

test('Deterministic rubric evaluator returns sub-scores and rationale', () => {
  const res = evaluateDeterministicRubric(
    'Inflation has eased over the past year and the Committee decided to lower the target range by 1/2 percentage point.'
  );
  assert.equal(res.score < 0, true);
  assert.equal(res.rationale.length > 15, true);
});
