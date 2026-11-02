/**
 * Pure NLP scoring & sentence diff engine for Fed Whisperer.
 * Implements:
 *  1. Dictionary-based FOMC tone scoring with local negation inversion
 *  2. Semantic embedding-axis projection onto canonical Hawkish-Dovish anchors
 *  3. Deterministic rubric fallback & LLM rubric integration
 *  4. Inter-method agreement (Pearson r, Spearman rho, 3-class Cohen's kappa, delta sign match)
 *  5. Sentence-level SequenceMatcher diff mode & word-level change highlighting
 */

export interface DictionarySentenceScore {
  sentence: string;
  score: number;
  hawkishWeight: number;
  dovishWeight: number;
  matchedHawkish: string[];
  matchedDovish: string[];
}

export interface DictionaryStatementScore {
  score: number;
  hawkishWeight: number;
  dovishWeight: number;
  matchedHawkish: string[];
  matchedDovish: string[];
  sentenceScores: DictionarySentenceScore[];
}

export interface EmbeddingStatementScore {
  score: number;
  rawProjection: number;
  sentenceProjections: { sentence: string; score: number }[];
}

export interface LLMRubricScore {
  score: number;
  inflation_stance: number;
  growth_labor_stance: number;
  forward_guidance_stance: number;
  rationale: string;
  model: string;
}

export interface SentenceDiffItem {
  tag: 'unchanged' | 'modified' | 'added' | 'removed';
  prevSentence: string | null;
  currSentence: string | null;
  prevScore: number;
  currScore: number;
  deltaScore: number;
  similarity: number;
  wordTokens?: { text: string; type: 'same' | 'added' | 'removed' }[];
}

export interface StatementDiffReport {
  prevDate: string;
  currDate: string;
  items: SentenceDiffItem[];
  overallPrevScore: number;
  overallCurrScore: number;
  netToneDelta: number;
  changedSentencesDelta: number;
  turnoverRatio: number;
}

export interface PairwiseAgreement {
  pairLabel: string;
  methodA: string;
  methodB: string;
  pearsonR: number;
  spearmanRho: number;
  cohenKappa: number;
  deltaSignAgreement: number;
}

export const HAWKISH_LEXICON: Record<string, number> = {
  'ongoing increases': 1.8,
  'additional policy firming': 1.8,
  'unacceptably high': 1.7,
  'further increases': 1.6,
  'extremely tight': 1.6,
  'restrictive': 1.6,
  'inflationary pressures': 1.5,
  'tightening': 1.5,
  'tighten': 1.5,
  'overheating': 1.5,
  'upside risks': 1.4,
  'firming': 1.4,
  'attentive to inflation': 1.4,
  'price pressures': 1.3,
  'above target': 1.3,
  'elevated': 1.2,
  'persistent': 1.2,
  'vigilant': 1.2,
  'raise': 1.2,
  'raised': 1.2,
  'strong labor': 1.1,
  'imbalances': 1.1,
  'strongly committed': 1.1,
  'rapid': 1.0,
  'robust': 0.9,
  'higher': 0.8,
  'increases': 0.8,
  'increase': 0.7,
};

export const DOVISH_LEXICON: Record<string, number> = {
  'lower the target range': 1.9,
  'reduce the target range': 1.7,
  'accommodative': 1.6,
  'accommodation': 1.5,
  'transitory': 1.5,
  'downside risks': 1.4,
  'easing': 1.4,
  'disinflation': 1.4,
  'patient': 1.4,
  'cut': 1.4,
  'progress on inflation': 1.4,
  'sustainably toward 2 percent': 1.3,
  'further progress': 1.2,
  'roughly in balance': 1.1,
  'better balance': 1.0,
  'ease': 1.3,
  'eased': 1.3,
  'lowered': 1.3,
  'lowering': 1.3,
  'softened': 1.3,
  'subdued': 1.3,
  'weakness': 1.3,
  'slowed': 1.2,
  'slowing': 1.2,
  'moderated': 1.1,
  'moderating': 1.1,
  'lower': 1.0,
  'reduce': 0.9,
  'support': 0.9,
  'supporting': 0.9,
  'below': 0.9,
  'anchored': 0.8,
  'ample': 0.8,
};

const NEGATION_WORDS = new Set([
  'not',
  'no',
  'never',
  'neither',
  'nor',
  'less',
  'without',
  'hardly',
  'little',
]);

export const HAWKISH_ANCHORS: string[] = [
  'Inflation remains elevated and upside risks to prices require further policy firming and rate increases.',
  'Tight labor market conditions and strong price pressures warrant maintaining a restrictive monetary policy stance.',
  'The Committee is prepared to tighten monetary policy further if inflationary pressures persist above target.',
  'Ongoing increases in the target range for the federal funds rate will be appropriate to return inflation to 2 percent.',
];

export const DOVISH_ANCHORS: string[] = [
  'Economic activity has slowed and downside risks to employment warrant lowering the target range and accommodative policy.',
  'Inflation has eased substantially toward the 2 percent objective, allowing the Committee to reduce policy restraint.',
  'Weakening labor demand and disinflationary pressures support a more accommodative stance of monetary policy.',
  'The Committee will support maximum employment by cutting the target rate and maintaining ample liquidity.',
];

export function splitSentences(text: string): string[] {
  const cleaned = text.replace(/\s+/g, ' ').trim();
  if (!cleaned) return [];
  const raw = cleaned.split(/(?<=[.!?])\s+(?=[A-Z])/);
  return raw
    .map((s) => s.trim())
    .filter(
      (s) =>
        s.length >= 20 &&
        !s.toLowerCase().startsWith('for media inquiries') &&
        !s.toLowerCase().startsWith('implementation note')
    );
}

function escapeRegExp(str: string): string {
  return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function isNegated(prefixText: string, windowSize = 3): boolean {
  const tokens = prefixText.toLowerCase().match(/[a-z]+/g) || [];
  const slice = tokens.slice(Math.max(0, tokens.length - windowSize));
  return slice.some((t) => NEGATION_WORDS.has(t));
}

export function scoreSentenceDictionary(sentence: string): DictionarySentenceScore {
  const lower = sentence.toLowerCase();
  let hWeight = 0;
  let dWeight = 0;
  const matchedHawkish: string[] = [];
  const matchedDovish: string[] = [];

  for (const [phrase, weight] of Object.entries(HAWKISH_LEXICON)) {
    const regex = new RegExp(`\\b${escapeRegExp(phrase)}\\b`, 'g');
    let match: RegExpExecArray | null;
    while ((match = regex.exec(lower)) !== null) {
      const prefix = lower.slice(0, match.index);
      if (isNegated(prefix)) {
        dWeight += weight * 0.8;
        matchedDovish.push(`NOT_${phrase}`);
      } else {
        hWeight += weight;
        matchedHawkish.push(phrase);
      }
    }
  }

  for (const [phrase, weight] of Object.entries(DOVISH_LEXICON)) {
    const regex = new RegExp(`\\b${escapeRegExp(phrase)}\\b`, 'g');
    let match: RegExpExecArray | null;
    while ((match = regex.exec(lower)) !== null) {
      const prefix = lower.slice(0, match.index);
      if (isNegated(prefix)) {
        hWeight += weight * 0.8;
        matchedHawkish.push(`NOT_${phrase}`);
      } else {
        dWeight += weight;
        matchedDovish.push(phrase);
      }
    }
  }

  const denom = hWeight + dWeight;
  const rawNet = denom > 0 ? (hWeight - dWeight) / (denom + 1.5) : 0;
  const score = Number(Math.max(-1, Math.min(1, rawNet)).toFixed(4));

  return {
    sentence,
    score,
    hawkishWeight: Number(hWeight.toFixed(4)),
    dovishWeight: Number(dWeight.toFixed(4)),
    matchedHawkish,
    matchedDovish,
  };
}

export function scoreTextDictionary(text: string): DictionaryStatementScore {
  const sentences = splitSentences(text);
  const targetSentences = sentences.length > 0 ? sentences : text.trim() ? [text.trim()] : [];

  let totalH = 0;
  let totalD = 0;
  const allH: string[] = [];
  const allD: string[] = [];
  const sentenceScores: DictionarySentenceScore[] = [];

  for (const s of targetSentences) {
    const res = scoreSentenceDictionary(s);
    totalH += res.hawkishWeight;
    totalD += res.dovishWeight;
    allH.push(...res.matchedHawkish);
    allD.push(...res.matchedDovish);
    sentenceScores.push(res);
  }

  const denom = totalH + totalD;
  const overall = denom > 0 ? (totalH - totalD) / (denom + 2.0) : 0;

  return {
    score: Number(Math.max(-1, Math.min(1, overall)).toFixed(4)),
    hawkishWeight: Number(totalH.toFixed(4)),
    dovishWeight: Number(totalD.toFixed(4)),
    matchedHawkish: allH,
    matchedDovish: allD,
    sentenceScores,
  };
}

/**
 * Semantic Embedding Axis Scorer (Method B):
 * Builds a high-dimensional n-gram & monetary concept space, embeds the Hawkish and Dovish
 * anchor sets, computes the normalized direction vector v_axis = normalize(mu_hawk - mu_dove),
 * and projects each statement sentence onto v_axis.
 */
const SEMANTIC_CONCEPT_CLUSTERS: { name: string; terms: string[]; polarity: number }[] = [
  {
    name: 'inflation_persistence',
    terms: ['elevated', 'persistent', 'unacceptably', 'pressures', 'upside', 'imbalances', 'above'],
    polarity: 1.0,
  },
  {
    name: 'rate_tightening',
    terms: ['increases', 'ongoing', 'firming', 'raise', 'raised', 'tighten', 'tightening', 'restrictive', 'sufficiently'],
    polarity: 1.15,
  },
  {
    name: 'labor_tightness',
    terms: ['robust', 'strong', 'tight', 'low', 'unemployment', 'solid', 'rapid'],
    polarity: 0.65,
  },
  {
    name: 'inflation_disinflation',
    terms: ['eased', 'easing', 'moderated', 'disinflation', 'subdued', 'transitory', 'anchored', 'closer'],
    polarity: -1.0,
  },
  {
    name: 'policy_accommodation',
    terms: ['accommodative', 'accommodation', 'lower', 'lowered', 'lowering', 'cut', 'reduce', 'restraint', 'patient', 'support'],
    polarity: -1.15,
  },
  {
    name: 'growth_downside',
    terms: ['slowed', 'slowing', 'softened', 'downside', 'weakness', 'moderating', 'uncertain'],
    polarity: -0.8,
  },
];

function extractConceptVector(text: string, vocab: string[]): number[] {
  const lower = text.toLowerCase();
  const words = lower.match(/[a-z]+/g) || [];
  const bigrams: string[] = [];
  for (let i = 0; i < words.length - 1; i++) {
    bigrams.push(`${words[i]} ${words[i + 1]}`);
  }
  const tokenSet = new Set([...words, ...bigrams]);

  const vec: number[] = [];
  for (const term of vocab) {
    vec.push(tokenSet.has(term) ? 1.0 : 0.0);
  }

  // Append semantic concept cluster activations
  for (const cluster of SEMANTIC_CONCEPT_CLUSTERS) {
    let hits = 0;
    for (const t of cluster.terms) {
      if (tokenSet.has(t)) hits += 1;
    }
    vec.push(hits * Math.abs(cluster.polarity));
  }

  let norm = Math.sqrt(vec.reduce((acc, v) => acc + v * v, 0));
  if (norm === 0) norm = 1;
  return vec.map((v) => v / norm);
}

function buildEmbeddingAxis(): { vocab: string[]; axis: number[] } {
  const vocabSet = new Set<string>();
  for (const anchor of [...HAWKISH_ANCHORS, ...DOVISH_ANCHORS]) {
    const words = anchor.toLowerCase().match(/[a-z]+/g) || [];
    words.forEach((w) => {
      if (w.length > 2) vocabSet.add(w);
    });
    for (let i = 0; i < words.length - 1; i++) {
      vocabSet.add(`${words[i]} ${words[i + 1]}`);
    }
  }
  const vocab = Array.from(vocabSet);

  const hVecs = HAWKISH_ANCHORS.map((a) => extractConceptVector(a, vocab));
  const dVecs = DOVISH_ANCHORS.map((a) => extractConceptVector(a, vocab));

  const dim = hVecs[0].length;
  const diff = new Array(dim).fill(0);
  for (let j = 0; j < dim; j++) {
    const hMean = hVecs.reduce((s, v) => s + v[j], 0) / hVecs.length;
    const dMean = dVecs.reduce((s, v) => s + v[j], 0) / dVecs.length;
    diff[j] = hMean - dMean;
  }

  let norm = Math.sqrt(diff.reduce((s, v) => s + v * v, 0));
  if (norm === 0) norm = 1;
  return {
    vocab,
    axis: diff.map((v) => v / norm),
  };
}

const CACHED_AXIS = buildEmbeddingAxis();

export function scoreSentenceEmbedding(sentence: string): number {
  const vec = extractConceptVector(sentence, CACHED_AXIS.vocab);
  let dot = 0;
  for (let i = 0; i < vec.length; i++) {
    dot += vec[i] * CACHED_AXIS.axis[i];
  }
  return Number(Math.tanh(dot * 2.6).toFixed(4));
}

export function scoreTextEmbedding(text: string): EmbeddingStatementScore {
  const sentences = splitSentences(text);
  const target = sentences.length > 0 ? sentences : text.trim() ? [text.trim()] : [];
  if (target.length === 0) {
    return { score: 0, rawProjection: 0, sentenceProjections: [] };
  }

  const sentenceProjections = target.map((s) => ({
    sentence: s,
    score: scoreSentenceEmbedding(s),
  }));

  const meanProj =
    sentenceProjections.reduce((acc, item) => acc + item.score, 0) / sentenceProjections.length;
  const overall = Number(Math.tanh(meanProj * 1.45).toFixed(4));

  return {
    score: overall,
    rawProjection: Number(meanProj.toFixed(4)),
    sentenceProjections,
  };
}

/**
 * Deterministic rubric evaluator for offline/instant preview when evaluating custom statements.
 */
export function evaluateDeterministicRubric(text: string): LLMRubricScore {
  const dict = scoreTextDictionary(text);
  const emb = scoreTextEmbedding(text);
  const blended = Number(Math.max(-1, Math.min(1, 0.55 * dict.score + 0.45 * emb.score)).toFixed(4));

  const lower = text.toLowerCase();
  const infTerms = ['inflation', 'prices', 'price', 'elevated', 'disinflation', '2 percent'];
  const laborTerms = ['employment', 'labor', 'unemployment', 'job', 'activity', 'spending', 'indicators'];
  const guidanceTerms = ['target range', 'federal funds rate', 'committee', 'future', 'ongoing', 'firming', 'restraint'];

  const sentences = splitSentences(text);
  const scoreSub = (keywords: string[]) => {
    const matching = sentences.filter((s) => keywords.some((k) => s.toLowerCase().includes(k)));
    if (matching.length === 0) return blended;
    const avg = matching.reduce((acc, s) => acc + scoreSentenceDictionary(s).score, 0) / matching.length;
    return Number(Math.max(-1, Math.min(1, avg)).toFixed(4));
  };

  const inflation_stance = scoreSub(infTerms);
  const growth_labor_stance = scoreSub(laborTerms);
  const forward_guidance_stance = scoreSub(guidanceTerms);

  const keyPhrases = [...dict.matchedHawkish.slice(0, 2), ...dict.matchedDovish.slice(0, 2)];
  const posture =
    blended >= 0.45
      ? 'Strongly hawkish policy posture'
      : blended >= 0.15
      ? 'Moderately hawkish tilt'
      : blended <= -0.45
      ? 'Strongly dovish accommodation'
      : blended <= -0.15
      ? 'Moderately dovish recalibration'
      : 'Balanced / data-dependent posture';

  const rationale =
    keyPhrases.length > 0
      ? `${posture} (${blended >= 0 ? '+' : ''}${blended.toFixed(2)}), driven by language including "${keyPhrases.join('", "')}"${lower.includes('target range') ? ' alongside explicit federal funds target range guidance.' : '.'}`
      : `${posture} (${blended >= 0 ? '+' : ''}${blended.toFixed(2)}) based on balanced assessment of economic activity and inflation risks.`;

  return {
    score: blended,
    inflation_stance,
    growth_labor_stance,
    forward_guidance_stance,
    rationale,
    model: 'rubric-deterministic-v1',
  };
}

/**
 * Compute word-level diff tokens between two modified sentences.
 */
export function computeWordDiff(
  prevSentence: string,
  currSentence: string
): { text: string; type: 'same' | 'added' | 'removed' }[] {
  const a = prevSentence.trim().split(/\s+/);
  const b = currSentence.trim().split(/\s+/);
  const m = a.length;
  const n = b.length;

  const dp: number[][] = Array.from({ length: m + 1 }, () => new Array(n + 1).fill(0));
  for (let i = m - 1; i >= 0; i--) {
    for (let j = n - 1; j >= 0; j--) {
      if (a[i].toLowerCase() === b[j].toLowerCase()) {
        dp[i][j] = 1 + dp[i + 1][j + 1];
      } else {
        dp[i][j] = Math.max(dp[i + 1][j], dp[i][j + 1]);
      }
    }
  }

  const result: { text: string; type: 'same' | 'added' | 'removed' }[] = [];
  let i = 0;
  let j = 0;
  while (i < m && j < n) {
    if (a[i].toLowerCase() === b[j].toLowerCase()) {
      result.push({ text: b[j], type: 'same' });
      i++;
      j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      result.push({ text: a[i], type: 'removed' });
      i++;
    } else {
      result.push({ text: b[j], type: 'added' });
      j++;
    }
  }
  while (i < m) {
    result.push({ text: a[i++], type: 'removed' });
  }
  while (j < n) {
    result.push({ text: b[j++], type: 'added' });
  }
  return result;
}

function sentenceSimilarity(a: string, b: string): number {
  const wordsA = new Set(a.toLowerCase().match(/[a-z0-9]+/g) || []);
  const wordsB = new Set(b.toLowerCase().match(/[a-z0-9]+/g) || []);
  if (wordsA.size === 0 && wordsB.size === 0) return 1;
  let intersection = 0;
  for (const w of wordsA) {
    if (wordsB.has(w)) intersection++;
  }
  return (2 * intersection) / (wordsA.size + wordsB.size);
}

/**
 * Align and diff two consecutive FOMC statements at the sentence level,
 * highlighting unchanged, modified, added, and removed sentences and scoring the shift.
 */
export function computeStatementDiff(
  prevDate: string,
  prevText: string,
  currDate: string,
  currText: string,
  scorerMode: 'dictionary' | 'embedding' = 'dictionary'
): StatementDiffReport {
  const prevSentences = splitSentences(prevText);
  const currSentences = splitSentences(currText);

  const scoreFn = (s: string): number =>
    scorerMode === 'dictionary' ? scoreSentenceDictionary(s).score : scoreSentenceEmbedding(s);

  const usedCurr = new Set<number>();
  const items: SentenceDiffItem[] = [];

  for (let i = 0; i < prevSentences.length; i++) {
    const pSent = prevSentences[i];
    const pScore = scoreFn(pSent);

    // Find best matching sentence in currSentences
    let bestJ = -1;
    let bestSim = 0;
    for (let j = 0; j < currSentences.length; j++) {
      if (usedCurr.has(j)) continue;
      const sim = sentenceSimilarity(pSent, currSentences[j]);
      if (sim > bestSim) {
        bestSim = sim;
        bestJ = j;
      }
    }

    if (bestJ !== -1 && bestSim >= 0.96) {
      usedCurr.add(bestJ);
      const cSent = currSentences[bestJ];
      const cScore = scoreFn(cSent);
      items.push({
        tag: 'unchanged',
        prevSentence: pSent,
        currSentence: cSent,
        prevScore: pScore,
        currScore: cScore,
        deltaScore: 0,
        similarity: Number(bestSim.toFixed(4)),
      });
    } else if (bestJ !== -1 && bestSim >= 0.42) {
      usedCurr.add(bestJ);
      const cSent = currSentences[bestJ];
      const cScore = scoreFn(cSent);
      items.push({
        tag: 'modified',
        prevSentence: pSent,
        currSentence: cSent,
        prevScore: pScore,
        currScore: cScore,
        deltaScore: Number((cScore - pScore).toFixed(4)),
        similarity: Number(bestSim.toFixed(4)),
        wordTokens: computeWordDiff(pSent, cSent),
      });
    } else {
      items.push({
        tag: 'removed',
        prevSentence: pSent,
        currSentence: null,
        prevScore: pScore,
        currScore: 0,
        deltaScore: Number((-pScore).toFixed(4)),
        similarity: 0,
      });
    }
  }

  for (let j = 0; j < currSentences.length; j++) {
    if (!usedCurr.has(j)) {
      const cSent = currSentences[j];
      const cScore = scoreFn(cSent);
      items.push({
        tag: 'added',
        prevSentence: null,
        currSentence: cSent,
        prevScore: 0,
        currScore: cScore,
        deltaScore: Number(cScore.toFixed(4)),
        similarity: 0,
      });
    }
  }

  const prevValid = items.filter((it) => it.prevSentence !== null);
  const currValid = items.filter((it) => it.currSentence !== null);
  const overallPrevScore =
    prevValid.length > 0
      ? Number((prevValid.reduce((s, it) => s + it.prevScore, 0) / prevValid.length).toFixed(4))
      : 0;
  const overallCurrScore =
    currValid.length > 0
      ? Number((currValid.reduce((s, it) => s + it.currScore, 0) / currValid.length).toFixed(4))
      : 0;

  const changedItems = items.filter((it) => it.tag !== 'unchanged');
  const changedSentencesDelta =
    changedItems.length > 0
      ? Number(
          (changedItems.reduce((s, it) => s + it.deltaScore, 0) / changedItems.length).toFixed(4)
        )
      : 0;

  const turnoverRatio =
    items.length > 0 ? Number((changedItems.length / items.length).toFixed(4)) : 0;

  return {
    prevDate,
    currDate,
    items,
    overallPrevScore,
    overallCurrScore,
    netToneDelta: Number((overallCurrScore - overallPrevScore).toFixed(4)),
    changedSentencesDelta,
    turnoverRatio,
  };
}

/**
 * Inter-method Agreement Calculator:
 * Computes Pearson r, Spearman rank rho, 3-class Cohen's kappa, and first-difference directional agreement.
 */
function pearsonCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n < 2) return 0;
  const mx = x.reduce((a, b) => a + b, 0) / n;
  const my = y.reduce((a, b) => a + b, 0) / n;
  let num = 0;
  let dx2 = 0;
  let dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = x[i] - mx;
    const dy = y[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom > 0 ? num / denom : 0;
}

function rankArray(arr: number[]): number[] {
  const indexed = arr.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const ranks = new Array(arr.length).fill(0);
  let i = 0;
  while (i < indexed.length) {
    let j = i;
    while (j + 1 < indexed.length && indexed[j + 1].v === indexed[i].v) {
      j++;
    }
    const avgRank = (i + j) / 2 + 1;
    for (let k = i; k <= j; k++) {
      ranks[indexed[k].i] = avgRank;
    }
    i = j + 1;
  }
  return ranks;
}

function cohenKappa3Class(x: number[], y: number[], threshold = 0.12): number {
  const n = x.length;
  if (n === 0) return 0;
  const toClass = (v: number) => (v > threshold ? 2 : v < -threshold ? 0 : 1);
  const cx = x.map(toClass);
  const cy = y.map(toClass);

  const countsA = [0, 0, 0];
  const countsB = [0, 0, 0];
  let agree = 0;
  for (let i = 0; i < n; i++) {
    if (cx[i] === cy[i]) agree++;
    countsA[cx[i]]++;
    countsB[cy[i]]++;
  }
  const p0 = agree / n;
  let pe = 0;
  for (let k = 0; k < 3; k++) {
    pe += (countsA[k] / n) * (countsB[k] / n);
  }
  return pe < 1 ? (p0 - pe) / (1 - pe) : 1;
}

export function computeMethodAgreementMatrix(
  records: { dictScore: number; embedScore: number; llmScore: number }[]
): PairwiseAgreement[] {
  const dict = records.map((r) => r.dictScore);
  const embed = records.map((r) => r.embedScore);
  const llm = records.map((r) => r.llmScore);

  const pairs: { label: string; aName: string; bName: string; a: number[]; b: number[] }[] = [
    { label: 'Dictionary vs. Embedding Axis', aName: 'Dictionary (A)', bName: 'Embedding Axis (B)', a: dict, b: embed },
    { label: 'Dictionary vs. LLM Rubric', aName: 'Dictionary (A)', bName: 'LLM Rubric (C)', a: dict, b: llm },
    { label: 'Embedding Axis vs. LLM Rubric', aName: 'Embedding Axis (B)', bName: 'LLM Rubric (C)', a: embed, b: llm },
  ];

  return pairs.map(({ label, aName, bName, a, b }) => {
    const r = pearsonCorrelation(a, b);
    const rho = pearsonCorrelation(rankArray(a), rankArray(b));
    const kappa = cohenKappa3Class(a, b);

    let signMatches = 0;
    const diffs = Math.max(0, a.length - 1);
    for (let i = 1; i < a.length; i++) {
      const da = Math.sign(a[i] - a[i - 1]);
      const db = Math.sign(b[i] - b[i - 1]);
      if (da === db) signMatches++;
    }
    const deltaSignAgreement = diffs > 0 ? signMatches / diffs : 0;

    return {
      pairLabel: label,
      methodA: aName,
      methodB: bName,
      pearsonR: Number(r.toFixed(3)),
      spearmanRho: Number(rho.toFixed(3)),
      cohenKappa: Number(kappa.toFixed(3)),
      deltaSignAgreement: Number(deltaSignAgreement.toFixed(3)),
    };
  });
}
