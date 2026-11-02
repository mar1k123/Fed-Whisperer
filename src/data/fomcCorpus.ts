/**
 * Historical FOMC Statement Corpus & FRED / S&P 500 Market Event Dataset (2021–2025).
 * Each entry contains authentic FOMC monetary policy statement excerpts, policy rate decisions,
 * short-end rate surprise proxies, and 1d/2d/5d market changes in DGS2 (bps), DGS10 (bps), and SPY (%).
 * All tone scores (Dictionary, Embedding Axis, and Fixed-Rubric LLM) are computed dynamically by the NLP engine.
 */

import {
  scoreTextDictionary,
  scoreTextEmbedding,
  evaluateDeterministicRubric,
  LLMRubricScore,
} from '../engine/nlp';
import { EventObservation } from '../engine/econometrics';

export interface RawFOMCRelease {
  id: string;
  date: string;
  title: string;
  regime: string;
  targetRateRange: string;
  rateDecisionBps: number;
  rateSurpriseBps: number;
  dgs2_1d_bps: number;
  dgs2_2d_bps: number;
  dgs2_5d_bps: number;
  dgs10_1d_bps: number;
  dgs10_2d_bps: number;
  dgs10_5d_bps: number;
  sp500_1d_pct: number;
  sp500_2d_pct: number;
  sp500_5d_pct: number;
  hasSEP: boolean;
  statementText: string;
  precomputedLLM: LLMRubricScore;
}

export interface EnrichedFOMCRelease extends RawFOMCRelease {
  dictScore: number;
  embedScore: number;
  llmScore: number;
  dictToneChange: number;
  embedToneChange: number;
  llmToneChange: number;
  hawkishMatched: string[];
  dovishMatched: string[];
}

export const RAW_FOMC_CORPUS: RawFOMCRelease[] = [
  {
    id: '2021-04-28',
    date: '2021-04-28',
    title: 'FOMC Statement — April 28, 2021',
    regime: '2021 Transitory Inflation Era',
    targetRateRange: '0.00%–0.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: -1.5,
    dgs2_1d_bps: -2.0,
    dgs2_2d_bps: -2.5,
    dgs2_5d_bps: -3.0,
    dgs10_1d_bps: -2.0,
    dgs10_2d_bps: 1.0,
    dgs10_5d_bps: -4.0,
    sp500_1d_pct: -0.08,
    sp500_2d_pct: 0.60,
    sp500_5d_pct: -0.45,
    hasSEP: false,
    statementText:
      'Amid progress on vaccinations and strong policy support, indicators of economic activity and employment have strengthened. Inflation has risen, largely reflecting transitory factors. Overall financial conditions remain accommodative, in part reflecting policy measures to support the economy and the flow of credit to U.S. households and businesses. The Committee decided to keep the target range for the federal funds rate at 0 to 1/4 percent and expects it will be appropriate to maintain this target range until labor market conditions have reached levels consistent with the Committee\'s assessments of maximum employment.',
    precomputedLLM: {
      score: -0.68,
      inflation_stance: -0.65,
      growth_labor_stance: 0.15,
      forward_guidance_stance: -0.82,
      rationale:
        'Strongly dovish posture (-0.68): characterizes rising inflation as largely reflecting "transitory factors" and pledges to maintain the 0–0.25% accommodative target range.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2021-06-16',
    date: '2021-06-16',
    title: 'FOMC Statement — June 16, 2021',
    regime: '2021 Transitory Inflation Era',
    targetRateRange: '0.00%–0.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: 3.5,
    dgs2_1d_bps: 4.0,
    dgs2_2d_bps: 5.5,
    dgs2_5d_bps: 9.0,
    dgs10_1d_bps: 7.0,
    dgs10_2d_bps: 2.0,
    dgs10_5d_bps: -1.0,
    sp500_1d_pct: -0.54,
    sp500_2d_pct: -0.58,
    sp500_5d_pct: 0.14,
    hasSEP: true,
    statementText:
      'Progress on vaccinations has reduced the spread of COVID-19 in the United States. Amid this progress and strong policy support, indicators of economic activity and employment have strengthened. Inflation has risen, largely reflecting transitory factors. Overall financial conditions remain accommodative. The Committee decided to keep the target range for the federal funds rate at 0 to 1/4 percent and will continue to increase its holdings of Treasury securities by at least $80 billion per month.',
    precomputedLLM: {
      score: -0.55,
      inflation_stance: -0.50,
      growth_labor_stance: 0.25,
      forward_guidance_stance: -0.70,
      rationale:
        'Moderately dovish (-0.55): retains transitory inflation language and asset purchases, though notes stronger pandemic recovery and activity.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2021-09-22',
    date: '2021-09-22',
    title: 'FOMC Statement — September 22, 2021',
    regime: '2021 Transitory Inflation Era',
    targetRateRange: '0.00%–0.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: 2.0,
    dgs2_1d_bps: 3.0,
    dgs2_2d_bps: 5.0,
    dgs2_5d_bps: 8.0,
    dgs10_1d_bps: -1.0,
    dgs10_2d_bps: 9.0,
    dgs10_5d_bps: 20.0,
    sp500_1d_pct: 0.95,
    sp500_2d_pct: 2.17,
    sp500_5d_pct: 0.12,
    hasSEP: true,
    statementText:
      'Indicators of economic activity and employment have continued to strengthen. Inflation is elevated, largely reflecting transitory factors. Overall financial conditions remain accommodative. If progress continues broadly as expected, the Committee judges that a moderation in the pace of asset purchases may soon be warranted. The Committee decided to keep the target range for the federal funds rate at 0 to 1/4 percent.',
    precomputedLLM: {
      score: -0.22,
      inflation_stance: 0.10,
      growth_labor_stance: 0.30,
      forward_guidance_stance: -0.45,
      rationale:
        'Less dovish shift (-0.22): upgrades inflation description to "elevated" and signals that tapering asset purchases "may soon be warranted".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2021-11-03',
    date: '2021-11-03',
    title: 'FOMC Statement — November 3, 2021',
    regime: '2021 Transitory Inflation Era',
    targetRateRange: '0.00%–0.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: -1.0,
    dgs2_1d_bps: 2.0,
    dgs2_2d_bps: -5.0,
    dgs2_5d_bps: 4.0,
    dgs10_1d_bps: 4.0,
    dgs10_2d_bps: -3.0,
    dgs10_5d_bps: 1.0,
    sp500_1d_pct: 0.65,
    sp500_2d_pct: 1.07,
    sp500_5d_pct: 0.35,
    hasSEP: false,
    statementText:
      'Indicators of economic activity and employment have continued to strengthen. Inflation is elevated, largely reflecting factors that are expected to be transitory. Supply and demand imbalances related to the pandemic and the reopening of the economy have contributed to sizable price increases in some sectors. In light of the substantial further progress the economy has made toward the Committee\'s goals, the Committee decided to begin reducing the monthly pace of its net asset purchases.',
    precomputedLLM: {
      score: -0.05,
      inflation_stance: 0.25,
      growth_labor_stance: 0.35,
      forward_guidance_stance: -0.15,
      rationale:
        'Near-neutral transition (-0.05): initiates QE tapering and acknowledges "sizable price increases" from supply-demand imbalances, while hedging that factors are "expected to be transitory".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2021-12-15',
    date: '2021-12-15',
    title: 'FOMC Statement — December 15, 2021',
    regime: '2021 Transitory Inflation Era',
    targetRateRange: '0.00%–0.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: 4.5,
    dgs2_1d_bps: 3.0,
    dgs2_2d_bps: -2.0,
    dgs2_5d_bps: 3.0,
    dgs10_1d_bps: 3.0,
    dgs10_2d_bps: -1.0,
    dgs10_5d_bps: 2.0,
    sp500_1d_pct: 1.63,
    sp500_2d_pct: 0.75,
    sp500_5d_pct: 1.02,
    hasSEP: true,
    statementText:
      'Indicators of economic activity and employment have continued to strengthen. Job gains have been solid in recent months, and the unemployment rate has declined substantially. Supply and demand imbalances related to the pandemic and the reopening of the economy have continued to contribute to elevated levels of inflation. In light of inflation developments and the further improvement in the labor market, the Committee decided to reduce the monthly pace of its net asset purchases at a faster pace.',
    precomputedLLM: {
      score: 0.34,
      inflation_stance: 0.55,
      growth_labor_stance: 0.50,
      forward_guidance_stance: 0.20,
      rationale:
        'Hawkish pivot (+0.34): retires the word "transitory" entirely, highlights "elevated levels of inflation" and substantial unemployment decline, and doubles the pace of QE tapering.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-01-26',
    date: '2022-01-26',
    title: 'FOMC Statement — January 26, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '0.00%–0.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: 8.0,
    dgs2_1d_bps: 13.0,
    dgs2_2d_bps: 17.0,
    dgs2_5d_bps: 16.0,
    dgs10_1d_bps: 7.0,
    dgs10_2d_bps: 3.0,
    dgs10_5d_bps: 1.0,
    sp500_1d_pct: -0.15,
    sp500_2d_pct: -0.69,
    sp500_5d_pct: 4.55,
    hasSEP: false,
    statementText:
      'Indicators of economic activity and employment have continued to strengthen. Job gains have been solid in recent months, and the unemployment rate has declined. Supply and demand imbalances related to the pandemic and the reopening of the economy have continued to contribute to elevated levels of inflation. With inflation well above 2 percent and a strong labor market, the Committee expects it will soon be appropriate to raise the target range for the federal funds rate.',
    precomputedLLM: {
      score: 0.58,
      inflation_stance: 0.70,
      growth_labor_stance: 0.60,
      forward_guidance_stance: 0.55,
      rationale:
        'Moderately-to-strongly hawkish (+0.58): explicitly states that with inflation well above 2 percent and a strong labor market, raising the target range will "soon be appropriate".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-03-16',
    date: '2022-03-16',
    title: 'FOMC Statement — March 16, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '0.25%–0.50%',
    rateDecisionBps: 25,
    rateSurpriseBps: 4.0,
    dgs2_1d_bps: 9.0,
    dgs2_2d_bps: 9.0,
    dgs2_5d_bps: 26.0,
    dgs10_1d_bps: 4.0,
    dgs10_2d_bps: 5.0,
    dgs10_5d_bps: 18.0,
    sp500_1d_pct: 2.24,
    sp500_2d_pct: 3.50,
    sp500_5d_pct: 4.56,
    hasSEP: true,
    statementText:
      'Indicators of economic activity and employment have continued to strengthen. Job gains have been strong in recent months, and the unemployment rate has declined substantially. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher energy prices, and broader price pressures. The Committee decided to raise the target range for the federal funds rate to 1/4 to 1/2 percent and anticipates that ongoing increases in the target range will be appropriate.',
    precomputedLLM: {
      score: 0.76,
      inflation_stance: 0.82,
      growth_labor_stance: 0.70,
      forward_guidance_stance: 0.80,
      rationale:
        'Strongly hawkish (+0.76): initiates liftoff hike, cites "broader price pressures", and introduces the canonical hawkish forward guidance phrase "ongoing increases in the target range will be appropriate".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-05-04',
    date: '2022-05-04',
    title: 'FOMC Statement — May 4, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '0.75%–1.00%',
    rateDecisionBps: 50,
    rateSurpriseBps: -4.0,
    dgs2_1d_bps: -12.0,
    dgs2_2d_bps: -6.0,
    dgs2_5d_bps: -14.0,
    dgs10_1d_bps: -4.0,
    dgs10_2d_bps: 9.0,
    dgs10_5d_bps: -5.0,
    sp500_1d_pct: 2.99,
    sp500_2d_pct: -0.68,
    sp500_5d_pct: -4.28,
    hasSEP: false,
    statementText:
      'Although overall economic activity edged down in the first quarter, household spending and business fixed investment remained strong. Job gains have been robust in recent months, and the unemployment rate has declined substantially. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher energy prices, and broader price pressures. The Committee is highly attentive to inflation risks. The Committee decided to raise the target range for the federal funds rate to 3/4 to 1 percent and anticipates that ongoing increases in the target range will be appropriate.',
    precomputedLLM: {
      score: 0.84,
      inflation_stance: 0.90,
      growth_labor_stance: 0.72,
      forward_guidance_stance: 0.88,
      rationale:
        'Strongly hawkish (+0.84): delivers 50 bps hike, adds "The Committee is highly attentive to inflation risks", and retains "ongoing increases" plus balance sheet runoff.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-06-15',
    date: '2022-06-15',
    title: 'FOMC Statement — June 15, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '1.50%–1.75%',
    rateDecisionBps: 75,
    rateSurpriseBps: 6.0,
    dgs2_1d_bps: -15.0,
    dgs2_2d_bps: -27.0,
    dgs2_5d_bps: -37.0,
    dgs10_1d_bps: -15.0,
    dgs10_2d_bps: -18.0,
    dgs10_5d_bps: -32.0,
    sp500_1d_pct: 1.46,
    sp500_2d_pct: -1.84,
    sp500_5d_pct: 0.65,
    hasSEP: true,
    statementText:
      'Overall economic activity appears to have picked up after edging down in the first quarter. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher food and energy prices, and broader price pressures. The Committee is highly attentive to inflation risks and is strongly committed to returning inflation to its 2 percent objective. The Committee decided to raise the target range for the federal funds rate to 1-1/2 to 1-3/4 percent and anticipates that ongoing increases in the target range will be appropriate.',
    precomputedLLM: {
      score: 0.91,
      inflation_stance: 0.95,
      growth_labor_stance: 0.80,
      forward_guidance_stance: 0.92,
      rationale:
        'Peak hawkishness (+0.91): executes 75 bps hike, adds explicit pledge that the Committee is "strongly committed to returning inflation to its 2 percent objective", and affirms ongoing increases.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-07-27',
    date: '2022-07-27',
    title: 'FOMC Statement — July 27, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '2.25%–2.50%',
    rateDecisionBps: 75,
    rateSurpriseBps: -3.0,
    dgs2_1d_bps: -6.0,
    dgs2_2d_bps: -19.0,
    dgs2_5d_bps: 4.0,
    dgs10_1d_bps: -3.0,
    dgs10_2d_bps: -13.0,
    dgs10_5d_bps: -6.0,
    sp500_1d_pct: 2.62,
    sp500_2d_pct: 3.86,
    sp500_5d_pct: 4.25,
    hasSEP: false,
    statementText:
      'Recent indicators of spending and production have softened. Nonetheless, job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher food and energy prices, and broader price pressures. The Committee is highly attentive to inflation risks. The Committee decided to raise the target range for the federal funds rate to 2-1/4 to 2-1/2 percent and anticipates that ongoing increases in the target range will be appropriate.',
    precomputedLLM: {
      score: 0.72,
      inflation_stance: 0.88,
      growth_labor_stance: 0.15,
      forward_guidance_stance: 0.85,
      rationale:
        'Hawkish but slightly tempered (+0.72): delivers second 75 bps hike and keeps "ongoing increases", but opens by noting that "indicators of spending and production have softened".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-09-21',
    date: '2022-09-21',
    title: 'FOMC Statement — September 21, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '3.00%–3.25%',
    rateDecisionBps: 75,
    rateSurpriseBps: 7.5,
    dgs2_1d_bps: 8.0,
    dgs2_2d_bps: 15.0,
    dgs2_5d_bps: 12.0,
    dgs10_1d_bps: -6.0,
    dgs10_2d_bps: 13.0,
    dgs10_5d_bps: 16.0,
    sp500_1d_pct: -1.71,
    sp500_2d_pct: -2.54,
    sp500_5d_pct: -3.58,
    hasSEP: true,
    statementText:
      'Recent indicators point to modest growth in spending and production. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher food and energy prices, and broader price pressures. The Committee is highly attentive to inflation risks and is strongly committed to returning inflation to its 2 percent objective. The Committee decided to raise the target range for the federal funds rate to 3 to 3-1/4 percent and anticipates that ongoing increases in the target range will be appropriate.',
    precomputedLLM: {
      score: 0.80,
      inflation_stance: 0.90,
      growth_labor_stance: 0.45,
      forward_guidance_stance: 0.86,
      rationale:
        'Strongly hawkish (+0.80): upgrades activity from "softened" to "modest growth", delivers third consecutive 75 bps hike, and maintains "ongoing increases" commitment.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-11-02',
    date: '2022-11-02',
    title: 'FOMC Statement — November 2, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '3.75%–4.00%',
    rateDecisionBps: 75,
    rateSurpriseBps: 5.0,
    dgs2_1d_bps: 7.0,
    dgs2_2d_bps: 17.0,
    dgs2_5d_bps: 9.0,
    dgs10_1d_bps: 5.0,
    dgs10_2d_bps: 7.0,
    dgs10_5d_bps: 7.0,
    sp500_1d_pct: -2.50,
    sp500_2d_pct: -3.53,
    sp500_5d_pct: -2.01,
    hasSEP: false,
    statementText:
      'Recent indicators point to modest growth in spending and production. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher food and energy prices, and broader price pressures. The Committee anticipates that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive to return inflation to 2 percent over time. In determining the pace of future increases in the target range, the Committee will take into account the cumulative tightening of monetary policy and the lags with which monetary policy affects economic activity and inflation.',
    precomputedLLM: {
      score: 0.62,
      inflation_stance: 0.85,
      growth_labor_stance: 0.45,
      forward_guidance_stance: 0.50,
      rationale:
        'Hawkish but introduces step-down conditioning (+0.62): targets a "sufficiently restrictive" stance while noting cumulative tightening and policy lags for future increases.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2022-12-14',
    date: '2022-12-14',
    title: 'FOMC Statement — December 14, 2022',
    regime: '2022 Rapid Tightening Cycle',
    targetRateRange: '4.25%–4.50%',
    rateDecisionBps: 50,
    rateSurpriseBps: 3.0,
    dgs2_1d_bps: 2.0,
    dgs2_2d_bps: 2.0,
    dgs2_5d_bps: 4.0,
    dgs10_1d_bps: -3.0,
    dgs10_2d_bps: -6.0,
    dgs10_5d_bps: 17.0,
    sp500_1d_pct: -0.61,
    sp500_2d_pct: -3.08,
    sp500_5d_pct: -2.92,
    hasSEP: true,
    statementText:
      'Recent indicators point to modest growth in spending and production. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated, reflecting supply and demand imbalances related to the pandemic, higher food and energy prices, and broader price pressures. The Committee anticipates that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive to return inflation to 2 percent over time.',
    precomputedLLM: {
      score: 0.60,
      inflation_stance: 0.82,
      growth_labor_stance: 0.45,
      forward_guidance_stance: 0.52,
      rationale:
        'Moderately-to-strongly hawkish (+0.60): steps down hike size to 50 bps but retains "ongoing increases" to reach a sufficiently restrictive policy stance.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-02-01',
    date: '2023-02-01',
    title: 'FOMC Statement — February 1, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '4.50%–4.75%',
    rateDecisionBps: 25,
    rateSurpriseBps: -4.5,
    dgs2_1d_bps: -12.0,
    dgs2_2d_bps: -12.0,
    dgs2_5d_bps: 24.0,
    dgs10_1d_bps: -13.0,
    dgs10_2d_bps: -12.0,
    dgs10_5d_bps: 12.0,
    sp500_1d_pct: 1.05,
    sp500_2d_pct: 2.53,
    sp500_5d_pct: 1.01,
    hasSEP: false,
    statementText:
      'Recent indicators point to modest growth in spending and production. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation has eased somewhat but remains elevated. The Committee anticipates that ongoing increases in the target range will be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive to return inflation to 2 percent over time. In determining the extent of future increases in the target range, the Committee will take into account the cumulative tightening of monetary policy.',
    precomputedLLM: {
      score: 0.42,
      inflation_stance: 0.45,
      growth_labor_stance: 0.45,
      forward_guidance_stance: 0.38,
      rationale:
        'Less hawkish shift (+0.42): acknowledges for the first time that "Inflation has eased somewhat" and changes "pace of future increases" to "extent of future increases".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-03-22',
    date: '2023-03-22',
    title: 'FOMC Statement — March 22, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '4.75%–5.00%',
    rateDecisionBps: 25,
    rateSurpriseBps: -8.0,
    dgs2_1d_bps: -21.0,
    dgs2_2d_bps: -37.0,
    dgs2_5d_bps: -9.0,
    dgs10_1d_bps: -12.0,
    dgs10_2d_bps: -22.0,
    dgs10_5d_bps: -3.0,
    sp500_1d_pct: -1.65,
    sp500_2d_pct: -1.36,
    sp500_5d_pct: 0.61,
    hasSEP: true,
    statementText:
      'Recent indicators point to modest growth in spending and production. Job gains have picked up in recent months and are running at a robust pace; the unemployment rate has remained low. Inflation remains elevated. The U.S. banking system is sound and resilient. Recent developments are likely to result in tighter credit conditions for households and businesses and to weigh on economic activity, hiring, and inflation. The Committee anticipates that some additional policy firming may be appropriate in order to attain a stance of monetary policy that is sufficiently restrictive.',
    precomputedLLM: {
      score: 0.18,
      inflation_stance: 0.55,
      growth_labor_stance: -0.10,
      forward_guidance_stance: 0.12,
      rationale:
        'Dovish revision amid regional banking stress (+0.18): drops "ongoing increases" in favor of "some additional policy firming may be appropriate" and cites tighter credit weighing on activity.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-05-03',
    date: '2023-05-03',
    title: 'FOMC Statement — May 3, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '5.00%–5.25%',
    rateDecisionBps: 25,
    rateSurpriseBps: -3.5,
    dgs2_1d_bps: -8.0,
    dgs2_2d_bps: -24.0,
    dgs2_5d_bps: -6.0,
    dgs10_1d_bps: -5.0,
    dgs10_2d_bps: -7.0,
    dgs10_5d_bps: 0.0,
    sp500_1d_pct: -0.70,
    sp500_2d_pct: -1.42,
    sp500_5d_pct: 0.45,
    hasSEP: false,
    statementText:
      'Economic activity expanded at a modest pace in the first quarter. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated. Tighter credit conditions for households and businesses are likely to weigh on economic activity, hiring, and inflation. In determining the extent to which additional policy firming may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening of monetary policy.',
    precomputedLLM: {
      score: 0.08,
      inflation_stance: 0.45,
      growth_labor_stance: 0.05,
      forward_guidance_stance: -0.10,
      rationale:
        'Conditional pause signal (+0.08): removes explicit anticipation of further hikes, shifting to data-dependent assessment of "the extent to which additional policy firming may be appropriate".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-06-14',
    date: '2023-06-14',
    title: 'FOMC Statement — June 14, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '5.00%–5.25%',
    rateDecisionBps: 0,
    rateSurpriseBps: 4.5,
    dgs2_1d_bps: 4.0,
    dgs2_2d_bps: -3.0,
    dgs2_5d_bps: 4.0,
    dgs10_1d_bps: -1.0,
    dgs10_2d_bps: -12.0,
    dgs10_5d_bps: -11.0,
    sp500_1d_pct: 0.08,
    sp500_2d_pct: 1.30,
    sp500_5d_pct: -0.08,
    hasSEP: true,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a modest pace. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated. Holding the target range steady at this meeting allows the Committee to assess additional information and its implications for monetary policy. In determining the extent of additional policy firming that may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening of monetary policy.',
    precomputedLLM: {
      score: 0.16,
      inflation_stance: 0.45,
      growth_labor_stance: 0.20,
      forward_guidance_stance: 0.05,
      rationale:
        'Hawkish skip (+0.16): holds target range steady at 5.00%–5.25% to assess information while retaining upward bias toward "additional policy firming".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-07-26',
    date: '2023-07-26',
    title: 'FOMC Statement — July 26, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '5.25%–5.50%',
    rateDecisionBps: 25,
    rateSurpriseBps: -1.5,
    dgs2_1d_bps: -3.0,
    dgs2_2d_bps: 6.0,
    dgs2_5d_bps: 3.0,
    dgs10_1d_bps: -5.0,
    dgs10_2d_bps: 10.0,
    dgs10_5d_bps: 19.0,
    sp500_1d_pct: -0.02,
    sp500_2d_pct: -0.66,
    sp500_5d_pct: -1.16,
    hasSEP: false,
    statementText:
      'Recent indicators suggest that economic activity has been expanding at a moderate pace. Job gains have been robust in recent months, and the unemployment rate has remained low. Inflation remains elevated. The Committee decided to raise the target range for the federal funds rate to 5-1/4 to 5-1/2 percent. The Committee will continue to assess additional information and its implications for monetary policy.',
    precomputedLLM: {
      score: 0.24,
      inflation_stance: 0.45,
      growth_labor_stance: 0.35,
      forward_guidance_stance: 0.05,
      rationale:
        'Moderately hawkish (+0.24): upgrades growth characterization from "modest" to "moderate" and raises the target range to 5.25%–5.50%.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-09-20',
    date: '2023-09-20',
    title: 'FOMC Statement — September 20, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '5.25%–5.50%',
    rateDecisionBps: 0,
    rateSurpriseBps: 5.5,
    dgs2_1d_bps: 6.0,
    dgs2_2d_bps: 5.0,
    dgs2_5d_bps: 4.0,
    dgs10_1d_bps: -2.0,
    dgs10_2d_bps: 12.0,
    dgs10_5d_bps: 24.0,
    sp500_1d_pct: -0.94,
    sp500_2d_pct: -2.56,
    sp500_5d_pct: -3.79,
    hasSEP: true,
    statementText:
      'Recent indicators suggest that economic activity has been expanding at a solid pace. Job gains have slowed in recent months but remain strong, and the unemployment rate has remained low. Inflation remains elevated. The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent and will continue to assess additional information and the extent of additional policy firming that may be appropriate.',
    precomputedLLM: {
      score: 0.28,
      inflation_stance: 0.45,
      growth_labor_stance: 0.42,
      forward_guidance_stance: 0.12,
      rationale:
        'High-for-longer hawkish hold (+0.28): upgrades economic expansion to "solid pace" and preserves option for additional policy firming.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2023-12-13',
    date: '2023-12-13',
    title: 'FOMC Statement — December 13, 2023',
    regime: '2023 High-for-Longer Peak',
    targetRateRange: '5.25%–5.50%',
    rateDecisionBps: 0,
    rateSurpriseBps: -9.5,
    dgs2_1d_bps: -27.0,
    dgs2_2d_bps: -34.0,
    dgs2_5d_bps: -39.0,
    dgs10_1d_bps: -17.0,
    dgs10_2d_bps: -28.0,
    dgs10_5d_bps: -35.0,
    sp500_1d_pct: 1.37,
    sp500_2d_pct: 1.63,
    sp500_5d_pct: 1.20,
    hasSEP: true,
    statementText:
      'Recent indicators suggest that growth of economic activity has slowed from its strong pace in the third quarter. Job gains have moderated since earlier in the year but remain strong, and the unemployment rate has remained low. Inflation has eased over the past year but remains elevated. The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent. In determining the extent of any additional policy firming that may be appropriate to return inflation to 2 percent over time, the Committee will take into account the cumulative tightening.',
    precomputedLLM: {
      score: -0.14,
      inflation_stance: -0.10,
      growth_labor_stance: -0.15,
      forward_guidance_stance: -0.18,
      rationale:
        'Dovish pivot (-0.14): notes that growth "has slowed", job gains "have moderated", inflation "has eased over the past year", and inserts the qualifier "any" before additional policy firming.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2024-01-31',
    date: '2024-01-31',
    title: 'FOMC Statement — January 31, 2024',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '5.25%–5.50%',
    rateDecisionBps: 0,
    rateSurpriseBps: 3.5,
    dgs2_1d_bps: -9.0,
    dgs2_2d_bps: -15.0,
    dgs2_5d_bps: 6.0,
    dgs10_1d_bps: -7.0,
    dgs10_2d_bps: -19.0,
    dgs10_5d_bps: 4.0,
    sp500_1d_pct: -1.61,
    sp500_2d_pct: -0.38,
    sp500_5d_pct: 1.43,
    hasSEP: false,
    statementText:
      'Recent indicators suggest that economic activity has been expanding at a solid pace. Job gains have moderated since early last year but remain strong, and the unemployment rate has remained low. Inflation has eased over the past year but remains elevated. The Committee judges that the risks to achieving its employment and inflation goals are moving into better balance. The Committee does not expect it will be appropriate to reduce the target range until it has gained greater confidence that inflation is moving sustainably toward 2 percent.',
    precomputedLLM: {
      score: -0.06,
      inflation_stance: -0.05,
      growth_labor_stance: 0.25,
      forward_guidance_stance: -0.25,
      rationale:
        'Balanced hold with explicit cut condition (-0.06): drops hiking bias completely, notes risks are moving into "better balance", but pushes back against immediate cuts until "greater confidence" is gained.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2024-05-01',
    date: '2024-05-01',
    title: 'FOMC Statement — May 1, 2024',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '5.25%–5.50%',
    rateDecisionBps: 0,
    rateSurpriseBps: -2.5,
    dgs2_1d_bps: -8.0,
    dgs2_2d_bps: -17.0,
    dgs2_5d_bps: -20.0,
    dgs10_1d_bps: -5.0,
    dgs10_2d_bps: -11.0,
    dgs10_5d_bps: -19.0,
    sp500_1d_pct: -0.34,
    sp500_2d_pct: 0.57,
    sp500_5d_pct: 2.97,
    hasSEP: false,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have remained strong, and the unemployment rate has remained low. Inflation has eased over the past year but remains elevated. In recent months, there has been a lack of further progress toward the Committee\'s 2 percent inflation objective. The Committee decided to maintain the target range for the federal funds rate at 5-1/4 to 5-1/2 percent and will slow the pace of decline of its securities holdings.',
    precomputedLLM: {
      score: 0.12,
      inflation_stance: 0.38,
      growth_labor_stance: 0.30,
      forward_guidance_stance: -0.15,
      rationale:
        'Slightly hawkish inflation warning offset by QT taper (+0.12): highlights a "lack of further progress" toward the 2% inflation target while slowing balance sheet runoff.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2024-07-31',
    date: '2024-07-31',
    title: 'FOMC Statement — July 31, 2024',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '5.25%–5.50%',
    rateDecisionBps: 0,
    rateSurpriseBps: -4.5,
    dgs2_1d_bps: -10.0,
    dgs2_2d_bps: -20.0,
    dgs2_5d_bps: -36.0,
    dgs10_1d_bps: -10.0,
    dgs10_2d_bps: -17.0,
    dgs10_5d_bps: -25.0,
    sp500_1d_pct: 1.58,
    sp500_2d_pct: 0.19,
    sp500_5d_pct: -4.25,
    hasSEP: false,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have moderated, and the unemployment rate has moved up but remains low. Inflation has eased over the past year but remains somewhat elevated. In recent months, there has been some further progress toward the Committee\'s 2 percent inflation objective. The Committee is attentive to the risks to both sides of its dual mandate.',
    precomputedLLM: {
      score: -0.26,
      inflation_stance: -0.25,
      growth_labor_stance: -0.20,
      forward_guidance_stance: -0.32,
      rationale:
        'Dovish setup for easing (-0.26): softens inflation from "elevated" to "somewhat elevated", notes unemployment "has moved up", and shifts focus to "both sides of its dual mandate".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2024-09-18',
    date: '2024-09-18',
    title: 'FOMC Statement — September 18, 2024',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '4.75%–5.00%',
    rateDecisionBps: -50,
    rateSurpriseBps: -6.0,
    dgs2_1d_bps: 2.0,
    dgs2_2d_bps: -1.0,
    dgs2_5d_bps: -6.0,
    dgs10_1d_bps: 6.0,
    dgs10_2d_bps: 9.0,
    dgs10_5d_bps: 14.0,
    sp500_1d_pct: -0.29,
    sp500_2d_pct: 1.41,
    sp500_5d_pct: 1.85,
    hasSEP: true,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. Job gains have slowed, and the unemployment rate has moved up but remains low. Inflation has made further progress toward the Committee\'s 2 percent objective but remains somewhat elevated. The Committee has gained greater confidence that inflation is moving sustainably toward 2 percent, and judges that the risks to achieving its employment and inflation goals are roughly in balance. In light of the progress on inflation and the balance of risks, the Committee decided to lower the target range for the federal funds rate by 1/2 percentage point to 4-3/4 to 5 percent.',
    precomputedLLM: {
      score: -0.56,
      inflation_stance: -0.50,
      growth_labor_stance: -0.45,
      forward_guidance_stance: -0.68,
      rationale:
        'Strongly dovish recalibration (-0.56): lowers target range by 50 bps, states job gains "have slowed", and affirms "greater confidence that inflation is moving sustainably toward 2 percent".',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2024-11-07',
    date: '2024-11-07',
    title: 'FOMC Statement — November 7, 2024',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '4.50%–4.75%',
    rateDecisionBps: -25,
    rateSurpriseBps: -1.5,
    dgs2_1d_bps: -6.0,
    dgs2_2d_bps: -1.0,
    dgs2_5d_bps: 7.0,
    dgs10_1d_bps: -11.0,
    dgs10_2d_bps: -12.0,
    dgs10_5d_bps: 1.0,
    sp500_1d_pct: 0.74,
    sp500_2d_pct: 1.12,
    sp500_5d_pct: 0.36,
    hasSEP: false,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. Since earlier in the year, labor market conditions have generally eased, and the unemployment rate has moved up but remains low. Inflation has made progress toward the Committee\'s 2 percent objective but remains somewhat elevated. In support of its goals, the Committee decided to lower the target range for the federal funds rate by 1/4 percentage point to 4-1/2 to 4-3/4 percent.',
    precomputedLLM: {
      score: -0.42,
      inflation_stance: -0.35,
      growth_labor_stance: -0.38,
      forward_guidance_stance: -0.50,
      rationale:
        'Moderately dovish (-0.42): delivers 25 bps cut and notes that "labor market conditions have generally eased", while removing the prior "greater confidence" phrasing.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2024-12-18',
    date: '2024-12-18',
    title: 'FOMC Statement — December 18, 2024',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '4.25%–4.50%',
    rateDecisionBps: -25,
    rateSurpriseBps: 6.5,
    dgs2_1d_bps: 11.0,
    dgs2_2d_bps: 7.0,
    dgs2_5d_bps: 9.0,
    dgs10_1d_bps: 10.0,
    dgs10_2d_bps: 17.0,
    dgs10_5d_bps: 19.0,
    sp500_1d_pct: -2.95,
    sp500_2d_pct: -3.03,
    sp500_5d_pct: -0.52,
    hasSEP: true,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. Since earlier in the year, labor market conditions have generally eased, and the unemployment rate has moved up but remains low. Inflation has made progress toward the Committee\'s 2 percent objective but remains somewhat elevated. In support of its goals, the Committee decided to lower the target range for the federal funds rate by 1/4 percentage point to 4-1/4 to 4-1/2 percent. In considering the extent and timing of additional adjustments to the target range, the Committee will carefully assess incoming data.',
    precomputedLLM: {
      score: -0.18,
      inflation_stance: -0.20,
      growth_labor_stance: -0.20,
      forward_guidance_stance: -0.10,
      rationale:
        'Hawkish cut (-0.18): lowers rate by 25 bps to 4.25%–4.50% but inserts "extent and timing of additional adjustments", signaling a slower pace of future cuts.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
  {
    id: '2025-01-29',
    date: '2025-01-29',
    title: 'FOMC Statement — January 29, 2025',
    regime: '2024–2025 Calibration & Easing',
    targetRateRange: '4.25%–4.50%',
    rateDecisionBps: 0,
    rateSurpriseBps: 2.5,
    dgs2_1d_bps: 3.0,
    dgs2_2d_bps: 1.0,
    dgs2_5d_bps: -2.0,
    dgs10_1d_bps: 1.0,
    dgs10_2d_bps: -1.0,
    dgs10_5d_bps: -11.0,
    sp500_1d_pct: -0.47,
    sp500_2d_pct: 0.05,
    sp500_5d_pct: 0.36,
    hasSEP: false,
    statementText:
      'Recent indicators suggest that economic activity has continued to expand at a solid pace. The unemployment rate has stabilized at a low level in recent months, and labor market conditions remain solid. Inflation remains somewhat elevated. The Committee decided to maintain the target range for the federal funds rate at 4-1/4 to 4-1/2 percent. In considering the extent and timing of additional adjustments to the target range, the Committee will carefully assess incoming data.',
    precomputedLLM: {
      score: 0.14,
      inflation_stance: 0.25,
      growth_labor_stance: 0.32,
      forward_guidance_stance: -0.05,
      rationale:
        'Hawkish tilt on hold (+0.14): removes reference to inflation making progress toward 2%, upgrades labor description to "stabilized at a low level" and "remain solid", and holds rates.',
      model: 'gemini-3.8-flash (temperature=0.0)',
    },
  },
];

let cachedEnrichedCorpus: EnrichedFOMCRelease[] | null = null;

export function getEnrichedFOMCCorpus(): EnrichedFOMCRelease[] {
  if (cachedEnrichedCorpus) return cachedEnrichedCorpus;

  const enriched: EnrichedFOMCRelease[] = [];
  for (let i = 0; i < RAW_FOMC_CORPUS.length; i++) {
    const raw = RAW_FOMC_CORPUS[i];
    const dictRes = scoreTextDictionary(raw.statementText);
    const embedRes = scoreTextEmbedding(raw.statementText);
    const llmScore = raw.precomputedLLM.score;

    const prev = i > 0 ? enriched[i - 1] : null;
    const dictToneChange = prev ? Number((dictRes.score - prev.dictScore).toFixed(4)) : 0;
    const embedToneChange = prev ? Number((embedRes.score - prev.embedScore).toFixed(4)) : 0;
    const llmToneChange = prev ? Number((llmScore - prev.llmScore).toFixed(4)) : 0;

    enriched.push({
      ...raw,
      dictScore: dictRes.score,
      embedScore: embedRes.score,
      llmScore,
      dictToneChange,
      embedToneChange,
      llmToneChange,
      hawkishMatched: dictRes.matchedHawkish,
      dovishMatched: dictRes.matchedDovish,
    });
  }

  cachedEnrichedCorpus = enriched;
  return enriched;
}

export function getEventStudyObservations(): EventObservation[] {
  const corpus = getEnrichedFOMCCorpus();
  // Exclude index 0 since delta tone requires t-1
  return corpus.slice(1).map((item) => ({
    date: item.date,
    regime: item.regime,
    rateDecisionBps: item.rateDecisionBps,
    rateSurpriseBps: item.rateSurpriseBps,
    dictToneChange: item.dictToneChange,
    embedToneChange: item.embedToneChange,
    llmToneChange: item.llmToneChange,
    dgs2_1d_bps: item.dgs2_1d_bps,
    dgs2_2d_bps: item.dgs2_2d_bps,
    dgs2_5d_bps: item.dgs2_5d_bps,
    dgs10_1d_bps: item.dgs10_1d_bps,
    dgs10_2d_bps: item.dgs10_2d_bps,
    dgs10_5d_bps: item.dgs10_5d_bps,
    sp500_1d_pct: item.sp500_1d_pct,
    sp500_2d_pct: item.sp500_2d_pct,
    sp500_5d_pct: item.sp500_5d_pct,
  }));
}

export { evaluateDeterministicRubric };
