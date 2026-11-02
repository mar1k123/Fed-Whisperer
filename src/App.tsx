/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useMemo, useState } from 'react';
import { getEnrichedFOMCCorpus, getEventStudyObservations } from './data/fomcCorpus';
import { TimelineView } from './components/TimelineView';
import { DiffModeView } from './components/DiffModeView';
import { EventStudyView } from './components/EventStudyView';
import { PythonPackageView } from './components/PythonPackageView';

type ActiveTab = 'timeline' | 'diff' | 'event-study' | 'python-package';

export default function App() {
  const corpus = useMemo(() => getEnrichedFOMCCorpus(), []);
  const observations = useMemo(() => getEventStudyObservations(), []);

  const [activeTab, setActiveTab] = useState<ActiveTab>('timeline');
  const [selectedReleaseId, setSelectedReleaseId] = useState<string>('2024-09-18');
  const [diffPair, setDiffPair] = useState<{ currId: string; prevId: string }>({
    currId: '2024-09-18',
    prevId: '2024-07-31',
  });

  const handleOpenDiffForRelease = (currId: string, prevId: string) => {
    setDiffPair({ currId, prevId });
    setActiveTab('diff');
  };

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="flex items-center justify-between gap-8 px-6 py-4 bg-white border-b border-slate-200">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('timeline');
          }}
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
        >
          Fed Whisperer
        </a>

        {/* Zone 2: 4 clean single-line navigation links */}
        <nav className="flex items-center gap-6 text-sm font-medium text-slate-600" aria-label="Main navigation">
          <button
            type="button"
            onClick={() => setActiveTab('timeline')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'timeline'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Tone Timeline
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('diff')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'diff'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Statement Diff
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('event-study')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'event-study'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Event Study
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('python-package')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'python-package'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Python Package
          </button>
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => setActiveTab('diff')}
            className="px-4 py-2 text-xs font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Score Statement
          </button>
        </div>
      </header>

      {/* Main Content Container */}
      <main className="flex-1 max-w-[1360px] w-full mx-auto px-6 py-8">
        {activeTab === 'timeline' && (
          <TimelineView
            corpus={corpus}
            selectedReleaseId={selectedReleaseId}
            onSelectRelease={setSelectedReleaseId}
            onOpenDiffForRelease={handleOpenDiffForRelease}
          />
        )}

        {activeTab === 'diff' && (
          <DiffModeView
            corpus={corpus}
            currReleaseId={diffPair.currId}
            prevReleaseId={diffPair.prevId}
            onChangePair={(currId, prevId) => setDiffPair({ currId, prevId })}
          />
        )}

        {activeTab === 'event-study' && (
          <EventStudyView observations={observations} />
        )}

        {activeTab === 'python-package' && <PythonPackageView />}
      </main>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-200 bg-white px-6 py-4 mt-12">
        <div className="max-w-[1360px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <span>
            Fed Whisperer · FOMC Statement NLP Tone & Treasury/Equity Event Study
          </span>
          <span>
            Methods: Dictionary Lexicon · Semantic Embedding Axis · Fixed-Rubric LLM (T=0.0) · White HC1 OLS
          </span>
        </div>
      </footer>
    </div>
  );
}
