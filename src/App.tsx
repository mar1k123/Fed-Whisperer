import React, { useMemo, useState } from 'react';
import { runFullSimulation, FraudPattern } from './engine/fraudEngine';
import { NetworkExplorerView } from './components/NetworkExplorerView';
import { EvaluationAblationView } from './components/EvaluationAblationView';
import { ExplainabilityLabView } from './components/ExplainabilityLabView';
import { PythonPackageView } from './components/PythonPackageView';

type ActiveTab = 'network' | 'evaluation' | 'explainability' | 'repository';

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>('network');
  const [seed, setSeed] = useState<number>(42);
  const [fraudRate, setFraudRate] = useState<number>(0.015);
  const [shiftedTactics, setShiftedTactics] = useState<boolean>(false);

  const bundle = useMemo(
    () =>
      runFullSimulation({
        seed,
        fraudRate,
        shiftedTactics,
      }),
    [seed, fraudRate, shiftedTactics]
  );

  const [selectedRingId, setSelectedRingId] = useState<string>('RING-001');
  const [selectedAccountId, setSelectedAccountId] = useState<string>(
    () => bundle.rings[0]?.accounts[0] || 'ACC-0001'
  );

  const handleQuickPatternJump = (pattern: Exclude<FraudPattern, 'legitimate'>) => {
    const match = bundle.rings.find((r) => r.pattern === pattern);
    if (match) {
      setSelectedRingId(match.ringId);
      setSelectedAccountId(match.accounts[0]);
      setActiveTab('network');
    }
  };

  const gatEval = bundle.standardEvaluation.find((m) => m.modelName.includes('GAT'));

  return (
    <div className="min-h-screen flex flex-col bg-slate-50 text-slate-900">
      {/* Strict 3-Zone Top Bar Contract */}
      <header className="flex items-center justify-between gap-8 px-6 py-3.5 bg-white border-b border-slate-200">
        {/* Zone 1: Single text element wordmark */}
        <a
          href="#top"
          onClick={(e) => {
            e.preventDefault();
            setActiveTab('network');
          }}
          className="text-lg font-bold tracking-tight text-slate-900 whitespace-nowrap shrink-0"
        >
          Fraud Ring Finder
        </a>

        {/* Zone 2: 4 clean single-line navigation links */}
        <nav
          className="hidden md:flex items-center gap-6 text-sm font-medium text-slate-600"
          aria-label="Main navigation"
        >
          <button
            type="button"
            onClick={() => setActiveTab('network')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'network'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Network Graph
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('evaluation')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'evaluation'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Evaluation &amp; Ablation
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('explainability')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'explainability'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            SHAP &amp; GNNExplainer
          </button>
          <button
            type="button"
            onClick={() => setActiveTab('repository')}
            className={`py-1 transition-colors whitespace-nowrap shrink-0 border-b-2 focus:outline-none focus-visible:ring-2 focus-visible:ring-slate-900 ${
              activeTab === 'repository'
                ? 'border-slate-900 text-slate-900 font-semibold'
                : 'border-transparent hover:text-slate-900'
            }`}
          >
            Python Repository
          </button>
        </nav>

        {/* Zone 3: 1 primary action */}
        <div className="flex items-center gap-3 shrink-0">
          <button
            type="button"
            onClick={() => setActiveTab('evaluation')}
            className="px-4 py-2 text-xs font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
          >
            Run Ablation Suite
          </button>
        </div>
      </header>

      {/* PC Workstation Body: Left Desktop Control Sidebar + Main Viewport */}
      <div className="flex-1 flex flex-col lg:flex-row max-w-[1440px] w-full mx-auto">
        {/* Desktop Program Left Sidebar Menu & Generator Parameters */}
        <aside
          className="w-full lg:w-64 shrink-0 bg-white border-b lg:border-b-0 lg:border-r border-slate-200 p-5 space-y-6"
          aria-label="Workstation controls and menu"
        >
          {/* Program Modules Menu */}
          <div className="space-y-1.5">
            <div className="text-xs font-semibold text-slate-400 px-2">
              Workstation Modules
            </div>
            {(
              [
                { id: 'network', label: '01. Ring Graph & Risk Card' },
                { id: 'evaluation', label: '02. PR-AUC & Ablation' },
                { id: 'explainability', label: '03. SHAP & GNNExplainer' },
                { id: 'repository', label: '04. Python Files & README' },
              ] as const
            ).map((item) => (
              <button
                key={item.id}
                type="button"
                onClick={() => setActiveTab(item.id)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-medium transition-colors whitespace-nowrap shrink-0 ${
                  activeTab === item.id
                    ? 'bg-slate-900 text-white font-semibold'
                    : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                {item.label}
              </button>
            ))}
          </div>

          {/* Seeded Synthetic Generator Parameters */}
          <div className="space-y-3 pt-4 border-t border-slate-200">
            <div className="text-xs font-semibold text-slate-400 px-1">
              Synthetic Generator Config
            </div>

            <div className="space-y-1">
              <label htmlFor="sidebar-seed" className="block text-xs font-medium text-slate-700">
                Deterministic Seed
              </label>
              <select
                id="sidebar-seed"
                value={seed}
                onChange={(e) => setSeed(Number(e.target.value))}
                className="w-full px-2.5 py-1.5 text-xs font-mono bg-slate-50 border border-slate-300 rounded-md text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-900"
              >
                <option value={42}>Seed 42 (Default Benchmark)</option>
                <option value={101}>Seed 101 (High Mule Density)</option>
                <option value={2026}>Seed 2026 (Dense Device Cliques)</option>
                <option value={777}>Seed 777 (Burst Stress Test)</option>
              </select>
            </div>

            <div className="space-y-1">
              <div className="flex items-center justify-between text-xs">
                <label htmlFor="sidebar-fraud-rate" className="font-medium text-slate-700">
                  Planted Fraud Rate
                </label>
                <span className="font-mono font-semibold text-slate-900 tabular-nums">
                  {(fraudRate * 100).toFixed(1)}%
                </span>
              </div>
              <input
                id="sidebar-fraud-rate"
                type="range"
                min={0.005}
                max={0.02}
                step={0.0025}
                value={fraudRate}
                onChange={(e) => setFraudRate(Number(e.target.value))}
                className="w-full accent-slate-900 cursor-pointer"
              />
              <div className="flex justify-between text-[10px] font-mono text-slate-400">
                <span>0.5%</span>
                <span>1.25%</span>
                <span>2.0%</span>
              </div>
            </div>

            <div className="pt-1">
              <label className="flex items-center justify-between gap-2 cursor-pointer text-xs font-medium text-slate-700">
                <span>Post-Training Shift</span>
                <input
                  type="checkbox"
                  checked={shiftedTactics}
                  onChange={(e) => setShiftedTactics(e.target.checked)}
                  className="rounded border-slate-300 text-slate-900 focus:ring-slate-900"
                />
              </label>
              <p className="text-[11px] text-slate-500 mt-1">
                Elongates mule chains to 6 hops &amp; rotates 50% shared devices in test window.
              </p>
            </div>
          </div>

          {/* Planted Fraud Ring Presets */}
          <div className="space-y-1.5 pt-4 border-t border-slate-200">
            <div className="text-xs font-semibold text-slate-400 px-1">
              Inspect Planted Topologies
            </div>
            <button
              type="button"
              onClick={() => handleQuickPatternJump('money_mule_chain')}
              className="w-full text-left px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              Money-Mule Layering Chain
            </button>
            <button
              type="button"
              onClick={() => handleQuickPatternJump('shared_device_ring')}
              className="w-full text-left px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              Shared-Device Collusive Ring
            </button>
            <button
              type="button"
              onClick={() => handleQuickPatternJump('burst_activity')}
              className="w-full text-left px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              High-Velocity Burst Drain
            </button>
            <button
              type="button"
              onClick={() => handleQuickPatternJump('circular_transfer')}
              className="w-full text-left px-2.5 py-1.5 text-xs text-slate-700 hover:bg-slate-100 rounded-md transition-colors whitespace-nowrap shrink-0"
            >
              Circular Wash Transfer Loop
            </button>
          </div>

          {/* Dataset Summary Card */}
          <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg space-y-1.5 text-xs font-mono tabular-nums">
            <div className="text-slate-500 font-sans font-medium">Active Network Snapshot</div>
            <div className="flex justify-between">
              <span className="text-slate-500">Accounts:</span>
              <span className="font-semibold text-slate-900">{bundle.accounts.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Devices / Merch:</span>
              <span className="font-semibold text-slate-900">
                {bundle.devices.length} / {bundle.merchants.length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Transactions:</span>
              <span className="font-semibold text-slate-900">
                {bundle.transactions.length.toLocaleString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">Planted Rings:</span>
              <span className="font-semibold text-rose-700">{bundle.rings.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-slate-500">GAT PR-AUC:</span>
              <span className="font-semibold text-slate-900">
                {gatEval ? gatEval.prAuc.toFixed(4) : '—'}
              </span>
            </div>
          </div>
        </aside>

        {/* Main Workspace Canvas */}
        <main className="flex-1 min-w-0 px-6 py-7">
          {activeTab === 'network' && (
            <NetworkExplorerView
              bundle={bundle}
              selectedAccountId={selectedAccountId}
              onSelectAccount={setSelectedAccountId}
              selectedRingId={selectedRingId}
              onSelectRing={setSelectedRingId}
            />
          )}

          {activeTab === 'evaluation' && (
            <EvaluationAblationView
              bundle={bundle}
              onToggleDistributionShift={setShiftedTactics}
            />
          )}

          {activeTab === 'explainability' && (
            <ExplainabilityLabView
              bundle={bundle}
              selectedAccountId={selectedAccountId}
              onSelectAccount={setSelectedAccountId}
            />
          )}

          {activeTab === 'repository' && <PythonPackageView />}
        </main>
      </div>

      {/* Quiet Footer */}
      <footer className="border-t border-slate-200 bg-white px-6 py-4">
        <div className="max-w-[1440px] mx-auto flex flex-col sm:flex-row items-center justify-between gap-2 text-xs text-slate-500">
          <span>
            Fraud Ring Finder · Graph ML Transaction Network Detection &amp; Ablation Suite
          </span>
          <span>
            Stack: Python 3.10+ · PyTorch Geometric (GraphSAGE, GAT, GNNExplainer) · XGBoost · SHAP · NetworkX · Streamlit · PyVis
          </span>
        </div>
      </footer>
    </div>
  );
}
