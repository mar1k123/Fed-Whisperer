import React, { useMemo } from 'react';
import {
  SimulationBundle,
  computeAccountShapExplanation,
  computeGnnExplainerSubgraph,
} from '../engine/fraudEngine';

interface ExplainabilityLabViewProps {
  bundle: SimulationBundle;
  selectedAccountId: string;
  onSelectAccount: (accountId: string) => void;
}

export const ExplainabilityLabView: React.FC<ExplainabilityLabViewProps> = ({
  bundle,
  selectedAccountId,
  onSelectAccount,
}) => {
  const flaggedAccounts = useMemo(
    () => bundle.testFeatures.filter((f) => f.isFraud),
    [bundle.testFeatures]
  );

  const activeFeature = useMemo(() => {
    return (
      bundle.testFeatures.find((f) => f.accountId === selectedAccountId) ||
      flaggedAccounts[0] ||
      bundle.testFeatures[0]
    );
  }, [bundle.testFeatures, flaggedAccounts, selectedAccountId]);

  const shapList = useMemo(
    () => computeAccountShapExplanation(activeFeature, bundle.testFeatures),
    [activeFeature, bundle.testFeatures]
  );

  const gnnSubgraph = useMemo(
    () =>
      computeGnnExplainerSubgraph(
        activeFeature.accountId,
        bundle.accounts,
        bundle.transactions
      ),
    [activeFeature.accountId, bundle.accounts, bundle.transactions]
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <p className="text-xs font-medium text-slate-500">
            03. Dual Model Explainability: TreeSHAP Feature Attribution vs. GNNExplainer Subgraph Masks
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-0.5">
            SHAP &amp; GNNExplainer Forensic Comparison Lab
          </h1>
        </div>

        <div className="flex items-center gap-3">
          <label htmlFor="explain-acc-select" className="text-xs font-semibold text-slate-700">
            Target Account:
          </label>
          <select
            id="explain-acc-select"
            value={activeFeature.accountId}
            onChange={(e) => onSelectAccount(e.target.value)}
            className="px-3 py-1.5 text-xs font-mono bg-white border border-slate-300 rounded-lg text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-900"
          >
            {flaggedAccounts.map((acc) => (
              <option key={acc.accountId} value={acc.accountId}>
                {acc.accountId} · {acc.fraudPattern} ({acc.ringId})
              </option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 6 Cols: TreeSHAP Local + Global Attributions */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div>
              <h2 className="text-xs font-bold text-slate-900">
                XGBoost TreeSHAP Attribution Waterfall ({activeFeature.accountId})
              </h2>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Marginal log-odds contribution of Tabular vs. Hand-Crafted Graph features
              </p>
            </div>
            <span className="text-xs font-mono text-slate-600">shap.TreeExplainer</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 bg-slate-50/50">
                  <th className="py-2.5 px-4 font-medium">Feature Name</th>
                  <th className="py-2.5 px-3 font-medium">Domain</th>
                  <th className="py-2.5 px-3 font-medium text-right">Account Value</th>
                  <th className="py-2.5 px-3 font-medium text-right">Local SHAP</th>
                  <th className="py-2.5 px-4 font-medium text-right">Global Mean |SHAP|</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                {shapList.map((row) => (
                  <tr key={row.feature} className="hover:bg-slate-50">
                    <td className="py-2.5 px-4 font-semibold text-slate-900">{row.feature}</td>
                    <td className="py-2.5 px-3 text-slate-500 capitalize">{row.category}</td>
                    <td className="py-2.5 px-3 text-right text-slate-700">{row.rawValue}</td>
                    <td
                      className={`py-2.5 px-3 text-right font-bold ${
                        row.shapValue >= 0 ? 'text-rose-700' : 'text-sky-700'
                      }`}
                    >
                      {row.shapValue >= 0 ? `+${row.shapValue.toFixed(4)}` : row.shapValue.toFixed(4)}
                    </td>
                    <td className="py-2.5 px-4 text-right text-slate-600">
                      {row.globalMeanAbsShap.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Right 6 Cols: GNNExplainer 2-Hop Computational Subgraph */}
        <div className="lg:col-span-6 bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="px-5 py-3.5 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
            <div>
              <h2 className="text-xs font-bold text-slate-900">
                PyG GNNExplainer 2-Hop Subgraph Edge Mask ({activeFeature.accountId})
              </h2>
              <p className="text-[11px] text-slate-500 mt-0.5">
                Mutual-information maximization over message-passing edges <code className="font-mono">G_S</code>
              </p>
            </div>
            <span className="text-xs font-mono text-slate-600">GNNExplainer(hops=2)</span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left border-collapse text-xs">
              <thead>
                <tr className="border-b border-slate-200 text-slate-500 bg-slate-50/50">
                  <th className="py-2.5 px-4 font-medium">Edge / Link ID</th>
                  <th className="py-2.5 px-3 font-medium">Source → Target</th>
                  <th className="py-2.5 px-3 font-medium">Relation</th>
                  <th className="py-2.5 px-4 font-medium text-right">Mask Importance w_e</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                {gnnSubgraph.map((edge) => (
                  <tr key={edge.edgeId} className="hover:bg-slate-50">
                    <td className="py-2.5 px-4 text-slate-700">{edge.edgeId}</td>
                    <td className="py-2.5 px-3 font-semibold text-slate-900">
                      {edge.source} → {edge.target}
                    </td>
                    <td className="py-2.5 px-3 text-slate-600">
                      {edge.entityType === 'shared_device'
                        ? 'Shared Device'
                        : `Tx (${edge.channel?.toUpperCase()})`}
                    </td>
                    <td className="py-2.5 px-4 text-right font-bold text-rose-700">
                      {edge.importanceWeight.toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
};
