import React, { useMemo, useState } from 'react';
import {
  SimulationBundle,
  FraudPattern,
  computeAccountShapExplanation,
  computeGnnExplainerSubgraph,
} from '../engine/fraudEngine';
import { ZoomIn, ZoomOut, RotateCcw, Search } from 'lucide-react';

interface NetworkExplorerViewProps {
  bundle: SimulationBundle;
  selectedAccountId: string;
  onSelectAccount: (accountId: string) => void;
  selectedRingId: string;
  onSelectRing: (ringId: string) => void;
}

const PATTERN_LABELS: Record<FraudPattern, string> = {
  legitimate: 'Legitimate Baseline',
  money_mule_chain: 'Money-Mule Chain',
  shared_device_ring: 'Shared-Device Ring',
  burst_activity: 'Burst Activity Drain',
  circular_transfer: 'Circular Transfer Loop',
};

const PATTERN_NODE_FILL: Record<FraudPattern, string> = {
  legitimate: '#475569',
  money_mule_chain: '#e11d48',
  shared_device_ring: '#d97706',
  burst_activity: '#7c3aed',
  circular_transfer: '#0284c7',
};

export const NetworkExplorerView: React.FC<NetworkExplorerViewProps> = ({
  bundle,
  selectedAccountId,
  onSelectAccount,
  selectedRingId,
  onSelectRing,
}) => {
  const [patternFilter, setPatternFilter] = useState<'all' | Exclude<FraudPattern, 'legitimate'>>('all');
  const [zoom, setZoom] = useState<number>(1);
  const [showEdgeAmounts, setShowEdgeAmounts] = useState<boolean>(true);
  const [highlightExplainerSubgraph, setHighlightExplainerSubgraph] = useState<boolean>(true);
  const [searchQuery, setSearchQuery] = useState<string>('');

  const filteredRings = useMemo(() => {
    if (patternFilter === 'all') return bundle.rings;
    return bundle.rings.filter((r) => r.pattern === patternFilter);
  }, [bundle.rings, patternFilter]);

  const activeRing = useMemo(() => {
    return (
      filteredRings.find((r) => r.ringId === selectedRingId) ||
      filteredRings[0] ||
      bundle.rings[0]
    );
  }, [filteredRings, selectedRingId, bundle.rings]);

  const activeAccountFeature = useMemo(() => {
    return (
      bundle.testFeatures.find((f) => f.accountId === selectedAccountId) ||
      bundle.testFeatures.find((f) => f.accountId === activeRing?.accounts[0]) ||
      bundle.testFeatures[0]
    );
  }, [bundle.testFeatures, selectedAccountId, activeRing]);

  const activeAccountNode = useMemo(() => {
    return (
      bundle.accounts.find((a) => a.accountId === activeAccountFeature.accountId) ||
      bundle.accounts[0]
    );
  }, [bundle.accounts, activeAccountFeature]);

  const activePreds = bundle.nodePredictions[activeAccountFeature.accountId] || {
    xgbTabular: 0.05,
    xgbGraph: 0.1,
    graphSageFocal: 0.15,
    gatFocal: 0.18,
  };

  const shapAttributions = useMemo(
    () => computeAccountShapExplanation(activeAccountFeature, bundle.testFeatures).slice(0, 6),
    [activeAccountFeature, bundle.testFeatures]
  );

  const gnnSubgraphEdges = useMemo(
    () =>
      computeGnnExplainerSubgraph(
        activeAccountFeature.accountId,
        bundle.accounts,
        bundle.transactions
      ),
    [activeAccountFeature.accountId, bundle.accounts, bundle.transactions]
  );

  // Build interactive PyVis-style subgraph layout around activeRing + 1-hop context nodes
  const graphLayout = useMemo(() => {
    if (!activeRing) return { nodes: [], edges: [] };
    const ringSet = new Set<string>(activeRing.accounts);

    // Collect ring transactions + up to 10 background neighboring transactions for context
    const ringTxs = bundle.transactions.filter((tx) => tx.ringId === activeRing.ringId);
    const contextTxs = bundle.transactions
      .filter(
        (tx) =>
          tx.ringId !== activeRing.ringId &&
          (ringSet.has(tx.srcAccount) || ringSet.has(tx.dstAccount))
      )
      .slice(0, 8);

    const allTxs = [...ringTxs, ...contextTxs];
    const nodeIds = new Set<string>(activeRing.accounts);
    allTxs.forEach((tx) => {
      nodeIds.add(tx.srcAccount);
      nodeIds.add(tx.dstAccount);
    });

    const accMap = new Map(bundle.accounts.map((a) => [a.accountId, a]));
    const centerX = 360;
    const centerY = 230;

    const ringNodesList = Array.from(ringSet);
    const contextNodesList = Array.from(nodeIds).filter((id) => !ringSet.has(id));

    const nodes: Array<{
      id: string;
      label: string;
      kind: 'account' | 'device' | 'merchant';
      x: number;
      y: number;
      isFraud: boolean;
      pattern: FraudPattern;
      riskScore: number;
    }> = [];

    // Position ring accounts according to their motif geometry
    ringNodesList.forEach((id, idx) => {
      const acc = accMap.get(id);
      const preds = bundle.nodePredictions[id];
      let x = centerX;
      let y = centerY;

      if (activeRing.pattern === 'money_mule_chain') {
        const span = 320;
        const step = ringNodesList.length > 1 ? span / (ringNodesList.length - 1) : 0;
        x = centerX - span / 2 + idx * step;
        y = centerY + (idx % 2 === 0 ? -28 : 28);
      } else if (activeRing.pattern === 'shared_device_ring' || activeRing.pattern === 'burst_activity') {
        if (idx === 0) {
          x = centerX;
          y = centerY;
        } else {
          const angle = ((idx - 1) / Math.max(1, ringNodesList.length - 1)) * Math.PI * 2 - Math.PI / 2;
          x = centerX + Math.cos(angle) * 115;
          y = centerY + Math.sin(angle) * 95;
        }
      } else {
        // circular_transfer
        const angle = (idx / Math.max(1, ringNodesList.length)) * Math.PI * 2 - Math.PI / 4;
        x = centerX + Math.cos(angle) * 110;
        y = centerY + Math.sin(angle) * 95;
      }

      nodes.push({
        id,
        label: id,
        kind: 'account',
        x,
        y,
        isFraud: acc?.isFraud ?? true,
        pattern: acc?.fraudPattern ?? activeRing.pattern,
        riskScore: preds?.gatFocal ?? 0.85,
      });
    });

    // Add shared device node if present
    if (activeRing.sharedDeviceId) {
      nodes.push({
        id: activeRing.sharedDeviceId,
        label: activeRing.sharedDeviceId,
        kind: 'device',
        x: centerX - 235,
        y: centerY - 140,
        isFraud: true,
        pattern: 'shared_device_ring',
        riskScore: 0.92,
      });
    }

    // Add collusive merchant node if present
    if (activeRing.merchantId) {
      nodes.push({
        id: activeRing.merchantId,
        label: activeRing.merchantId,
        kind: 'merchant',
        x: centerX + 235,
        y: centerY - 140,
        isFraud: true,
        pattern: 'shared_device_ring',
        riskScore: 0.88,
      });
    }

    // Position 1-hop legitimate context accounts in dedicated top/bottom outer slots (guaranteed non-overlapping)
    const outerSlots = [
      { x: 115, y: 78 },
      { x: 275, y: 62 },
      { x: 445, y: 62 },
      { x: 605, y: 78 },
      { x: 115, y: 362 },
      { x: 275, y: 376 },
      { x: 445, y: 376 },
      { x: 605, y: 362 },
    ];
    contextNodesList.slice(0, outerSlots.length).forEach((id, idx) => {
      const acc = accMap.get(id);
      const preds = bundle.nodePredictions[id];
      const slot = outerSlots[idx];
      nodes.push({
        id,
        label: id,
        kind: 'account',
        x: slot.x,
        y: slot.y,
        isFraud: acc?.isFraud ?? false,
        pattern: acc?.fraudPattern ?? 'legitimate',
        riskScore: preds?.gatFocal ?? 0.08,
      });
    });

    const nodePos = new Map(nodes.map((n) => [n.id, n]));
    const edges: Array<{
      id: string;
      source: string;
      target: string;
      amountLabel: string;
      isFraud: boolean;
      isDeviceLink: boolean;
    }> = [];

    allTxs.forEach((tx) => {
      if (nodePos.has(tx.srcAccount) && nodePos.has(tx.dstAccount)) {
        edges.push({
          id: tx.txId,
          source: tx.srcAccount,
          target: tx.dstAccount,
          amountLabel: `$${tx.amount.toLocaleString()} (${tx.channel.toUpperCase()})`,
          isFraud: tx.isFraud,
          isDeviceLink: false,
        });
      }
    });

    if (activeRing.sharedDeviceId) {
      ringNodesList.forEach((accId) => {
        edges.push({
          id: `DEV-${activeRing.sharedDeviceId}-${accId}`,
          source: activeRing.sharedDeviceId!,
          target: accId,
          amountLabel: 'Shared Device Auth',
          isFraud: true,
          isDeviceLink: true,
        });
      });
    }

    return { nodes, edges };
  }, [activeRing, bundle]);

  const handleAccountSearch = (e: React.FormEvent) => {
    e.preventDefault();
    const q = searchQuery.trim().toUpperCase();
    if (!q) return;
    const matchAcc = bundle.accounts.find((a) => a.accountId.toUpperCase().includes(q));
    if (matchAcc) {
      onSelectAccount(matchAcc.accountId);
      if (matchAcc.ringId) {
        onSelectRing(matchAcc.ringId);
      }
    }
  };

  return (
    <div className="space-y-6">
      {/* Section Header + Pattern Filter Controls */}
      <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <p className="text-xs font-medium text-slate-500">
            01. Interactive Transaction Topology & Forensic Account Inspector
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-0.5">
            Planted Fraud Ring Graph & Multi-Model Risk Card
          </h1>
        </div>

        {/* Functional Segmented Filter Bar */}
        <div
          className="flex items-center gap-1 p-1 bg-slate-100 rounded-lg border border-slate-200 overflow-x-auto shrink-0"
          role="group"
          aria-label="Filter by fraud ring pattern"
        >
          {(
            [
              { id: 'all', label: 'All Rings' },
              { id: 'money_mule_chain', label: 'Mule Chains' },
              { id: 'shared_device_ring', label: 'Shared Devices' },
              { id: 'burst_activity', label: 'Burst Drains' },
              { id: 'circular_transfer', label: 'Circular Loops' },
            ] as const
          ).map((item) => (
            <button
              key={item.id}
              type="button"
              onClick={() => {
                setPatternFilter(item.id);
                const firstMatch =
                  item.id === 'all'
                    ? bundle.rings[0]
                    : bundle.rings.find((r) => r.pattern === item.id);
                if (firstMatch) {
                  onSelectRing(firstMatch.ringId);
                  onSelectAccount(firstMatch.accounts[0]);
                }
              }}
              className={`px-3 py-1.5 text-xs font-medium rounded-md transition-colors whitespace-nowrap shrink-0 ${
                patternFilter === item.id
                  ? 'bg-slate-900 text-white shadow-xs'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              {item.label}
            </button>
          ))}
        </div>
      </div>

      {/* Main 12-Col Forensic Split: 7 Cols PyVis-Style Interactive Graph, 5 Cols Per-Account Risk Card */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 7 Cols: Interactive Graph Canvas + Ring Selector Table */}
        <div className="lg:col-span-7 space-y-5">
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            {/* Graph Toolbar */}
            <div className="flex flex-wrap items-center justify-between gap-3 px-4 py-3 bg-slate-50 border-b border-slate-200">
              <div className="flex items-center gap-3">
                <label htmlFor="ring-select" className="text-xs font-semibold text-slate-700">
                  Flagged Ring:
                </label>
                <select
                  id="ring-select"
                  value={activeRing?.ringId || ''}
                  onChange={(e) => {
                    const nextRing = bundle.rings.find((r) => r.ringId === e.target.value);
                    if (nextRing) {
                      onSelectRing(nextRing.ringId);
                      onSelectAccount(nextRing.accounts[0]);
                    }
                  }}
                  className="px-2.5 py-1.5 text-xs font-mono bg-white border border-slate-300 rounded-md text-slate-900 focus:outline-none focus:ring-2 focus:ring-slate-900"
                >
                  {filteredRings.map((r) => (
                    <option key={r.ringId} value={r.ringId}>
                      {r.ringId} · {PATTERN_LABELS[r.pattern]} ({r.splitWindow.toUpperCase()})
                    </option>
                  ))}
                </select>
              </div>

              <div className="flex items-center gap-2">
                <button
                  type="button"
                  onClick={() => setShowEdgeAmounts((v) => !v)}
                  className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 ${
                    showEdgeAmounts
                      ? 'bg-slate-900 text-white border-slate-900'
                      : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                  }`}
                >
                  Edge Labels
                </button>
                <button
                  type="button"
                  onClick={() => setHighlightExplainerSubgraph((v) => !v)}
                  className={`px-2.5 py-1 text-xs font-medium rounded border transition-colors whitespace-nowrap shrink-0 ${
                    highlightExplainerSubgraph
                      ? 'bg-rose-700 text-white border-rose-700'
                      : 'bg-white text-slate-700 border-slate-300 hover:bg-slate-100'
                  }`}
                >
                  GNNExplainer Mask
                </button>
                <button
                  type="button"
                  aria-label="Zoom in network graph"
                  onClick={() => setZoom((z) => Math.min(1.45, Number((z + 0.15).toFixed(2))))}
                  className="p-1.5 text-slate-600 bg-white border border-slate-300 rounded hover:bg-slate-100"
                >
                  <ZoomIn className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  aria-label="Zoom out network graph"
                  onClick={() => setZoom((z) => Math.max(0.7, Number((z - 0.15).toFixed(2))))}
                  className="p-1.5 text-slate-600 bg-white border border-slate-300 rounded hover:bg-slate-100"
                >
                  <ZoomOut className="w-3.5 h-3.5" />
                </button>
                <button
                  type="button"
                  aria-label="Reset graph zoom"
                  onClick={() => setZoom(1)}
                  className="p-1.5 text-slate-600 bg-white border border-slate-300 rounded hover:bg-slate-100"
                >
                  <RotateCcw className="w-3.5 h-3.5" />
                </button>
              </div>
            </div>

            {/* Interactive SVG Network Viewport (Dark Forensic Slate Surface) */}
            <div className="relative bg-slate-950 h-[450px] w-full overflow-hidden select-none">
              <svg
                viewBox="0 0 720 460"
                className="w-full h-full"
                data-testid="fraud-network-svg"
                aria-label="Interactive transaction network subgraph"
              >
                <defs>
                  <marker
                    id="arrow-fraud"
                    viewBox="0 0 10 10"
                    refX="22"
                    refY="5"
                    markerWidth="6"
                    markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#fb7185" />
                  </marker>
                  <marker
                    id="arrow-legit"
                    viewBox="0 0 10 10"
                    refX="20"
                    refY="5"
                    markerWidth="5"
                    markerHeight="5"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 1 L 10 5 L 0 9 z" fill="#64748b" />
                  </marker>
                </defs>

                <g
                  transform={`translate(${360 * (1 - zoom)}, ${230 * (1 - zoom)}) scale(${zoom})`}
                >
                  {/* Edges */}
                  {graphLayout.edges.map((edge, idx) => {
                    const src = graphLayout.nodes.find((n) => n.id === edge.source);
                    const dst = graphLayout.nodes.find((n) => n.id === edge.target);
                    if (!src || !dst) return null;

                    const midX = (src.x + dst.x) / 2;
                    const midY = (src.y + dst.y) / 2 - (idx % 2 === 0 ? 8 : -10);
                    const isSelectedEdge =
                      highlightExplainerSubgraph &&
                      (edge.source === activeAccountFeature.accountId ||
                        edge.target === activeAccountFeature.accountId);

                    return (
                      <g key={edge.id}>
                        <line
                          x1={src.x}
                          y1={src.y}
                          x2={dst.x}
                          y2={dst.y}
                          stroke={
                            edge.isDeviceLink
                              ? '#f59e0b'
                              : edge.isFraud
                              ? isSelectedEdge
                                ? '#fb7185'
                                : '#e11d48'
                              : '#334155'
                          }
                          strokeWidth={edge.isFraud ? (isSelectedEdge ? 3.2 : 2.2) : 1.2}
                          strokeDasharray={edge.isDeviceLink ? '5,4' : undefined}
                          markerEnd={
                            edge.isDeviceLink
                              ? undefined
                              : edge.isFraud
                              ? 'url(#arrow-fraud)'
                              : 'url(#arrow-legit)'
                          }
                        />
                        {showEdgeAmounts && edge.isFraud && (
                          <text
                            x={midX}
                            y={midY}
                            textAnchor="middle"
                            className="text-[10px] font-mono fill-slate-300 pointer-events-none"
                          >
                            {edge.amountLabel}
                          </text>
                        )}
                      </g>
                    );
                  })}

                  {/* Nodes */}
                  {graphLayout.nodes.map((node) => {
                    const isSelected = node.id === activeAccountFeature.accountId;
                    const fill =
                      node.kind === 'device'
                        ? '#d97706'
                        : node.kind === 'merchant'
                        ? '#0d9488'
                        : PATTERN_NODE_FILL[node.pattern];

                    return (
                      <g
                        key={node.id}
                        transform={`translate(${node.x}, ${node.y})`}
                        onClick={() => {
                          if (node.kind === 'account') {
                            onSelectAccount(node.id);
                          }
                        }}
                        className={node.kind === 'account' ? 'cursor-pointer' : ''}
                      >
                        {isSelected && (
                          <circle
                            r={24}
                            fill="none"
                            stroke="#f8fafc"
                            strokeWidth={2}
                            strokeDasharray="4,3"
                          />
                        )}
                        <circle
                          r={node.isFraud ? 17 : 12}
                          fill={fill}
                          stroke={isSelected ? '#ffffff' : '#0f172a'}
                          strokeWidth={isSelected ? 2.5 : 1.5}
                        />
                        <text
                          y={node.isFraud ? 29 : 24}
                          textAnchor="middle"
                          className="text-[11px] font-mono font-semibold fill-slate-100"
                        >
                          {node.label}
                        </text>
                        {node.kind === 'account' && (
                          <text
                            y={3}
                            textAnchor="middle"
                            className="text-[9px] font-mono font-bold fill-white pointer-events-none"
                          >
                            {Math.round(node.riskScore * 100)}%
                          </text>
                        )}
                      </g>
                    );
                  })}
                </g>
              </svg>

              {/* Clean Legend Bar inside Viewport */}
              <div className="absolute bottom-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 px-3 py-2 bg-slate-900/90 border border-slate-800 rounded-lg text-[11px] text-slate-300">
                <div className="flex flex-wrap items-center gap-3">
                  <span>
                    Ring: <strong className="font-mono text-white">{activeRing?.ringId}</strong>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span>
                    Topology:{' '}
                    <strong className="text-rose-300">
                      {activeRing ? PATTERN_LABELS[activeRing.pattern] : ''}
                    </strong>
                  </span>
                  <span aria-hidden="true">·</span>
                  <span className="font-mono tabular-nums">
                    Volume: ${activeRing?.totalVolume.toLocaleString()} ({activeRing?.numTransactions}{' '}
                    txs)
                  </span>
                </div>
                <div className="text-slate-400">Click any account node to inspect risk card</div>
              </div>
            </div>
          </div>

          {/* Ring Member Accounts Table */}
          <div className="bg-white border border-slate-200 rounded-xl overflow-hidden">
            <div className="px-4 py-3 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <h2 className="text-xs font-semibold text-slate-800">
                Accounts in Selected Ring ({activeRing?.ringId}) — Click Row to Inspect
              </h2>
              <span className="text-xs text-slate-500 font-mono tabular-nums">
                Window: {activeRing?.splitWindow.toUpperCase()}
              </span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="border-b border-slate-200 text-slate-500 bg-slate-50/60">
                    <th className="py-2.5 px-3 font-medium">Account ID</th>
                    <th className="py-2.5 px-3 font-medium">Primary Device</th>
                    <th className="py-2.5 px-3 font-medium text-right">Out / In Tx</th>
                    <th className="py-2.5 px-3 font-medium text-right">PageRank</th>
                    <th className="py-2.5 px-3 font-medium text-right">XGB Tabular</th>
                    <th className="py-2.5 px-3 font-medium text-right">GraphSAGE</th>
                    <th className="py-2.5 px-3 font-medium text-right">GAT Score</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-200 font-mono tabular-nums">
                  {activeRing?.accounts.map((accId) => {
                    const feat = bundle.testFeatures.find((f) => f.accountId === accId);
                    const acc = bundle.accounts.find((a) => a.accountId === accId);
                    const preds = bundle.nodePredictions[accId];
                    const isSelected = accId === activeAccountFeature.accountId;
                    return (
                      <tr
                        key={accId}
                        onClick={() => onSelectAccount(accId)}
                        className={`cursor-pointer transition-colors ${
                          isSelected ? 'bg-slate-900 text-white' : 'hover:bg-slate-50 text-slate-800'
                        }`}
                      >
                        <td className="py-2.5 px-3 font-semibold">{accId}</td>
                        <td className="py-2.5 px-3">{acc?.primaryDeviceId}</td>
                        <td className="py-2.5 px-3 text-right">
                          {feat?.txOutCount ?? 0} / {feat?.txInCount ?? 0}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          {(feat?.graphPageRank ?? 1).toFixed(2)}
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          {((preds?.xgbTabular ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right">
                          {((preds?.graphSageFocal ?? 0) * 100).toFixed(1)}%
                        </td>
                        <td className="py-2.5 px-3 text-right font-bold">
                          {((preds?.gatFocal ?? 0) * 100).toFixed(1)}%
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* Right 5 Cols: Per-Account Risk Card with SHAP + GNNExplainer Subgraph */}
        <div className="lg:col-span-5 space-y-5">
          <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-5">
            {/* Search & Account Header */}
            <div className="flex items-center justify-between gap-2 border-b border-slate-200 pb-4">
              <div>
                <div className="text-xs text-slate-500">
                  Per-Account Forensic Risk Card ·{' '}
                  <span>{PATTERN_LABELS[activeAccountFeature.fraudPattern]}</span>
                </div>
                <h2
                  className="text-xl font-bold font-mono text-slate-900 mt-0.5"
                  data-testid="inspected-account-id"
                >
                  {activeAccountFeature.accountId}
                </h2>
              </div>

              <form onSubmit={handleAccountSearch} className="flex items-center gap-1.5">
                <label htmlFor="account-search-input" className="sr-only">
                  Jump to Account ID
                </label>
                <input
                  id="account-search-input"
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="ACC-0012"
                  className="w-28 px-2.5 py-1.5 text-xs font-mono border border-slate-300 rounded-md focus:outline-none focus:ring-2 focus:ring-slate-900"
                />
                <button
                  type="submit"
                  aria-label="Find account"
                  className="p-1.5 bg-slate-900 text-white rounded-md hover:bg-slate-800"
                >
                  <Search className="w-3.5 h-3.5" />
                </button>
              </form>
            </div>

            {/* Metadata Line (Zero-Pill Clean Inline Text) */}
            <div className="flex flex-wrap items-center gap-2 text-xs text-slate-600 font-mono tabular-nums">
              <span>Ring: {activeAccountFeature.ringId || 'None'}</span>
              <span aria-hidden="true">·</span>
              <span>Device: {activeAccountNode.primaryDeviceId}</span>
              <span aria-hidden="true">·</span>
              <span>KYC Level {activeAccountFeature.kycLevel}</span>
              <span aria-hidden="true">·</span>
              <span>Tenure: {activeAccountFeature.accountAgeDays}d</span>
            </div>

            {/* 4-Model Risk Probability Comparison */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 pt-1">
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-[11px] text-slate-500">GAT (Focal)</div>
                <div className="text-lg font-bold font-mono tabular-nums text-rose-700 mt-0.5">
                  {(activePreds.gatFocal * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-[11px] text-slate-500">GraphSAGE</div>
                <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                  {(activePreds.graphSageFocal * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-[11px] text-slate-500">XGB + Graph</div>
                <div className="text-lg font-bold font-mono tabular-nums text-slate-900 mt-0.5">
                  {(activePreds.xgbGraph * 100).toFixed(1)}%
                </div>
              </div>
              <div className="p-3 bg-slate-50 border border-slate-200 rounded-lg">
                <div className="text-[11px] text-slate-500">XGB Tabular</div>
                <div className="text-lg font-bold font-mono tabular-nums text-slate-600 mt-0.5">
                  {(activePreds.xgbTabular * 100).toFixed(1)}%
                </div>
              </div>
            </div>

            {/* TreeSHAP Feature Attributions */}
            <div className="space-y-2.5 pt-2 border-t border-slate-200">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-900">
                  SHAP Feature Attributions (XGBoost + Graph)
                </h3>
                <span className="text-[11px] text-slate-500 font-mono">Log-odds Δ</span>
              </div>

              <div className="space-y-2">
                {shapAttributions.map((attr) => {
                  const pct = Math.min(100, Math.round(Math.abs(attr.shapValue) * 65));
                  const isPositiveRisk = attr.shapValue >= 0;
                  return (
                    <div key={attr.feature} className="space-y-1">
                      <div className="flex items-center justify-between text-xs font-mono tabular-nums">
                        <span className="text-slate-700 truncate">
                          {attr.feature}{' '}
                          <span className="text-slate-400">({attr.rawValue})</span>
                        </span>
                        <span
                          className={
                            isPositiveRisk ? 'text-rose-700 font-semibold' : 'text-sky-700'
                          }
                        >
                          {isPositiveRisk ? `+${attr.shapValue.toFixed(3)}` : attr.shapValue.toFixed(3)}
                        </span>
                      </div>
                      <div className="w-full h-1.5 bg-slate-100 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full ${
                            isPositiveRisk ? 'bg-rose-600' : 'bg-sky-600'
                          }`}
                          style={{ width: `${Math.max(6, pct)}%` }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* GNNExplainer 2-Hop Computational Subgraph Attribution */}
            <div className="space-y-2.5 pt-3 border-t border-slate-200">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-semibold text-slate-900">
                  GNNExplainer 2-Hop Subgraph Edge Importance
                </h3>
                <span className="text-[11px] text-slate-500 font-mono">Mask w_e ∈ [0, 1]</span>
              </div>

              <div className="divide-y divide-slate-200 border border-slate-200 rounded-lg overflow-hidden">
                {gnnSubgraphEdges.slice(0, 5).map((edge) => (
                  <div
                    key={edge.edgeId}
                    className="px-3 py-2 text-xs font-mono tabular-nums flex items-center justify-between bg-white"
                  >
                    <div className="truncate pr-2">
                      <span className="font-semibold text-slate-900">{edge.source}</span>
                      <span className="text-slate-400"> → </span>
                      <span className="text-slate-800">{edge.target}</span>
                      {edge.amount !== undefined && (
                        <span className="text-slate-500"> (${edge.amount.toLocaleString()})</span>
                      )}
                    </div>
                    <span className="font-semibold text-rose-700 shrink-0">
                      w={edge.importanceWeight.toFixed(2)}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
