import React, { useEffect, useState } from 'react';
import { Copy, Check, Download } from 'lucide-react';

const FILE_DESCRIPTIONS: Record<string, string> = {
  'README.md':
    'Full project documentation: why synthetic data is used, what that implies, how to plug in real data, and zero-invented-numbers reproduction guide.',
  'configs/config.yaml':
    'Configurable seed, accounts/devices/merchants counts, fraud rate (0.5%–2.0%), pattern weights, temporal splits, and XGBoost/GraphSAGE/GAT hyperparameters.',
  'requirements.txt':
    'Python 3.10+ dependencies (torch, torch-geometric, networkx, scikit-learn, xgboost, shap, streamlit, pyvis, pytest).',
  'app.py':
    'Interactive Streamlit + PyVis application highlighting flagged fraud rings and displaying per-account SHAP & GNNExplainer risk cards.',
  'desktop_app.py':
    'Native PC desktop window launcher (opens frameless OS window via PyWebView or Chromium/Edge --app mode).',
  'desktop/main.cjs':
    'Electron desktop application wrapper for launching the Fraud Ring Finder workstation.',
  'src/fraud_ring_finder/__init__.py':
    'Python package version & exports.',
  'src/fraud_ring_finder/config.py':
    'Typed dataclass loader for configs/config.yaml.',
  'src/fraud_ring_finder/data/generator.py':
    'Deterministic seeded transaction network generator planting money-mule chains, shared-device rings, burst activity, and circular transfers + distribution shift.',
  'src/fraud_ring_finder/features/graph_features.py':
    'Strictly leak-free temporal feature builder (tabular features + degree, PageRank, shared-device count, flow pass-through, and cycle motifs).',
  'src/fraud_ring_finder/models/baselines.py':
    'Tabular XGBoost and Graph-Feature-Enhanced XGBoost baselines with automatic class-imbalance scaling.',
  'src/fraud_ring_finder/models/gnn.py':
    'PyTorch Geometric GraphSAGE and Multi-Head GAT models trained with Focal Loss and Weighted BCE Loss.',
  'src/fraud_ring_finder/evaluation/metrics.py':
    'Temporal split zero-leakage verifier, PR-AUC, Recall @ Fixed Precision, and Fraud Pattern Ablation evaluator.',
  'src/fraud_ring_finder/explainability/explainer.py':
    'TreeSHAP local/global feature attributions for XGBoost and PyG GNNExplainer 2-hop computational subgraphs.',
  'src/fraud_ring_finder/pipeline.py':
    'CLI end-to-end training, evaluation, ablation, and distribution-shift benchmark runner.',
  'tests/test_fraud_ring_finder.py':
    'Pytest suite verifying bitwise generator determinism, fraud rate bounds, and zero future temporal leakage.',
};

export const PythonPackageView: React.FC = () => {
  const [files, setFiles] = useState<Record<string, string>>({});
  const [selectedFile, setSelectedFile] = useState<string>('README.md');
  const [copied, setCopied] = useState<boolean>(false);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    let active = true;
    fetch('/api/project-files')
      .then((r) => r.json())
      .then((data: { files?: Record<string, string> }) => {
        if (active && data.files) {
          setFiles(data.files);
        }
      })
      .catch(() => {})
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, []);

  const fileList = Object.keys(FILE_DESCRIPTIONS);
  const activeContent = files[selectedFile] || 'Loading file content...';

  const handleCopy = () => {
    if (!files[selectedFile]) return;
    navigator.clipboard.writeText(files[selectedFile]);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const handleDownload = () => {
    const content = files[selectedFile];
    if (!content) return;
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = selectedFile.replace(/\//g, '_');
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-4">
        <div>
          <p className="text-xs font-medium text-slate-500">
            04. Full Python 3.10+ Repository Source Tree &amp; Documentation
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-0.5">
            Project Files (src/fraud_ring_finder/, configs/, tests/, app.py, README.md)
          </h1>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 4 cols: File Tree */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-4 space-y-1">
          <div className="text-xs font-semibold text-slate-500 px-2 py-1.5">
            Repository Files ({fileList.length})
          </div>
          {fileList.map((pathKey) => {
            const isSelected = pathKey === selectedFile;
            return (
              <button
                key={pathKey}
                type="button"
                onClick={() => setSelectedFile(pathKey)}
                className={`w-full text-left px-3 py-2 rounded-lg text-xs font-mono transition-colors flex items-center justify-between ${
                  isSelected
                    ? 'bg-slate-900 text-white font-semibold'
                    : 'text-slate-700 hover:bg-slate-100'
                }`}
              >
                <span className="truncate">{pathKey}</span>
              </button>
            );
          })}
        </div>

        {/* Right 8 cols: Code Viewer */}
        <div className="lg:col-span-8 bg-white border border-slate-200 rounded-xl overflow-hidden">
          <div className="flex items-center justify-between gap-4 px-5 py-3.5 bg-slate-50 border-b border-slate-200">
            <div>
              <div className="text-xs font-mono font-bold text-slate-900">{selectedFile}</div>
              <div className="text-xs text-slate-500 mt-0.5">
                {FILE_DESCRIPTIONS[selectedFile]}
              </div>
            </div>

            <div className="flex items-center gap-2 shrink-0">
              <button
                type="button"
                onClick={handleCopy}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0"
              >
                {copied ? (
                  <Check className="w-3.5 h-3.5 text-emerald-600" />
                ) : (
                  <Copy className="w-3.5 h-3.5" />
                )}
                {copied ? 'Copied' : 'Copy File'}
              </button>
              <button
                type="button"
                onClick={handleDownload}
                className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-white bg-slate-900 rounded-lg hover:bg-slate-800 transition-colors whitespace-nowrap shrink-0"
              >
                <Download className="w-3.5 h-3.5" />
                Download
              </button>
            </div>
          </div>

          <pre className="p-5 text-xs font-mono text-slate-800 overflow-x-auto leading-relaxed max-h-[640px]">
            <code>{loading ? 'Loading project source files...' : activeContent}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
