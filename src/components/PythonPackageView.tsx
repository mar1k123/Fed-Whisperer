import React, { useEffect, useState } from 'react';
import { Copy, Check } from 'lucide-react';

const FILE_DESCRIPTIONS: Record<string, string> = {
  'README.md': 'Full documentation, architecture diagram, desktop launch guide, empirical results, and limitations.',
  'configs/config.yaml': 'Scraper rate limits, FRED series IDs, equity tickers, embedding anchors, and LOPO regimes.',
  'requirements.txt': 'Python 3.10+ dependencies (pandas, beautifulsoup4, sentence-transformers, statsmodels, google-genai, fredapi, yfinance, streamlit).',
  'desktop_app.py': 'Native PC desktop launcher (opens standalone OS window via PyWebView or native Chromium/Edge app mode).',
  'desktop/main.cjs': 'Electron desktop application wrapper for running the interactive React + Express UI as a PC app.',
  'app.py': 'Full 4-tab Streamlit interactive dashboard with Plotly charts, Diff Mode, and statsmodels OLS.',
  'src/fed_whisperer/__init__.py': 'Package version metadata.',
  'src/fed_whisperer/data/scraper.py': 'Robots.txt-compliant FOMC statement scraper with rate limiting and local disk caching.',
  'src/fed_whisperer/data/market.py': 'FRED Treasury yield (DGS2, DGS10) and yfinance S&P 500 loader + multi-window event calculator.',
  'src/fed_whisperer/scoring/dictionary.py': 'Method A: Domain monetary policy lexicon with 3-token negation window.',
  'src/fed_whisperer/scoring/embeddings.py': 'Method B: Sentence-Transformers semantic projection onto Hawkish-Dovish anchor axis.',
  'src/fed_whisperer/scoring/llm.py': 'Method C: Fixed-rubric Google Gemini API scorer (google-genai SDK, temperature=0.0).',
  'src/fed_whisperer/scoring/agreement.py': 'Pairwise Pearson r, Spearman rho, 3-class Cohen’s kappa, and ΔTone sign concordance.',
  'src/fed_whisperer/analysis/diff.py': 'Diff Mode: SequenceMatcher sentence alignment and edited-sentence tone delta.',
  'src/fed_whisperer/analysis/event_study.py': 'Statsmodels OLS event study regression with HC1 robust standard errors and 95% CIs.',
  'src/fed_whisperer/analysis/robustness.py': 'Window sensitivity (1d/2d/5d), Leave-One-Period-Out (LOPO), and Monte Carlo placebo dates.',
  'tests/test_fed_whisperer.py': 'Pytest unit test suite covering all scorers, diff engine, and econometric regressions.',
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

  return (
    <div className="space-y-6">
      <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-4 border-b border-slate-200 pb-5">
        <div>
          <p className="text-xs font-medium text-slate-500">
            04. Python 3.10+ Package Source Tree & Econometric Documentation
          </p>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900 mt-1">
            Repository Files (src/, configs/, tests/, app.py, README.md)
          </h1>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* Left 4 cols: File Tree */}
        <div className="lg:col-span-4 bg-white border border-slate-200 rounded-xl p-4 space-y-1">
          <div className="text-xs font-semibold text-slate-500 px-2 py-1.5">
            Project Files ({fileList.length})
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

            <button
              type="button"
              onClick={handleCopy}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 text-xs font-medium text-slate-700 bg-white border border-slate-200 rounded-lg hover:bg-slate-100 transition-colors whitespace-nowrap shrink-0"
            >
              {copied ? <Check className="w-3.5 h-3.5 text-emerald-600" /> : <Copy className="w-3.5 h-3.5" />}
              {copied ? 'Copied' : 'Copy File'}
            </button>
          </div>

          <pre className="p-5 text-xs font-mono text-slate-800 overflow-x-auto leading-relaxed max-h-[640px]">
            <code>{loading ? 'Loading project source files...' : activeContent}</code>
          </pre>
        </div>
      </div>
    </div>
  );
};
