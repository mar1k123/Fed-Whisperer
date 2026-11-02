import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PYTHON_PROJECT_FILES = [
  'README.md',
  'configs/config.yaml',
  'requirements.txt',
  'app.py',
  'desktop_app.py',
  'desktop/main.cjs',
  'src/fraud_ring_finder/__init__.py',
  'src/fraud_ring_finder/config.py',
  'src/fraud_ring_finder/data/generator.py',
  'src/fraud_ring_finder/features/graph_features.py',
  'src/fraud_ring_finder/models/baselines.py',
  'src/fraud_ring_finder/models/gnn.py',
  'src/fraud_ring_finder/evaluation/metrics.py',
  'src/fraud_ring_finder/explainability/explainer.py',
  'src/fraud_ring_finder/pipeline.py',
  'tests/test_fraud_ring_finder.py',
];

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '2mb' }));

  // Endpoint to read the actual Python project files from disk
  app.get('/api/project-files', (_req, res) => {
    try {
      const files: Record<string, string> = {};
      for (const relPath of PYTHON_PROJECT_FILES) {
        const fullPath = path.join(__dirname, relPath);
        if (fs.existsSync(fullPath)) {
          files[relPath] = fs.readFileSync(fullPath, 'utf-8');
        }
      }
      res.json({ files });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Failed to read project files',
      });
    }
  });

  if (process.env.NODE_ENV === 'production') {
    const distPath = path.join(__dirname, 'dist');
    app.use(express.static(distPath));
    app.get('*', (_req, res) => {
      res.sendFile(path.join(distPath, 'index.html'));
    });
  } else {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  }

  const port = Number(process.env.PORT) || 3000;
  app.listen(port, '0.0.0.0', () => {
    console.log(`Fraud Ring Finder server listening on http://localhost:${port}`);
  });
}

startServer();
