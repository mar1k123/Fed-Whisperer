import express from 'express';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { GoogleGenAI, Type } from '@google/genai';
import dotenv from 'dotenv';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const FIXED_MONETARY_RUBRIC = `You are a senior quantitative monetary economist evaluating Federal Open Market Committee (FOMC) policy statements.
Score the statement on a continuous scale from -1.00 (maximally dovish) to +1.00 (maximally hawkish) using this strict 5-point anchor rubric:

-1.00 to -0.60 (Strongly Dovish): Explicit rate cuts, QE/balance-sheet expansion, emphasis on downside employment risks, or characterization of inflation as subdued/below target.
-0.59 to -0.20 (Moderately Dovish): Signaling patience, noting slowing economic activity or easing inflation pressures, preparing to reduce policy restraint.
-0.19 to +0.19 (Neutral / Balanced): Balanced two-sided risks to employment and inflation, data-dependent posture without bias toward tightening or easing.
+0.20 to +0.59 (Moderately Hawkish): Highlighting elevated inflation, tight labor markets, readiness to maintain restrictive policy or consider additional firming.
+0.60 to +1.00 (Strongly Hawkish): Active rate hikes, expectation of "ongoing increases" in the target range, quantitative tightening acceleration, or strong vigilance against upside inflation risks.`;

const PYTHON_PROJECT_FILES = [
  'README.md',
  'configs/config.yaml',
  'requirements.txt',
  'desktop_app.py',
  'desktop/main.cjs',
  'app.py',
  'src/fed_whisperer/__init__.py',
  'src/fed_whisperer/data/scraper.py',
  'src/fed_whisperer/data/market.py',
  'src/fed_whisperer/scoring/dictionary.py',
  'src/fed_whisperer/scoring/embeddings.py',
  'src/fed_whisperer/scoring/llm.py',
  'src/fed_whisperer/scoring/agreement.py',
  'src/fed_whisperer/analysis/diff.py',
  'src/fed_whisperer/analysis/event_study.py',
  'src/fed_whisperer/analysis/robustness.py',
  'tests/test_fed_whisperer.py',
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
      res.status(500).json({ error: err instanceof Error ? err.message : 'Failed to read project files' });
    }
  });

  // Live Gemini Rubric Scoring endpoint (Method C with temperature = 0 equivalent deterministic seed)
  app.post('/api/score-llm', async (req, res) => {
    try {
      const { text } = req.body as { text?: string };
      if (!text || typeof text !== 'string' || text.trim().length < 20) {
        res.status(400).json({ error: 'Please provide an FOMC statement excerpt of at least 20 characters.' });
        return;
      }

      const apiKey = process.env.GEMINI_API_KEY;
      if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
        res.status(503).json({
          error: 'GEMINI_API_KEY is not configured in environment.',
          fallbackAvailable: true,
        });
        return;
      }

      const ai = new GoogleGenAI({
        apiKey,
        httpOptions: {
          headers: {
            'User-Agent': 'aistudio-build',
          },
        },
      });

      const response = await ai.models.generateContent({
        model: 'gemini-3.8-flash',
        contents: `Evaluate the following FOMC monetary policy statement text under the fixed rubric:\n\n${text}`,
        config: {
          systemInstruction: FIXED_MONETARY_RUBRIC,
          seed: 42,
          responseMimeType: 'application/json',
          responseSchema: {
            type: Type.OBJECT,
            properties: {
              score: {
                type: Type.NUMBER,
                description: 'Overall tone score in [-1.0, 1.0]',
              },
              inflation_stance: {
                type: Type.NUMBER,
                description: 'Inflation assessment sub-score in [-1.0, 1.0]',
              },
              growth_labor_stance: {
                type: Type.NUMBER,
                description: 'Economic growth and labor market sub-score in [-1.0, 1.0]',
              },
              forward_guidance_stance: {
                type: Type.NUMBER,
                description: 'Policy rate and balance sheet forward guidance sub-score in [-1.0, 1.0]',
              },
              rationale: {
                type: Type.STRING,
                description: 'Concise 1-2 sentence explanation citing specific phrases from the statement.',
              },
            },
            required: [
              'score',
              'inflation_stance',
              'growth_labor_stance',
              'forward_guidance_stance',
              'rationale',
            ],
          },
        },
      });

      const parsed = JSON.parse((response.text ?? '{}').trim());
      res.json({
        score: Math.max(-1, Math.min(1, Number(parsed.score ?? 0))),
        inflation_stance: Math.max(-1, Math.min(1, Number(parsed.inflation_stance ?? 0))),
        growth_labor_stance: Math.max(-1, Math.min(1, Number(parsed.growth_labor_stance ?? 0))),
        forward_guidance_stance: Math.max(-1, Math.min(1, Number(parsed.forward_guidance_stance ?? 0))),
        rationale: String(parsed.rationale ?? ''),
        model: 'gemini-3.8-flash',
      });
    } catch (err) {
      res.status(500).json({
        error: err instanceof Error ? err.message : 'Gemini rubric evaluation failed.',
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
    console.log(`Fed Whisperer server listening on http://localhost:${port}`);
  });
}

startServer();
