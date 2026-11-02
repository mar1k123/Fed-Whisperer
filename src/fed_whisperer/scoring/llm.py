"""Method C: LLM scoring with a fixed monetary policy rubric and temperature 0 via google-genai."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

FIXED_MONETARY_RUBRIC = """You are a senior quantitative monetary economist evaluating Federal Open Market Committee (FOMC) policy statements.
Score the statement on a continuous scale from -1.00 (maximally dovish) to +1.00 (maximally hawkish) using this strict 5-point anchor rubric:

-1.00 to -0.60 (Strongly Dovish): Explicit rate cuts, QE/balance-sheet expansion, emphasis on downside employment risks, or characterization of inflation as subdued/below target.
-0.59 to -0.20 (Moderately Dovish): Signaling patience, noting slowing economic activity or easing inflation pressures, preparing to reduce policy restraint.
-0.19 to +0.19 (Neutral / Balanced): Balanced two-sided risks to employment and inflation, data-dependent posture without bias toward tightening or easing.
+0.20 to +0.59 (Moderately Hawkish): Highlighting elevated inflation, tight labor markets, readiness to maintain restrictive policy or consider additional firming.
+0.60 to +1.00 (Strongly Hawkish): Active rate hikes, expectation of "ongoing increases" in the target range, quantitative tightening acceleration, or strong vigilance against upside inflation risks.

Return ONLY valid JSON with keys:
- "score": float in [-1.0, 1.0]
- "inflation_stance": float in [-1.0, 1.0]
- "growth_labor_stance": float in [-1.0, 1.0]
- "forward_guidance_stance": float in [-1.0, 1.0]
- "rationale": concise 1-2 sentence explanation citing specific phrases from the statement.
"""


@dataclass
class LLMScoreResult:
    """Structured output from Gemini rubric evaluation."""

    score: float
    inflation_stance: float
    growth_labor_stance: float
    forward_guidance_stance: float
    rationale: str
    model: str


class GeminiRubricScorer:
    """Evaluates FOMC statements using the official `google-genai` SDK at temperature=0.0."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.5-flash",
        cache_dir: str | Path = ".cache/fed_whisperer/llm_scores",
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _cache_file(self, text: str) -> Path:
        digest = hashlib.sha256((self.model + "::" + text).encode("utf-8")).hexdigest()[:20]
        return self.cache_dir / f"llm_{digest}.json"

    def score_text(self, text: str, use_cache: bool = True) -> LLMScoreResult:
        """Score an FOMC statement or diff excerpt using Gemini with temperature=0.0."""
        cache_path = self._cache_file(text)
        if use_cache and cache_path.exists():
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            return LLMScoreResult(**data)

        if not self.api_key:
            raise EnvironmentError(
                "GEMINI_API_KEY is required for GeminiRubricScorer. "
                "Set GEMINI_API_KEY in your environment."
            )

        from google import genai
        from google.genai import types

        client = genai.Client(api_key=self.api_key)
        response = client.models.generate_content(
            model=self.model,
            contents=f"FOMC Statement Text:\n\n{text}",
            config=types.GenerateContentConfig(
                system_instruction=FIXED_MONETARY_RUBRIC,
                temperature=0.0,
                response_mime_type="application/json",
            ),
        )

        payload = json.loads(response.text or "{}")
        result = LLMScoreResult(
            score=round(max(-1.0, min(1.0, float(payload.get("score", 0.0)))), 4),
            inflation_stance=round(float(payload.get("inflation_stance", 0.0)), 4),
            growth_labor_stance=round(float(payload.get("growth_labor_stance", 0.0)), 4),
            forward_guidance_stance=round(float(payload.get("forward_guidance_stance", 0.0)), 4),
            rationale=str(payload.get("rationale", "")),
            model=self.model,
        )
        cache_path.write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")
        return result
