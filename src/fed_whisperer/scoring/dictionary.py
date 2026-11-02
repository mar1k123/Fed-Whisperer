"""Method A: Central-bank dictionary-based hawkish/dovish tone scorer with negation handling."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List

from src.fed_whisperer.data.scraper import split_sentences

# Curated FOMC-specific monetary policy lexicon
HAWKISH_TERMS: Dict[str, float] = {
    "elevated": 1.2,
    "tighten": 1.5,
    "tightening": 1.5,
    "restrictive": 1.6,
    "firming": 1.4,
    "upside risks": 1.4,
    "inflationary pressures": 1.5,
    "price pressures": 1.3,
    "overheating": 1.5,
    "persistent": 1.2,
    "unacceptably high": 1.7,
    "ongoing increases": 1.8,
    "further increases": 1.6,
    "additional policy firming": 1.8,
    "rapid": 1.0,
    "robust": 0.9,
    "strong labor": 1.1,
    "extremely tight": 1.6,
    "imbalances": 1.1,
    "vigilant": 1.2,
    "attentive to inflation": 1.4,
    "strongly committed": 1.1,
    "above target": 1.3,
    "higher": 0.8,
    "increase": 0.7,
    "increases": 0.8,
    "raise": 1.2,
    "raised": 1.2,
}

DOVISH_TERMS: Dict[str, float] = {
    "lower the target range": 1.9,
    "reduce the target range": 1.7,
    "accommodative": 1.6,
    "accommodation": 1.5,
    "ease": 1.3,
    "eased": 1.3,
    "easing": 1.4,
    "lower": 1.0,
    "lowered": 1.3,
    "lowering": 1.3,
    "reduce": 0.9,
    "cut": 1.4,
    "progress on inflation": 1.4,
    "sustainably toward 2 percent": 1.3,
    "further progress": 1.2,
    "roughly in balance": 1.1,
    "better balance": 1.0,
    "downside risks": 1.4,
    "slowed": 1.2,
    "slowing": 1.2,
    "moderated": 1.1,
    "moderating": 1.1,
    "softened": 1.3,
    "subdued": 1.3,
    "transitory": 1.5,
    "patient": 1.4,
    "support": 0.9,
    "supporting": 0.9,
    "disinflation": 1.4,
    "anchored": 0.8,
    "weakness": 1.3,
    "ample": 0.8,
    "below": 0.9,
}

NEGATIONS = {"not", "no", "never", "neither", "nor", "less", "without", "hardly", "little"}


@dataclass
class DictionaryScoreResult:
    """Output of the dictionary-based FOMC tone scorer."""

    score: float  # Continuous index in [-1.0, +1.0]
    hawkish_weight: float
    dovish_weight: float
    matched_hawkish: List[str]
    matched_dovish: List[str]
    sentence_scores: List[float]


class DictionaryToneScorer:
    """Scores text along [-1, +1] using a domain lexicon with local negation inversion."""

    def __init__(
        self,
        hawkish_lexicon: Dict[str, float] | None = None,
        dovish_lexicon: Dict[str, float] | None = None,
        negation_window: int = 3,
    ) -> None:
        self.hawkish = hawkish_lexicon or HAWKISH_TERMS
        self.dovish = dovish_lexicon or DOVISH_TERMS
        self.negation_window = negation_window

    def _is_negated(self, tokens: List[str], token_idx: int) -> bool:
        start = max(0, token_idx - self.negation_window)
        window = tokens[start:token_idx]
        return any(w in NEGATIONS for w in window)

    def score_sentence(self, sentence: str) -> Dict[str, float | List[str]]:
        """Score a single sentence and return hawkish/dovish weights and net score."""
        lower = sentence.lower()
        tokens = re.findall(r"[a-z]+", lower)

        h_weight = 0.0
        d_weight = 0.0
        h_matches: List[str] = []
        d_matches: List[str] = []

        # Match multi-word and single-word phrases
        for phrase, weight in self.hawkish.items():
            for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", lower):
                prefix_tokens = re.findall(r"[a-z]+", lower[: match.start()])
                if self._is_negated(prefix_tokens, len(prefix_tokens)):
                    d_weight += weight * 0.8
                    d_matches.append(f"NOT_{phrase}")
                else:
                    h_weight += weight
                    h_matches.append(phrase)

        for phrase, weight in self.dovish.items():
            for match in re.finditer(r"\b" + re.escape(phrase) + r"\b", lower):
                prefix_tokens = re.findall(r"[a-z]+", lower[: match.start()])
                if self._is_negated(prefix_tokens, len(prefix_tokens)):
                    h_weight += weight * 0.8
                    h_matches.append(f"NOT_{phrase}")
                else:
                    d_weight += weight
                    d_matches.append(phrase)

        denom = h_weight + d_weight
        net = (h_weight - d_weight) / (denom + 1.5) if denom > 0 else 0.0
        net = max(-1.0, min(1.0, net))
        return {
            "score": round(float(net), 4),
            "hawkish_weight": round(float(h_weight), 4),
            "dovish_weight": round(float(d_weight), 4),
            "matched_hawkish": h_matches,
            "matched_dovish": d_matches,
        }

    def score_text(self, text: str) -> DictionaryScoreResult:
        """Compute statement-level and sentence-level dictionary tone scores."""
        sentences = split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]

        total_h = 0.0
        total_d = 0.0
        all_h: List[str] = []
        all_d: List[str] = []
        sent_scores: List[float] = []

        for s in sentences:
            res = self.score_sentence(s)
            total_h += float(res["hawkish_weight"])
            total_d += float(res["dovish_weight"])
            all_h.extend(res["matched_hawkish"])  # type: ignore[arg-type]
            all_d.extend(res["matched_dovish"])  # type: ignore[arg-type]
            sent_scores.append(float(res["score"]))

        denom = total_h + total_d
        overall = (total_h - total_d) / (denom + 2.0) if denom > 0 else 0.0
        overall = max(-1.0, min(1.0, overall))

        return DictionaryScoreResult(
            score=round(float(overall), 4),
            hawkish_weight=round(float(total_h), 4),
            dovish_weight=round(float(total_d), 4),
            matched_hawkish=all_h,
            matched_dovish=all_d,
            sentence_scores=sent_scores,
        )
