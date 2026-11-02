"""Method B: Embedding-based scoring along a semantic Hawkish-Dovish axis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from src.fed_whisperer.data.scraper import split_sentences

DEFAULT_HAWKISH_ANCHORS = [
    "Inflation remains elevated and upside risks to prices require further policy firming and rate increases.",
    "Tight labor market conditions and strong price pressures warrant maintaining a restrictive monetary policy stance.",
    "The Committee is prepared to tighten monetary policy further if inflationary pressures persist above target.",
    "Ongoing increases in the target range for the federal funds rate will be appropriate to return inflation to 2 percent.",
]

DEFAULT_DOVISH_ANCHORS = [
    "Economic activity has slowed and downside risks to employment warrant lowering the target range and accommodative policy.",
    "Inflation has eased substantially toward the 2 percent objective, allowing the Committee to reduce policy restraint.",
    "Weakening labor demand and disinflationary pressures support a more accommodative stance of monetary policy.",
    "The Committee will support maximum employment by cutting the target rate and maintaining ample liquidity.",
]


@dataclass
class EmbeddingScoreResult:
    """Output of semantic axis projection."""

    score: float  # Continuous value in [-1.0, +1.0]
    raw_projection: float
    sentence_projections: List[float]


class EmbeddingAxisScorer:
    """
    Projects sentences onto a semantic direction vector:
        v_axis = normalize(mean(E(hawkish_anchors)) - mean(E(dovish_anchors)))
    Uses SentenceTransformers (`all-MiniLM-L6-v2`) when installed, with a deterministic
    TF-IDF + SVD fallback for lightweight/offline test environments.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        hawkish_anchors: Optional[List[str]] = None,
        dovish_anchors: Optional[List[str]] = None,
        use_transformer: bool = True,
    ) -> None:
        self.model_name = model_name
        self.hawkish_anchors = hawkish_anchors or DEFAULT_HAWKISH_ANCHORS
        self.dovish_anchors = dovish_anchors or DEFAULT_DOVISH_ANCHORS
        self._model = None

        if use_transformer:
            try:
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(self.model_name)
            except Exception:
                self._model = None

        self._axis_vector: Optional[np.ndarray] = None
        self._vectorizer = None
        self._init_axis()

    def _encode(self, texts: List[str]) -> np.ndarray:
        if self._model is not None:
            vecs = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
            return np.asarray(vecs, dtype=np.float64)

        from sklearn.feature_extraction.text import TfidfVectorizer

        if self._vectorizer is None:
            self._vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
            self._vectorizer.fit(self.hawkish_anchors + self.dovish_anchors)

        mat = self._vectorizer.transform(texts).toarray().astype(np.float64)
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return mat / norms

    def _init_axis(self) -> None:
        h_vecs = self._encode(self.hawkish_anchors)
        d_vecs = self._encode(self.dovish_anchors)
        diff = h_vecs.mean(axis=0) - d_vecs.mean(axis=0)
        norm = np.linalg.norm(diff)
        self._axis_vector = diff / (norm if norm > 0 else 1.0)

    def score_sentence(self, sentence: str) -> float:
        """Project a single sentence onto the hawkish-dovish axis and scale to [-1, 1]."""
        vec = self._encode([sentence])[0]
        raw = float(np.dot(vec, self._axis_vector))
        # Cosine projection onto unit difference axis typically spans [-0.45, +0.45]
        scaled = float(np.tanh(raw * 3.0))
        return round(scaled, 4)

    def score_text(self, text: str) -> EmbeddingScoreResult:
        """Score a full FOMC statement by averaging its sentence projections."""
        sentences = split_sentences(text)
        if not sentences and text.strip():
            sentences = [text.strip()]

        if not sentences:
            return EmbeddingScoreResult(score=0.0, raw_projection=0.0, sentence_projections=[])

        vecs = self._encode(sentences)
        raw_projs = vecs @ self._axis_vector
        scaled_projs = [round(float(np.tanh(val * 3.0)), 4) for val in raw_projs]
        mean_raw = float(np.mean(raw_projs))
        overall = round(float(np.tanh(mean_raw * 3.2)), 4)

        return EmbeddingScoreResult(
            score=overall,
            raw_projection=round(mean_raw, 4),
            sentence_projections=scaled_projs,
        )
