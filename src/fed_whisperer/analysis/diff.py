"""Diff Mode: Sentence-level diff alignment and tone shift scoring between consecutive FOMC statements."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import Callable, List, Optional


@dataclass
class SentenceDiffItem:
    """Represents a single aligned sentence comparison between statement t-1 and statement t."""

    tag: str  # "unchanged", "modified", "added", "removed"
    prev_sentence: Optional[str]
    curr_sentence: Optional[str]
    prev_score: float
    curr_score: float
    delta_score: float
    similarity: float


@dataclass
class StatementDiffReport:
    """Aggregated diff report between two consecutive FOMC statements."""

    prev_date: str
    curr_date: str
    items: List[SentenceDiffItem]
    overall_prev_score: float
    overall_curr_score: float
    net_tone_delta: float
    changed_sentences_delta: float
    turnover_ratio: float


def compute_statement_diff(
    prev_date: str,
    prev_sentences: List[str],
    curr_date: str,
    curr_sentences: List[str],
    sentence_scorer: Callable[[str], float],
    modification_similarity_threshold: float = 0.45,
) -> StatementDiffReport:
    """
    Align sentences between consecutive FOMC statements using SequenceMatcher,
    classify pairs into unchanged/modified/added/removed, and compute the tone delta
    attributable specifically to edited/added/removed sentences.
    """
    matcher = SequenceMatcher(a=prev_sentences, b=curr_sentences, autojunk=False)
    items: List[SentenceDiffItem] = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for prev_s, curr_s in zip(prev_sentences[i1:i2], curr_sentences[j1:j2]):
                sc = round(float(sentence_scorer(curr_s)), 4)
                items.append(
                    SentenceDiffItem(
                        tag="unchanged",
                        prev_sentence=prev_s,
                        curr_sentence=curr_s,
                        prev_score=sc,
                        curr_score=sc,
                        delta_score=0.0,
                        similarity=1.0,
                    )
                )
        elif tag == "replace":
            prev_chunk = prev_sentences[i1:i2]
            curr_chunk = curr_sentences[j1:j2]
            paired = min(len(prev_chunk), len(curr_chunk))

            for k in range(paired):
                p_s = prev_chunk[k]
                c_s = curr_chunk[k]
                sim = SequenceMatcher(a=p_s.lower(), b=c_s.lower()).ratio()
                p_sc = round(float(sentence_scorer(p_s)), 4)
                c_sc = round(float(sentence_scorer(c_s)), 4)
                if sim >= modification_similarity_threshold:
                    items.append(
                        SentenceDiffItem(
                            tag="modified",
                            prev_sentence=p_s,
                            curr_sentence=c_s,
                            prev_score=p_sc,
                            curr_score=c_sc,
                            delta_score=round(c_sc - p_sc, 4),
                            similarity=round(sim, 4),
                        )
                    )
                else:
                    items.append(
                        SentenceDiffItem(
                            tag="removed",
                            prev_sentence=p_s,
                            curr_sentence=None,
                            prev_score=p_sc,
                            curr_score=0.0,
                            delta_score=round(-p_sc, 4),
                            similarity=0.0,
                        )
                    )
                    items.append(
                        SentenceDiffItem(
                            tag="added",
                            prev_sentence=None,
                            curr_sentence=c_s,
                            prev_score=0.0,
                            curr_score=c_sc,
                            delta_score=round(c_sc, 4),
                            similarity=0.0,
                        )
                    )

            for p_s in prev_chunk[paired:]:
                p_sc = round(float(sentence_scorer(p_s)), 4)
                items.append(
                    SentenceDiffItem(
                        tag="removed",
                        prev_sentence=p_s,
                        curr_sentence=None,
                        prev_score=p_sc,
                        curr_score=0.0,
                        delta_score=round(-p_sc, 4),
                        similarity=0.0,
                    )
                )
            for c_s in curr_chunk[paired:]:
                c_sc = round(float(sentence_scorer(c_s)), 4)
                items.append(
                    SentenceDiffItem(
                        tag="added",
                        prev_sentence=None,
                        curr_sentence=c_s,
                        prev_score=0.0,
                        curr_score=c_sc,
                        delta_score=round(c_sc, 4),
                        similarity=0.0,
                    )
                )
        elif tag == "delete":
            for p_s in prev_sentences[i1:i2]:
                p_sc = round(float(sentence_scorer(p_s)), 4)
                items.append(
                    SentenceDiffItem(
                        tag="removed",
                        prev_sentence=p_s,
                        curr_sentence=None,
                        prev_score=p_sc,
                        curr_score=0.0,
                        delta_score=round(-p_sc, 4),
                        similarity=0.0,
                    )
                )
        elif tag == "insert":
            for c_s in curr_sentences[j1:j2]:
                c_sc = round(float(sentence_scorer(c_s)), 4)
                items.append(
                    SentenceDiffItem(
                        tag="added",
                        prev_sentence=None,
                        curr_sentence=c_s,
                        prev_score=0.0,
                        curr_score=c_sc,
                        delta_score=round(c_sc, 4),
                        similarity=0.0,
                    )
                )

    prev_scores = [it.prev_score for it in items if it.prev_sentence is not None]
    curr_scores = [it.curr_score for it in items if it.curr_sentence is not None]
    overall_prev = sum(prev_scores) / len(prev_scores) if prev_scores else 0.0
    overall_curr = sum(curr_scores) / len(curr_scores) if curr_scores else 0.0

    changed = [it.delta_score for it in items if it.tag != "unchanged"]
    changed_delta = sum(changed) / len(changed) if changed else 0.0

    total_items = len(items)
    changed_count = sum(1 for it in items if it.tag != "unchanged")
    turnover = changed_count / total_items if total_items > 0 else 0.0

    return StatementDiffReport(
        prev_date=prev_date,
        curr_date=curr_date,
        items=items,
        overall_prev_score=round(overall_prev, 4),
        overall_curr_score=round(overall_curr, 4),
        net_tone_delta=round(overall_curr - overall_prev, 4),
        changed_sentences_delta=round(changed_delta, 4),
        turnover_ratio=round(turnover, 4),
    )
