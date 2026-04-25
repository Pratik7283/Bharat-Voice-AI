from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import unicodedata
from typing import Any

import regex as re


@dataclass(frozen=True)
class PronunciationAlignmentStep:
    operation: str
    reference: str | None
    recognized: str | None


@dataclass(frozen=True)
class PronunciationScoreResult:
    score: int
    feedback_hi: str
    transcript: str
    target_phrase: str
    language_code: str
    distance: int
    reference_length: int
    normalized_distance: float
    alignment: list[PronunciationAlignmentStep]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["alignment"] = [asdict(step) for step in self.alignment]
        return payload


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKC", text).strip().lower()
    normalized = re.sub(r"[^\p{L}\p{N}\p{M}\s]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _tokenize_units(text: str) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    if " " in normalized:
        tokens = [token for token in normalized.split(" ") if token]
    else:
        tokens = [cluster for cluster in re.findall(r"\X", normalized) if cluster.strip()]

    return tokens


def _levenshtein_alignment(reference: list[str], hypothesis: list[str]) -> tuple[int, list[PronunciationAlignmentStep]]:
    rows = len(reference) + 1
    cols = len(hypothesis) + 1
    dp = [[0] * cols for _ in range(rows)]
    backtrace: list[list[tuple[int, int, str] | None]] = [[None] * cols for _ in range(rows)]

    for i in range(1, rows):
        dp[i][0] = i
        backtrace[i][0] = (i - 1, 0, "delete")
    for j in range(1, cols):
        dp[0][j] = j
        backtrace[0][j] = (0, j - 1, "insert")

    for i in range(1, rows):
        for j in range(1, cols):
            if reference[i - 1] == hypothesis[j - 1]:
                cost = 0
            else:
                cost = 1

            options = [
                (dp[i - 1][j] + 1, i - 1, j, "delete"),
                (dp[i][j - 1] + 1, i, j - 1, "insert"),
                (dp[i - 1][j - 1] + cost, i - 1, j - 1, "match" if cost == 0 else "substitute"),
            ]
            best_cost, prev_i, prev_j, op = min(options, key=lambda item: (item[0], 0 if item[3] == "match" else 1))
            dp[i][j] = best_cost
            backtrace[i][j] = (prev_i, prev_j, op)

    alignment: list[PronunciationAlignmentStep] = []
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        step = backtrace[i][j]
        if step is None:
            break
        prev_i, prev_j, op = step
        if op == "match":
            alignment.append(
                PronunciationAlignmentStep(
                    operation="match",
                    reference=reference[i - 1],
                    recognized=hypothesis[j - 1],
                )
            )
        elif op == "substitute":
            alignment.append(
                PronunciationAlignmentStep(
                    operation="substitute",
                    reference=reference[i - 1],
                    recognized=hypothesis[j - 1],
                )
            )
        elif op == "delete":
            alignment.append(
                PronunciationAlignmentStep(
                    operation="delete",
                    reference=reference[i - 1],
                    recognized=None,
                )
            )
        else:
            alignment.append(
                PronunciationAlignmentStep(
                    operation="insert",
                    reference=None,
                    recognized=hypothesis[j - 1],
                )
            )
        i, j = prev_i, prev_j

    alignment.reverse()
    return dp[-1][-1], alignment


def _summarize_feedback(alignment: list[PronunciationAlignmentStep], max_items: int = 3) -> str:
    hints: list[str] = []
    for step in alignment:
        if step.operation == "substitute" and step.reference and step.recognized:
            hints.append(f"'{step.reference}' ke badle '{step.recognized}' suna gaya")
        elif step.operation == "delete" and step.reference:
            hints.append(f"'{step.reference}' chhut gaya")
        elif step.operation == "insert" and step.recognized:
            hints.append(f"Extra '{step.recognized}' suna gaya")
        if len(hints) >= max_items:
            break
    return "; ".join(hints)


def _build_feedback_hi(score: int, alignment: list[PronunciationAlignmentStep]) -> str:
    if score >= 9:
        base = "बहुत बढ़िया! आपका उच्चारण लगभग सही है."
    elif score >= 7:
        base = "अच्छा प्रयास। कुछ ध्वनियों को और साफ़ बोलें."
    elif score >= 5:
        base = "ठीक-ठाक प्रयास। कुछ शब्द या ध्वनियाँ अलग सुनाई दीं."
    else:
        base = "अभी अभ्यास की ज़रूरत है। शब्दों को धीमे, साफ़ और स्थिर तरीके से बोलें."

    hint = _summarize_feedback(alignment)
    if hint:
        base = f"{base} ध्यान दें: {hint}."
    return base


def score_pronunciation(user_audio: str | Path, target_phrase: str, language_code: str) -> PronunciationScoreResult:
    from asr import get_asr_service

    audio_path = Path(user_audio)
    transcript = get_asr_service().transcribe(audio_path, language_code=language_code)

    reference_tokens = _tokenize_units(target_phrase)
    recognized_tokens = _tokenize_units(transcript)

    if not reference_tokens:
        raise ValueError("target_phrase is empty")

    distance, alignment = _levenshtein_alignment(reference_tokens, recognized_tokens)
    normalized_distance = min(1.0, distance / max(1, len(reference_tokens)))
    score = max(1, min(10, round(10 - (normalized_distance * 9))))
    feedback_hi = _build_feedback_hi(score, alignment)

    return PronunciationScoreResult(
        score=score,
        feedback_hi=feedback_hi,
        transcript=transcript,
        target_phrase=target_phrase,
        language_code=language_code,
        distance=distance,
        reference_length=len(reference_tokens),
        normalized_distance=normalized_distance,
        alignment=alignment,
    )
