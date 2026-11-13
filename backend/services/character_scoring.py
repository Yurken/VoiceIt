"""Evidence-based cast labels, independent of model size and book length.

Confidence describes support for a speaker attribution, not how often a name
was returned. Repeated guesses therefore do not become high-confidence facts.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

_UNRESOLVED = {"", "未识别角色", "unknown", "unknown speaker", "narrator", "narration", "旁白"}
_SOURCE_METHODS = {"rule", "source_tag", "repair_source_tag"}
_FALLBACK_METHODS = {"source_fallback", "empty_result_fallback"}


def _attribution_score(segment: dict) -> float:
    metadata = segment.get("analysis_metadata") or {}
    speaker = str(segment.get("speaker") or "").strip().casefold()
    if segment.get("flagged") or speaker in _UNRESOLVED or metadata.get("method") in _FALLBACK_METHODS:
        return 0.25

    evidence = metadata.get("speaker_evidence")
    excerpt = metadata.get("source_excerpt")
    has_source_evidence = (
        isinstance(evidence, str)
        and bool(evidence.strip())
        and isinstance(excerpt, str)
        and evidence in excerpt
    )
    # speaker_verified may only be set by source-tag validation. It permits a
    # verified alias while keeping arbitrary model evidence below high trust.
    verified = metadata.get("speaker_verified") is True
    identity_in_evidence = has_source_evidence and (speaker in evidence.casefold() or verified)
    if metadata.get("method") in _SOURCE_METHODS and verified and has_source_evidence:
        return 0.95
    # An unflagged response is a usable guess, but schema validity alone is not
    # evidence that the model chose the right person.
    return 0.75 if identity_in_evidence else 0.65


def score_character(segments: Sequence[dict]) -> float:
    """Mean support across this character's spoken lines (narration excluded).

    Every attributed line carries the same weight. A long speech or one good
    annotation cannot hide many uncertain speaker assignments.
    """
    scores = [_attribution_score(segment) for segment in segments if segment.get("type") == "dialogue"]
    return round(sum(scores) / len(scores), 4) if scores else 0.0


def _line_count(character: dict) -> int:
    try:
        return max(0, int(character.get("dialogue_count") or 0))
    except (TypeError, ValueError, OverflowError):
        return 0


def assign_character_roles(roster: Sequence[dict]) -> list[dict]:
    """Return copied rows with major/minor roles based on dialogue prominence.

    A major role needs 2% of the book's identified dialogue and 5% of its most
    talkative character's lines. The floor is two lines for a short story and
    five once the book has at least 100 lines. Unresolved speakers and the
    narrator do not contribute to prominence or receive a major-role label.
    """
    identified = [
        character
        for character in roster
        if not character.get("is_narrator")
        and str(character.get("name") or "").strip().casefold() not in _UNRESOLVED
    ]
    counts = [_line_count(character) for character in identified]
    total = sum(counts)
    floor = 5 if total >= 100 else 2
    cutoff = max(floor, math.ceil(total * 0.02), math.ceil(max(counts, default=0) * 0.05))
    return [
        {
            **character,
            "role": "major" if character in identified and _line_count(character) >= cutoff else "minor",
        }
        for character in roster
    ]
