# ruff: noqa: RUF001
"""Rebuild nested quotation boundaries before any audio/story is created.

Only segment structure changes; source text, cast rows and voice profiles stay
intact. The caller owns backup, commit and rollback.
"""

from __future__ import annotations

import copy
import uuid
from datetime import UTC, datetime
from itertools import pairwise

from sqlalchemy.orm import Session

from ..database import Book, BookCharacter, BookSegment, Chapter


def _quoted_ranges(text: str) -> list[tuple[int, int]]:
    from .character_identity import quoted_ranges

    return quoted_ranges(text)


def _quote_is_speech(text: str, start: int, end: int) -> bool:
    from .character_identity import quote_is_speech

    return quote_is_speech(text, start, end)


def _coverage(text: str) -> str:
    return "".join(text.split())


def _source_positions(source: str, segments: list[BookSegment]) -> list[tuple[int, int, BookSegment]]:
    if _coverage("".join(segment.text for segment in segments)) != _coverage(source):
        raise ValueError("原有片段未完整保留章节原文，未重建引号边界")
    positioned, cursor = [], 0
    for segment in segments:
        start = source.find(segment.text, cursor)
        if start < 0:
            raise ValueError("原有片段无法逐字定位原文，未重建引号边界")
        end = start + len(segment.text)
        positioned.append((start, end, segment))
        cursor = end
    return positioned


def _new_spans(source: str, ranges: list[tuple[int, int]]) -> list[tuple[int, int, str]]:
    spans, cursor = [], 0

    def add(start: int, end: int, kind: str) -> None:
        raw = source[start:end]
        left = len(raw) - len(raw.lstrip())
        right = len(raw.rstrip())
        if right > left:
            spans.append((start + left, start + right, kind))

    for start, end in ranges:
        if not cursor <= start < end <= len(source):
            raise ValueError("引号解析返回了重叠或越界的原文区间")
        add(cursor, start, "narration")
        add(start, end, "dialogue" if _quote_is_speech(source, start, end) else "narration")
        cursor = end
    add(cursor, len(source), "narration")
    if _coverage("".join(source[start:end] for start, end, _ in spans)) != _coverage(source):
        raise ValueError("新引号边界未完整保留章节原文")
    return spans


def _preserve_narration_boundaries(
    source: str,
    spans: list[tuple[int, int, str]],
    positioned: list[tuple[int, int, BookSegment]],
    quote_ranges: list[tuple[int, int]],
) -> list[tuple[int, int, str]]:
    """Keep existing safe narration chunks outside top-level quoted ranges."""
    old_boundaries = {position for start, end, _ in positioned for position in (start, end)}
    quotes = set(quote_ranges)
    result = []
    for start, end, kind in spans:
        if kind != "narration" or (start, end) in quotes:
            result.append((start, end, kind))
            continue
        boundaries = sorted({start, end, *(p for p in old_boundaries if start < p < end)})
        for left, right in pairwise(boundaries):
            raw = source[left:right]
            trimmed_left = left + len(raw) - len(raw.lstrip())
            trimmed_right = left + len(raw.rstrip())
            if trimmed_left < trimmed_right:
                result.append((trimmed_left, trimmed_right, kind))
    if _coverage("".join(source[start:end] for start, end, _ in result)) != _coverage(source):
        raise ValueError("旁白边界未完整保留章节原文")
    return result


def repair_book_quote_boundaries(book_id: str, db: Session, *, dry_run: bool = True) -> dict:
    """Rebuild only chapters containing nested quotes; flush without commit.

    Refuses the entire book if any segment has audio or any chapter is linked
    to a story, including chapters whose quotation boundaries would not change.
    An exact existing source span/type retains its ID. New dialogue spans
    inherit the most-overlapping old dialogue's voice settings for the following
    source-attribution refinement pass to validate.
    """
    with db.no_autoflush:
        book = db.get(Book, book_id)
        if book is None:
            raise ValueError("未找到这本书")
        if book.status in {"analyzing", "generating", "exporting"}:
            raise ValueError("小说有正在运行的任务，未重建引号边界")
        chapters = db.query(Chapter).filter_by(book_id=book_id).order_by(Chapter.number, Chapter.id).all()
        if not chapters:
            raise ValueError("这本书没有章节")
        chapter_segments = [
            db.query(BookSegment).filter_by(chapter_id=chapter.id).order_by(BookSegment.order, BookSegment.id).all()
            for chapter in chapters
        ]
        if any(chapter.story_id for chapter in chapters):
            raise ValueError("这本书已有故事时间线，未重建引号边界")
        if any(segment.generation_id for segments in chapter_segments for segment in segments):
            raise ValueError("这本书已有生成音频，未重建引号边界")
        narrators = db.query(BookCharacter).filter_by(book_id=book_id, is_narrator=True).all()
        narrator = max(narrators, key=lambda row: row.dialogue_count or 0, default=None)

        plans = []
        examples = []
        before_count = sum(len(segments) for segments in chapter_segments)
        after_count = before_count
        reused_count = new_count = removed_count = nested_count = 0
        for chapter, segments in zip(chapters, chapter_segments, strict=True):
            source = chapter.raw_text
            ranges = _quoted_ranges(source)
            nested = [
                (start, end)
                for start, end in ranges
                if _quoted_ranges(source[start + 1:end - 1])
            ]
            if not nested:
                continue
            if narrator is None:
                raise ValueError("这本书尚无旁白角色，未重建引号边界")
            positioned = _source_positions(source, segments)
            new_spans = _preserve_narration_boundaries(source, _new_spans(source, ranges), positioned, ranges)
            old_by_span = {(start, end, segment.type, segment.text): segment for start, end, segment in positioned}
            new_rows, reused_ids = [], set()
            for order, (start, end, kind) in enumerate(new_spans):
                text = source[start:end]
                old = old_by_span.get((start, end, kind, text))
                if old is not None:
                    new_rows.append((old, order, kind))
                    reused_ids.add(old.id)
                    continue
                overlapping = [
                    (min(end, old_end) - max(start, old_start), segment)
                    for old_start, old_end, segment in positioned
                    if old_start < end and old_end > start and segment.type == "dialogue"
                ]
                inherited = max(overlapping, key=lambda pair: pair[0])[1] if overlapping else None
                metadata = copy.deepcopy(inherited.analysis_metadata or {}) if inherited is not None else {}
                metadata.update(
                    method="quote_boundary_repair",
                    speaker_verified=False,
                    quote_boundary_repair={"inherited_segment_id": inherited.id if inherited else None,
                                           "source_start": start, "source_end": end},
                )
                row = BookSegment(
                    id=str(uuid.uuid4()), chapter_id=chapter.id, order=order, text=text, type=kind,
                    character_id=(inherited.character_id if inherited is not None and kind == "dialogue" else narrator.id),
                    emotion=inherited.emotion if inherited is not None else None,
                    emotion_intensity=inherited.emotion_intensity if inherited is not None else None,
                    delivery=inherited.delivery if inherited is not None else None,
                    voice_stage=inherited.voice_stage if inherited is not None and kind == "dialogue" else None,
                    flagged=kind == "dialogue", analysis_metadata=metadata, audio_status="none",
                )
                new_rows.append((row, order, kind))
            removed = [segment for segment in segments if segment.id not in reused_ids]
            new_ids = [row.id for row, _, _ in new_rows]
            if len(new_ids) != len(set(new_ids)):
                raise ValueError("重建引号产生了重复片段编号")
            after_count += len(new_rows) - len(segments)
            reused_count += len(reused_ids)
            new_count += len(new_rows) - len(reused_ids)
            removed_count += len(removed)
            nested_count += len(nested)
            for start, end in nested:
                if len(examples) < 10:
                    examples.append({"chapter": chapter.number, "start": start, "end": end,
                                     "excerpt": source[start:min(end, start + 160)]})
            plans.append({"chapter": chapter, "rows": new_rows, "removed": removed,
                          "before": len(segments), "nested": len(nested)})

        report = {
            "book_id": book_id, "title": book.title, "dry_run": dry_run, "source_preserved": True,
            "counts": {"affected_chapters": len(plans), "segments_before": before_count,
                       "segments_after": after_count, "reused_affected_segments": reused_count,
                       "new_segments": new_count, "removed_segments": removed_count, "nested_ranges": nested_count},
            "affected_chapters": [{"id": plan["chapter"].id, "number": plan["chapter"].number,
                                   "title": plan["chapter"].title, "segments_before": plan["before"],
                                   "segments_after": len(plan["rows"]), "nested_ranges": plan["nested"]}
                                  for plan in plans],
            "nested_examples": examples,
        }
        if dry_run:
            return report

    for plan in plans:
        for segment in plan["removed"]:
            db.delete(segment)
        for row, order, kind in plan["rows"]:
            row.order = order
            if kind == "narration":
                row.character_id = narrator.id
            db.add(row)
    if plans:
        book.updated_at = datetime.now(UTC).replace(tzinfo=None)
        db.flush()
    return report
