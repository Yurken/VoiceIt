# ruff: noqa: RUF001
"""Apply source-validated cast refinements without replacing text or voices.

This service never commits. The caller owns the transaction and should back up
the database before applying a reviewed dry-run plan.
"""

from __future__ import annotations

import copy
import unicodedata
from collections import Counter
from collections.abc import Sequence
from datetime import UTC, datetime

from sqlalchemy.orm import Session

from ..database import Book, BookCharacter, BookSegment, Chapter

_UNKNOWN = "未识别角色"
_BUSY = {"analyzing", "generating", "exporting"}


def _normalized(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().casefold()


def _refine_segments(chapter_texts: Sequence[str], chapter_segments: Sequence[list[dict]], roster: Sequence[dict]):
    # Import on use: repair tooling and its CPU tests do not initialise a model.
    from .character_refinement import refine_character_segments

    return refine_character_segments(chapter_texts, chapter_segments, roster)


def _coverage(text: str) -> str:
    return "".join(text.split())


def _validate_segments(chapters: list[Chapter], segments: list[list[dict]], before: list[list[dict]]) -> None:
    if len(segments) != len(chapters):
        raise ValueError("角色修复未完整覆盖全部章节")
    for chapter, refined, original in zip(chapters, segments, before, strict=True):
        if [(s.get("id"), s.get("text"), s.get("order")) for s in refined] != [
            (s["id"], s["text"], s["order"]) for s in original
        ]:
            raise ValueError("角色修复不得更改、删除或重排原文片段")
        if _coverage("".join(s["text"] for s in refined)) != _coverage(chapter.raw_text):
            raise ValueError(f"章节 {chapter.number} 的片段未完整保留原文，未应用修复")
        if any(s.get("type") not in {"dialogue", "narration"} for s in refined):
            raise ValueError("角色修复返回了无效片段类型")


def repair_book_cast(book_id: str, db: Session, *, dry_run: bool = True) -> dict:
    """Refine one inactive book's cast; dry-run returns a serializable report.

    Text, segment IDs/order, generated files and profile rows are preserved.
    Canonical character rows retain their IDs and voice bindings. If an
    assignment/type changes, its existing audio is marked stale for rendering.
    ``dry_run=False`` flushes changes but leaves commit/rollback to the caller.
    """
    with db.no_autoflush:
        book = db.get(Book, book_id)
        if book is None:
            raise ValueError("未找到这本书")
        if book.status in _BUSY:
            raise ValueError("小说正在分析、生成或导出，请等待任务完成后修复角色")
        chapters = db.query(Chapter).filter_by(book_id=book_id).order_by(Chapter.number, Chapter.id).all()
        if not chapters:
            raise ValueError("这本书没有章节")
        rows = db.query(BookCharacter).filter_by(book_id=book_id).all()
        characters_by_id = {row.id: row for row in rows}
        chapter_rows = [
            db.query(BookSegment).filter_by(chapter_id=chapter.id).order_by(BookSegment.order, BookSegment.id).all()
            for chapter in chapters
        ]

        chapter_segments = [
            [
                {
                    "id": segment.id,
                    "order": segment.order,
                    "text": segment.text,
                    "type": segment.type,
                    "speaker": (
                        characters_by_id[segment.character_id].name
                        if segment.character_id in characters_by_id and not characters_by_id[segment.character_id].is_narrator
                        else _UNKNOWN if segment.type == "dialogue" else None
                    ),
                    "flagged": segment.flagged,
                    "emotion": segment.emotion,
                    "intensity": segment.emotion_intensity,
                    "delivery": segment.delivery,
                    "voice_stage": segment.voice_stage,
                    "analysis_metadata": copy.deepcopy(segment.analysis_metadata or {}),
                }
                for segment in segments
            ]
            for segments in chapter_rows
        ]
        before = copy.deepcopy(chapter_segments)
        # Fail before refinement as well: pre-existing missing text must not be
        # hidden by a repair that only changes character identities.
        _validate_segments(chapters, chapter_segments, before)
        roster = [
            {
                "id": row.id,
                "name": row.name,
                "aliases": list(row.aliases or []),
                "dialogue_count": row.dialogue_count,
                "confidence": row.confidence,
                "role": row.role,
                "is_narrator": row.is_narrator,
                "profile_id": row.profile_id,
            }
            for row in rows
            if not row.is_narrator
        ]
        result = _refine_segments([chapter.raw_text for chapter in chapters], chapter_segments, roster)
        _validate_segments(chapters, chapter_segments, before)

        output_characters = copy.deepcopy(result.get("characters") or [])
        name_map = result.get("name_map") or {}
        normalized_map = {_normalized(name): target for name, target in name_map.items()}
        output_by_name = {_normalized(character["name"]): character for character in output_characters}
        if len(output_by_name) != len(output_characters) or not all(output_by_name):
            raise ValueError("角色修复返回了重复或空白姓名")

        target_names = {
            _normalized(segment.get("speaker") or _UNKNOWN)
            for segments in chapter_segments
            for segment in segments
            if segment["type"] == "dialogue"
        }
        if target_names - set(output_by_name):
            raise ValueError("角色修复的对白未对应完整角色表")

        # Prefer exact canonical identities, then a mapped/verified alias row.
        # Original line counts break ties; a renamed real person keeps a voice.
        chosen: dict[str, BookCharacter | None] = {}
        for key in output_by_name:
            exact = [row for row in rows if not row.is_narrator and _normalized(row.name) == key]
            mapped = [
                row
                for row in rows
                if not row.is_narrator
                and (
                    _normalized(normalized_map.get(_normalized(row.name))) == key
                    or any(_normalized(alias) == key for alias in (row.aliases or []))
                )
            ]
            candidates = exact or mapped
            chosen[key] = max(candidates, key=lambda row: row.dialogue_count or 0) if candidates else None
        chosen_ids = [row.id for row in chosen.values() if row is not None]
        if len(chosen_ids) != len(set(chosen_ids)):
            raise ValueError("一个旧角色被映射到多个不同人物，未应用修复")

        narrator = max((row for row in rows if row.is_narrator), key=lambda row: row.dialogue_count or 0, default=None)
        referenced_after = set(chosen_ids)
        if narrator is not None:
            referenced_after.add(narrator.id)
        mapped_old = {
            row.id
            for row in rows
            if not row.is_narrator
            and (_normalized(row.name) in normalized_map or any(_normalized(a) in normalized_map for a in row.aliases or []))
        }
        canonical_orphans = {
            row.id
            for row in rows
            if _normalized(normalized_map.get(_normalized(row.name))) == _normalized(row.name)
        }
        removable = mapped_old - referenced_after - canonical_orphans
        # Protect any unexpected cross-book reference as well as this book.
        outside_references = {
            character_id
            for (character_id,) in db.query(BookSegment.character_id)
            .filter(BookSegment.character_id.in_(removable), BookSegment.chapter_id.notin_([c.id for c in chapters]))
            .all()
        } if removable else set()
        removable -= outside_references

        reassigned = retyped = restaged = stale = 0
        dialogue_counts: Counter[str] = Counter()
        narration_count = 0
        for old_segments, refined_segments in zip(chapter_rows, chapter_segments, strict=True):
            for segment, refined in zip(old_segments, refined_segments, strict=True):
                target_key = _normalized(refined.get("speaker") or _UNKNOWN) if refined["type"] == "dialogue" else None
                target = chosen[target_key] if target_key else narrator
                changed_assignment = target is None or segment.character_id != target.id
                changed_type = segment.type != refined["type"]
                changed_stage = segment.voice_stage != refined.get("voice_stage")
                reassigned += changed_assignment
                retyped += changed_type
                restaged += changed_stage
                stale += bool((changed_assignment or changed_type or changed_stage) and segment.generation_id)
                if target_key:
                    dialogue_counts[target_key] += 1
                else:
                    narration_count += 1

        removed_names = [row.name for row in rows if row.id in removable]
        retained_orphans = [
            row.name
            for row in rows
            if not row.is_narrator and row.id not in removable and row.id not in referenced_after
        ]
        report = {
            "book_id": book_id,
            "title": book.title,
            "dry_run": dry_run,
            "source_preserved": True,
            "counts": {
                "characters_before": sum(not row.is_narrator for row in rows),
                "characters_after": sum(not row.is_narrator for row in rows) - len(removable) + sum(row is None for row in chosen.values()),
                "segments": sum(len(segments) for segments in chapter_rows),
                "reassigned_segments": reassigned,
                "retyped_segments": retyped,
                "restaged_segments": restaged,
                "stale_audio": stale,
                "removed_characters": len(removable),
            },
            "characters": output_characters,
            "names": [character["name"] for character in output_characters],
            "name_map": copy.deepcopy(name_map),
            "changes": copy.deepcopy(result.get("changes") or []),
            "removed_names": removed_names,
            "retained_unmapped_names": retained_orphans,
            "preserved_bindings": [
                {"name": output_by_name[key]["name"], "character_id": row.id, "profile_id": row.profile_id}
                for key, row in chosen.items()
                if row is not None
            ],
        }
        if dry_run:
            return report

    if narrator is None:
        narrator = BookCharacter(book_id=book_id, name="旁白", is_narrator=True, confidence=1.0)
        db.add(narrator)
    for key, character in output_by_name.items():
        row = chosen[key]
        if row is None:
            row = BookCharacter(book_id=book_id, name=character["name"], is_narrator=False)
            if key == _normalized(_UNKNOWN):
                row.profile_id = narrator.profile_id
            db.add(row)
            chosen[key] = row
        row.name = character["name"]
        row.aliases = list(character.get("aliases") or [])
        row.confidence = character.get("confidence")
        row.role = character.get("role")
        row.dialogue_count = dialogue_counts[key]
    narrator.dialogue_count = narration_count
    for row in rows:
        if row.id not in referenced_after and row.id not in removable:
            row.dialogue_count = 0
            if not row.is_narrator:
                row.role = "minor"
                row.confidence = 0.0
    db.flush()

    for old_segments, refined_segments in zip(chapter_rows, chapter_segments, strict=True):
        for segment, refined in zip(old_segments, refined_segments, strict=True):
            target = chosen[_normalized(refined.get("speaker") or _UNKNOWN)] if refined["type"] == "dialogue" else narrator
            if (
                segment.character_id != target.id
                or segment.type != refined["type"]
                or segment.voice_stage != refined.get("voice_stage")
            ) and segment.generation_id:
                segment.audio_status = "stale"
            segment.character_id = target.id
            segment.type = refined["type"]
            segment.voice_stage = refined.get("voice_stage")
            segment.flagged = bool(refined.get("flagged"))
            segment.analysis_metadata = copy.deepcopy(refined.get("analysis_metadata") or {})
    db.flush()
    for row in rows:
        if row.id in removable and not db.query(BookSegment.id).filter_by(character_id=row.id).first():
            db.delete(row)
    book.updated_at = datetime.now(UTC).replace(tzinfo=None)
    if stale and book.status == "ready":
        book.status = "analyzed"
    db.flush()
    return report
