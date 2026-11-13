# ruff: noqa: RUF001
"""Orchestration service for book analysis (B4).

Wires together literary analysis (B2) + voice casting (B3) into a single
background task that materialises BookCharacter, VoiceProfile, and BookSegment
rows and publishes SSE progress events on the per-book channel.

GPU coordination strategy (documented per task spec):
    Analysis runs the LLM (literary_analysis.analyze_book) which needs the
    GPU.  Before the LLM pass we defensively unload the TTS model so both
    pipelines never contend for the same VRAM at the same time.  The unload
    calls are wrapped in try/except so they are silent no-ops when no model
    is currently loaded (e.g. during tests) — the TTS backend reloads on
    demand the next time a generation is requested.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import uuid
from dataclasses import asdict
from datetime import UTC, datetime

from sqlalchemy.orm import Session, load_only

logger = logging.getLogger(__name__)


def load_analysis_request(book_id: str) -> dict[str, str] | None:
    """Read the durable options for a run without changing its checkpoints."""
    from ..config import get_data_dir

    path = get_data_dir() / "analysis" / book_id / "request.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        if value.get("model_size") not in {"0.6B", "1.7B", "4B"}:
            return None
        if value.get("voice_mode") not in {"preset", "designed"}:
            return None
        if not isinstance(value.get("narrator_voice_id"), str) or not value["narrator_voice_id"]:
            return None
        if not isinstance(value.get("started_at"), str) or not value["started_at"]:
            return None
        return {key: value[key] for key in ("model_size", "voice_mode", "narrator_voice_id", "started_at")}
    except (OSError, ValueError, TypeError):
        return None


def resolve_analysis_request(
    book_id: str,
    model_size: str | None = None,
    narrator_voice_id: str | None = None,
    *,
    voice_mode: str | None = None,
    resume: bool = False,
) -> dict[str, str]:
    """Persist effective options; omitted retry fields retain the original run.

    Explicit caller options win. A new analysis uses normal defaults, while a
    retry can recover the previous voice design mode and narrator after restart.
    The original chapter checkpoint version and contents are not changed.
    """
    from ..config import get_data_dir

    previous = load_analysis_request(book_id) if resume else None
    saved = previous or {}
    request = {
        "model_size": model_size or saved.get("model_size") or "1.7B",
        "narrator_voice_id": narrator_voice_id or saved.get("narrator_voice_id") or "auto",
        "voice_mode": voice_mode or saved.get("voice_mode") or "preset",
        "started_at": saved.get("started_at") or datetime.now(UTC).isoformat(),
    }
    if request["model_size"] not in {"0.6B", "1.7B", "4B"}:
        raise ValueError("不支持的小说分析模型大小")
    if request["voice_mode"] not in {"preset", "designed"}:
        raise ValueError("不支持的角色声音模式")
    if request != previous:
        directory = get_data_dir() / "analysis" / book_id
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "request.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
        temporary.replace(path)
    return request


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _format_eta(seconds: float) -> str:
    """Human-readable time-remaining, e.g. '45s', '3m 20s', '1h 05m'."""
    secs = max(1, int(seconds))
    if secs < 60:
        return f"{secs}秒"
    minutes, secs = divmod(secs, 60)
    if minutes < 60:
        return f"{minutes}分{secs:02d}秒" if secs else f"{minutes}分钟"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}小时{minutes:02d}分"


def _unload_tts_for_analysis() -> None:
    """Unload the TTS model before the LLM pass to avoid GPU contention.

    Wrapped in a broad except so tests (which never load a model) and cold
    starts (where no backend has been initialised yet) are no-ops rather than
    errors.  The TTS backend reloads on demand when the next generation runs.
    """
    from .. import backends

    # Do not call a getter here: unloading an unused model must not initialise
    # a new backend (or touch Metal during CPU-only unit tests).
    loaded = list(backends._tts_backends.values())
    if backends._tts_backend is not None:
        loaded.append(backends._tts_backend)
    for backend in {id(b): b for b in loaded}.values():
        try:
            backend.unload_model()
        except Exception:
            logger.debug("TTS unload skipped", exc_info=True)


def _unload_llm_after_analysis() -> None:
    from .. import backends

    for backend in list(backends._llm_backends.values()):
        try:
            backend.unload_model()
        except Exception:
            logger.debug("LLM unload skipped", exc_info=True)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


async def run_analysis_task(
    book_id: str,
    *,
    model_size: str = "1.7B",
    narrator_voice_id: str = "auto",
    voice_mode: str = "preset",
    resume: bool = False,
    db: Session,
) -> None:
    """Reserve the inference device for the complete analysis lifecycle."""
    from .book_events import publish
    from .generation import inference_lock

    publish(
        book_id, {"type": "analysis_progress", "stage": "detect", "progress": 0, "message": "正在等待模型推理资源…"}
    )
    async with inference_lock():
        try:
            await _run_analysis_task_unlocked(
                book_id,
                model_size=model_size,
                narrator_voice_id=narrator_voice_id,
                voice_mode=voice_mode,
                resume=resume,
                db=db,
            )
        finally:
            _unload_llm_after_analysis()


async def _run_analysis_task_unlocked(
    book_id: str,
    *,
    model_size: str = "1.7B",
    narrator_voice_id: str = "auto",
    voice_mode: str = "preset",
    resume: bool = False,
    db: Session,
) -> None:
    """Full analysis pipeline: detect → cast → materialise → publish events.

    Args:
        book_id:           Primary key of the book to analyse.
        model_size:        LLM size to use for literary analysis ("0.6B","1.7B","4B").
        narrator_voice_id: Voice hint for the narrator ("auto" or a VoiceProfile id).
                           When a real profile id is given the narrator is pre-assigned
                           before casting so cast_book leaves it alone.
        db:                SQLAlchemy session — caller owns the lifecycle.

    Status transitions:
        analyzing → analyzed   on success
        analyzing → error      on any unhandled exception

    Events published on the book channel (contract-04):
        analysis_progress  (stage: detect|reconcile|profile|cast, progress 0-100)
        character_detected (incremental, with total)
        analysis_complete  (character_count, chapter_count)
        error              (on failure)
    """
    from .. import database
    from . import book_events, literary_analysis, voice_casting
    from .book_character_structure import _recount_dialogue
    from .ingestion import detect_text_language

    def _pub(payload: dict) -> None:
        book_events.publish(book_id, payload)

    try:
        # ── Stage: detect ─────────────────────────────────────────────────
        # The first LLM call lazily loads a multi-GB model onto the GPU/CPU,
        # which is silent and CPU-heavy — say so, so it doesn't look hung.
        _pub(
            {
                "type": "analysis_progress",
                "stage": "detect",
                "progress": 0,
                "message": "正在加载本地 Qwen 语言模型，首次加载可能需要一分钟…",
            }
        )

        # GPU coordination: unload TTS before the LLM analysis pass.
        # See module docstring for rationale.
        _unload_tts_for_analysis()

        repair_legacy_book_chapters(db, book_id)

        # Fetch chapters in reading order
        chapters = (
            db.query(database.Chapter)
            .filter_by(book_id=book_id)
            .options(load_only(database.Chapter.id, database.Chapter.number, database.Chapter.book_id))
            .order_by(database.Chapter.number)
            .all()
        )
        if not chapters:
            raise ValueError("这本书没有可分析的正文，请重新上传包含文字的小说。")

        class ChapterTexts:
            """Load one chapter at a time instead of eagerly copying a whole novel."""

            def __len__(self):
                return len(chapters)

            def __getitem__(self, index):
                if isinstance(index, slice):
                    return [chapter.raw_text for chapter in chapters[index]]
                return chapters[index].raw_text

            def __iter__(self):
                for chapter in chapters:
                    yield chapter.raw_text
                    db.expire(chapter, ["raw_text"])

        chapter_texts = ChapterTexts()
        language = detect_text_language("\n".join(ch.raw_text[:4000] for ch in chapters[:4]))
        chapter_ids = [ch.id for ch in chapters]

        # Forward analyze_book's fine-grained progress to the book channel.
        # Its 0..1 fraction drives the (long-running) detect stage's bar 0→100;
        # once it returns, the fast reconcile/profile/cast events take over.
        # An ETA is appended once enough has run to extrapolate.
        _analysis_started = time.monotonic()
        _restored_fraction = 0.0

        def _on_analysis_progress(fraction: float, message: str) -> None:
            nonlocal _analysis_started, _restored_fraction
            if "已恢复第" in message:
                _restored_fraction = fraction
                _analysis_started = time.monotonic()
            elapsed = time.monotonic() - _analysis_started
            completed_now = fraction - _restored_fraction
            if fraction >= 0.04 and completed_now >= 0.004 and elapsed >= 5.0:
                remaining = elapsed * (1.0 - fraction) / completed_now
                message = f"{message}  ·  预计还需 {_format_eta(remaining)}"
            _pub(
                {
                    "type": "analysis_progress",
                    "stage": "detect",
                    "progress": round(fraction * 100),
                    "message": message,
                }
            )

        existing_characters = db.query(database.BookCharacter).filter_by(book_id=book_id).all()
        initial_roster = [
            {"name": c.name, "aliases": c.aliases or [], "dialogue_count": 0, "confidence": c.confidence}
            for c in existing_characters
            if not c.is_narrator
        ]

        def _save_chapter_checkpoint(index, result):
            from ..config import get_data_dir

            chapter = chapters[index]
            directory = get_data_dir() / "analysis" / book_id
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{chapter.id}.json"
            temporary = path.with_suffix(".json.tmp")
            payload = {
                "version": 2,
                "chapter_id": chapter.id,
                "model_size": model_size,
                "source_sha256": hashlib.sha256(chapter.raw_text.encode()).hexdigest(),
                "analysis": asdict(result),
            }
            temporary.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            temporary.replace(path)

        def _load_chapter_checkpoint(index, source_text):
            if not resume:
                return None
            from ..config import get_data_dir

            path = get_data_dir() / "analysis" / book_id / f"{chapters[index].id}.json"
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if (
                    payload.get("version") != 2
                    or payload.get("model_size") != model_size
                    or payload.get("source_sha256") != hashlib.sha256(source_text.encode()).hexdigest()
                ):
                    return None
                data = payload["analysis"]
                result = literary_analysis.ChapterAnalysis(
                    segments=data["segments"],
                    characters=data["characters"],
                    flagged=data.get("flagged", False),
                    warnings=data.get("warnings", []),
                )
                expected = "".join(source_text.split())
                actual = "".join("".join(s["text"].split()) for s in result.segments)
                if actual != expected:
                    return None
                return result
            except (OSError, ValueError, TypeError, KeyError):
                return None

        analysis = await literary_analysis.analyze_book(
            chapter_texts,
            model_size=model_size,
            progress_cb=_on_analysis_progress,
            initial_characters=initial_roster,
            checkpoint_cb=_save_chapter_checkpoint,
            checkpoint_load=_load_chapter_checkpoint,
        )

        _pub({"type": "analysis_progress", "stage": "detect", "progress": 100})

        # ── Stage: reconcile ──────────────────────────────────────────────
        _pub({"type": "analysis_progress", "stage": "reconcile", "progress": 45})

        global_chars = analysis.characters  # [{name, dialogue_count, confidence}, ...]

        # ── Stage: profile (materialise BookCharacter rows) ───────────────
        _pub({"type": "analysis_progress", "stage": "profile", "progress": 50})

        book = db.query(database.Book).filter_by(id=book_id).first()
        if book is None:
            raise ValueError(f"Book {book_id!r} disappeared during analysis")

        if len(analysis.chapters) != len(chapters):
            raise ValueError("章节分析结果不完整，未覆盖全书；原有角色和音频已保留，请重试。")
        # A successful analysis is never allowed to materialise an empty book.
        for chapter, result in zip(chapters, analysis.chapters, strict=True):
            if not any(s.get("text", "").strip() for s in result.segments):
                result.segments = [
                    {
                        "type": "narration",
                        "text": chapter.raw_text,
                        "order": 0,
                        "speaker": None,
                        "flagged": True,
                        "analysis_metadata": {
                            "method": "empty_result_fallback",
                            "source_text": chapter.raw_text,
                            "warnings": ["模型未返回片段，已保留整章原文。"],
                        },
                    }
                ]
                result.flagged = True
                result.warnings = ["模型未返回片段，已保留整章原文，请检查角色。"]

        # Re-analysis updates identities in place. Bound profiles, manual names
        # and character IDs survive; new aliases can resolve an existing voice.
        existing = db.query(database.BookCharacter).filter_by(book_id=book_id).all()

        def norm(value):
            return literary_analysis.normalize_character_name(value).casefold()

        by_name = {}
        for row in existing:
            if row.is_narrator:
                continue
            for name in [row.name, *(row.aliases or [])]:
                by_name.setdefault(norm(name), row)
        narrator = next((row for row in existing if row.is_narrator), None)
        if narrator is None:
            narrator = database.BookCharacter(
                book_id=book_id, name="旁白", is_narrator=True, dialogue_count=0, confidence=1.0
            )
            db.add(narrator)
        char_rows = [narrator]
        name_to_id = {}
        total_chars = len(global_chars) + 1
        for char_data in global_chars:
            name = char_data["name"]
            candidates = [name, *char_data.get("aliases", [])]
            char = next((by_name[norm(n)] for n in candidates if norm(n) in by_name), None)
            if char is None:
                char = database.BookCharacter(book_id=book_id, name=name, is_narrator=False)
                db.add(char)
            old_name = char.name
            char.name = name
            char.dialogue_count = char_data.get("dialogue_count", 0)
            char.confidence = char_data.get("confidence")
            char.aliases = sorted(
                {
                    a
                    for a in [*(char.aliases or []), *char_data.get("aliases", []), old_name]
                    if a and norm(a) != norm(name)
                }
            )
            for field, key in [
                ("color", "color"),
                ("gender", "gender"),
                ("age_range", "age_estimate"),
                ("vocal_description", "vocal_description"),
                ("archetype", "archetype"),
                ("role", "role"),
            ]:
                value = char_data.get(key)
                if value is not None:
                    setattr(char, field, value)
            db.flush()
            name_to_id[name] = char.id
            for alias in char.aliases:
                name_to_id[alias] = char.id
            if char not in char_rows:
                char_rows.append(char)
            for n in [name, *char.aliases]:
                by_name[norm(n)] = char
        db.flush()
        for char in char_rows:
            _pub(
                {
                    "type": "character_detected",
                    "character": {
                        "id": char.id,
                        "name": char.name,
                        "color": char.color,
                        "dialogue_count": char.dialogue_count,
                        "confidence": char.confidence,
                    },
                    "total": total_chars,
                }
            )

        _pub({"type": "analysis_progress", "stage": "profile", "progress": 70})

        # ── Stage: cast (VoiceProfile rows via voice_casting) ─────────────
        _pub({"type": "analysis_progress", "stage": "cast", "progress": 75})

        # Fix 2: if narrator_voice_id is a real profile id, pre-assign the
        # narrator before calling cast_book so casting leaves it alone.
        if narrator_voice_id and narrator_voice_id not in ("auto", None):
            narrator_profile = db.get(database.VoiceProfile, narrator_voice_id)
            if narrator_profile is not None:
                narrator.profile_id = narrator_voice_id
                db.flush()

        # Build co-occurrence set from per-chapter character mentions
        cooccurrence: set[tuple[str, str]] = set()
        for ch_analysis in analysis.chapters:
            present = {c["name"] for c in ch_analysis.characters}
            # map name → char_id using the rows we just created
            ids_in_chapter = {name_to_id[n] for n in present if n in name_to_id}
            ids_list = sorted(ids_in_chapter)
            for i, a_id in enumerate(ids_list):
                for b_id in ids_list[i + 1 :]:
                    cooccurrence.add((a_id, b_id))

        voice_casting.cast_book(
            book_id,
            cooccurrence=cooccurrence,
            language=language,
            voice_mode=voice_mode,
            db=db,
        )

        _pub({"type": "analysis_progress", "stage": "cast", "progress": 90})

        # ── Materialise BookSegment rows ───────────────────────────────────
        # Replace only after the full analysis has succeeded; generated files
        # remain in history, while new segments explicitly start without audio.
        book_chapters = db.query(database.Chapter.id).filter_by(book_id=book_id)
        book_stories = (
            db.query(database.Chapter.story_id).filter_by(book_id=book_id).filter(database.Chapter.story_id.isnot(None))
        )
        old_generations = db.query(database.BookSegment.generation_id).filter(
            database.BookSegment.chapter_id.in_(book_chapters), database.BookSegment.generation_id.isnot(None)
        )
        # Remove only this book's derived timeline links. Generations/files and
        # copies used by other stories remain available in history.
        db.query(database.StoryItem).filter(
            database.StoryItem.story_id.in_(book_stories), database.StoryItem.generation_id.in_(old_generations)
        ).delete(synchronize_session=False)
        db.query(database.BookSegment).filter(database.BookSegment.chapter_id.in_(book_chapters)).delete(
            synchronize_session=False
        )
        db.flush()

        for chapter_id, ch_analysis in zip(chapter_ids, analysis.chapters, strict=True):
            for seg_data in ch_analysis.segments:
                speaker_name = seg_data.get("speaker")
                char_id: str | None = None
                if speaker_name and speaker_name.lower() != "narrator":
                    char_id = name_to_id.get(speaker_name)
                    if char_id is None:
                        # Fallback: use narrator for unresolved speakers
                        char_id = narrator.id
                else:
                    # Narration segments → narrator character
                    char_id = narrator.id

                segment = database.BookSegment(
                    chapter_id=chapter_id,
                    character_id=char_id,
                    type=seg_data.get("type", "narration"),
                    order=seg_data.get("order", 0),
                    text=seg_data.get("text", ""),
                    emotion=seg_data.get("emotion"),
                    emotion_intensity=seg_data.get("intensity"),
                    delivery=seg_data.get("delivery"),
                    voice_stage=seg_data.get("voice_stage"),
                    analysis_metadata=seg_data.get("analysis_metadata", {}),
                    flagged=seg_data.get("flagged", ch_analysis.flagged),
                    audio_status="none",
                )
                db.add(segment)

        db.commit()

        # Fix 4: recompute every character's dialogue_count from the materialized
        # BookSegment rows using the same rule as merge/split/delete operations.
        for row in char_rows:
            _recount_dialogue(row.id, db)
        active_ids = {row.id for row in char_rows}
        identity_map = getattr(analysis, "identity_map", {})
        for row in existing:
            if (
                row.id not in active_ids
                and row.name in identity_map
                and identity_map[row.name] != row.name
                and not db.query(database.BookSegment.id).filter_by(character_id=row.id).first()
            ):
                db.delete(row)
        db.commit()

        # Reload char_rows to get profile_ids populated by cast_book
        db.expire_all()
        char_rows = db.query(database.BookCharacter).filter_by(book_id=book_id).all()

        # ── Flip status + emit completion ──────────────────────────────────
        book = db.query(database.Book).filter_by(id=book_id).first()
        if book is not None:
            book.status = "analyzed"
            db.commit()

        chapter_count = len(chapters)
        character_count = len(char_rows)

        _pub(
            {
                "type": "analysis_complete",
                "character_count": character_count,
                "chapter_count": chapter_count,
                "segment_count": sum(len(c.segments) for c in analysis.chapters),
                "flagged_chapters": sum(bool(c.flagged) for c in analysis.chapters),
                "warnings": list(
                    dict.fromkeys(
                        [
                            *getattr(analysis, "warnings", []),
                            *(w for c in analysis.chapters for w in getattr(c, "warnings", [])),
                        ]
                    )
                ),
            }
        )

    except Exception as exc:
        logger.exception("Analysis task failed for book %s", book_id)
        db.rollback()
        try:
            db_book = db.query(database.Book).filter_by(id=book_id).first()
            if db_book is not None:
                db_book.status = "error"
                db.commit()
        except Exception:
            logger.debug("Failed to persist analysis error state", exc_info=True)

        try:
            book_events.publish(
                book_id,
                {
                    "type": "error",
                    "stage": "analyze",
                    "message": str(exc),
                },
            )
        except Exception:
            logger.debug("Failed to publish analysis error", exc_info=True)


# ---------------------------------------------------------------------------
# Synchronous enqueue entry-point
# ---------------------------------------------------------------------------


def enqueue_analysis(
    book_id: str,
    model_size: str | None,
    narrator_voice_id: str | None,
    *,
    voice_mode: str | None = None,
) -> str:
    """Flip book status to 'analyzing' and enqueue the analysis background task.

    The status flip is synchronous and committed before this function returns
    so that concurrent requests see the 409 guard immediately.

    Args:
        book_id:           Book primary key.
        model_size:        LLM size string (e.g. "1.7B").  Defaults to "1.7B".
        narrator_voice_id: Voice hint for narrator; defaults to "auto".

    Returns:
        A task_id string (a fresh uuid4, opaque to clients).
    """
    from .. import database
    from . import task_queue

    # Flip status synchronously — race-safe guard
    db = next(database.get_db())
    resume = False
    try:
        repair_legacy_book_chapters(db, book_id)
        book = db.query(database.Book).filter_by(id=book_id).first()
        if book is not None:
            resume = book.status == "error"
        request = resolve_analysis_request(
            book_id, model_size, narrator_voice_id, voice_mode=voice_mode, resume=resume
        )
        if book is not None:
            book.status = "analyzing"
            db.commit()
    finally:
        db.close()

    # Launch the pipeline in the background with a fresh session
    async def _run():
        fresh_db = next(database.get_db())
        try:
            await run_analysis_task(
                book_id,
                model_size=request["model_size"],
                narrator_voice_id=request["narrator_voice_id"],
                voice_mode=request["voice_mode"],
                resume=resume,
                db=fresh_db,
            )
        finally:
            fresh_db.close()

    task_queue.create_background_task(_run())
    task_id = str(uuid.uuid4())
    return task_id


def recover_interrupted_analyses(db: Session) -> int:
    """Startup hook: a process restart has no surviving analysis tasks.

    Preserve characters/audio and completed chapter checkpoints, but let the UI
    retry instead of leaving an orphaned 'analyzing' status and permanent 409.
    """
    from ..database import Book

    count = db.query(Book).filter_by(status="analyzing").update({"status": "error"}, synchronize_session=False)
    db.commit()
    return count


def repair_legacy_book_chapters(db: Session, book_id: str) -> int:
    """Reparse unrendered legacy imports without changing the book ID or source.

    Empty Story placeholders are safe. Any materialised segment or story audio
    protects the existing chapter layout. A lossless source backup precedes edits.
    """
    from .. import database
    from ..config import get_data_dir
    from . import ingestion

    chapters = db.query(database.Chapter).filter_by(book_id=book_id).order_by(database.Chapter.number).all()
    if not chapters:
        return 0
    ids = [chapter.id for chapter in chapters]
    if db.query(database.BookSegment).filter(database.BookSegment.chapter_id.in_(ids)).first() is not None:
        return 0
    story_ids = [chapter.story_id for chapter in chapters if chapter.story_id]
    if (
        story_ids
        and db.query(database.StoryItem).filter(database.StoryItem.story_id.in_(story_ids)).first() is not None
    ):
        return 0
    reparsed = []
    changed = False
    for chapter in chapters:
        normalized = chapter.raw_text.replace("\r\n", "\n").replace("\r", "\n")
        parts = ingestion._split_into_chapters(normalized, default_title=chapter.title)
        if not parts:
            continue
        changed |= len(parts) != 1 or parts[0].text != chapter.raw_text or parts[0].word_count != chapter.word_count
        reparsed.append((chapter, parts))
    if not changed:
        return 0
    directory = get_data_dir() / "analysis" / book_id
    directory.mkdir(parents=True, exist_ok=True)
    backup = directory / "source-before-repair.json"
    if not backup.exists():
        temporary = backup.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps(
                {
                    "book_id": book_id,
                    "chapters": [
                        {
                            "id": c.id,
                            "number": c.number,
                            "title": c.title,
                            "raw_text": c.raw_text,
                            "story_id": c.story_id,
                        }
                        for c in chapters
                    ],
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        temporary.replace(backup)
    number = 0
    for chapter, parts in reparsed:
        for index, part in enumerate(parts):
            number += 1
            row = chapter if index == 0 else database.Chapter(book_id=book_id)
            row.number, row.title, row.raw_text, row.word_count = number, part.title, part.text, part.word_count
            if index:
                db.add(row)
    book = db.get(database.Book, book_id)
    if book is not None and book.status == "analyzed":
        book.status = "imported"
    db.commit()
    return number
