"""Book generation service (D1).

Lazily materialises audio for a chapter or an entire book by:
1. Creating a Story row (once per chapter) and linking it back on Chapter.story_id.
2. For each unrendered BookSegment, creating a Generation row and a StoryItem.
3. Enqueuing each Generation through the serial TTS queue.

The endpoint returns immediately (202); synthesis runs in the background.

Key helpers exposed for reuse by D3/D4:
    compose_instruct(segment) → str | None
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import zlib
from collections import deque
from typing import TYPE_CHECKING, Optional

from sqlalchemy.orm import Session

from ..utils.voice_stages import normalize_voice_stage

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Profiles choose their own engine; unconfigured book voices default to Qwen.
_DEFAULT_ENGINE = "qwen_custom_voice"
_DEFAULT_MODEL_SIZE = "1.7B"


def stable_voice_seed(profile_id: str) -> int:
    return zlib.crc32(profile_id.encode("utf-8")) & 0x7fffffff


# ---------------------------------------------------------------------------
# Public helper: compose instruct string
# ---------------------------------------------------------------------------


def compose_instruct(segment) -> Optional[str]:
    """Fold emotion + emotion_intensity + delivery into a single instruct string.

    This mirrors how VoiceIt's existing generation composes instruct/emotion
    so the TTS engine receives a consistent prompt regardless of whether audio
    is triggered manually or from the book pipeline.

    Args:
        segment: A BookSegment ORM row (or any object with .emotion,
                 .emotion_intensity, .delivery attributes).

    Returns:
        A composed instruct string, or ``None`` if all fields are blank.
    """
    parts: list[str] = []

    emotion = getattr(segment, "emotion", None)
    intensity = getattr(segment, "emotion_intensity", None)
    delivery = getattr(segment, "delivery", None)

    if emotion:
        if intensity is not None:
            # Express intensity as an adverb modifier
            if intensity >= 0.8:
                modifier = "very "
            elif intensity >= 0.5:
                modifier = ""
            else:
                modifier = "slightly "
            parts.append(f"{modifier}{emotion}")
        else:
            parts.append(emotion)

    if delivery:
        parts.append(delivery)

    stage = normalize_voice_stage(getattr(segment, "voice_stage", None))
    if stage:
        parts.append(f"保持同一角色的基础音色，以{stage}阶段自然朗读，不必强调具体岁数")

    if not parts:
        return None

    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Completion hooks — segment-status lifecycle + book drain
# ---------------------------------------------------------------------------


async def _per_generation_completion_hook(
    segment_id: str,
    book_id: str,
    inner_coro,
) -> None:
    """Per-generation wrapper that handles the full segment + book lifecycle.

    Responsibilities (in order):
    1. Await *inner_coro* (the real ``run_generation`` call).
    2. Flip the owning ``BookSegment.audio_status`` to ``"completed"`` on
       success, or ``"error"`` on failure.  This is the step that was
       previously missing — ``run_generation`` only updates ``Generation``
       rows, never ``BookSegment`` rows, so without this hook segments
       remained ``"pending"`` forever.
    3. When all segments in the chapter have settled, reflow the chapter's
       ``StoryItem.start_time_ms`` values to real cumulative milliseconds
       (using each ``Generation.duration``) so that the D5 read-along
       ``useStoryPlayback`` can schedule clips at correct times instead of
       using the order-counter placeholder values (0, 1, 2, …) written at
       generation-materialisation time.
    4. Publish ``generation_progress`` / ``generation_complete`` SSE events
       and reset ``book.status`` to ``"analyzed"`` once all book segments
       settle (the book-level drain logic shared with the progress-events
       feature).

    Args:
        segment_id: Primary key of the ``BookSegment`` whose status to flip.
        book_id:    The ``Book`` primary key for drain / SSE logic.
        inner_coro: The ``run_generation`` coroutine to await.
    """
    from .. import database
    from . import book_events

    success = False
    cancelled = False
    try:
        await inner_coro
        success = True
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception:
        logger.exception(
            "run_generation raised inside per-generation completion hook "
            "(segment %s, book %s)",
            segment_id,
            book_id,
        )

    finally:
        # ── Step 2: flip BookSegment.audio_status ─────────────────────────────
        db: Session = next(database.get_db())
        try:
            segment = db.query(database.BookSegment).filter_by(id=segment_id).first()
            if segment is not None:
                gen = db.query(database.Generation).filter_by(id=segment.generation_id).first()
                success = success and gen is not None and gen.status == "completed"
                if not success and gen is not None and gen.status in {"pending", "loading_model", "generating"}:
                    gen.status = "failed"
                    gen.error = gen.error or ("Generation cancelled" if cancelled else "Generation failed before completion")
                segment.audio_status = "completed" if success else "error"
                db.commit()

            # ── Step 3: reflow StoryItem times if the whole chapter settled ───
            if segment is not None:
                chapter_id = segment.chapter_id
                _reflow_chapter_story_times(chapter_id, db)

            # ── Step 4: book-level progress events and drain ──────────────────
            _publish_progress_and_drain(book_id, db, book_events)

        except Exception:
            logger.exception(
                "Per-generation completion hook failed for segment %s book %s",
                segment_id,
                book_id,
            )
        finally:
            db.close()


def _reflow_chapter_story_times(chapter_id: str, db: Session) -> None:
    """Reflow StoryItem.start_time_ms for a chapter to real cumulative ms.

    Only executes when ALL segments in the chapter have settled (audio_status
    is ``"completed"`` or ``"error"``).  Segments with ``"error"`` status have
    no duration, so they contribute 0 ms (their StoryItem start time is still
    updated for ordering correctness even though no audio exists).

    Uses the ``Generation.duration`` field written by ``run_generation``
    after synthesis.  A gap of 0 ms is used between segments (the Story
    timeline editor manages gaps; the export step manages its own pauses).

    Args:
        chapter_id: The Chapter whose Story's StoryItems to reflow.
        db:         Active SQLAlchemy session (will be committed on change).
    """
    from .. import database

    # Only reflow when all segments have settled (no pending/generating left)
    segments = (
        db.query(database.BookSegment)
        .filter_by(chapter_id=chapter_id)
        .order_by(database.BookSegment.order)
        .all()
    )
    if not segments:
        return

    in_flight = sum(1 for s in segments if s.audio_status in {"pending", "generating"})
    if in_flight > 0:
        return  # Chapter not fully settled yet — defer reflow

    # Find the chapter's Story
    chapter = db.query(database.Chapter).filter_by(id=chapter_id).first()
    if chapter is None or chapter.story_id is None:
        return

    # Fetch StoryItems ordered by current start_time_ms (reading order)
    items = (
        db.query(database.StoryItem)
        .join(database.BookSegment, database.BookSegment.generation_id == database.StoryItem.generation_id)
        .filter(database.StoryItem.story_id == chapter.story_id)
        .order_by(database.BookSegment.order)
        .all()
    )
    if not items:
        return

    # Build a map from generation_id → Generation.duration (ms)
    gen_ids = [item.generation_id for item in items]
    gens = (
        db.query(database.Generation)
        .filter(database.Generation.id.in_(gen_ids))
        .all()
    )
    duration_map: dict[str, int] = {
        g.id: int((g.duration or 0.0) * 1000) for g in gens
    }

    # Reflow: assign cumulative start times
    current_ms = 0
    changed = False
    for item in items:
        if item.start_time_ms != current_ms:
            item.start_time_ms = current_ms
            changed = True
        duration_ms = duration_map.get(item.generation_id, 0)
        current_ms += duration_ms  # gap between items is 0; editor manages gaps

    if changed:
        try:
            db.commit()
            logger.info(
                "Chapter %s: reflowed %d StoryItem start_time_ms values "
                "(total chapter duration ~%d ms)",
                chapter_id,
                len(items),
                current_ms,
            )
        except Exception:
            logger.exception(
                "Failed to commit StoryItem reflow for chapter %s", chapter_id
            )
            db.rollback()


def _publish_progress_and_drain(book_id: str, db: Session, book_events) -> None:
    """Publish generation_progress / generation_complete events and drain book status.

    Computes per-chapter segment counts, publishes SSE events, and resets
    ``book.status`` to ``"analyzed"`` when all segments have settled.

    This is the book-level logic previously in ``_generation_with_completion_hook``
    — extracted so it can be called from ``_per_generation_completion_hook``
    after the segment status has already been flipped.

    Args:
        book_id:     The Book to inspect and potentially reset.
        db:          Active session (must already reflect updated segment statuses).
        book_events: The ``book_events`` module (passed in so tests can patch it).
    """
    from .. import database

    chapters = (
        db.query(database.Chapter)
        .filter_by(book_id=book_id)
        .order_by(database.Chapter.number)
        .all()
    )

    # Pre-compute book-wide totals
    chapter_segments: dict[str, list] = {}
    book_total = 0
    book_completed = 0
    for chapter in chapters:
        segs = (
            db.query(database.BookSegment)
            .filter_by(chapter_id=chapter.id)
            .all()
        )
        chapter_segments[chapter.id] = segs
        book_total += len(segs)
        book_completed += sum(1 for s in segs if s.audio_status == "completed")

    overall_progress = book_completed / book_total if book_total > 0 else 0.0

    published_complete_ids: set[str] = set()

    for chapter in chapters:
        segments = chapter_segments[chapter.id]
        total = len(segments)
        completed = sum(1 for s in segments if s.audio_status == "completed")
        errors = sum(1 for s in segments if s.audio_status == "error")

        book_events.publish(
            book_id,
            {
                "type": "generation_progress",
                "chapter_id": chapter.id,
                "completed": completed,
                "errors": errors,
                "total": total,
                "overall_progress": overall_progress,
            },
        )

        chapter_in_flight = sum(
            1 for s in segments if s.audio_status in {"pending", "generating"}
        )
        if chapter_in_flight == 0 and chapter.id not in published_complete_ids:
            published_complete_ids.add(chapter.id)
            book_events.publish(
                book_id,
                {
                    "type": "generation_complete",
                    "chapter_id": chapter.id,
                },
            )

    # Reset book.status if all segments have settled
    in_flight = (
        db.query(database.BookSegment)
        .join(database.Chapter, database.BookSegment.chapter_id == database.Chapter.id)
        .filter(
            database.Chapter.book_id == book_id,
            database.BookSegment.audio_status.in_({"pending", "generating"}),
        )
        .count()
    )
    if in_flight == 0:
        book = db.query(database.Book).filter_by(id=book_id).first()
        if book is not None and book.status == "generating":
            book.status = "analyzed"
            db.commit()
            logger.info(
                "Book %s: all segments settled — status reset to 'analyzed'", book_id
            )


async def _generation_with_completion_hook(
    book_id: str,
    inner_coro,
) -> None:
    """Book-level wrapper used by the progress-events tests.

    This hook is called by ``test_generation_progress_events.py`` directly
    (it pre-sets segment statuses before calling).  For production code the
    enqueue path uses ``_per_generation_completion_hook`` instead, which
    additionally flips the segment's ``audio_status`` and reflowed StoryItem
    times — see the docstring on that function.

    Kept for backward-compat with the D2 progress-events tests which call
    it directly after manually setting segment statuses to 'completed'.

    Args:
        book_id:    The Book primary key whose status we manage.
        inner_coro: The run_generation coroutine to await.
    """
    from .. import database
    from . import book_events

    try:
        await inner_coro
    except Exception:
        logger.exception("run_generation raised inside completion hook")
    finally:
        db: Session = next(database.get_db())
        try:
            _publish_progress_and_drain(book_id, db, book_events)
        except Exception:
            logger.exception(
                "Completion hook failed to reset book.status for book %s", book_id
            )
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Core materialization: generate a single chapter
# ---------------------------------------------------------------------------


def generate_chapter(
    chapter_id: str,
    *,
    engine: Optional[str] = None,
    model_size: Optional[str] = None,
    overwrite_errors: bool = False,
    db: Session,
) -> tuple[int, list[str]]:
    """Lazily materialise and enqueue audio for all unrendered segments in a chapter.

    Creates:
    - A ``Story`` row (if the chapter has none yet) and links it to the chapter.
    - A ``Generation`` row per unrendered segment (source="book_import").
    - A ``StoryItem`` row linking each generation into the chapter's story.

    Sets each segment's ``generation_id`` and flips ``audio_status`` to
    ``"pending"`` before returning.

    Args:
        chapter_id:       Primary key of the Chapter to render.
        engine:           TTS engine to use (defaults to character's profile engine
                          or ``_DEFAULT_ENGINE``).
        model_size:       Model size string (defaults to ``_DEFAULT_MODEL_SIZE``).
        overwrite_errors: When True, re-enqueue segments whose
                          ``audio_status`` is ``"error"``.
        db:               Active SQLAlchemy session.

    Returns:
        A tuple of ``(queued_count, generation_ids)`` — number of segments
        enqueued and the list of new Generation ids (so the caller can return
        a meaningful ``queued_segments`` count).
    """
    from .. import database
    chapter = db.query(database.Chapter).filter_by(id=chapter_id).first()
    if chapter is None:
        raise ValueError(f"Chapter {chapter_id!r} not found")

    # ── Lazy Story creation ───────────────────────────────────────────────
    if chapter.story_id is None:
        story = database.Story(
            id=str(uuid.uuid4()),
            name=chapter.title or f"Chapter {chapter.number}",
        )
        db.add(story)
        db.flush()
        chapter.story_id = story.id
        db.flush()
    else:
        story = db.query(database.Story).filter_by(id=chapter.story_id).first()

    # ── Fetch segments in reading order ──────────────────────────────────
    eligible_statuses = {"none", "stale"}
    if overwrite_errors:
        eligible_statuses.add("error")

    segments = (
        db.query(database.BookSegment)
        .filter(
            database.BookSegment.chapter_id == chapter_id,
            database.BookSegment.audio_status.in_(eligible_statuses),
        )
        .order_by(database.BookSegment.order)
        .all()
    )

    # Determine the next start_time_ms for story ordering.
    # We place each new item after the last existing one (gap = 0 since we
    # don't know durations yet; the export step will re-sequence).
    existing_item_count = (
        db.query(database.StoryItem).filter_by(story_id=story.id).count()
    )
    next_order = existing_item_count  # used as a proxy for start_time_ms ordering

    queued_count = 0
    generation_ids: list[str] = []

    for idx, segment in enumerate(segments):
        # Resolve character → profile_id
        profile_id = _resolve_profile_id(segment, db)
        if profile_id is None:
            logger.warning(
                "Segment %s has no resolvable profile_id; skipping", segment.id
            )
            continue

        # Compose TTS instruct from emotion + delivery
        instruct = compose_instruct(segment)

        # Resolve engine for this segment
        effective_engine = engine or _resolve_engine_for_profile(profile_id, db)
        effective_model_size = (model_size or _DEFAULT_MODEL_SIZE) if effective_engine in {"qwen", "qwen_custom_voice"} else "default"
        from .ingestion import detect_text_language
        language = detect_text_language(segment.text)

        # Create the Generation row (status starts as "pending" here; the
        # queue worker will flip it to "generating" / "completed" / "failed")
        gen_id = str(uuid.uuid4())
        generation = database.Generation(
            id=gen_id,
            profile_id=profile_id,
            text=segment.text,
            language=language,
            seed=stable_voice_seed(profile_id),
            engine=effective_engine,
            model_size=effective_model_size,
            instruct=instruct,
            source="book_import",
            status="pending",
        )
        db.add(generation)
        db.flush()

        if segment.generation_id:
            db.query(database.StoryItem).filter_by(story_id=story.id, generation_id=segment.generation_id).delete()

        # Place the generation in the story at reading order position
        story_item = database.StoryItem(
            id=str(uuid.uuid4()),
            story_id=story.id,
            generation_id=gen_id,
            start_time_ms=next_order + idx,  # strictly increasing
            track=0,
        )
        db.add(story_item)

        # Link segment → generation and flip status to pending
        segment.generation_id = gen_id
        segment.audio_status = "pending"

        db.flush()

        generation_ids.append(gen_id)
        queued_count += 1

    db.commit()

    return queued_count, generation_ids


# ---------------------------------------------------------------------------
# Book-level generate: iterate chapters
# ---------------------------------------------------------------------------


def generate_book(
    book_id: str,
    *,
    engine: Optional[str] = None,
    model_size: Optional[str] = None,
    overwrite_errors: bool = False,
    chapter_ids: list[str] | None = None,
    db: Session,
) -> tuple[int, list[str]]:
    """Enqueue audio for every chapter in *book_id*, reusing generate_chapter.

    Args:
        book_id:          Primary key of the Book to render.
        engine:           TTS engine (passed through to each chapter).
        model_size:       Model size string (passed through).
        overwrite_errors: Re-enqueue error segments when True.
        db:               Active SQLAlchemy session.

    Returns:
        A tuple of ``(total_queued, all_generation_ids)`` across all chapters.
    """
    from .. import database

    chapters = (
        db.query(database.Chapter)
        .filter_by(book_id=book_id)
        .order_by(database.Chapter.number)
        .all()
    )

    if chapter_ids is not None:
        from fastapi import HTTPException
        known = {chapter.id for chapter in chapters}
        if not chapter_ids or not set(chapter_ids) <= known:
            raise HTTPException(status_code=400, detail="请选择属于本书的有效章节")
        chapters = [chapter for chapter in chapters if chapter.id in set(chapter_ids)]

    total_queued = 0
    all_gen_ids: list[str] = []

    for chapter in chapters:
        count, gen_ids = generate_chapter(
            chapter.id,
            engine=engine,
            model_size=model_size,
            overwrite_errors=overwrite_errors,
            db=db,
        )
        total_queued += count
        all_gen_ids.extend(gen_ids)

    return total_queued, all_gen_ids


# ---------------------------------------------------------------------------
# Public enqueue entry-points: 409 guard + status flip + drain reset
# ---------------------------------------------------------------------------


def enqueue_chapter_generation(
    book_id: str,
    chapter_id: str,
    db: Session,
    *,
    engine: Optional[str] = None,
    model_size: Optional[str] = None,
    overwrite_errors: bool = False,
) -> tuple[str, int]:
    """409-guard, flip book status, materialise and enqueue a chapter.

    This is the single place that owns the book status lifecycle for
    chapter-level generation:
    1. Checks book.status != 'generating' (409 guard).
    2. Flips book.status = 'generating' synchronously (race-safe).
    3. Calls generate_chapter to materialise segments.
    4. Wraps each enqueued coroutine in the completion hook that resets
       book.status to 'analyzed' once all its segments settle.

    Args:
        book_id:          Book primary key.
        chapter_id:       Chapter primary key.
        db:               Caller-supplied session (managed by route via Depends).
        engine:           TTS engine override.
        model_size:       Model size override.
        overwrite_errors: Re-enqueue error segments when True.

    Returns:
        ``(task_id, queued_segments)``

    Raises:
        HTTPException 404 if book or chapter not found.
        HTTPException 409 if book is already generating.
    """
    from .. import database
    from . import task_queue
    from .generation import run_generation

    book = db.query(database.Book).filter_by(id=book_id).first()
    if book is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Book not found")

    if book.status == "generating":
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Book is already generating")

    chapter = db.query(database.Chapter).filter_by(id=chapter_id, book_id=book_id).first()
    if chapter is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Chapter not found")

    _preflight_generation(book, db, [chapter_id], engine)

    # Flip status synchronously — race-safe 409 guard
    _claim_book_generation(book, db)

    try:
        queued, generation_ids = generate_chapter(
            chapter_id,
            engine=engine,
            model_size=model_size,
            overwrite_errors=overwrite_errors,
            db=db,
        )
    except BaseException:
        _recover_failed_materialization(book_id, db)
        raise

    # Enqueue each generation through the TTS queue, wrapped with the
    # per-generation completion hook that flips BookSegment.audio_status,
    # reflowed StoryItem times, then runs the book-level drain/progress logic.
    for gen_id in generation_ids:
        gen = db.query(database.Generation).filter_by(id=gen_id).first()
        if gen is None:
            continue

        # Look up the segment that owns this generation so the hook can flip
        # its audio_status.
        seg = db.query(database.BookSegment).filter_by(generation_id=gen_id).first()
        if seg is None:
            logger.warning(
                "No BookSegment found for generation_id %s — skipping enqueue", gen_id
            )
            continue

        inner = run_generation(
            generation_id=gen_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine,
            model_size=gen.model_size or _DEFAULT_MODEL_SIZE,
            seed=gen.seed,
            instruct=gen.instruct,
            mode="generate",
        )

        wrapped = _per_generation_completion_hook(
            segment_id=seg.id,
            book_id=book_id,
            inner_coro=inner,
        )
        wrapped = task_queue.with_coro_cleanup(wrapped, inner)

        try:
            task_queue.enqueue_generation(gen_id, wrapped)
        except Exception:
            wrapped.close()
            inner.close()
            logger.exception("Failed to enqueue generation %s", gen_id)
            _mark_generation_error(gen_id, "Queue unavailable", db)

    # If nothing was queued (e.g. all segments already rendered) reset now.
    _reset_book_status_if_settled(book_id, db)

    task_id = str(uuid.uuid4())
    return task_id, queued


def enqueue_book_generation(
    book_id: str,
    db: Session,
    *,
    engine: Optional[str] = None,
    model_size: Optional[str] = None,
    overwrite_errors: bool = False,
    chapter_ids: list[str] | None = None,
    concurrency: int = 2,
) -> tuple[str, int]:
    """409-guard, flip book status, materialise and enqueue all chapters.

    Mirrors enqueue_chapter_generation but for whole-book generation.

    Args:
        book_id:          Book primary key.
        db:               Caller-supplied session (managed by route via Depends).
        engine:           TTS engine override.
        model_size:       Model size override.
        overwrite_errors: Re-enqueue error segments when True.

    Returns:
        ``(task_id, total_queued_segments)``

    Raises:
        HTTPException 404 if book not found.
        HTTPException 409 if book is already generating.
    """
    from .. import database
    from . import task_queue
    from .generation import run_generation

    book = db.query(database.Book).filter_by(id=book_id).first()
    if book is None:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Book not found")

    if book.status == "generating":
        from fastapi import HTTPException
        raise HTTPException(status_code=409, detail="Book is already generating")

    _preflight_generation(book, db, chapter_ids, engine)

    # Flip status synchronously — race-safe 409 guard
    _claim_book_generation(book, db)

    try:
        total, generation_ids = generate_book(
            book_id,
            engine=engine,
            model_size=model_size,
            overwrite_errors=overwrite_errors,
            chapter_ids=chapter_ids,
            db=db,
        )
    except BaseException:
        _recover_failed_materialization(book_id, db)
        raise

    # Round-robin across chapters lets several chapters make progress together.
    by_chapter: dict[str, deque] = {}
    chapter_by_generation = dict(db.query(database.BookSegment.generation_id, database.BookSegment.chapter_id).join(database.Chapter).filter(database.Chapter.book_id == book_id, database.BookSegment.generation_id.isnot(None)).all())
    for gen_id in generation_ids:
        chapter_id = chapter_by_generation.get(gen_id)
        if chapter_id is not None:
            by_chapter.setdefault(chapter_id, deque()).append(gen_id)
    generation_ids = []
    active = deque(by_chapter.values())
    while active:
        chapter_queue = active.popleft()
        generation_ids.append(chapter_queue.popleft())
        if chapter_queue:
            active.append(chapter_queue)

    # Enqueue each generation through the TTS queue, wrapped with the
    # per-generation completion hook that flips BookSegment.audio_status,
    # reflowed StoryItem times, then runs the book-level drain/progress logic.
    for gen_id in generation_ids:
        gen = db.query(database.Generation).filter_by(id=gen_id).first()
        if gen is None:
            continue

        # Look up the segment that owns this generation so the hook can flip
        # its audio_status.
        seg = db.query(database.BookSegment).filter_by(generation_id=gen_id).first()
        if seg is None:
            logger.warning(
                "No BookSegment found for generation_id %s — skipping enqueue", gen_id
            )
            continue

        inner = run_generation(
            generation_id=gen_id,
            profile_id=gen.profile_id,
            text=gen.text,
            language=gen.language,
            engine=gen.engine,
            model_size=gen.model_size or _DEFAULT_MODEL_SIZE,
            seed=gen.seed,
            instruct=gen.instruct,
            mode="generate",
        )

        wrapped = _per_generation_completion_hook(
            segment_id=seg.id,
            book_id=book_id,
            inner_coro=inner,
        )
        wrapped = task_queue.with_coro_cleanup(wrapped, inner)

        try:
            task_queue.enqueue_generation(gen_id, wrapped, group_id=book_id, concurrency=concurrency)
        except Exception:
            wrapped.close()
            inner.close()
            logger.exception("Failed to enqueue generation %s", gen_id)
            _mark_generation_error(gen_id, "Queue unavailable", db)

    # If nothing was queued (e.g. all segments already rendered) reset now.
    _reset_book_status_if_settled(book_id, db)

    task_id = str(uuid.uuid4())
    return task_id, total


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _reset_book_status_if_settled(book_id: str, db: Session) -> None:
    """Synchronously reset book.status to 'analyzed' if no segments are in-flight.

    Used when no generations were enqueued (nothing to await), so the
    async completion hook would never fire.
    """
    from .. import database

    in_flight = (
        db.query(database.BookSegment)
        .join(database.Chapter, database.BookSegment.chapter_id == database.Chapter.id)
        .filter(
            database.Chapter.book_id == book_id,
            database.BookSegment.audio_status.in_({"pending", "generating"}),
        )
        .count()
    )
    if in_flight == 0:
        book = db.query(database.Book).filter_by(id=book_id).first()
        if book is not None and book.status == "generating":
            book.status = "analyzed"
            db.commit()


def settle_cancelled_book_generation(generation_id: str, db: Session) -> None:
    """Settle book rows even when a queued job never entered its coroutine."""
    from .. import database
    from . import book_events

    segments = db.query(database.BookSegment).filter_by(generation_id=generation_id).all()
    books = set()
    chapters = set()
    for segment in segments:
        if segment.audio_status in {"pending", "generating"}:
            segment.audio_status = "error"
        chapter = db.get(database.Chapter, segment.chapter_id)
        if chapter is not None:
            chapters.add(chapter.id)
            books.add(chapter.book_id)
    if not segments:
        return
    db.commit()
    for chapter_id in chapters:
        _reflow_chapter_story_times(chapter_id, db)
    for book_id in books:
        _publish_progress_and_drain(book_id, db, book_events)


def _recover_failed_materialization(book_id: str, db: Session) -> None:
    """Rollback a failed chapter and settle earlier committed, unqueued chapters."""
    from .. import database

    db.rollback()
    pending = db.query(database.BookSegment).join(database.Chapter).filter(
        database.Chapter.book_id == book_id,
        database.BookSegment.audio_status.in_({"pending", "generating"}),
    ).all()
    for segment in pending:
        segment.audio_status = "error"
        generation = db.get(database.Generation, segment.generation_id) if segment.generation_id else None
        if generation is not None and generation.status in {"pending", "loading_model", "generating"}:
            generation.status = "failed"
            generation.error = "章节任务创建失败，请重试"
    db.commit()
    _reset_book_status_if_settled(book_id, db)


def _claim_book_generation(book, db):
    from fastapi import HTTPException

    from .. import database

    changed = db.query(database.Book).filter(database.Book.id == book.id, database.Book.status == book.status).update({"status": "generating"}, synchronize_session=False)
    db.commit()
    if not changed:
        raise HTTPException(status_code=409, detail="这本书的任务状态已变化，请刷新后重试")


def _preflight_generation(book, db, chapter_ids, engine):
    from fastapi import HTTPException

    from .. import database
    from .profiles import validate_profile_engine

    if book.status in {"analyzing", "exporting"}:
        raise HTTPException(status_code=409, detail="请等待当前分析或导出完成")
    chapter_query = db.query(database.Chapter.id).filter_by(book_id=book.id)
    known = {row[0] for row in chapter_query.all()}
    if chapter_ids is not None and (not chapter_ids or not set(chapter_ids) <= known):
        raise HTTPException(status_code=400, detail="请选择属于本书的有效章节")
    segments = db.query(database.BookSegment).join(database.Chapter).filter(database.Chapter.book_id == book.id)
    if chapter_ids is not None:
        segments = segments.filter(database.BookSegment.chapter_id.in_(chapter_ids))
    if not segments.count():
        raise HTTPException(status_code=409, detail="尚无朗读片段，请先分析小说")
    profiles = db.query(database.VoiceProfile).join(database.BookCharacter, database.BookCharacter.profile_id == database.VoiceProfile.id).filter(database.BookCharacter.book_id == book.id).all()
    profile_by_id = {profile.id: profile for profile in profiles}
    character_profiles = dict(db.query(database.BookCharacter.id, database.BookCharacter.profile_id).filter_by(book_id=book.id).all())
    for character_id, in segments.with_entities(database.BookSegment.character_id).distinct():
        profile = profile_by_id.get(character_profiles.get(character_id))
        if profile is None:
            raise HTTPException(status_code=422, detail="存在未绑定声音的角色，请先分配角色声音")
        try:
            validate_profile_engine(profile, engine or _resolve_engine_for_profile(profile.id, db))
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc


def _resolve_profile_id(segment, db) -> Optional[str]:
    """Return the VoiceProfile id for a segment's assigned character."""
    from .. import database

    if segment.character_id is None:
        return None

    char = db.query(database.BookCharacter).filter_by(id=segment.character_id).first()
    if char is None:
        return None

    return char.profile_id


def _resolve_engine_for_profile(profile_id: str, db) -> str:
    """Return the default engine for a profile, falling back to _DEFAULT_ENGINE."""
    from .. import database

    profile = db.query(database.VoiceProfile).filter_by(id=profile_id).first()
    if profile is None:
        return _DEFAULT_ENGINE

    fallback = {"cloned": "qwen", "designed": "breeze"}.get(profile.voice_type, _DEFAULT_ENGINE)
    return (
        getattr(profile, "default_engine", None)
        or getattr(profile, "preset_engine", None)
        or fallback
    )


def _mark_generation_error(generation_id: str, error: str, db) -> None:
    """Best-effort: flip a Generation row to error status."""
    from .. import database

    try:
        gen = db.query(database.Generation).filter_by(id=generation_id).first()
        if gen is not None:
            gen.status = "failed"
            gen.error = error
            db.query(database.BookSegment).filter_by(generation_id=generation_id).update({"audio_status": "error"})
            db.commit()
    except Exception:
        logger.exception("Failed to mark generation %s as error", generation_id)
