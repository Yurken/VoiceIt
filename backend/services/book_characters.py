"""Services for character roster, voice-assignment, voice-options, and preview.

Implements the business logic for Contract 02 character endpoints.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime

from sqlalchemy import or_
from sqlalchemy.orm import Session

from ..database import Book, BookCharacter, VoiceProfile
from ..models import (
    CharacterResponse,
    CharacterUpdate,
    VoiceOptionsResponse,
    VoiceProfileSummary,
)
from ..services import voice_casting as vc
from ..services.generation import run_generation
from ..services.task_queue import enqueue_generation
from ..utils.tasks import get_task_manager
from ..utils.voice_stages import normalize_voice_stage

# ---------------------------------------------------------------------------
# Internal helpers (thin seams for test monkeypatching)
# ---------------------------------------------------------------------------


def _get_qwen_voices() -> list[tuple]:
    return vc.get_qwen_voices()


def _unique_profile_name(base: str, db: Session) -> str:
    """Return *base* if not taken in DB, else append (2), (3) …"""
    name = base
    n = 1
    while db.query(VoiceProfile).filter_by(name=name).first() is not None:
        n += 1
        name = f"{base} ({n})"
    return name


# ---------------------------------------------------------------------------
# Serializer
# ---------------------------------------------------------------------------


def character_to_response(char: BookCharacter, db: Session) -> CharacterResponse:
    """Convert a BookCharacter ORM row to a CharacterResponse pydantic model.

    voice_type / voice_label / is_library are derived from the assigned
    VoiceProfile when present; they are null / False when unassigned.
    """
    voice_type: str | None = None
    voice_label: str | None = None
    is_library: bool = False

    if char.profile_id:
        profile = db.get(VoiceProfile, char.profile_id)
        if profile is not None:
            voice_type = getattr(profile, "voice_type", None) or "cloned"
            voice_label = profile.name
            is_library = bool(profile.is_library)

    return CharacterResponse(
        id=char.id,
        name=char.name,
        color=char.color,
        profile_id=char.profile_id,
        voice_type=voice_type,
        voice_label=voice_label,
        is_library=is_library,
        is_narrator=bool(char.is_narrator),
        role=char.role,
        gender=char.gender,
        age_range=normalize_voice_stage(char.age_range),
        vocal_description=char.vocal_description,
        archetype=char.archetype,
        dialogue_count=char.dialogue_count or 0,
        confidence=char.confidence,
        aliases=char.aliases or [],
    )


# ---------------------------------------------------------------------------
# Roster
# ---------------------------------------------------------------------------


def get_character_roster(book_id: str, db: Session) -> list[CharacterResponse]:
    """Return CharacterResponse list for all characters in the book.

    Raises ValueError with a '404' sentinel if the book doesn't exist.
    """
    book = db.get(Book, book_id)
    if book is None:
        raise ValueError("404:book not found")

    chars = (
        db.query(BookCharacter)
        .filter_by(book_id=book_id)
        .order_by(BookCharacter.is_narrator.desc(), BookCharacter.dialogue_count.desc())
        .all()
    )
    return [character_to_response(c, db) for c in chars]


# ---------------------------------------------------------------------------
# Update character (rename / recolor / assign voice)
# ---------------------------------------------------------------------------


def update_character(
    book_id: str,
    char_id: str,
    data: CharacterUpdate,
    db: Session,
) -> CharacterResponse:
    """Apply a CharacterUpdate to the specified character.

    Assignment modes (mutually exclusive):
      - profile_id     → assign existing profile directly
      - design_prompt  → create an explicitly requested Breeze designed profile
      - preset_voice_id → create a Qwen preset profile and assign it

    Other fields (name, color, is_narrator) may be updated independently.

    Raises ValueError with '404' sentinel for unknown book/char.
    """
    book = db.get(Book, book_id)
    if book is None:
        raise ValueError("404:book not found")

    char = db.get(BookCharacter, char_id)
    if char is None or char.book_id != book_id:
        raise ValueError("404:character not found")

    # Enforce "exactly one of" constraint for voice-assignment fields
    prompt_id = getattr(data, "prompt_id", None)
    voice_fields = [data.profile_id, data.design_prompt, data.preset_voice_id, prompt_id]
    original_profile_id = char.profile_id
    if book.status == "generating" and any(value is not None for value in voice_fields):
        from fastapi import HTTPException

        raise HTTPException(status_code=409, detail="正在生成朗读，请完成后再修改角色声音")
    if sum(1 for f in voice_fields if f is not None) > 1:
        raise ValueError(
            "400:profile_id、design_prompt、preset_voice_id、prompt_id 只能选择一项"
        )

    # Simple field updates
    if data.name is not None:
        char.name = data.name
    if data.color is not None:
        char.color = data.color
    if data.is_narrator is not None:
        char.is_narrator = data.is_narrator

    # Voice assignment — only one of profile_id / design_prompt / preset_voice_id
    if data.profile_id is not None:
        # Assign an existing profile directly
        existing = db.get(VoiceProfile, data.profile_id)
        if existing is None:
            raise ValueError("400:profile not found")
        char.profile_id = data.profile_id

    elif data.design_prompt is not None or prompt_id is not None:
        # Breeze is the explicit VoiceDesign path; default automatic casting
        # continues to use the existing Qwen CustomVoice inventory.
        prompt_text = data.design_prompt
        if prompt_id is not None:
            from .voice_prompts import get_prompt

            prompt = get_prompt(db, prompt_id)
            if prompt is None:
                raise ValueError("400:声音设计提示词不存在")
            prompt_text = prompt.prompt
        new_profile = vc.create_designed_profile(char, book, prompt_text, db)
        char.profile_id = new_profile.id

    elif data.preset_voice_id is not None:
        # A manually chosen Qwen speaker remains fixed for every chapter.
        new_profile = vc.create_preset_profile(char, book, data.preset_voice_id, db)
        char.profile_id = new_profile.id

    if char.profile_id != original_profile_id:
        from ..database import BookSegment

        db.query(BookSegment).filter(
            BookSegment.character_id == char.id,
            BookSegment.generation_id.isnot(None),
        ).update({"audio_status": "stale"}, synchronize_session=False)

    db.commit()
    db.refresh(char)
    return character_to_response(char, db)


# ---------------------------------------------------------------------------
# Voice options
# ---------------------------------------------------------------------------


def get_voice_options(book_id: str, db: Session) -> VoiceOptionsResponse:
    """Return three-section voice picker payload for a book.

    library  — library profiles and existing global profiles
    book     — profiles with book_id == this book
    presets  — all Qwen CustomVoice speakers, with descriptions

    Raises ValueError with '404' sentinel if book not found.
    """
    book = db.get(Book, book_id)
    if book is None:
        raise ValueError("404:book not found")

    library_profiles = (
        db.query(VoiceProfile)
        .filter(or_(VoiceProfile.is_library.is_(True), VoiceProfile.book_id.is_(None)))
        .filter(~VoiceProfile.name.like("_preview_%"))
        .all()
    )
    book_profiles = (
        db.query(VoiceProfile)
        .filter(VoiceProfile.book_id == book_id)
        .all()
    )

    qwen_presets = [
        {
            "voice_id": vid,
            "name": name,
            "gender": gender,
            "language": lang,
            "description": description,
            "engine": vc.DEFAULT_ENGINE,
        }
        for vid, name, gender, lang, description in _get_qwen_voices()
    ]

    return VoiceOptionsResponse(
        library=[VoiceProfileSummary.model_validate(p) for p in library_profiles],
        book=[VoiceProfileSummary.model_validate(p) for p in book_profiles],
        presets=qwen_presets,
    )


# ---------------------------------------------------------------------------
# Save to library
# ---------------------------------------------------------------------------


def save_character_to_library(char_id: str, db: Session) -> VoiceProfileSummary:
    """Promote the character's assigned voice profile to the global library.

    Flips profile.is_library=True and profile.book_id=None on the *same*
    profile row (promotion, not copy).

    Raises ValueError with '404' sentinel if character not found.
    Raises ValueError with '400' sentinel if character has no profile.
    """
    char = db.get(BookCharacter, char_id)
    if char is None:
        raise ValueError("404:character not found")

    if not char.profile_id:
        raise ValueError("400:character has no voice profile assigned")

    profile = db.get(VoiceProfile, char.profile_id)
    if profile is None:
        raise ValueError("400:assigned profile not found")

    profile.is_library = True
    profile.book_id = None
    profile.updated_at = datetime.utcnow()

    db.commit()
    db.refresh(profile)
    return VoiceProfileSummary.model_validate(profile)


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

_PREVIEW_TEXT = "你好，这是我的声音。故事即将开始，请和我一起走进书中的世界。"


def profile_engine(profile: VoiceProfile) -> str:
    if profile.voice_type == "preset":
        return profile.preset_engine or vc.DEFAULT_ENGINE
    if profile.voice_type == "designed":
        return profile.default_engine or "breeze"
    return profile.default_engine or "qwen"


async def _clean_finished_candidate_previews(char_id: str, keep_name: str, db: Session) -> None:
    """Reap only this character's unused, finished candidates and their files."""
    from ..database import BookSegment, Generation, StoryItem
    from .history import delete_generation
    from .profiles import delete_profile

    prefix = f"_preview_{char_id}_"
    candidates = db.query(VoiceProfile).filter(VoiceProfile.name.like(prefix + "%")).all()
    for profile in candidates:
        if not profile.name.startswith(prefix) or profile.name == keep_name:
            continue
        if db.query(BookCharacter).filter_by(profile_id=profile.id).first():
            continue
        generations = db.query(Generation).filter_by(profile_id=profile.id).all()
        if any(g.status in {"pending", "queued", "generating", "processing"} or g.source != "book_preview" for g in generations):
            continue
        ids = [g.id for g in generations]
        if ids and (db.query(BookSegment).filter(BookSegment.generation_id.in_(ids)).first() or db.query(StoryItem).filter(StoryItem.generation_id.in_(ids)).first()):
            continue
        for generation in generations:
            await delete_generation(generation.id, db)
        await delete_profile(profile.id, db)


def _resolve_preview_profile(
    char: BookCharacter,
    profile_id: str | None,
    preset_voice_id: str | None,
    design_prompt: str | None,
    db: Session,
) -> tuple[str, str, str]:
    """Resolve (effective_profile_id, engine, voice_type) for a preview request.

    Candidate priority (first supplied wins):
      1. profile_id   — audition an existing profile by id
      2. preset_voice_id — audition a preset voice (no DB row created)
      3. design_prompt   — audition a designed voice (no DB row created)
      4. <none>          — use the character's assigned voice

    Raises ValueError('400:...') when nothing is resolvable.
    Raises ValueError('404:...') when an explicit profile_id is not found.
    """
    if profile_id is not None:
        prof = db.get(VoiceProfile, profile_id)
        if prof is None:
            raise ValueError("404:preview profile not found")
        engine = profile_engine(prof)
        vt = getattr(prof, "voice_type", None) or "cloned"
        return prof.id, engine, vt

    if preset_voice_id is not None:
        if preset_voice_id not in vc._get_preset_ids(vc.DEFAULT_ENGINE):
            raise ValueError("400:未知的 Qwen 内置声音")
        return "_preset_" + preset_voice_id, vc.DEFAULT_ENGINE, "preset"

    if design_prompt is not None:
        return "_designed_" + design_prompt[:32], "breeze", "designed"

    # Fallback: character's assigned voice
    if not char.profile_id:
        raise ValueError("400:character has no voice profile assigned")
    prof = db.get(VoiceProfile, char.profile_id)
    if prof is None:
        raise ValueError("400:assigned profile not found")
    engine = profile_engine(prof)
    vt = getattr(prof, "voice_type", None) or "cloned"
    return prof.id, engine, vt


async def preview_character_voice(
    char_id: str,
    text: str | None,
    db: Session,
    *,
    profile_id: str | None = None,
    preset_voice_id: str | None = None,
    design_prompt: str | None = None,
    emotion: str | None = None,
    instruct: str | None = None,
    prompt_id: str | None = None,
) -> dict:
    """Synthesize a short preview clip through the serial TTS queue.

    When a candidate voice is supplied (profile_id / preset_voice_id /
    design_prompt), that voice is auditioned without persisting a new profile.
    Without any candidate the character's assigned voice is used.

    Returns {generation_id, audio_path}.

    Raises ValueError with '404' sentinel if character not found.
    Raises ValueError with '400' sentinel if no voice is resolvable.
    """
    char = db.get(BookCharacter, char_id)
    if char is None:
        raise ValueError("404:character not found")

    if prompt_id is not None:
        if any(value is not None for value in (profile_id, preset_voice_id, design_prompt)):
            raise ValueError("400:每次试听只能指定一种声音来源")
        from .voice_prompts import get_prompt

        prompt = get_prompt(db, prompt_id)
        if prompt is None:
            raise ValueError("400:声音设计提示词不存在")
        design_prompt = prompt.prompt

    effective_profile_id, engine, _voice_type = _resolve_preview_profile(
        char, profile_id, preset_voice_id, design_prompt, db
    )

    preview_text = text or _PREVIEW_TEXT
    language = "zh" if any("\u4e00" <= c <= "\u9fff" for c in preview_text) else "en"
    generation_id = str(uuid.uuid4())

    # A verbatim instruct (e.g. composed by the book pipeline's compose_instruct,
    # so preview matches what generation will render) takes precedence; otherwise
    # wrap a bare emotion as a tone hint.
    if instruct is None and emotion:
        instruct = f"Speak with a {emotion} tone."

    # Reuse a stable candidate profile so repeated auditions share one reference
    # and seed. The permanent character assignment remains untouched.
    is_ephemeral = effective_profile_id.startswith(("_preset_", "_designed_"))

    if is_ephemeral:
        candidate_hash = hashlib.sha256(f"{engine}:{preset_voice_id or design_prompt}".encode()).hexdigest()[:16]
        candidate_name = f"_preview_{char.id}_{candidate_hash}"
        await _clean_finished_candidate_previews(char.id, candidate_name, db)
        tmp_profile = db.query(VoiceProfile).filter_by(name=candidate_name).first()
        if tmp_profile is None and preset_voice_id is not None:
            tmp_profile = VoiceProfile(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, candidate_name)),
                name=candidate_name,
                language="zh",
                voice_type="preset",
                preset_engine=vc.DEFAULT_ENGINE,
                preset_voice_id=preset_voice_id,
                default_engine=vc.DEFAULT_ENGINE,
                is_library=False,
            )
        elif tmp_profile is None:
            tmp_profile = VoiceProfile(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, candidate_name)),
                name=candidate_name,
                language="zh",
                voice_type="designed",
                design_prompt=design_prompt,
                default_engine="breeze",
                description=design_prompt,
                is_library=False,
            )
        db.add(tmp_profile)
        db.flush()
        gen_profile_id = tmp_profile.id
    else:
        gen_profile_id = effective_profile_id

    # Create a pending generation row
    from ..database import Generation as DBGeneration
    gen_row = DBGeneration(
        id=generation_id,
        profile_id=gen_profile_id,
        text=preview_text,
        language=language,
        engine=engine,
        audio_path="",
        duration=0,
        status="generating",
        source="book_preview",
        created_at=datetime.utcnow(),
    )
    db.add(gen_row)
    db.commit()

    # Register in task manager
    task_manager = get_task_manager()
    task_manager.start_generation(
        task_id=generation_id,
        profile_id=gen_profile_id,
        text=preview_text,
    )

    # Build generation params — for ephemeral candidates override profile fields
    from .book_generation import stable_voice_seed

    gen_kwargs: dict = dict(
        generation_id=generation_id,
        profile_id=gen_profile_id,
        text=preview_text,
        language=language,
        engine=engine,
        model_size=None,
        seed=stable_voice_seed(gen_profile_id),
        normalize=True,
        instruct=instruct,
        mode="generate",
    )

    # Enqueue via the serial TTS queue
    enqueue_generation(
        generation_id,
        run_generation(**gen_kwargs),
    )

    return {
        "generation_id": generation_id,
        "audio_path": "",
    }
