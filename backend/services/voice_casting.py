"""Stable, playable Qwen casting for an entire book.

A character keeps one profile across chapters.  Casting reuses installed Qwen
preset profiles, respects existing assignments, and chooses a matching speaker
before reusing one when the finite preset inventory is exhausted.
"""

from __future__ import annotations

import logging
import re
import uuid
from datetime import datetime

from sqlalchemy.orm import Session

from ..database import Book, BookCharacter, VoiceProfile

logger = logging.getLogger(__name__)
DEFAULT_ENGINE = "qwen_custom_voice"


def _get_preset_ids(engine: str) -> list[str]:
    from . import profiles as profiles_service

    return sorted(profiles_service._get_preset_voice_ids(engine))


def get_qwen_voices() -> list[tuple]:
    """Return the backend's real (id, name, gender, language, description) inventory."""
    from ..backends.qwen_custom_voice_backend import QWEN_CUSTOM_VOICES

    return list(QWEN_CUSTOM_VOICES)


def _unique_name(base: str, db: Session) -> str:
    name = base
    n = 1
    while db.query(VoiceProfile).filter_by(name=name).first() is not None:
        n += 1
        name = f"{base} ({n})"
    return name


def _gender(character: BookCharacter, prompt: str = "") -> str | None:
    gender = (character.gender or "").strip().lower()
    if gender in {"female", "woman", "girl", "女", "女性"}:
        return "female"
    if gender in {"male", "man", "boy", "男", "男性"}:
        return "male"
    if re.search(r"female|woman|girl|女", prompt, re.I):
        return "female"
    if re.search(r"\bmale\b|\bman\b|\bboy\b|男", prompt, re.I):
        return "male"
    return None


def choose_voice_id(
    character: BookCharacter,
    *,
    language: str = "zh",
    prompt: str = "",
    used: dict[str, str] | None = None,
    cooccurrence: set[tuple[str, str]] | None = None,
) -> str:
    """Choose deterministically from playable Qwen speakers using character traits.

    Matching gender takes priority; within that group narrator separation,
    scene distinctness and inventory usage are considered before timbre traits.
    Reuse is intentional for large casts: CustomVoice has nine speaker identities.
    """
    inventory = set(_get_preset_ids(DEFAULT_ENGINE))
    voices = [voice for voice in get_qwen_voices() if voice[0] in inventory]
    if not voices:
        raise ValueError("400:没有可用的 Qwen 内置声音，请先在声音档案中启用 Qwen CustomVoice")
    used = used or {}
    cooccurrence = cooccurrence or set()
    traits = " ".join(filter(None, [prompt, character.vocal_description, character.archetype, character.age_range])).lower()
    gender = _gender(character, traits)
    old = bool(re.search(r"elder|old|senior|aged|老年|年迈|苍老|沧桑|中老年", traits))
    ages = [int(n) for n in re.findall(r"\d+", character.age_range or "")]
    old = old or bool(ages and min(ages) >= 55)
    warm = bool(re.search(r"warm|gentle|soft|温柔|温暖|柔和|沉稳|低沉|醇厚", traits))
    bright = bool(re.search(r"bright|lively|energetic|活泼|明亮|开朗|清脆|俏皮", traits))

    def score(voice: tuple) -> tuple[int, str]:
        voice_id, _name, voice_gender, native_language, _description = voice
        value = 0
        if gender and voice_gender != gender:
            value += 1000
        # Keep the narrator audibly separate even without an explicit scene edge.
        matches = [cid for cid, vid in used.items() if vid == voice_id]
        value += len(matches) * 40
        for cid in matches:
            if character.is_narrator or (character.id, cid) in cooccurrence or (cid, character.id) in cooccurrence:
                value += 100
        if native_language != language:
            value += 12
        if character.is_narrator and voice_id == ("Uncle_Fu" if language == "zh" else "Ryan"):
            value -= 20
        if old and voice_id == "Uncle_Fu":
            value -= 25
        if warm and voice_id in {"Serena", "Uncle_Fu", "Sohee"}:
            value -= 15
        if bright and voice_id in {"Vivian", "Eric", "Aiden", "Ono_Anna"}:
            value -= 15
        if not old and voice_id == "Uncle_Fu" and not character.is_narrator:
            value += 6
        return value, voice_id

    return min(voices, key=score)[0]


def create_preset_profile(
    character: BookCharacter,
    book: Book,
    voice_id: str,
    db: Session,
) -> VoiceProfile:
    """Create a book-owned, playable Qwen profile for a selected speaker."""
    if voice_id not in _get_preset_ids(DEFAULT_ENGINE):
        raise ValueError("400:未知的 Qwen 内置声音")
    profile = VoiceProfile(
        id=str(uuid.uuid4()),
        name=_unique_name(f"{character.name} — {book.title}", db),
        description=character.vocal_description,
        language="zh",
        voice_type="preset",
        preset_engine=DEFAULT_ENGINE,
        preset_voice_id=voice_id,
        default_engine=DEFAULT_ENGINE,
        book_id=book.id,
        is_library=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(profile)
    db.flush()
    return profile


def create_designed_profile(
    character: BookCharacter,
    book: Book,
    prompt: str,
    db: Session,
) -> VoiceProfile:
    """Create an explicitly requested Breeze voice, separate from default casting."""
    profile = VoiceProfile(
        id=str(uuid.uuid4()),
        name=_unique_name(f"{character.name} — {book.title}", db),
        description=prompt,
        language="zh",
        voice_type="designed",
        design_prompt=prompt,
        default_engine="breeze",
        book_id=book.id,
        is_library=False,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
    )
    db.add(profile)
    db.flush()
    return profile


_create_designed_profile = create_designed_profile
_create_preset_profile = create_preset_profile


def _existing_qwen_profile(voice_id: str, book_id: str, db: Session) -> VoiceProfile | None:
    profiles = (
        db.query(VoiceProfile)
        .filter_by(voice_type="preset", preset_engine=DEFAULT_ENGINE, preset_voice_id=voice_id)
        .all()
    )
    # Never borrow a private voice from another book. Existing global voices are
    # valid library choices even if created before is_library was introduced.
    profiles = [p for p in profiles if p.book_id is None or p.book_id == book_id]
    return min(profiles, key=lambda p: (p.book_id is not None, not p.is_library, p.name, p.id), default=None)


def cast_book(
    book_id: str,
    *,
    cooccurrence: set[tuple[str, str]],
    language: str,
    db: Session,
    voice_mode: str = "preset",
) -> None:
    """Cast unassigned characters once, preserving bindings on every later run."""
    book = db.get(Book, book_id)
    if book is None:
        raise ValueError(f"Book {book_id!r} not found")
    characters = db.query(BookCharacter).filter_by(book_id=book_id).all()
    used: dict[str, str] = {}
    for character in characters:
        profile = db.get(VoiceProfile, character.profile_id) if character.profile_id else None
        if profile and profile.voice_type == "preset" and profile.preset_engine == DEFAULT_ENGINE:
            used[character.id] = profile.preset_voice_id
        elif character.profile_id and profile is None:
            # A deleted voice must not leave a character silently unrenderable.
            character.profile_id = None

    ordered = sorted(characters, key=lambda c: (not c.is_narrator, c.role != "major", -(c.dialogue_count or 0), c.name, c.id))
    for character in ordered:
        if character.profile_id:
            continue
        if voice_mode == "designed":
            from .voice_prompts import match_voice_prompt

            prompt = match_voice_prompt(
                db,
                gender=character.gender,
                age_range=character.age_range,
                description=character.vocal_description or "",
                is_narrator=bool(character.is_narrator),
            )
            description = character.vocal_description or ""
            design_prompt = "\n".join(filter(None, [prompt.prompt if prompt else None, description]))
            if not design_prompt:
                raise ValueError("400:未找到适合该角色的设计提示词，请先导入提示词库或填写角色声音描述")
            profile = create_designed_profile(character, book, design_prompt, db)
            character.profile_id = profile.id
            continue
        voice_id = choose_voice_id(character, language=language, used=used, cooccurrence=cooccurrence)
        if voice_id in used.values():
            logger.warning("书籍 %s 的角色 %s 复用 Qwen 声线 %s：匹配的内置声线数量有限", book.id, character.name, voice_id)
        profile = _existing_qwen_profile(voice_id, book.id, db)
        if profile is None:
            profile = create_preset_profile(character, book, voice_id, db)
        character.profile_id = profile.id
        used[character.id] = voice_id
    db.commit()
