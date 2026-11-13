"""Book casting must produce playable, stable voices from the existing inventory."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, Book, BookCharacter, VoiceProfile
from backend.services import voice_casting as casting


@pytest.fixture
def db(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'casting.db'}")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    yield session
    session.close()
    engine.dispose()


def book_cast(db, characters):
    book = Book(title="雨城", source_format="txt", status="analyzed")
    db.add(book)
    db.flush()
    chars = [BookCharacter(book_id=book.id, **attrs) for attrs in characters]
    db.add_all(chars)
    db.commit()
    return book, chars


def profile(db, char):
    db.refresh(char)
    return db.get(VoiceProfile, char.profile_id)


def test_qwen_casts_narrator_leads_and_minors_without_designed_or_kokoro(db):
    book, chars = book_cast(db, [
        {"name": "旁白", "is_narrator": True},
        {"name": "林青", "role": "major", "gender": "female", "vocal_description": "温柔清澈"},
        {"name": "阿宁", "role": "minor", "gender": "male", "age_range": "18-25"},
    ])
    casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    profiles = [profile(db, c) for c in chars]
    assert all(p.voice_type == "preset" and p.preset_engine == "qwen_custom_voice" for p in profiles)
    assert all(p.default_engine == "qwen_custom_voice" for p in profiles)
    assert len({p.preset_voice_id for p in profiles}) == 3
    assert profiles[0].preset_voice_id == "Uncle_Fu"
    assert profiles[1].preset_voice_id == "Serena"


def test_existing_nine_global_qwen_profiles_are_reused_even_without_library_flag(db):
    for vid, name, _gender, lang, _desc in casting.get_qwen_voices():
        db.add(VoiceProfile(id="global-" + vid, name=name, voice_type="preset", preset_engine=casting.DEFAULT_ENGINE,
                            preset_voice_id=vid, default_engine=casting.DEFAULT_ENGINE, language=lang, is_library=False))
    db.commit()
    book, chars = book_cast(db, [{"name": "旁白", "is_narrator": True}, {"name": "少女", "gender": "female"}])
    casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    assert all(c.profile_id.startswith("global-") for c in chars)
    assert db.query(VoiceProfile).count() == 9


def test_matching_respects_gender_age_and_traits(db):
    book, chars = book_cast(db, [
        {"name": "爷爷", "gender": "male", "age_range": "65-80", "vocal_description": "苍老醇厚"},
        {"name": "妹妹", "gender": "female", "age_range": "14-18", "vocal_description": "开朗活泼"},
    ])
    casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    assert profile(db, chars[0]).preset_voice_id == "Uncle_Fu"
    assert profile(db, chars[1]).preset_voice_id == "Vivian"


def test_major_characters_get_distinct_matching_voices_before_minor_reuse(db):
    book, chars = book_cast(db, [
        {"name": "乙", "gender": "female", "role": "major", "dialogue_count": 100},
        {"name": "甲", "gender": "female", "role": "major", "dialogue_count": 200},
        {"name": "配角", "gender": "female", "role": "minor"},
    ])
    casting.cast_book(book.id, cooccurrence={(chars[0].id, chars[1].id)}, language="zh", db=db)
    assert len({profile(db, c).preset_voice_id for c in chars}) == 3


def test_reanalysis_keeps_whole_book_identity_and_manual_binding(db):
    book, chars = book_cast(db, [{"name": "少女", "gender": "female"}, {"name": "男主", "gender": "male"}])
    manual = VoiceProfile(name="手动克隆声音", voice_type="cloned", default_engine="qwen")
    db.add(manual)
    db.flush()
    chars[0].profile_id = manual.id
    db.commit()
    casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    before = [c.profile_id for c in chars]
    count = db.query(VoiceProfile).count()
    chars[1].age_range = "60-80"
    casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    assert [c.profile_id for c in chars] == before
    assert chars[0].profile_id == manual.id
    assert db.query(VoiceProfile).count() == count


def test_exhausted_inventory_reuses_playable_voice_and_warns(db, monkeypatch, caplog):
    monkeypatch.setattr(casting, "_get_preset_ids", lambda engine: ["Serena"])
    book, chars = book_cast(db, [{"name": "甲", "gender": "female"}, {"name": "乙", "gender": "female"}])
    casting.cast_book(book.id, cooccurrence={(chars[0].id, chars[1].id)}, language="zh", db=db)
    assert all(profile(db, c).preset_voice_id == "Serena" for c in chars)
    assert "内置声线数量有限" in caplog.text
    assert db.query(VoiceProfile).count() == 1


def test_missing_inventory_fails_clearly_without_fake_designed_profiles(db, monkeypatch):
    monkeypatch.setattr(casting, "_get_preset_ids", lambda engine: [])
    book, chars = book_cast(db, [{"name": "旁白", "is_narrator": True}])
    with pytest.raises(ValueError, match="没有可用的 Qwen"):
        casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    assert chars[0].profile_id is None
    assert db.query(VoiceProfile).count() == 0


def test_deleted_binding_is_repaired_and_invalid_preset_rejected(db):
    book, chars = book_cast(db, [{"name": "阿宁", "profile_id": "deleted"}])
    casting.cast_book(book.id, cooccurrence=set(), language="zh", db=db)
    assert profile(db, chars[0]) is not None
    with pytest.raises(ValueError, match="未知的 Qwen"):
        casting.create_preset_profile(chars[0], book, "af_heart", db)


def test_invalid_book_id_raises_value_error(db):
    with pytest.raises(ValueError, match="not found"):
        casting.cast_book("missing", cooccurrence=set(), language="zh", db=db)


def test_explicit_designed_mode_matches_catalogue_and_keeps_breeze_profile(db):
    from backend.database.models import VoiceDesignPrompt

    db.add_all([
        VoiceDesignPrompt(id="old-man", name="老先生", category="旁白", description="沉稳年迈男声", prompt="苍老沉稳的男性声音", gender="male", age_range="senior"),
        VoiceDesignPrompt(id="young-woman", name="少女", category="角色", description="年轻女性温柔", prompt="温柔的年轻女性声音", gender="female", age_range="young"),
    ])
    db.commit()
    book, chars = book_cast(db, [{"name": "林青", "gender": "female", "age_range": "young", "vocal_description": "温柔亲切"}])
    casting.cast_book(book.id, cooccurrence=set(), language="zh", voice_mode="designed", db=db)
    selected = profile(db, chars[0])
    assert selected.voice_type == "designed"
    assert selected.default_engine == "breeze"
    assert "温柔的年轻女性声音" in selected.design_prompt
    assert "温柔亲切" in selected.design_prompt
    original_id = selected.id
    casting.cast_book(book.id, cooccurrence=set(), language="zh", voice_mode="designed", db=db)
    assert profile(db, chars[0]).id == original_id
