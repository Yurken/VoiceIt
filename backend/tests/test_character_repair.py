# ruff: noqa: RUF001
"""Existing cast repairs preserve source, voice identities and transaction control."""

import copy
import json

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, Book, BookCharacter, BookSegment, Chapter, Generation, VoiceProfile
from backend.services import character_repair as repair


@pytest.fixture
def cast(tmp_path, monkeypatch):
    engine = create_engine(f"sqlite:///{tmp_path / 'repair.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    book = Book(title="既有角色修复", source_format="txt", status="analyzed")
    other = Book(title="另一本书", source_format="txt", status="analyzed")
    db.add_all([book, other])
    profiles = [VoiceProfile(name=f"保留音色{index}", voice_type="preset") for index in range(5)]
    db.add_all(profiles)
    db.flush()
    narrator = BookCharacter(book_id=book.id, name="旁白", is_narrator=True, profile_id=profiles[0].id)
    canonical = BookCharacter(book_id=book.id, name="陆鸣", profile_id=profiles[1].id, dialogue_count=100)
    polluted = BookCharacter(book_id=book.id, name="陆鸣赶紧", profile_id=profiles[2].id, dialogue_count=2)
    renamed = BookCharacter(book_id=book.id, name="陆瑾则", aliases=["陆瑾"], profile_id=profiles[3].id, dialogue_count=20)
    rejected = BookCharacter(book_id=book.id, name="关心的", profile_id=profiles[4].id, dialogue_count=2)
    unknown = BookCharacter(book_id=book.id, name="未识别角色", profile_id=profiles[0].id)
    manual = BookCharacter(book_id=book.id, name="手动角色", profile_id=profiles[1].id, dialogue_count=0)
    unrelated = BookCharacter(book_id=other.id, name="别书人物", profile_id=profiles[2].id, dialogue_count=12)
    db.add_all([narrator, canonical, polluted, renamed, rejected, unknown, manual, unrelated])
    db.flush()
    texts = ["陆鸣说：", "“你好。”", "陆瑾沉吟：", "“再想想。”", "一个词。", "“谁？”"]
    chapter = Chapter(book_id=book.id, number=1, raw_text="".join(texts))
    db.add(chapter)
    db.flush()
    owners = [narrator, polluted, narrator, renamed, rejected, rejected]
    kinds = ["narration", "dialogue", "narration", "dialogue", "dialogue", "dialogue"]
    segments = []
    for index, (text, owner, kind) in enumerate(zip(texts, owners, kinds, strict=True)):
        generation = Generation(profile_id=owner.profile_id, text=text, language="zh", audio_path=f"kept-{index}.wav", duration=1)
        db.add(generation)
        db.flush()
        segment = BookSegment(chapter_id=chapter.id, character_id=owner.id, type=kind, order=index, text=text,
                              generation_id=generation.id, audio_status="completed")
        db.add(segment)
        segments.append(segment)
    db.commit()

    def refine(chapter_texts, chapter_segments, roster):
        assert chapter_texts == [chapter.raw_text]
        assert {row["name"] for row in roster} == {"陆鸣", "陆鸣赶紧", "陆瑾则", "关心的", "未识别角色", "手动角色"}
        refined = chapter_segments[0]
        refined[1]["speaker"] = "陆鸣"
        refined[3]["speaker"] = "陆瑾"
        refined[4].update(type="narration", speaker=None)
        refined[5].update(speaker="未识别角色", flagged=True)
        for segment in refined:
            segment["analysis_metadata"]["method"] = "repair_source_tag"
        return {
            "characters": [
                {"name": "陆鸣", "aliases": [], "confidence": 0.95, "dialogue_count": 1, "role": "major"},
                {"name": "陆瑾", "aliases": [], "confidence": 0.95, "dialogue_count": 1, "role": "minor"},
                {"name": "未识别角色", "aliases": [], "confidence": 0.25, "dialogue_count": 1, "role": "minor"},
            ],
            "name_map": {"陆鸣": "陆鸣", "陆鸣赶紧": "陆鸣", "陆瑾则": "陆瑾", "关心的": None,
                         "未识别角色": None, "手动角色": "手动角色"},
            "changes": [{"name": "陆鸣赶紧", "canonical": "陆鸣"}],
        }

    monkeypatch.setattr(repair, "_refine_segments", refine)
    yield {
        "db": db, "factory": factory, "book": book, "other": other, "chapter": chapter, "segments": segments,
        "canonical": canonical, "polluted": polluted, "renamed": renamed, "rejected": rejected,
        "unknown": unknown, "narrator": narrator, "manual": manual, "unrelated": unrelated, "refine": refine,
    }
    db.close()
    engine.dispose()


def test_dry_run_is_serializable_and_does_not_mutate_database(cast):
    db, book = cast["db"], cast["book"]
    before = [(s.id, s.character_id, s.type, s.text, copy.deepcopy(s.analysis_metadata)) for s in cast["segments"]]
    report = repair.repair_book_cast(book.id, db)
    assert report["dry_run"] is True
    assert report["counts"]["characters_before"] == 6
    assert report["counts"]["characters_after"] == 4  # 3 active + unused manual identity
    assert report["counts"]["reassigned_segments"] == 3
    assert report["counts"]["retyped_segments"] == 1
    assert report["counts"]["stale_audio"] == 3
    assert set(report["removed_names"]) == {"陆鸣赶紧", "关心的"}
    assert report["retained_unmapped_names"] == ["手动角色"]
    json.dumps(report, ensure_ascii=False)
    assert before == [(s.id, s.character_id, s.type, s.text, s.analysis_metadata) for s in cast["segments"]]
    assert not db.dirty
    assert not db.new
    assert not db.deleted


def test_apply_keeps_canonical_ids_profiles_source_and_generated_files(cast):
    db = cast["db"]
    source = [(s.id, s.order, s.text, s.generation_id) for s in cast["segments"]]
    canonical_id, canonical_profile = cast["canonical"].id, cast["canonical"].profile_id
    renamed_id, renamed_profile = cast["renamed"].id, cast["renamed"].profile_id
    rejected_id, polluted_id = cast["rejected"].id, cast["polluted"].id
    repair.repair_book_cast(cast["book"].id, db, dry_run=False)
    assert source == [(s.id, s.order, s.text, s.generation_id) for s in cast["segments"]]
    assert db.get(BookCharacter, canonical_id).profile_id == canonical_profile
    renamed = db.get(BookCharacter, renamed_id)
    assert renamed.name == "陆瑾"
    assert renamed.profile_id == renamed_profile
    assert renamed.aliases == []
    assert [s.character_id for s in cast["segments"]] == [cast["narrator"].id, canonical_id, cast["narrator"].id,
                                                         renamed_id, cast["narrator"].id, cast["unknown"].id]
    assert [s.audio_status for s in cast["segments"]] == ["completed", "stale", "completed", "completed", "stale", "stale"]
    assert db.get(BookCharacter, polluted_id) is None
    assert db.get(BookCharacter, rejected_id) is None
    assert db.query(VoiceProfile).count() == 5
    assert db.query(Generation).count() == 6
    assert all(row.audio_path.startswith("kept-") for row in db.query(Generation).all())
    assert db.get(BookCharacter, cast["manual"].id) is not None
    assert db.get(BookCharacter, cast["unrelated"].id).dialogue_count == 12
    assert cast["narrator"].dialogue_count == 3


def test_apply_leaves_commit_and_rollback_to_caller(cast):
    db = cast["db"]
    polluted_id = cast["polluted"].id
    repair.repair_book_cast(cast["book"].id, db, dry_run=False)
    with cast["factory"]() as reader:
        assert reader.get(BookCharacter, polluted_id) is not None
    db.rollback()
    assert db.get(BookCharacter, polluted_id) is not None
    assert cast["segments"][1].character_id == polluted_id
    assert cast["renamed"].name == "陆瑾则"


def test_exact_canonical_identity_wins_over_a_more_frequent_polluted_name(cast):
    cast["canonical"].dialogue_count = 1
    cast["polluted"].dialogue_count = 1000
    repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert cast["segments"][1].character_id == cast["canonical"].id


def test_retained_unused_identity_has_live_count_and_minor_role(cast):
    cast["manual"].dialogue_count = 100
    cast["manual"].role = "major"
    repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert cast["manual"].dialogue_count == 0
    assert cast["manual"].role == "minor"


def test_voice_stage_repair_is_persisted_and_invalidates_existing_audio(cast, monkeypatch):
    segment = cast["segments"][3]
    segment.voice_stage = "老年"
    segment.analysis_metadata = {"voice_stage_evidence": "错误人物的年龄", "voice_stage_inherited": True}

    def reset_stage(texts, segments, roster):
        result = cast["refine"](texts, segments, roster)
        refined = segments[0][3]
        refined["voice_stage"] = None
        refined["analysis_metadata"].pop("voice_stage_evidence", None)
        refined["analysis_metadata"].pop("voice_stage_inherited", None)
        return result

    monkeypatch.setattr(repair, "_refine_segments", reset_stage)
    report = repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert segment.character_id == cast["renamed"].id
    assert segment.voice_stage is None
    assert segment.audio_status == "stale"
    assert "voice_stage_evidence" not in segment.analysis_metadata
    assert "voice_stage_inherited" not in segment.analysis_metadata
    assert report["counts"]["restaged_segments"] == 1
    assert report["counts"]["stale_audio"] == 4


def test_same_identity_rename_preserves_unchanged_stage_and_audio(cast):
    segment = cast["segments"][3]
    segment.voice_stage = "中年"
    repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert cast["renamed"].name == "陆瑾"
    assert segment.voice_stage == "中年"
    assert segment.audio_status == "completed"


@pytest.mark.parametrize("status", ["analyzing", "generating", "exporting"])
def test_repair_refuses_active_work(cast, status):
    cast["book"].status = status
    with pytest.raises(ValueError, match="任务完成"):
        repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)


def test_preexisting_source_gap_prevents_repair(cast):
    cast["segments"][0].text = "丢字"
    with pytest.raises(ValueError, match="未完整保留原文"):
        repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert cast["renamed"].name == "陆瑾则"


@pytest.mark.parametrize("mutation", ["text", "id", "order", "remove", "kind"])
def test_refinement_cannot_change_source_or_segment_identity(cast, monkeypatch, mutation):
    def bad_refine(texts, segments, roster):
        result = cast["refine"](texts, segments, roster)
        if mutation == "remove":
            segments[0].pop()
        else:
            segments[0][0]["type" if mutation == "kind" else mutation] = "invalid"
        return result

    monkeypatch.setattr(repair, "_refine_segments", bad_refine)
    with pytest.raises(ValueError, match="角色修复"):
        repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert cast["segments"][1].character_id == cast["polluted"].id
    assert cast["renamed"].name == "陆瑾则"


def test_missing_refined_character_prevents_partial_changes(cast, monkeypatch):
    def incomplete(texts, segments, roster):
        result = cast["refine"](texts, segments, roster)
        result["characters"].pop()
        return result

    monkeypatch.setattr(repair, "_refine_segments", incomplete)
    with pytest.raises(ValueError, match="完整角色表"):
        repair.repair_book_cast(cast["book"].id, cast["db"], dry_run=False)
    assert cast["renamed"].name == "陆瑾则"


def test_external_reference_protects_an_old_row_from_deletion(cast):
    db = cast["db"]
    other_chapter = Chapter(book_id=cast["other"].id, number=1, raw_text="别处。")
    db.add(other_chapter)
    db.flush()
    reference = BookSegment(chapter_id=other_chapter.id, character_id=cast["polluted"].id,
                            type="dialogue", text="别处。")
    db.add(reference)
    db.commit()
    report = repair.repair_book_cast(cast["book"].id, db, dry_run=False)
    assert "陆鸣赶紧" not in report["removed_names"]
    assert db.get(BookCharacter, reference.character_id) is not None
    assert reference.type == "dialogue"
    assert reference.text == "别处。"


def test_missing_unknown_row_creates_one_separate_from_narrator(cast):
    db = cast["db"]
    db.delete(cast["unknown"])
    db.commit()

    def refine(texts, segments, roster):
        # The original fixture's fake checks the roster. Substitute its absent
        # unknown candidate only inside this pure input snapshot.
        return cast["refine"](texts, segments, [*roster, {"name": "未识别角色"}])

    original = repair._refine_segments
    repair._refine_segments = refine
    try:
        repair.repair_book_cast(cast["book"].id, db, dry_run=False)
    finally:
        repair._refine_segments = original
    unknown = db.query(BookCharacter).filter_by(book_id=cast["book"].id, name="未识别角色").one()
    assert not unknown.is_narrator
    assert unknown.id != cast["narrator"].id
    assert unknown.profile_id == cast["narrator"].profile_id
    assert cast["segments"][5].character_id == unknown.id
