# ruff: noqa: RUF001
"""Nested quotation reconstruction is lossless and never touches rendered books."""

import json
import re

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database import Base, Book, BookCharacter, BookSegment, Chapter, Generation, Story, VoiceProfile
from backend.services.character_quote_repair import repair_book_quote_boundaries


@pytest.fixture
def book_cast(tmp_path):
    engine = create_engine(f"sqlite:///{tmp_path / 'quotes.db'}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    book = Book(title="嵌套引号", source_format="txt", status="analyzed")
    profile = VoiceProfile(name="已绑定声音", voice_type="preset")
    db.add_all([book, profile])
    db.flush()
    narrator = BookCharacter(book_id=book.id, name="旁白", is_narrator=True, profile_id=profile.id)
    speaker = BookCharacter(book_id=book.id, name="陆瑾", profile_id=profile.id)
    db.add_all([narrator, speaker])
    db.commit()
    yield {"db": db, "factory": factory, "book": book, "narrator": narrator, "speaker": speaker}
    db.close()
    engine.dispose()


def _seed_chapter(cast, source, number=1):
    db = cast["db"]
    chapter = Chapter(book_id=cast["book"].id, number=number, raw_text=source)
    db.add(chapter)
    db.flush()
    # The old non-nesting regexp cuts the first same-style inner closing quote
    # as if it closed the whole utterance.
    spans, cursor = [], 0
    for match in re.finditer(r'“[^”]*”|「[^」]*」|『[^』]*』|"[^"\n]*"', source):
        if source[cursor:match.start()].strip():
            spans.append((source[cursor:match.start()].strip(), "narration"))
        spans.append((match.group(), "dialogue"))
        cursor = match.end()
    if source[cursor:].strip():
        spans.append((source[cursor:].strip(), "narration"))
    rows = []
    for order, (text, kind) in enumerate(spans):
        row = BookSegment(chapter_id=chapter.id, type=kind, text=text, order=order,
                          character_id=cast["speaker" if kind == "dialogue" else "narrator"].id,
                          emotion="cold" if kind == "dialogue" else None,
                          emotion_intensity=0.6 if kind == "dialogue" else None,
                          delivery="低声" if kind == "dialogue" else None,
                          voice_stage="中年" if kind == "dialogue" else None,
                          analysis_metadata={"method": "llm_source_annotation", "source_excerpt": source,
                                             "original_order": order})
        db.add(row)
        rows.append(row)
    db.commit()
    return chapter, rows


def _read(cast, chapter):
    return cast["db"].query(BookSegment).filter_by(chapter_id=chapter.id).order_by(BookSegment.order).all()


def test_dry_run_reports_nested_chapter_without_writing(book_cast):
    source = "陆瑾道：“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”"
    chapter, original = _seed_chapter(book_cast, source)
    before = [(s.id, s.text, s.type) for s in original]
    report = repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"])
    assert report["counts"]["affected_chapters"] == 1
    assert report["counts"]["nested_ranges"] == 1
    assert report["counts"]["segments_before"] > report["counts"]["segments_after"]
    assert report["source_preserved"] is True
    json.dumps(report, ensure_ascii=False)
    assert before == [(s.id, s.text, s.type) for s in _read(book_cast, chapter)]
    assert not book_cast["db"].new
    assert not book_cast["db"].dirty
    assert not book_cast["db"].deleted


@pytest.mark.parametrize(
    ("source", "utterance"),
    [
        ("陆瑾道：“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”", "“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”"),
        ("陆瑾道：“你说的「金乌『玄光』」就是它。”", "“你说的「金乌『玄光』」就是它。”"),
        ("陆瑾道：「你说的“金乌玄光”就是它。」", "「你说的“金乌玄光”就是它。」"),
    ],
)
def test_same_style_and_mixed_nested_utterance_stays_whole(book_cast, source, utterance):
    chapter, _ = _seed_chapter(book_cast, source)
    before_characters = [(r.id, r.name, r.profile_id) for r in book_cast["db"].query(BookCharacter).all()]
    repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    rows = _read(book_cast, chapter)
    assert [r.text for r in rows if r.type == "dialogue"] == [utterance]
    assert "".join("".join(r.text.split()) for r in rows) == "".join(source.split())
    assert [r.order for r in rows] == list(range(len(rows)))
    assert all(r.character_id == book_cast["narrator"].id for r in rows if r.type == "narration")
    assert before_characters == [(r.id, r.name, r.profile_id) for r in book_cast["db"].query(BookCharacter).all()]
    assert book_cast["db"].query(VoiceProfile).count() == 1


def test_unchanged_spans_keep_ids_and_new_utterance_inherits_largest_old_dialogue(book_cast):
    source = "夜色沉沉。\n陆瑾道：“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”\n陆鸣道：“是。”"
    chapter, old = _seed_chapter(book_cast, source)
    first_id, last_id = old[0].id, old[-1].id
    inherited = next(row for row in old if row.type == "dialogue")
    report = repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    rows = _read(book_cast, chapter)
    assert rows[0].id == first_id
    assert rows[-1].id == last_id
    repaired = next(row for row in rows if "你这是" in row.text)
    assert repaired.id not in {r.id for r in old}
    assert repaired.character_id == inherited.character_id
    assert repaired.emotion == "cold"
    assert repaired.emotion_intensity == 0.6
    assert repaired.delivery == "低声"
    assert repaired.voice_stage == "中年"
    assert repaired.analysis_metadata["quote_boundary_repair"]["inherited_segment_id"] == inherited.id
    assert repaired.analysis_metadata["speaker_verified"] is False
    assert report["counts"]["reused_affected_segments"] == 2


def test_non_nested_chapter_is_untouched_even_beside_nested_chapter(book_cast):
    ordinary, original = _seed_chapter(book_cast, "陆瑾道：“早上好。”", number=1)
    nested, _ = _seed_chapter(book_cast, "陆瑾道：“你说“是”就行。”", number=2)
    before = [(s.id, s.text, s.type, s.order, s.analysis_metadata) for s in original]
    report = repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    assert [c["id"] for c in report["affected_chapters"]] == [nested.id]
    assert before == [(s.id, s.text, s.type, s.order, s.analysis_metadata) for s in _read(book_cast, ordinary)]


def test_long_narration_keeps_old_safe_chunk_boundaries_and_ids(book_cast):
    first, second = "山风拂过。" * 200, "落叶飘下。" * 200
    source = first + second + "陆瑾道：“你说“是”就行。”"
    chapter, original = _seed_chapter(book_cast, source)
    original[0].text = first
    extra = [
        BookSegment(chapter_id=chapter.id, type="narration", text=second, order=1,
                    character_id=book_cast["narrator"].id),
        BookSegment(chapter_id=chapter.id, type="narration", text="陆瑾道：", order=2,
                    character_id=book_cast["narrator"].id),
    ]
    for row in original[1:]:
        row.order += 2
    book_cast["db"].add_all(extra)
    book_cast["db"].commit()
    expected_narration = [(original[0].id, first), (extra[0].id, second), (extra[1].id, "陆瑾道：")]
    repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    rows = _read(book_cast, chapter)
    assert [(row.id, row.text) for row in rows if row.type == "narration"] == expected_narration
    assert [row.text for row in rows if row.type == "dialogue"] == ["“你说“是”就行。”"]
    assert "".join(row.text for row in rows) == source


def test_book_without_nested_quotes_is_a_noop(book_cast):
    chapter, original = _seed_chapter(book_cast, "陆瑾道：“早上好。”")
    report = repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    assert report["counts"]["affected_chapters"] == 0
    assert [s.id for s in _read(book_cast, chapter)] == [s.id for s in original]
    assert not book_cast["db"].dirty


def test_rebuild_does_not_commit_and_can_be_rolled_back(book_cast):
    chapter, original = _seed_chapter(book_cast, "陆瑾道：“你说“是”就行。”")
    before = [(s.id, s.text, s.type, s.order) for s in original]
    repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    with book_cast["factory"]() as reader:
        assert reader.query(BookSegment).filter_by(chapter_id=chapter.id).count() == len(original)
    book_cast["db"].rollback()
    assert before == [(s.id, s.text, s.type, s.order) for s in _read(book_cast, chapter)]


def test_any_audio_in_the_book_refuses_rebuild_including_other_chapters(book_cast):
    _seed_chapter(book_cast, "陆瑾道：“你说“是”就行。”", number=1)
    _, ordinary = _seed_chapter(book_cast, "陆瑾道：“早上好。”", number=2)
    generation = Generation(profile_id=book_cast["speaker"].profile_id, text=ordinary[-1].text,
                            language="zh", audio_path="must-keep.wav", duration=1)
    book_cast["db"].add(generation)
    book_cast["db"].flush()
    ordinary[-1].generation_id = generation.id
    book_cast["db"].commit()
    with pytest.raises(ValueError, match="已有生成音频"):
        repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    assert book_cast["db"].get(Generation, generation.id).audio_path == "must-keep.wav"


def test_existing_story_link_refuses_rebuild(book_cast):
    chapter, _ = _seed_chapter(book_cast, "陆瑾道：“你说“是”就行。”")
    story = Story(name="用户时间线")
    book_cast["db"].add(story)
    book_cast["db"].flush()
    chapter.story_id = story.id
    book_cast["db"].commit()
    with pytest.raises(ValueError, match="已有故事时间线"):
        repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)


@pytest.mark.parametrize("status", ["analyzing", "generating", "exporting"])
def test_active_book_refuses_rebuild(book_cast, status):
    _seed_chapter(book_cast, "陆瑾道：“你说“是”就行。”")
    book_cast["book"].status = status
    with pytest.raises(ValueError, match="正在运行"):
        repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)


def test_source_gap_refuses_all_changes(book_cast):
    _, original = _seed_chapter(book_cast, "陆瑾道：“你说“是”就行。”")
    original[0].text = "缺少原文。"
    with pytest.raises(ValueError, match="未完整保留"):
        repair_book_quote_boundaries(book_cast["book"].id, book_cast["db"], dry_run=False)
    assert original[-1] in book_cast["db"]
