# ruff: noqa: RUF001
"""Source-grounded cast regressions from the first long novel import.

The fake model deliberately returns the malformed identities seen in that
import.  Correct source tags must win without inventing speakers, while the
original novel text remains complete.  No test loads a language model.
"""

import json

import pytest

from backend.services import literary_analysis as la


def _roster(*names):
    return [{"name": name, "aliases": []} for name in names]


def _bad_model(monkeypatch, speaker, *, identity=None):
    async def generate(prompt, schema, **kwargs):
        spans = json.loads(prompt.split("原文片段：", 1)[1])
        replies = []
        for span in spans:
            reply = {
                "segment_id": span["segment_id"],
                "speaker": speaker if span["type"] == "dialogue" else "narrator",
                "type": span["type"],
            }
            if span["type"] == "dialogue" and identity:
                reply.update(identity)
            replies.append(reply)
        return schema.model_validate({"speakers": replies})

    monkeypatch.setattr(la, "generate_structured", generate)


def _assert_detected_source(result, text):
    assert "".join("".join(entry.text.split()) for entry in result.speakers) == "".join(text.split())


def _assert_chapter_source(result, text):
    assert "".join("".join(segment["text"].split()) for segment in result.segments) == "".join(text.split())


def _dialogue_speakers(result):
    return [entry.speaker for entry in result.speakers if entry.type == "dialogue"]


@pytest.mark.asyncio
@pytest.mark.parametrize("modifier", ["赶紧", "连忙", "沉吟着", "正色"])
async def test_speaking_action_does_not_extend_a_known_name(monkeypatch, modifier):
    bad_name = f"陆鸣{modifier}"
    _bad_model(monkeypatch, bad_name)
    text = f"陆鸣{modifier}说道：“先等等。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster("陆鸣"))
    assert _dialogue_speakers(result) == ["陆鸣"]
    _assert_detected_source(result, text)


@pytest.mark.asyncio
@pytest.mark.parametrize("subject", ["陆瑾", "周宣"])
async def test_listener_phrase_is_resolved_per_source_subject(monkeypatch, subject):
    _bad_model(monkeypatch, "对着陆鸣")
    text = f"{subject}对着陆鸣说道：“你先坐下。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster("陆鸣", "陆瑾", "周宣"))
    assert _dialogue_speakers(result) == [subject]
    _assert_detected_source(result, text)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("subject", "direction", "bad_name"),
    [
        ("石统", "笑眯眯的道", "笑眯眯的"),
        ("祝镰", "笑眯眯的道", "笑眯眯的"),
        ("高玉", "轻笑道", "轻笑"),
        ("周茯苓", "轻笑道", "轻笑"),
        ("陆瑾", "关心的问道", "关心的"),
    ],
)
async def test_delivery_is_not_a_character(monkeypatch, subject, direction, bad_name):
    _bad_model(monkeypatch, bad_name)
    text = f"{subject}{direction}：“你可还好？”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster(subject, "陆鸣"))
    assert _dialogue_speakers(result) == [subject]
    _assert_detected_source(result, text)


@pytest.mark.asyncio
async def test_example_name_absent_from_source_cannot_create_a_character(monkeypatch):
    _bad_model(monkeypatch, "林默")
    text = "黑暗中传来声音：“你先走。”"
    chapter = await la.analyze_chapter(text, known_characters=[])
    assert "林默" not in {character["name"] for character in chapter.characters}
    dialogue = [segment for segment in chapter.segments if segment["type"] == "dialogue"]
    assert dialogue
    assert all(segment["flagged"] for segment in dialogue)
    _assert_chapter_source(chapter, text)


@pytest.mark.asyncio
async def test_quoted_technique_names_remain_narration(monkeypatch):
    _bad_model(monkeypatch, "生灵之焱")
    text = "陆鸣修炼“生灵之焱”，又翻阅“摔碑印”的口诀。"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster("陆鸣"))
    assert all(entry.type == "narration" for entry in result.speakers)
    _assert_detected_source(result, text)


@pytest.mark.asyncio
async def test_short_quoted_utterances_remain_dialogue(monkeypatch):
    _bad_model(monkeypatch, "未识别角色")
    text = "陆鸣说：“走！”陆瑾回答：“好。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster("陆鸣", "陆瑾"))
    assert _dialogue_speakers(result) == ["陆鸣", "陆瑾"]
    _assert_detected_source(result, text)


@pytest.mark.asyncio
async def test_action_phrase_cannot_become_the_canonical_identity(monkeypatch):
    text = "陆瑾本名未改。陆瑾则是沉吟着说道：“先等等。”"
    _bad_model(
        monkeypatch,
        "陆瑾",
        identity={
            "canonical_name": "陆瑾则是",
            "aliases": ["陆瑾"],
            "name_evidence": "陆瑾则是沉吟着说道",
        },
    )
    chapter = await la.analyze_chapter(text, known_characters=_roster("陆瑾"))
    assert {character["name"] for character in chapter.characters} == {"陆瑾"}
    assert {segment["speaker"] for segment in chapter.segments if segment["type"] == "dialogue"} == {"陆瑾"}
    _assert_chapter_source(chapter, text)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["李贯", "李贯一", "欧阳修", "慕容复", "Elizabeth Bennet"])
async def test_real_names_are_not_truncated_or_rejected(monkeypatch, name):
    _bad_model(monkeypatch, name)
    text = f'{name} said: "Come along."' if name.isascii() else f"{name}说道：“请进。”"
    known = list(dict.fromkeys(["李贯", "李贯一", name]))
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster(*known))
    assert _dialogue_speakers(result) == [name]
    _assert_detected_source(result, text)


@pytest.mark.asyncio
async def test_same_style_nested_quotations_stay_one_speakers_complete_utterance(monkeypatch):
    _bad_model(monkeypatch, "神通骨")
    utterance = "“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”"
    text = f"陆瑾道：{utterance}"
    chapter = await la.analyze_chapter(text, known_characters=_roster("陆瑾"))
    dialogue = [segment for segment in chapter.segments if segment["type"] == "dialogue"]
    assert [(segment["speaker"], segment["text"]) for segment in dialogue] == [("陆瑾", utterance)]
    assert {character["name"] for character in chapter.characters} == {"陆瑾"}
    _assert_chapter_source(chapter, text)


@pytest.mark.asyncio
async def test_mixed_nested_quote_styles_preserve_text_and_turn_boundaries(monkeypatch):
    _bad_model(monkeypatch, "金乌玄光")
    first = '“他说‘金乌玄光’来自「大金乌馆」，并非"神通骨"。”'
    second = "“当真？”"
    text = f"陆瑾说道：{first}\n周宣问：{second}"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster("陆瑾", "周宣"))
    dialogue = [entry for entry in result.speakers if entry.type == "dialogue"]
    assert [(entry.speaker, entry.text) for entry in dialogue] == [("陆瑾", first), ("周宣", second)]
    _assert_detected_source(result, text)


@pytest.mark.asyncio
async def test_quoted_subject_with_an_action_is_not_its_own_utterance(monkeypatch):
    _bad_model(monkeypatch, "神通骨")
    utterance = "“告诉我，你是谁？”"
    text = f"“焦炭”勉强的睁开眼，胸膛起伏。片刻后，有声音传来：{utterance}"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=[])
    subject = [entry for entry in result.speakers if "焦炭" in entry.text]
    assert subject
    assert all(entry.type == "narration" for entry in subject)
    dialogue = [entry for entry in result.speakers if entry.type == "dialogue"]
    assert [entry.text for entry in dialogue] == [utterance]
    assert all(entry.speaker != "神通骨" for entry in dialogue)
    _assert_detected_source(result, text)


@pytest.mark.parametrize(
    "utterance",
    [
        "“这是一段应当完整保留的台词，请随我继续向前走。”",
        "“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”",
    ],
)
def test_normal_utterance_crossing_a_window_boundary_is_kept_whole(utterance):
    text = "山" * 1000 + f"陆瑾道：{utterance}陆鸣点头，继续向前。"
    chunks = la.chunk_text(text, max_tokens=256, preserve_quotes=True)
    assert len(chunks) > 1
    assert sum(utterance in chunk for chunk in chunks) == 1
    assert "".join(chunks) == text
    assert all(len(chunk) <= 4096 for chunk in chunks)


@pytest.mark.asyncio
async def test_chapter_chunking_keeps_a_nested_utterance_at_the_window_edge(monkeypatch):
    _bad_model(monkeypatch, "陆瑾")
    utterance = "“你这是上神通“金乌玄光”，你们是“大金乌馆”的人。”"
    text = "山" * 1000 + f"陆瑾道：{utterance}陆鸣点头，继续向前。"
    chapter = await la.analyze_chapter(text, known_characters=_roster("陆瑾", "陆鸣"))
    dialogue = [segment for segment in chapter.segments if segment["type"] == "dialogue"]
    assert [(segment["speaker"], segment["text"]) for segment in dialogue] == [("陆瑾", utterance)]
    _assert_chapter_source(chapter, text)


def test_extreme_monologue_is_split_into_bounded_chunks_without_losing_text():
    max_tokens = 64
    limit = max_tokens * 4
    utterance = "“" + "言" * (limit * 5) + "。”"
    text = f"陆瑾道：{utterance}陆鸣静静听着。"
    chunks = la.chunk_text(text, max_tokens=max_tokens, preserve_quotes=True)
    assert len(chunks) > 1
    assert all(len(chunk) <= limit * 4 for chunk in chunks)
    assert not any(utterance in chunk for chunk in chunks)
    assert "".join(chunks) == text
