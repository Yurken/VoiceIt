# ruff: noqa: RUF001
"""Exercise the CPU refinement pass with actual source identity validation."""

import copy
import re

import pytest

from backend.services.character_refinement import refine_character_segments


def _roster(*names):
    return [{"name": name, "aliases": []} for name in names]


def _segments(text, speakers):
    """Fixture spans retain exact source text, including whitespace and tags."""
    result = []
    position = 0
    dialogue = 0
    for match in re.finditer(r"“[^”]*”", text):
        if match.start() > position:
            result.append({"type": "narration", "text": text[position:match.start()], "speaker": None})
        result.append({"type": "dialogue", "text": match.group(), "speaker": speakers[dialogue]})
        dialogue += 1
        position = match.end()
    if position < len(text):
        result.append({"type": "narration", "text": text[position:], "speaker": None})
    assert dialogue == len(speakers)
    for index, segment in enumerate(result):
        segment.update(id=f"segment-{index}", order=index, flagged=False,
                       analysis_metadata={"method": "llm_source_annotation"})
    return result


def _dialogue(segments):
    return [segment for segment in segments if segment["type"] == "dialogue"]


def _assert_source_unchanged(segments, original, text):
    assert [(s["id"], s["order"], s["text"]) for s in segments] == [
        (s["id"], s["order"], s["text"]) for s in original
    ]
    assert "".join(segment["text"] for segment in segments) == text


@pytest.mark.parametrize("subject", ["陆瑾", "周宣"])
def test_recipient_pollution_recovers_each_source_subject(subject):
    text = f"{subject}对着陆鸣说道：“你先坐下。”"
    segments = _segments(text, ["对着陆鸣"])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("陆瑾", "周宣", "陆鸣", "对着陆鸣"))
    assert [s["speaker"] for s in _dialogue(segments)] == [subject]
    assert [c["name"] for c in result["characters"]] == [subject]
    assert result["name_map"]["对着陆鸣"] is None
    assert not _dialogue(segments)[0]["flagged"]
    _assert_source_unchanged(segments, original, text)


def test_one_shared_bad_label_can_recover_different_people_across_chapters():
    texts = ["石统笑眯眯的道：“你来了。”", "祝镰笑眯眯的道：“随我走吧。”"]
    chapters = [_segments(text, ["笑眯眯的"]) for text in texts]
    originals = copy.deepcopy(chapters)
    result = refine_character_segments(texts, chapters, _roster("石统", "祝镰", "笑眯眯的"))
    assert [[s["speaker"] for s in _dialogue(chapter)] for chapter in chapters] == [["石统"], ["祝镰"]]
    assert {c["name"] for c in result["characters"]} == {"石统", "祝镰"}
    for text, segments, original in zip(texts, chapters, originals, strict=True):
        _assert_source_unchanged(segments, original, text)


def test_polluted_canonical_name_recovers_its_source_grounded_old_alias():
    text = "陆瑾沉吟着说道：“先等等。”"
    segments = _segments(text, ["陆瑾则是"])
    original = copy.deepcopy(segments)
    roster = [{"name": "陆瑾则是", "aliases": ["陆瑾"], "id": "old-real-person"}]
    result = refine_character_segments([text], [segments], roster)
    assert result["name_map"]["陆瑾则是"] == "陆瑾"
    assert [c["name"] for c in result["characters"]] == ["陆瑾"]
    assert [s["speaker"] for s in _dialogue(segments)] == ["陆瑾"]
    _assert_source_unchanged(segments, original, text)


def test_model_example_name_absent_from_the_book_becomes_flagged_unknown():
    text = "黑暗中传来声音：“你先走。”"
    segments = _segments(text, ["林默"])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("林默"))
    line = _dialogue(segments)[0]
    assert line["speaker"] == "未识别角色"
    assert line["flagged"] is True
    assert result["name_map"]["林默"] is None
    assert result["characters"][0]["confidence"] < 0.5
    assert result["characters"][0]["role"] == "minor"
    _assert_source_unchanged(segments, original, text)


def test_quoted_references_become_narration_but_short_speech_stays_dialogue():
    text = "陆鸣修炼“生灵之焱”，又翻阅“摔碑印”的口诀。陆鸣说：“走！”陆瑾回答：“好。”"
    segments = _segments(text, ["生灵之焱", "摔碑印", "陆鸣", "陆瑾"])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("陆鸣", "陆瑾", "生灵之焱", "摔碑印"))
    terms = [s for s in segments if s["text"] in {"“生灵之焱”", "“摔碑印”"}]
    assert all(s["type"] == "narration" for s in terms)
    assert all(s["speaker"] is None for s in terms)
    assert [s["speaker"] for s in _dialogue(segments)] == ["陆鸣", "陆瑾"]
    assert {c["name"] for c in result["characters"]} == {"陆鸣", "陆瑾"}
    _assert_source_unchanged(segments, original, text)


def test_ambiguous_multiple_actors_are_not_assigned_to_the_nearest_name():
    text = "陆瑾和周宣互相看了一眼，随即有人说道：“先等等。”"
    segments = _segments(text, ["关心的"])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("陆瑾", "周宣", "关心的"))
    line = _dialogue(segments)[0]
    assert line["speaker"] == "未识别角色"
    assert line["flagged"] is True
    assert {c["name"] for c in result["characters"]} == {"未识别角色"}
    _assert_source_unchanged(segments, original, text)


def test_confidence_and_major_role_follow_evidence_and_book_prominence():
    text = "陆鸣说：“一。”" * 100 + "周宣坐在一旁。“二。”“三。”"
    segments = _segments(text, ["陆鸣"] * 100 + ["周宣"] * 2)
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("陆鸣", "周宣"))
    characters = {c["name"]: c for c in result["characters"]}
    assert characters["陆鸣"]["confidence"] >= 0.8
    assert characters["陆鸣"]["role"] == "major"
    assert characters["周宣"]["confidence"] < 0.8
    assert characters["周宣"]["dialogue_count"] == 2
    assert characters["周宣"]["role"] == "minor"
    _assert_source_unchanged(segments, original, text)


def test_repeating_refinement_with_the_clean_roster_is_semantically_idempotent():
    text = "陆瑾对着陆鸣说道：“你来了。”陆鸣修炼“生灵之焱”。黑暗中传来声音：“等我。”"
    segments = _segments(text, ["对着陆鸣", "生灵之焱", "林默"])
    original = copy.deepcopy(segments)
    first = refine_character_segments([text], [segments], _roster("陆瑾", "陆鸣", "对着陆鸣", "生灵之焱", "林默"))
    refined = copy.deepcopy(segments)
    second = refine_character_segments([text], [segments], first["characters"])
    assert second["characters"] == first["characters"]
    assert second["changes"] == []
    assert segments == refined
    _assert_source_unchanged(segments, original, text)


def test_mismatched_chapter_lists_fail_before_mutating_segments():
    text = "陆鸣说：“走！”"
    segments = _segments(text, ["陆鸣"])
    original = copy.deepcopy(segments)
    with pytest.raises(ValueError, match="章节数量"):
        refine_character_segments([text, text], [segments], _roster("陆鸣"))
    assert segments == original


@pytest.mark.parametrize(
    ("bad_name", "context"),
    [
        ("揉着眼", "陆鸣揉着眼"),
        ("柔声", "屋内传来柔声"),
        ("感激", "陆鸣心存感激"),
        ("有了这", "有了这番安排"),
        ("火莲元炁", "陆鸣运转火莲元炁"),
        ("神通骨", "眼前的神通骨闪烁"),
        ("小雷光", "陆鸣施展小雷光"),
        ("凤阳府", "众人来到凤阳府"),
    ],
)
def test_old_roster_and_source_occurrence_are_not_proof_of_a_person(bad_name, context):
    text = f"{context}，黑暗中传来声音：“你来了。”"
    segments = _segments(text, [bad_name])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("陆鸣", bad_name))
    assert bad_name not in {character["name"] for character in result["characters"]}
    assert result["name_map"][bad_name] is None
    line = _dialogue(segments)[0]
    assert line["speaker"] == "未识别角色"
    assert line["flagged"] is True
    _assert_source_unchanged(segments, original, text)


@pytest.mark.parametrize("descriptor", ["中年男子", "灰衣中年男子", "白虎姐姐"])
def test_human_descriptions_and_titles_remain_valid_cast_candidates(descriptor):
    text = f"{descriptor}说道：“随我来。”"
    segments = _segments(text, [descriptor])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster(descriptor))
    assert [character["name"] for character in result["characters"]] == [descriptor]
    assert [line["speaker"] for line in _dialogue(segments)] == [descriptor]
    _assert_source_unchanged(segments, original, text)


@pytest.mark.parametrize("name", ["李贯一", "司徒青云", "Elizabeth Bennet"])
def test_strict_identity_bank_preserves_real_names(name):
    text = f'{name} said: “Come along.”' if name.isascii() else f"{name}说道：“随我来。”"
    segments = _segments(text, [name])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("李贯", name))
    assert [character["name"] for character in result["characters"]] == [name]
    assert [line["speaker"] for line in _dialogue(segments)] == [name]
    _assert_source_unchanged(segments, original, text)


def test_age_introduction_preserves_a_name_with_a_rare_surname():
    text = "十七岁的青禾轻声说：“初次见面。”"
    segments = _segments(text, ["青禾"])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("青禾"))
    assert [character["name"] for character in result["characters"]] == ["青禾"]
    assert [line["speaker"] for line in _dialogue(segments)] == ["青禾"]
    _assert_source_unchanged(segments, original, text)


@pytest.mark.parametrize("direction", ["惊讶", "缓缓"])
def test_source_tags_cannot_expand_the_closed_bank_with_action_names(direction):
    text = f"陆鸣{direction}说道：“你来了。”"
    segments = _segments(text, ["未识别角色"])
    original = copy.deepcopy(segments)
    result = refine_character_segments([text], [segments], _roster("陆鸣"))
    assert {character["name"] for character in result["characters"]} <= {"陆鸣", "未识别角色"}
    assert {line["speaker"] for line in _dialogue(segments)} <= {"陆鸣", "未识别角色"}
    assert f"陆鸣{direction}" not in result["name_map"]
    _assert_source_unchanged(segments, original, text)
