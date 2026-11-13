# ruff: noqa: RUF001
"""Fast analysis saves model calls while ambiguous dialogue keeps its context."""

import json

import pytest

from backend.services import literary_analysis as la


def _roster():
    return [{"name": "林舟", "aliases": ["阿默"]}, {"name": "青禾", "aliases": []}]


def _install_model(monkeypatch, *, speaker="林舟", entries=None):
    calls = []

    async def model(prompt, schema, **kwargs):
        source = json.loads(prompt.split("原文片段：", 1)[1])
        pending = [span for span in source if span.get("needs_analysis", True)]
        calls.append({"source": source, "pending": pending, "kwargs": kwargs})
        annotations = entries(pending) if entries else [
            {
                "segment_id": span["segment_id"],
                "speaker": speaker if span["type"] == "dialogue" else "narrator",
                "type": span["type"],
            }
            for span in pending
        ]
        return schema.model_validate({"speakers": annotations})

    monkeypatch.setattr(la, "generate_structured", model)
    return calls


def _assert_source(result, text):
    actual = "".join(entry.text for entry in result.speakers)
    assert "".join(actual.split()) == "".join(text.split())


def _dialogue(result):
    return [entry for entry in result.speakers if entry.type == "dialogue"]


@pytest.mark.asyncio
async def test_optional_null_fields_do_not_repeat_a_valid_speaker_request(monkeypatch):
    calls = []

    async def model(prompt, schema, **kwargs):
        calls.append(prompt)
        return schema.model_validate({"speakers": [{"i": 0, "s": "林舟", "a": None, "t": None}]})

    monkeypatch.setattr(la, "generate_structured", model)
    result, flagged = await la.detect_chunk("“不明台词。”", "1.7B", known_characters=_roster())
    assert len(calls) == 1
    assert not flagged
    assert result.speakers[0].speaker == "林舟"


@pytest.mark.asyncio
async def test_single_annotation_without_id_uses_the_only_requested_span(monkeypatch):
    calls = []

    async def model(prompt, schema, **kwargs):
        calls.append(json.loads(prompt.split("原文片段：", 1)[1]))
        return schema.model_validate({"speakers": [{"s": "林舟"}]})

    monkeypatch.setattr(la, "generate_structured", model)
    text = "山风吹过。“请等等。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert [[span["segment_id"] for span in call] for call in calls] == [[1]]
    assert not flagged
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_single_annotation_with_wrong_explicit_id_is_not_reassigned(monkeypatch):
    calls = []

    async def model(prompt, schema, **kwargs):
        calls.append(json.loads(prompt.split("原文片段：", 1)[1]))
        return schema.model_validate({"speakers": [{"i": 0, "s": "林舟"}]})

    monkeypatch.setattr(la, "generate_structured", model)
    text = "山风吹过。“请等等。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert [[span["segment_id"] for span in call] for call in calls] == [[1]]
    assert flagged
    assert [entry.speaker for entry in _dialogue(result)] == ["未识别角色"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_multiple_annotations_without_ids_are_not_assigned_by_position(monkeypatch):
    calls = []

    async def model(prompt, schema, **kwargs):
        calls.append(json.loads(prompt.split("原文片段：", 1)[1]))
        return schema.model_validate({"speakers": [{"s": "林舟"}, {"s": "青禾"}]})

    monkeypatch.setattr(la, "generate_structured", model)
    text = "山风吹过。“请等等。”落叶飘落。“再见。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert [[span["segment_id"] for span in call] for call in calls] == [[1, 3], [1], [3]]
    assert flagged
    assert [entry.speaker for entry in _dialogue(result)] == ["未识别角色", "未识别角色"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_truncated_batch_is_split_without_repeating_the_full_request(monkeypatch):
    calls = []

    async def model(prompt, schema, **kwargs):
        from backend.services.llm_structured import StructuredOutputError

        spans = json.loads(prompt.split("原文片段：", 1)[1])
        pending = [span for span in spans if span["needs_analysis"]]
        calls.append(len(pending))
        assert kwargs["retry_on_truncation"] is False
        assert kwargs["compact_schema"] is True
        if len(pending) > 2:
            raise StructuredOutputError("JSON truncated", truncated=True, attempts=1)
        return schema.model_validate({"speakers": [
            {"i": span["segment_id"], "s": "林舟" if span["segment_id"] % 2 == 0 else "青禾"}
            for span in pending
        ]})

    monkeypatch.setattr(la, "generate_structured", model)
    text = "“第一句。”“第二句。”“第三句。”“第四句。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert calls == [4, 2, 2]
    assert not flagged
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟", "青禾", "林舟", "青禾"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_model_failure_does_not_flag_a_known_rule_resolved_line(monkeypatch):
    from backend.services.llm_structured import StructuredOutputError

    async def fail(*args, **kwargs):
        raise StructuredOutputError("unavailable")

    monkeypatch.setattr(la, "generate_structured", fail)
    result = await la.analyze_chapter("林舟说：“已知台词。”“不明台词。”", known_characters=_roster())
    known, unknown = [segment for segment in result.segments if segment["type"] == "dialogue"]
    assert known["speaker"] == "林舟"
    assert known["flagged"] is False
    assert unknown["speaker"] == "未识别角色"
    assert unknown["flagged"] is True


@pytest.mark.asyncio
async def test_plain_narration_does_not_call_model(monkeypatch):
    calls = _install_model(monkeypatch)
    text = "山风拂过空院，落叶沿着石阶缓缓滚动。远处的灯火渐渐熄灭。"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert not calls
    assert all(entry.type == "narration" for entry in result.speakers)
    assert not flagged
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_known_explicit_speakers_skip_model_without_mixing_turns(monkeypatch):
    calls = _install_model(monkeypatch)
    text = "林舟说：“来吧。”青禾回答：“好。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert not calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟", "青禾"]
    assert not flagged
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_simple_delivery_tag_does_not_require_model(monkeypatch):
    calls = _install_model(monkeypatch)
    text = "林舟轻声说：“别惊动他们。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert not calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟"]
    assert not flagged
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_known_alias_reuses_the_canonical_character_without_model(monkeypatch):
    calls = _install_model(monkeypatch)
    text = "阿默说：“我回来了。”"
    result, flagged = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert not calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟"]
    assert not flagged
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_new_named_character_still_goes_to_model(monkeypatch):
    calls = _install_model(monkeypatch, speaker="黎妍")
    text = "黎妍说：“初次见面。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert calls
    assert [entry.speaker for entry in _dialogue(result)] == ["黎妍"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_tag_starting_next_turn_is_not_assigned_to_previous_question(monkeypatch):
    calls = _install_model(monkeypatch)
    text = "“你好吗？”青禾回答：“我很好。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟", "青禾"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_complex_action_does_not_override_model_with_last_mentioned_name(monkeypatch):
    calls = _install_model(monkeypatch)
    text = "林舟望着青禾，轻声说：“走吧。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_unquoted_script_dialogue_still_uses_model_and_preserves_text(monkeypatch):
    def annotations(pending):
        assert pending
        return [
            {"speaker": "林舟", "type": "dialogue", "text": "林舟：走吧。"},
            {"speaker": "青禾", "type": "dialogue", "text": "青禾：等等。"},
        ]

    calls = _install_model(monkeypatch, entries=annotations)
    text = "林舟：走吧。\n青禾：等等。"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟", "青禾"]
    _assert_source(result, text)


@pytest.mark.asyncio
async def test_identity_revelation_is_not_hidden_by_known_alias_fast_path(monkeypatch):
    def annotations(pending):
        return [
            {
                "segment_id": span["segment_id"],
                "speaker": "林舟" if span["type"] == "dialogue" else "narrator",
                "type": span["type"],
                "aliases": ["阿默"] if span["type"] == "dialogue" else [],
                "canonical_name": "林舟" if span["type"] == "dialogue" else None,
                "name_evidence": "林舟的小名是阿默" if span["type"] == "dialogue" else None,
            }
            for span in pending
        ]

    calls = _install_model(monkeypatch, entries=annotations)
    text = "林舟的小名是阿默。阿默说：“我回来了。”"
    result, _ = await la.detect_chunk(text, "1.7B", known_characters=_roster())
    assert calls
    assert [entry.speaker for entry in _dialogue(result)] == ["林舟"]
    _assert_source(result, text)
