# ruff: noqa: RUF001
"""Tests for the literary analysis pipeline (B2).

All LLM calls are mocked — this test suite never loads a model.
"""

import pytest

from backend.services import literary_analysis as la
from backend.services.llm_structured import StructuredOutputError

# ---------------------------------------------------------------------------
# Pure-function unit tests
# ---------------------------------------------------------------------------


def test_chunking_respects_budget():
    chunks = la.chunk_text("word " * 5000, max_tokens=512)
    assert len(chunks) > 1
    assert all(chunks)


def test_chunking_small_text_one_chunk():
    chunks = la.chunk_text("Short text.", max_tokens=2048)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_reconcile_merges_aliases():
    per_chunk = [
        {"speaker": "Holston", "confidence": 0.9},
        {"speaker": "the sheriff", "confidence": 0.6},
        {"speaker": "Holston", "confidence": 0.8},
    ]
    roster = la.reconcile_characters(per_chunk, aliases={"the sheriff": "Holston"})
    names = {c["name"] for c in roster}
    assert "Holston" in names
    holston = next(c for c in roster if c["name"] == "Holston")
    assert holston["dialogue_count"] == 3


def test_reconcile_without_aliases():
    per_chunk = [
        {"speaker": "Alice", "confidence": 0.9},
        {"speaker": "Bob", "confidence": 0.7},
        {"speaker": "Alice", "confidence": 0.85},
    ]
    roster = la.reconcile_characters(per_chunk, aliases={})
    names = {c["name"] for c in roster}
    assert "Alice" in names
    assert "Bob" in names
    alice = next(c for c in roster if c["name"] == "Alice")
    assert alice["dialogue_count"] == 2
    assert alice["confidence"] == pytest.approx(0.875)


def test_reconcile_confidence_keeps_uncertain_attributions_visible():
    per_chunk = [
        {"speaker": "Hero", "confidence": 0.5},
        {"speaker": "Hero", "confidence": 0.95},
        {"speaker": "Hero", "confidence": 0.3},
    ]
    roster = la.reconcile_characters(per_chunk, aliases={})
    hero = next(c for c in roster if c["name"] == "Hero")
    assert hero["confidence"] == pytest.approx((0.5 + 0.95 + 0.3) / 3)


def test_interrupted_utterance_splits(monkeypatch):
    # '"We are leaving," she said, "tonight."' → dialogue / narration / dialogue
    segs = la.split_into_segments('"We are leaving," she said, "tonight."', speaker="Jules")
    types = [s["type"] for s in segs]
    assert types == ["dialogue", "narration", "dialogue"]


def test_split_order_is_sequential():
    segs = la.split_into_segments('"We are leaving," she said, "tonight."', speaker="Jules")
    orders = [s["order"] for s in segs]
    assert orders == sorted(orders)
    assert orders[0] == 0


def test_em_dash_dialogue_detected():
    segs = la.split_into_segments("— We are leaving, she said.", speaker="Jules")
    assert any(s["type"] == "dialogue" for s in segs)


def test_simple_quoted_dialogue():
    segs = la.split_into_segments('"Hello," she said.', speaker="Alice")
    types = [s["type"] for s in segs]
    # At minimum we should get dialogue detected
    assert "dialogue" in types


def test_pure_narration_text():
    segs = la.split_into_segments("He walked down the road.", speaker=None)
    assert all(s["type"] == "narration" for s in segs)


def test_split_segment_has_required_keys():
    segs = la.split_into_segments('"Hello."', speaker="Bob")
    for s in segs:
        assert "type" in s
        assert "text" in s
        assert "order" in s
        assert s["type"] in ("dialogue", "narration")


# ---------------------------------------------------------------------------
# Async orchestrator tests (mocked LLM)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_failed_unknown_dialogue_preserves_source_and_flags(monkeypatch):
    async def boom(*a, **k):
        raise StructuredOutputError("bad")

    monkeypatch.setattr(la, "generate_structured", boom)
    result = await la.analyze_chapter("“快走！”", model_size="1.7B")
    assert result.segments  # produced something
    assert all(s["type"] == "dialogue" for s in result.segments)
    assert result.flagged is True


@pytest.mark.asyncio
async def test_analyze_chapter_returns_chapter_analysis(monkeypatch):


    # Craft a canned response matching the schema
    canned = {
        "speakers": [
            {
                "speaker": "Alice",
                "text": "Hello there",
                "emotion": "happy",
                "intensity": 0.7,
                "type": "dialogue",
            }
        ]
    }

    async def fake_generate(prompt, schema, **kwargs):
        return schema.model_validate(canned)

    monkeypatch.setattr(la, "generate_structured", fake_generate)
    result = await la.analyze_chapter('"Hello there," Alice said.', model_size="1.7B")
    assert hasattr(result, "segments")
    assert hasattr(result, "characters")
    assert hasattr(result, "flagged")


@pytest.mark.asyncio
async def test_analyze_chapter_flagged_false_on_success(monkeypatch):
    canned = {"speakers": [{"segment_id": 0, "speaker": "narrator", "type": "narration"}]}

    async def fake_generate(prompt, schema, **kwargs):
        return schema.model_validate(canned)

    monkeypatch.setattr(la, "generate_structured", fake_generate)
    result = await la.analyze_chapter("Narration text only.", model_size="1.7B")
    assert result.flagged is False


@pytest.mark.asyncio
async def test_analyze_book_aggregates_chapters(monkeypatch):
    canned = {
        "speakers": [
            {
                "speaker": "Bob",
                "text": "Let's go",
                "emotion": "eager",
                "intensity": 0.8,
                "type": "dialogue",
            }
        ]
    }

    async def fake_generate(prompt, schema, **kwargs):
        return schema.model_validate(canned)

    monkeypatch.setattr(la, "generate_structured", fake_generate)
    chapters = ["Chapter one text.", "Chapter two text."]
    result = await la.analyze_book(chapters, model_size="1.7B")
    assert hasattr(result, "chapters")
    assert len(result.chapters) == 2
    assert hasattr(result, "characters")


@pytest.mark.asyncio
async def test_analyze_book_reports_progress(monkeypatch):
    """analyze_book must drive progress_cb through detection and enrichment.

    Regression guard for the "appears hung" UX: the long LLM passes have to
    emit monotonic, in-range progress with human-readable messages.
    """
    canned = {
        "speakers": [
            {
                "speaker": "Bob",
                "text": "Let's go",
                "emotion": "eager",
                "intensity": 0.8,
                "type": "dialogue",
            }
        ]
    }

    async def fake_generate(prompt, schema, **kwargs):
        return schema.model_validate(canned)

    monkeypatch.setattr(la, "generate_structured", fake_generate)

    events: list[tuple[float, str]] = []
    chapters = ["Chapter one text.", "Chapter two text.", "Chapter three text."]
    await la.analyze_book(
        chapters,
        model_size="1.7B",
        progress_cb=lambda frac, msg: events.append((frac, msg)),
    )

    assert events, "progress_cb was never called"
    fractions = [f for f, _ in events]
    # In range, monotonic non-decreasing, and reaches completion.
    assert all(0.0 <= f <= 1.0 for f in fractions)
    assert fractions == sorted(fractions)
    assert fractions[-1] == 1.0
    # Per-chapter detection messages are surfaced.
    assert any("第 1/3 章" in msg for _, msg in events)
    # Enrichment phase reports per-character work past the detect band.
    assert any(f > la._DETECT_SPAN for f in fractions)


@pytest.mark.asyncio
async def test_analyze_book_progress_callback_errors_are_swallowed(monkeypatch):
    """A throwing progress_cb must never abort the analysis."""
    canned = {"speakers": []}

    async def fake_generate(prompt, schema, **kwargs):
        return schema.model_validate(canned)

    monkeypatch.setattr(la, "generate_structured", fake_generate)

    def boom(_frac, _msg):
        raise RuntimeError("ui exploded")

    result = await la.analyze_book(["Chapter."], model_size="1.7B", progress_cb=boom)
    assert hasattr(result, "chapters")


@pytest.mark.asyncio
async def test_enrich_profiles_degrades_gracefully(monkeypatch):
    """enrich_profiles must not crash even if LLM fails."""

    async def boom(*a, **k):
        raise StructuredOutputError("no profiles today")

    monkeypatch.setattr(la, "generate_structured", boom)
    roster = [{"name": "Alice", "dialogue_count": 1, "confidence": 0.9}]
    result = await la.enrich_profiles(roster, samples={"Alice": ["sample"]}, model_size="1.7B")
    assert len(result) == 1
    # Fields may be None but should be present
    assert "name" in result[0]


def test_budget_by_size_has_required_keys():
    assert "0.6B" in la.BUDGET_BY_SIZE
    assert "1.7B" in la.BUDGET_BY_SIZE
    assert "4B" in la.BUDGET_BY_SIZE
    assert la.BUDGET_BY_SIZE["1.7B"] >= la.BUDGET_BY_SIZE["0.6B"]
    assert la.BUDGET_BY_SIZE["4B"] >= la.BUDGET_BY_SIZE["1.7B"]


def test_chinese_unbroken_chunks_have_hard_bound_and_preserve_text():
    text = "林默" * 5001
    chunks = la.chunk_text(text, max_tokens=37)
    assert all(0 < len(chunk) <= 37 * 4 for chunk in chunks)
    assert "".join(chunks) == text


def test_chinese_quotes_and_unquoted_identified_speech():
    text = "“快走，”林默说，“天黑了。”"
    segments = la.split_into_segments(text, speaker="林默")
    assert [s["type"] for s in segments] == ["dialogue", "narration", "dialogue"]
    assert "".join(s["text"] for s in segments) == text
    assert la.split_into_segments("快走。", speaker="林默")[0]["type"] == "dialogue"


@pytest.mark.asyncio
async def test_empty_structured_output_preserves_source_and_flags(monkeypatch):
    async def empty(_prompt, schema, **kwargs):
        return schema.model_validate({"speakers": []})

    monkeypatch.setattr(la, "generate_structured", empty)
    text = "林默说：“快跑！”"
    result = await la.analyze_chapter(text)
    assert result.flagged
    assert "".join(s["text"] for s in result.segments) == text
    assert result.characters[0]["name"] == "林默"
    assert any(s["type"] == "dialogue" for s in result.segments)
    assert all(s["analysis_metadata"]["speaker_verified"] and not s["flagged"] for s in result.segments if s["type"] == "dialogue")
    assert not any(s["flagged"] for s in result.segments if s["analysis_metadata"]["method"] == "rule")


@pytest.mark.asyncio
async def test_numbered_annotation_keeps_narration_and_original_punctuation(monkeypatch):
    async def numbered(_prompt, schema, **kwargs):
        return schema.model_validate({"speakers": [{"segment_id": 1, "speaker": "林默", "type": "dialogue"}]})

    monkeypatch.setattr(la, "generate_structured", numbered)
    text = "林默回过头：“等一等。”风还在吹。"
    result = await la.analyze_chapter(text)
    assert "".join(s["text"] for s in result.segments) == text
    assert next(s for s in result.segments if s["type"] == "dialogue")["speaker"] == "林默"
    assert not result.flagged


@pytest.mark.asyncio
async def test_unquoted_dialogue_is_aligned_without_losing_remainder(monkeypatch):
    async def detected(_prompt, schema, **kwargs):
        return schema.model_validate({"speakers": [{"text": "明天出发。", "speaker": "林默", "type": "dialogue"}]})

    monkeypatch.setattr(la, "generate_structured", detected)
    text = "明天出发。他把信收好。"
    result = await la.analyze_chapter(text, known_characters=[{"name": "林默", "aliases": []}])
    assert [s["type"] for s in result.segments] == ["dialogue", "narration"]
    assert "".join(s["text"] for s in result.segments) == text
    assert result.characters[0]["name"] == "林默"


@pytest.mark.asyncio
async def test_rewritten_model_text_never_replaces_novel(monkeypatch):
    async def hallucinated(_prompt, schema, **kwargs):
        return schema.model_validate({"speakers": [{"text": "这不是原文。", "speaker": "张三", "type": "dialogue"}]})

    monkeypatch.setattr(la, "generate_structured", hallucinated)
    result = await la.analyze_chapter("不要开门。他把信收好。")
    assert result.flagged
    assert [s["text"] for s in result.segments] == ["不要开门。他把信收好。"]
    assert result.characters == []


@pytest.mark.asyncio
async def test_whole_book_aliases_and_explicit_age_stages_are_consistent(monkeypatch):
    calls = []
    responses = [
        {
            "segment_id": 1,
            "speaker": "林默",
            "type": "dialogue",
            "canonical_name": "林默",
            "aliases": ["阿默"],
            "name_evidence": "林默（小名阿默）",
            "voice_stage": "少年",
            "voice_stage_evidence": "十五岁",
        },
        {
            "segment_id": 1,
            "speaker": "阿默",
            "type": "dialogue",
            "voice_stage": "中年",
            "voice_stage_evidence": "四十五岁",
        },
        {"segment_id": 1, "speaker": "林默", "type": "dialogue"},
    ]

    async def detected(prompt, schema, **kwargs):
        if schema is la.ProfileSchema:
            return schema(gender="male", age_estimate=None, vocal_description="清澈沉静的男声")
        calls.append(prompt)
        return schema.model_validate({"speakers": [responses[len(calls) - 1]]})

    monkeypatch.setattr(la, "generate_structured", detected)
    result = await la.analyze_book(
        [
            "林默（小名阿默），十五岁。他说：“我会回来。”",
            "三十年后，四十五岁的阿默说：“我回来了。”",
            "他接着说：“别怕。”",
        ]
    )
    assert len(result.characters) == 1
    assert result.characters[0]["name"] == "林默"
    assert "阿默" in result.characters[0]["aliases"]
    dialogue = [s for c in result.chapters for s in c.segments if s["type"] == "dialogue"]
    assert {s["speaker"] for s in dialogue} == {"林默"}
    assert [s["voice_stage"] for s in dialogue] == ["少年", "中年", "中年"]
    assert dialogue[-1]["analysis_metadata"]["voice_stage_inherited"]
    assert "阿默" in calls[1]  # the next chapter receives the global roster


@pytest.mark.parametrize(("stage", "evidence", "expected"), [
    ("老年", None, "老年"),
    ("少年", "23岁", "少年"),
    ("青年", "毕业后开始工作", "青年"),
    ("中年", "人物已经历多年风雨", "中年"),
    ("儿童", None, "少年"),
    ("成年", None, "青年"),
    ("愤怒", "他低声说", None),
])
def test_age_stage_uses_broad_context_without_exact_age_proof(stage, evidence, expected):
    assert la._valid_stage(stage, evidence, "他23岁，他低声说。") == expected


@pytest.mark.asyncio
async def test_contextual_life_stages_continue_across_chapters_without_numeric_ages(monkeypatch):
    import json

    stages = iter(["少年", "青年", "中年", "老年", None])

    async def annotate(prompt, schema, **kwargs):
        if schema is la.ProfileSchema:
            return schema(age_estimate="young adult", vocal_description="清晰温和的声音")
        stage = next(stages)
        spans = json.loads(prompt.split("原文片段：", 1)[1])
        return schema.model_validate({"speakers": [
            {"segment_id": span["segment_id"], "speaker": "林舟" if span["type"] == "dialogue" else "narrator",
             "type": span["type"], "voice_stage": stage}
            for span in spans
        ]})

    monkeypatch.setattr(la, "generate_structured", annotate)
    source = [
        "少年林舟站在校门前。林舟说：“我要出发。”",
        "林舟毕业了，开始了第一份工作。林舟说：“新的生活开始了。”",
        "多年以后，林舟走过了半生。林舟说：“经历让我变得沉稳。”",
        "岁月流逝，林舟坐在院子里等孙子回家。林舟说：“慢些走。”",
        "林舟接着说：“茶还热着。”",
    ]
    result = await la.analyze_book(source)
    dialogue = [segment for chapter in result.chapters for segment in chapter.segments if segment["type"] == "dialogue"]
    assert [segment["voice_stage"] for segment in dialogue] == ["少年", "青年", "中年", "老年", "老年"]
    assert {segment["speaker"] for segment in dialogue} == {"林舟"}
    assert dialogue[-1]["analysis_metadata"]["voice_stage_inherited"]
    assert result.characters[0]["age_estimate"] == "青年"
    assert "".join(s["text"] for c in result.chapters for s in c.segments) == "".join(source)


@pytest.mark.asyncio
async def test_explicit_dialogue_tags_override_wrong_llm_and_chinese_ages(monkeypatch):
    """A small model's wrong type/speaker must not override explicit source tags."""
    import json

    chapters = [
        "第一章 雨夜的约定\n十七岁的林舟推开旧书店的门。\n林舟轻声说：“青禾，我找到那封信了。”\n十八岁的青禾抬起头，认真回答：“先别拆开，等爷爷回来一起看。”\n林舟点头说：“好，我会等。”",
        "第二章 十年后的灯光\n十年过去，二十七岁的林舟再次来到书店。柜台后的青禾已经二十八岁，声音依然温柔。\n林舟说：“我答应过你的事，今天终于做到了。”\n青禾微笑着回答：“我一直记得。进来吧，茶还是热的。”\n林舟望着窗边的旧灯，轻声说：“这次，我不走了。”",
    ]

    async def wrong_model(prompt, schema, **kwargs):
        if schema is la.ProfileSchema:
            return schema(vocal_description="清晰自然的声音")
        spans = json.loads(prompt.split("原文片段：", 1)[1])
        return schema.model_validate(
            {"speakers": [{"segment_id": span["segment_id"], "type": "dialogue", "speaker": "林舟"} for span in spans]}
        )

    monkeypatch.setattr(la, "generate_structured", wrong_model)
    result = await la.analyze_book(chapters)
    assert {c["name"] for c in result.characters} == {"林舟", "青禾"}
    for text, chapter in zip(chapters, result.chapters, strict=True):
        assert "".join(s["text"] for s in chapter.segments).replace("\n", "") == text.replace("\n", "")
        assert [s["speaker"] for s in chapter.segments if s["type"] == "dialogue"] == ["林舟", "青禾", "林舟"]
        assert all(s["type"] == "narration" for s in chapter.segments if s["text"].endswith("："))
    assert [s["voice_stage"] for s in result.chapters[0].segments if s["type"] == "dialogue"] == [
        "少年",
        "少年",
        "少年",
    ]
    assert [s["voice_stage"] for s in result.chapters[1].segments if s["type"] == "dialogue"] == [
        "青年",
        "青年",
        "青年",
    ]


@pytest.mark.asyncio
async def test_resume_checkpoint_skips_completed_chapter_llm(monkeypatch):
    saved = la.ChapterAnalysis(
        [
            {
                "type": "narration",
                "text": "第一章的原文。",
                "order": 0,
                "speaker": None,
                "analysis_metadata": {},
                "voice_stage": None,
            }
        ],
        [],
    )

    async def never_called(*args, **kwargs):
        raise AssertionError("A completed unchanged checkpoint should not invoke the model")

    monkeypatch.setattr(la, "generate_structured", never_called)
    writes = []
    result = await la.analyze_book(["第一章的原文。"], checkpoint_load=lambda index, text: saved,
                                   checkpoint_cb=lambda *args: writes.append(args))
    assert result.chapters[0] is saved
    assert result.chapters[0].segments[0]["text"] == "第一章的原文。"
    assert writes == []
