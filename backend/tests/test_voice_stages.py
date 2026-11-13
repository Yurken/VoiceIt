"""Voice life stages stay broad and keep old book data usable."""

import pytest

from backend.models import SegmentUpdate
from backend.utils.voice_stages import normalize_voice_stage


@pytest.mark.parametrize("value,expected", [
    ("少年", "少年"), ("青年", "青年"), ("中年", "中年"), ("老年", "老年"),
    ("儿童", "少年"), ("成年", "青年"), ("young", "青年"), ("adult", "青年"),
    ("十七岁的林舟", "少年"), ("27", "青年"), ("40s", "中年"), ("65-80", "老年"),
    ("二十岁的少年", "少年"), ("毕业后成为青年", "青年"),
    ("低沉一点", None), ("愤怒", None), (None, None), ("", None),
])
def test_normalizes_four_stages_and_legacy_values(value, expected):
    assert normalize_voice_stage(value) == expected


def test_segment_edit_accepts_broad_stages_and_normalizes_legacy_names():
    assert SegmentUpdate(voice_stage="青年").voice_stage == "青年"
    assert SegmentUpdate(voice_stage="成年").voice_stage == "青年"
    assert SegmentUpdate(voice_stage="儿童").voice_stage == "少年"
    assert SegmentUpdate(voice_stage=None).voice_stage is None


def test_generation_instruction_keeps_identity_and_uses_only_broad_stage():
    from types import SimpleNamespace

    from backend.services.book_generation import compose_instruct

    instruction = compose_instruct(SimpleNamespace(voice_stage="27岁"))
    assert "青年" in instruction
    assert "27" not in instruction
    assert "基础音色" in instruction
