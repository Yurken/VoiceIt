# ruff: noqa: RUF001
"""Cast labels reflect evidence and prominence rather than model optimism."""

import pytest

from backend.services.character_scoring import assign_character_roles, score_character


def _line(*, method="llm_source_annotation", evidence=None, flagged=False, speaker="林舟", kind="dialogue", verified=False):
    return {
        "type": kind,
        "speaker": speaker,
        "flagged": flagged,
        "analysis_metadata": {
            "method": method,
            "speaker_evidence": evidence,
            "source_excerpt": "林舟低声说：明日再见。",
            "speaker_verified": verified,
        },
    }


def test_unflagged_model_guess_is_not_high_confidence():
    assert 0.5 <= score_character([_line()] * 100) < 0.8


def test_rule_needs_verifiable_source_evidence_for_high_confidence():
    assert score_character([_line(method="rule", evidence="林舟低声说", verified=True)]) >= 0.8
    assert score_character([_line(method="rule", evidence="林舟低声说")]) < 0.8
    assert score_character([_line(method="rule")]) < 0.8
    assert score_character([_line(method="rule", evidence="青禾大声说", verified=True)]) < 0.8


@pytest.mark.parametrize("method", ["source_tag", "repair_source_tag"])
def test_verified_source_attribution_methods_support_high_confidence(method):
    assert score_character([_line(method=method, evidence="林舟低声说", verified=True)]) >= 0.8


def test_excerpt_without_identity_support_does_not_count_as_speaker_evidence():
    assert score_character([_line(evidence="明日再见")]) == score_character([_line()])


def test_verified_alias_can_support_identity_without_canonical_spelling():
    line = _line(method="source_tag", evidence="阿舟低声说", verified=True)
    line["analysis_metadata"]["source_excerpt"] = "阿舟低声说：明日再见。"
    assert score_character([line]) >= 0.8


def test_model_textual_evidence_remains_below_high_confidence():
    assert score_character([_line(evidence="林舟低声说")]) < 0.8


@pytest.mark.parametrize(
    "line",
    [
        _line(flagged=True, method="rule", evidence="林舟低声说", verified=True),
        _line(speaker="未识别角色"),
        _line(method="source_fallback"),
        _line(method="empty_result_fallback"),
    ],
)
def test_failed_or_unknown_attribution_is_low_confidence(line):
    assert score_character([line]) < 0.5


def test_one_supported_line_does_not_hide_many_failed_lines():
    lines = [_line(method="rule", evidence="林舟低声说", verified=True), *[_line(flagged=True) for _ in range(9)]]
    assert score_character(lines) < 0.5


def test_narration_does_not_inflate_speaker_confidence():
    assert score_character([_line(kind="narration"), _line(flagged=True)]) == 0.25
    assert score_character([]) == 0.0


def test_two_line_character_is_minor_in_a_long_book():
    roster = [{"name": "林舟", "dialogue_count": 1000}, {"name": "青禾", "dialogue_count": 200}]
    roster.extend({"name": f"访客{index}", "dialogue_count": 2} for index in range(200))
    result = assign_character_roles(roster)
    assert [character["name"] for character in result if character["role"] == "major"] == ["林舟", "青禾"]
    assert all("role" not in character for character in roster)


def test_two_line_characters_can_be_major_in_a_short_story():
    result = assign_character_roles([
        {"name": "林舟", "dialogue_count": 3},
        {"name": "青禾", "dialogue_count": 2},
        {"name": "访客", "dialogue_count": 1},
    ])
    assert [character["role"] for character in result] == ["major", "major", "minor"]


def test_prominence_uses_both_book_share_and_main_character_share():
    result = assign_character_roles([
        {"name": "主角", "dialogue_count": 1000},
        {"name": "配角", "dialogue_count": 40},
    ])
    # 40 is over 2% of the book but below 5% of the protagonist's dialogue.
    assert result[1]["role"] == "minor"


def test_unknown_and_narrator_do_not_affect_prominence():
    result = assign_character_roles([
        {"name": "旁白", "is_narrator": True, "dialogue_count": 1000},
        {"name": "未识别角色", "dialogue_count": 1000},
        {"name": "林舟", "dialogue_count": 3},
    ])
    assert [character["role"] for character in result] == ["minor", "minor", "major"]


def test_malformed_or_absent_counts_do_not_become_major():
    result = assign_character_roles([
        {"name": "林舟", "dialogue_count": 3},
        {"name": "访客", "dialogue_count": -2},
        {"name": "陌生人", "dialogue_count": "invalid"},
        {"name": "甲"},
    ])
    assert [character["role"] for character in result] == ["major", "minor", "minor", "minor"]
