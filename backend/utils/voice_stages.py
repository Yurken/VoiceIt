"""Broad voice life stages shared by analysis, editing, and synthesis."""

import re

VOICE_STAGES = ("少年", "青年", "中年", "老年")
_ALIASES = {
    "儿童": "少年", "童年": "少年", "幼年": "少年", "孩童": "少年", "少女": "少年",
    "年少": "少年", "青少年": "少年", "child": "少年", "childhood": "少年",
    "teen": "少年", "teenager": "少年", "teenage": "少年", "adolescent": "少年", "youth": "少年",
    "成年": "青年", "成人": "青年", "年轻": "青年", "青壮年": "青年",
    "adult": "青年", "young": "青年", "young adult": "青年", "young-adult": "青年",
    "middle": "中年", "middle-aged": "中年", "middle aged": "中年", "middle_aged": "中年",
    "elderly": "老年", "old": "老年", "senior": "老年", "older adult": "老年", "年迈": "老年", "老人": "老年",
}


def normalize_voice_stage(value: str | None) -> str | None:
    """Keep four labels; numeric legacy ages are only an approximate fallback.

    Contextual labels take precedence over numbers, so '二十岁的少年' stays
    少年. The approximate age buckets never invalidate a model's stage label.
    """
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().lower()
    if text in VOICE_STAGES:
        return text
    if text in _ALIASES:
        return _ALIASES[text]
    for pattern, stage in (
        (r"老年|年迈|年老|苍老|白发苍苍|垂暮|耄耋|古稀|花甲|\b(?:elderly|senior)\b", "老年"),
        (r"中年|不惑|知天命|\bmiddle[ -]?aged\b", "中年"),
        (r"少年|少女|年少|儿童|童年|幼年|孩童|\b(?:child|childhood|teen|teenager|teenage|adolescent)\b", "少年"),
        (r"青年|年轻|成年|成人|青壮年|\b(?:young|adult)\b", "青年"),
    ):
        if re.search(pattern, text):
            return stage
    age = re.search(
        r"(\d{1,3}|[一二三四五六七八九十百两零〇]{1,4})\s*(?:多|余)?岁|(\d{1,3})\s*(?:years?\s*old|s)\b",
        text,
    )
    if not age:
        age = re.fullmatch(r"(\d{1,3})(?:\s*[-~\u2013至到]\s*\d{1,3})?", text)
    if not age:
        return None
    raw = next(group for group in age.groups() if group is not None)
    if raw.isdigit():
        number = int(raw)
    else:
        digits = dict(zip("零〇一二三四五六七八九两", [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 2], strict=True))
        number, digit = 0, 0
        for char in raw:
            if char in {"十", "百"}:
                number += (digit or 1) * {"十": 10, "百": 100}[char]
                digit = 0
            else:
                digit = digits[char]
        number += digit
    return "少年" if number < 20 else "青年" if number < 40 else "中年" if number < 60 else "老年"
