# ruff: noqa: RUF001
"""Identity parsing must not convert a gesture, listener or quoted term to a person."""

import pytest

from backend.services.character_identity import (
    normalize_identity_candidate,
    quote_is_speech,
    quoted_ranges,
    source_speaker,
)


@pytest.mark.parametrize(("raw", "expected"), [
    ("陆鸣赶紧", "陆鸣"), ("陆鸣连忙", "陆鸣"), ("陆瑾则是", "陆瑾"),
    ("陆瑾沉吟", "陆瑾"), ("对着陆鸣", None), ("关心的", None),
    ("笑眯眯的", None), ("轻笑", None), ("s", None), ("dialogue", None),
    ("未知角色", None), ("好，既然你有这个信心，那我就信你", None),
    ("雷火馆", None), ("曜日金炎幡", None), ("生灵之焱", None),
    ("李贯一", "李贯一"), ("陆鸣", "陆鸣"), ("周茯苓", "周茯苓"),
    ("欧阳修", "欧阳修"), ("Alice", "Alice"), ("John Smith", "John Smith"),
    ("向晚", "向晚"),
    ("徐管事", "徐管事"), ("陆馆主", "陆馆主"), ("爷爷", "爷爷"),
])
def test_identity_shape_and_grammatical_suffixes(raw, expected):
    assert normalize_identity_candidate(raw, ["陆鸣", "陆瑾", "李贯"]) == expected


def test_vetted_fictional_names_survive_but_known_gesture_does_not():
    known = ["青禾", "林舟", "笑眯眯的", "陆鸣"]
    assert normalize_identity_candidate("青禾", known) == "青禾"
    assert normalize_identity_candidate("林舟", known) == "林舟"
    assert normalize_identity_candidate("笑眯眯的", known) is None
    assert normalize_identity_candidate("向陆鸣", known) is None
    assert normalize_identity_candidate("John Smith", ["John", "John Smith"]) == "John Smith"


@pytest.mark.parametrize("raw", [
    "突然", "羡慕", "提醒", "缓缓", "直接", "则是", "有一", "最终他", "惊讶", "失声",
    "嘲讽", "主动", "据他所", "而后冷", "抱拳", "自得", "当下", "再度", "正色",
    "认真的回", "解释", "咆哮", "当即", "嘟囔", "欣慰", "而是询", "接着", "只是",
    "急忙", "补充", "轻轻", "为了那", "毕竟这", "阴魂不散",
    "平静", "淡声", "诚声", "柔声", "悄声", "感激", "抱怨", "咬牙", "为难", "介绍", "惊叹",
    "淡笑", "就骂", "揉着眼", "颔首欣慰", "有些感慨", "咬着嘴唇", "又有着一", "催发这",
    "而这", "多谢爷爷", "在心中暗", "有了这", "有些",
])
def test_model_roster_cannot_legitimize_an_action_or_grammatical_phrase(raw):
    assert normalize_identity_candidate(raw, [raw]) is None
    assert normalize_identity_candidate(raw) is None


@pytest.mark.parametrize(("base", "clipped", "tag"), [
    ("陆鸣", "陆鸣苦", "陆鸣苦笑道"), ("陆鸣", "陆鸣回", "陆鸣回头说道"),
    ("黎妍", "黎妍淡", "黎妍淡淡道"), ("阎芝芝", "阎芝芝娇", "阎芝芝娇声道"),
    ("周宣", "周宣含", "周宣含笑道"), ("周茯苓", "周茯苓调", "周茯苓调笑道"),
    ("钟泰岳", "钟泰岳大", "钟泰岳大声道"),
])
def test_a_truncated_gesture_is_rejected_but_full_source_tag_recovers_known_subject(base, clipped, tag):
    assert normalize_identity_candidate(clipped, [base, clipped]) is None
    result = _speaker(tag + "：“好。”", [base, clipped])
    assert result
    assert result[0] == base


@pytest.mark.parametrize(("text", "expected"), [
    ("陆鸣苦笑道：“再等等。”", "陆鸣"),
    ("高玉笑道：“好。”", "高玉"),
    ("陆鸣赶紧说道：“再等等。”", "陆鸣"),
    ("陆鸣回头说道：“再等等。”", "陆鸣"),
    ("黎妍淡淡道：“好。”", "黎妍"),
    ("阎芝芝娇声道：“好。”", "阎芝芝"),
    ("周宣含笑道：“好。”", "周宣"),
    ("周茯苓调笑道：“好。”", "周茯苓"),
    ("钟泰岳大声道：“好。”", "钟泰岳"),
])
def test_first_appearance_does_not_split_a_speaking_word_into_the_name(text, expected):
    result = _speaker(text)
    assert result
    assert result[0] == expected


@pytest.mark.parametrize("action", [
    "缓缓", "惊讶", "摇", "闻言", "转头", "抬头", "惊喜", "皱眉", "先是", "起身", "突然",
    "微微", "淡笑", "又是", "安慰", "咧嘴", "一怔", "有些", "笑声", "又看", "就点", "讨饶", "讪笑", "淡然",
])
def test_an_unparsed_predicate_never_creates_a_longer_identity_from_known_subject(action):
    subject = "陆鸣"
    invented = subject + action
    assert normalize_identity_candidate(invented, [subject]) in {subject, None}
    result = _speaker(invented + "说道：“再等等。”", [subject])
    assert result is None or result[0] == subject


def test_an_unfamiliar_four_character_action_phrase_is_not_a_new_person():
    assert _speaker("都吸收掉道：“好。”", ["陆鸣"]) is None


def _speaker(text, known=()):
    start = text.index("“")
    end = text.index("”", start) + 1
    return source_speaker(text, start, end, known)


@pytest.mark.parametrize(("text", "known", "expected"), [
    ("陆鸣赶紧说道：“娘，儿子心怀大志！”", ["陆鸣"], "陆鸣"),
    ("陆瑾沉吟道：“你有什么办法？”", ["陆瑾"], "陆瑾"),
    ("陆瑾则是道：“看来黎妍没少出力。”", ["陆瑾"], "陆瑾"),
    ("陆鸣沉吟着说道：“先等等。”", ["陆鸣"], "陆鸣"),
    ("陆瑾则是沉吟着说道：“先等等。”", ["陆瑾"], "陆瑾"),
    ("陆瑾对着陆鸣说道：“你先坐下。”", ["陆瑾", "陆鸣"], "陆瑾"),
    ("周宣对着陆鸣说道：“你先坐下。”", ["周宣", "陆鸣"], "周宣"),
    ("周宣向陆鸣笑着说道：“你先坐下。”", ["周宣", "陆鸣"], "周宣"),
    ("十八岁的青禾抬起头，认真回答：“先别拆开。”", ["青禾", "林舟"], "青禾"),
    ("中年的青禾缓缓地说道：“我一直记得。”", ["青禾", "林舟"], "青禾"),
    ("白发苍苍的陆瑾低声说道：“别急。”", ["陆瑾"], "陆瑾"),
    ("陆瑾抖落烟灰，对着陆鸣道：“你也别怪她。”", ["陆瑾", "陆鸣"], "陆瑾"),
    ("周宣笑了笑，对着陆鸣问道：“你对丹道有兴趣吗？”", ["周宣", "陆鸣"], "周宣"),
    ("高玉哑然，轻笑道：“自然和你没法比。”", ["高玉"], "高玉"),
    ("周茯苓轻笑一声，看向两女，道：“看来你们并不知道。”", ["周茯苓"], "周茯苓"),
    ("祝镰笑眯眯的道：“什么规矩不规矩的？”", ["祝镰"], "祝镰"),
    ("陆青炎转过头，目光看向神色微微有些僵硬的方浪，面庞上浮现出一抹温和的笑容，关心的问道：“没受伤吧？”", ["陆青炎", "方浪"], "陆青炎"),
    ("“什么条件？”陆鸣赶紧问道。", ["陆鸣"], "陆鸣"),
    ("“你还要什么？”陆瑾抬眼，问道。", ["陆瑾"], "陆瑾"),
    ("Alice said：“Wait.”", ["Alice"], "Alice"),
    ("“Wait.” Alice said.", ["Alice"], "Alice"),
    ("“Wait.” said Alice.", ["Alice"], "Alice"),
    ("徐管事说道：“诸位请坐。”", [], "徐管事"),
    ("李贯一说道：“好。”", ["李贯"], "李贯一"),
])
def test_source_tag_preserves_subject_instead_of_listener_or_gesture(text, known, expected):
    result = _speaker(text, known)
    assert result
    assert result[0] == expected
    assert result[1] in text


@pytest.mark.parametrize("text", [
    "对着陆鸣道：“请坐。”",
    "笑眯眯的道：“你好。”",
    "陆鸣没有说道：“好。”",
    "陆鸣回忆起周宣曾经说道：“好。”",
    "如果陆鸣说道：“好。”",
    "林舟望着青禾说：“走吧。”",
    "陆鸣拉着陆瑾，周宣抬手说道：“别急。”",
    "“先前的回答。”周宣说道：“下一句。”",
    "“Wait.” Alice said: “Next.”",
])
def test_ambiguous_or_unasserted_tags_are_left_to_semantic_analysis(text):
    assert _speaker(text, ["陆鸣", "陆瑾", "周宣", "林舟", "青禾"]) is None


@pytest.mark.parametrize(("text", "expected"), [
    ("他伸出手，对着石台上的“生灵之焱”抓去。", False),
    ("两人进入“黑雨区”外围。", False),
    ("他开始修炼“曜日金炎幡”。", False),
    ("人们称之为“神通羽衣”。", False),
    ("他以“火焰刀”为名。", False),
    ("“金瞳”的变化很明显。", False),
    ("陆鸣问道：“曜日金炎幡？”", True),
    ("陆鸣说道：“好”", True),
    ("“好”陆鸣说道。", True),
    ("“走！”", True),
    ("“嗯”", True),
    ("“再来”", True),
    ("“原来如此。”他恍然大悟。", True),
    ("他望向手中的“禁旗”，没有说话。", False),
    ("陆鸣胸前悬着“雨珠”，微微发亮。", False),
    ("他最想得到“青玉翅”，可惜一直没有。", False),
    ("众人刚经历“黄风灵禁”，仍然惊魂未定。", False),
    ("他用尽“圣金粉”，没有剩下。", False),
    ("他去了“大商皇域”，再没有回来。", False),
    ("陆鸣轻声说“好”", True),
    ("他问：“好”", True),
    ("陆鸣拿着禁旗，开口说道：“禁旗”", True),
    ("“焦炭”勉强的睁开眼，有嘶哑的声音传出来：“告诉我...”", False),
    ("“黑衣男子”点了点头，转过身去。", False),
    ("“李白”看向众人。", False),
    ("“够了！”他猛然站起身来。", True),
    ("陆鸣说道：“焦炭”", True),
])
def test_named_references_are_not_utterances_but_short_dialogue_survives(text, expected):
    start = text.index("“")
    end = text.index("”", start) + 1
    assert quote_is_speech(text, start, end) is expected


@pytest.mark.parametrize("text", ["以禁旗为主，直接出发。", "先走，等我回来。", "我叫陆鸣。", "这是神通骨"])
def test_unquoted_dialogue_is_not_retyped_by_the_quote_classifier(text):
    assert quote_is_speech(text, 0, len(text)) is True


@pytest.mark.parametrize("quote", [
    '“你这是上神通“金乌玄光”...你们是“大金乌馆”的人。”',
    '"This is “Golden Crow”, from the “Big Academy”."',
    '“这叫"金乌玄光"，是门神通。”',
    '「这叫『金乌玄光』，来自“大金乌馆”。」',
    '“外层「中层『内层名词』的名字」继续说话。”',
    '“这不是结束\\”仍在说话。”',
    '"Use \\"literal quotes\\" in this sentence."',
])
def test_nested_and_escaped_quotes_form_one_outer_utterance(quote):
    text = "陆瑾道：" + quote + "尾声。"
    assert quoted_ranges(text) == [(len("陆瑾道："), len("陆瑾道：") + len(quote))]


def test_multiple_outer_quotations_keep_original_positions_and_all_source_text():
    text = '前文。“外层“名词”后文。”中间叙述。「另一个『名词』。」尾声。'
    ranges = quoted_ranges(text)
    assert [text[start:end] for start, end in ranges] == ['“外层“名词”后文。”', '「另一个『名词』。」']
    cursor = 0
    pieces = []
    for start, end in ranges:
        pieces.extend([text[cursor:start], text[start:end]])
        cursor = end
    pieces.append(text[cursor:])
    assert "".join(pieces) == text


def test_an_unmatched_outer_quote_does_not_hide_later_complete_quotation():
    text = '“开头没有闭合\n\n陆鸣说道：“这句完整。”结尾。'
    ranges = quoted_ranges(text)
    assert [text[start:end] for start, end in ranges] == ['“这句完整。”']


def test_unmatched_quote_text_is_not_returned_as_a_complete_utterance():
    assert quoted_ranges('前文“没有闭合的台词') == []
    text = '孤立的”后面是“完整台词。”'
    assert [text[start:end] for start, end in quoted_ranges(text)] == ['“完整台词。”']
