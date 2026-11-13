# ruff: noqa: RUF001, RUF003
"""Conservative identity checks and source-backed Chinese dialogue tags.

These helpers do not infer conversational turns.  A plausible name is only a
candidate: callers still have to ground a new identity in the book's source.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable
from dataclasses import dataclass, field
from functools import lru_cache

_CJK = r"\u3400-\u9fff"
_SURNAMES = set(
    "赵钱孙李周吴郑王冯陈褚卫蒋沈韩杨朱秦尤许何吕施张孔曹严华金魏陶姜"
    "戚谢邹喻柏水窦章云苏潘葛奚范彭郎鲁韦昌马苗凤花方俞任袁柳鲍史唐"
    "费廉岑薛雷贺倪汤滕殷罗毕郝邬安常乐于时傅皮卞齐康伍余元卜顾孟平"
    "黄和穆萧尹姚邵湛汪祁毛禹狄米贝明臧计伏成戴宋茅庞熊纪舒屈项祝"
    "董梁杜阮蓝闵席季麻强贾路娄危江童颜郭梅盛林刁钟徐邱骆高夏蔡田樊"
    "胡凌霍虞万支柯管卢莫经房裘缪干解应宗丁宣邓郁单杭洪包左石崔吉龚"
    "程嵇邢滑裴陆荣翁荀羊甄曲封芮储靳汲邴糜松井段富巫乌焦巴弓牧隗"
    "山谷车侯宓蓬全郗班仰秋仲伊宫宁仇栾暴甘钭厉戎祖武符刘景詹束龙叶"
    "幸司韶黎乔苍双闻莘党翟谭贡劳逄姬申扶堵冉宰郦雍郤璩桑桂濮牛寿"
    "通边扈燕冀浦尚农温别庄晏柴瞿阎充慕连茹习宦艾鱼容向古易慎戈廖庾"
    "终暨居衡步都耿满弘匡国文寇广禄阙东欧利师巩聂晁勾敖融冷訾辛阚"
    "简饶曾关查后荆游竺权逯盖益桓公楚南宫"
)
_COMPOUND_SURNAMES = (
    "欧阳", "司马", "上官", "诸葛", "夏侯", "东方", "皇甫", "尉迟", "公孙",
    "慕容", "司徒", "司空", "令狐", "端木", "独孤", "南宫", "宇文", "长孙",
    "闻人", "澹台", "轩辕", "百里", "东郭", "西门", "南门", "钟离",
)
_NON_IDENTITIES = {
    "s", "i", "dialogue", "narration", "narrator", "speaker", "unknown", "none", "null",
    "旁白", "叙述者", "解说", "未识别角色", "未知角色", "未知未识别角色", "未知", "不明",
    "他们", "她们", "它们", "自己", "有人", "众人", "那人", "对方", "大家", "两人", "两女",
    "轻笑", "微笑", "轻声", "低声", "大声", "小声", "关心", "关心的", "笑眯眯的",
    "笑吟吟的", "冷冷的", "认真", "点头", "摇头", "回答", "不禁", "忽然", "然后", "于是",
    "突然", "羡慕", "提醒", "缓缓", "直接", "则是", "有一", "最终他", "惊讶", "失声",
    "嘲讽", "主动", "据他所", "而后冷", "抱拳", "自得", "当下", "再度", "正色",
    "认真的回", "解释", "咆哮", "当即", "嘟囔", "欣慰", "而是询", "接着", "只是",
    "急忙", "补充", "轻轻", "为了那", "毕竟这", "阴魂不散", "震惊", "愕然", "讶然",
    "哑然", "默然", "沉默", "不满", "不屑", "担忧", "好奇", "关切", "疑惑", "恍然",
    "不由得", "不由", "连忙", "赶紧", "赶忙", "随即", "旋即", "随后", "最终", "终于",
    "冷笑", "苦笑", "含笑", "调笑", "娇笑", "娇声", "温和", "冷声", "朗声", "嘶哑",
    "笑道", "问道", "答道", "说道", "出声", "开口", "询问", "质问", "反问", "冷哼",
    "哈哈", "呵呵", "嘿嘿", "嗯嗯", "如此", "这个", "那个", "这般", "那般", "见状",
    "平静", "淡声", "诚声", "柔声", "悄声", "感激", "抱怨", "咬牙", "为难", "介绍", "惊叹",
    "淡笑", "就骂", "揉着眼", "颔首欣慰", "有些感慨", "咬着嘴唇", "又有着一", "催发这",
    "而这", "多谢爷爷", "在心中暗", "有了这", "有些",
}
_MODIFIER_WORDS = (
    "轻声", "低声", "大声", "小声", "微笑着", "笑着", "点了点头", "摇了摇头",
    "点头", "摇头", "冷冷地", "冷冷的", "冷冷", "缓缓地", "激动地", "严肃地",
    "哽咽着", "平静地", "认真地", "认真", "坚定地", "温柔地", "接着", "继续", "随后",
    "忽然", "又", "却", "叹了口气", "抬起头", "转过头", "回头", "苦笑着", "喃喃地",
    "赶紧", "连忙", "急忙", "连连", "赶忙", "则是", "也是", "便是", "就是", "只是", "沉吟着",
    "沉吟", "沉声", "哑然", "轻笑", "一笑", "笑了笑", "笑眯眯的", "笑吟吟的", "关心的", "抬眼",
    "没好气的", "没好气地", "尴尬的", "尴尬地", "戏谑的", "戏谑地", "好奇的", "好奇地",
    "平淡", "淡淡", "坦然", "讶然", "愕然", "愤然", "不由", "不禁", "耐心的", "耐心地",
    "苦笑", "含笑", "调笑", "娇笑", "娇声", "温和的", "温和地", "温和", "冷声", "朗声",
    "嘶哑的", "嘶哑着", "嘶哑", "嘲讽的", "嘲讽地", "惊讶的", "惊讶地", "失声",
    "轻轻", "正色", "解释", "补充", "咆哮", "嘟囔", "提醒", "羡慕的", "羡慕地", "欣慰的",
    "笑吟吟", "笑眯眯", "询问", "问询", "大笑", "冷笑", "不屑的", "疑惑的", "恍然的",
)
_MODIFIER = "(?:" + "|".join(re.escape(x) for x in sorted(_MODIFIER_WORDS, key=len, reverse=True)) + ")"
_SPEECH = r"(?:说道|问道|答道|喊道|叫道|笑道|回答|回应|反问|问询|低语|喃喃|说|问|道|喊|答)"
_UNSAFE = re.compile(
    r"没有|没说|未曾|并未|不是|假如|如果|仿佛|好像|想起|回忆|曾经|转述|据说|听说|说过|想道|心想"
    r"|\b(?:remembered|recalled|pretended|did\s+not|never|if)\b", re.I
)
_OBJECT_MARKER = re.compile(r"(?:对着|向着|朝着|对|向|朝|看向|望向|望着|看着|看了|扫了|看见|瞧着)")
_ACTION = re.compile(
    r"^(?:抖落烟灰|笑了笑|微微一笑|轻笑一声|沉吟片刻|转过头|抬起头|抬眼|点了点头|摇了摇头|"
    r"点头|摇头|回头|看向两女|看向众人|望向众人|扫视众人|皱眉|叹了口气)$"
)
_TITLES = {"爷爷", "奶奶", "父亲", "母亲", "爹爹", "娘亲", "爹", "娘", "大哥", "二哥", "大姐", "二姐",
           "大叔", "二叔", "三叔", "师父", "师傅", "师兄", "师姐", "师弟", "师妹", "先生", "殿下", "陛下"}
_TITLE_SUFFIX = r"(?:馆主|管事|长老|前辈|老师|师父|师傅|师兄|师姐|师弟|师妹|先生|夫人|姑娘|公子|小姐|大人|总管|掌柜|将军|大侠)"
_CLIPPED_ACTION = frozenset("苦回知淡娇含调大皱沉点轻朗温笑冷抬看望")
_ENGLISH_NAME = r"[A-Z][A-Za-z'’-]*(?:[ -][A-Z][A-Za-z'’-]*){0,4}"
_ENGLISH_SPEECH = r"(?:said|asked|replied|shouted|whispered)"
_PERSON_AGE_PREFIX = re.compile(
    r"^(?:(?:\d{1,3}|[一二三四五六七八九十百两零〇]{1,4})\s*岁的?|"
    r"(?:少年|少女|青年|年轻|中年|老年|年迈|年少|幼年|童年|年老|白发苍苍|垂暮)(?:时期|时代|时)?的?)"
)
_TAG_WORDS = tuple(set(_MODIFIER_WORDS) | set(re.findall(f"[{_CJK}]+", _SPEECH)))
_TAG_WORD_BOUNDARIES = re.compile("(?=(" + "|".join(re.escape(word) for word in sorted(_TAG_WORDS, key=len, reverse=True)) + "))")
_MAX_TAG_WORD = max(map(len, _TAG_WORDS))
_QUOTE_PAIRS = {'“': '”', '"': '"', '「': '」', '『': '』', '‘': '’'}
_QUOTED_SUBJECT_PREDICATE = re.compile(
    r"^(?:(?:勉强|缓缓|慢慢|微微|轻轻|艰难|无力|突然|随即|猛然|轻巧|沉默)(?:地|的)?){0,3}"
    r"(?:睁开|闭上|点头|点了点头|摇头|摇了摇头|看向|望着|望向|抬头|抬起头|转身|起身|站起|"
    r"伸出|走向|走到|冲向|挥手|拿起|放下|皱眉|咧嘴|眨眼|叹气|伸手|侧过身|倒下|躺在|"
    r"捂住|摸了摸|抓住|转过头|笑了笑|瞪着|盯着|晃动|摇晃|说道|问道|回答)"
)


@dataclass
class _QuotedNode:
    start: int
    closer: str
    end: int | None = None
    children: list[_QuotedNode] = field(default_factory=list)


def quoted_ranges(text: str) -> list[tuple[int, int]]:
    """Return complete outer quotations, including their delimiters.

    Distinct open/close marks support nesting with the same or mixed pairs.
    Straight double quotes toggle inside their current nesting level; escaped
    quotes are literal.  An unmatched opener contributes no range.  Complete
    children under an unmatched opener are salvaged, so a missing delimiter
    cannot hide every later balanced quotation.  All unmatched text stays with
    the caller, which must retain it as narration rather than dropping it.
    """
    roots: list[_QuotedNode] = []
    stack: list[_QuotedNode] = []
    backslashes = 0
    for index, char in enumerate(text):
        escaped = bool(backslashes % 2)
        backslashes = backslashes + 1 if char == "\\" else 0
        if escaped:
            continue
        if stack and char == stack[-1].closer:
            stack.pop().end = index + 1
        elif char in _QUOTE_PAIRS:
            node = _QuotedNode(index, _QUOTE_PAIRS[char])
            (stack[-1].children if stack else roots).append(node)
            stack.append(node)
    result: list[tuple[int, int]] = []
    pending = list(reversed(roots))
    while pending:
        node = pending.pop()
        if node.end is not None:
            result.append((node.start, node.end))
        else:
            pending.extend(reversed(node.children))
    return result


def _clean(value: str | None) -> str:
    return unicodedata.normalize("NFKC", value or "").strip().strip('“”"「」『』‘’')


@lru_cache(maxsize=256)
def _clean_known(names: tuple[str, ...]) -> frozenset[str]:
    """A book reuses the same roster for thousands of source annotations."""
    return frozenset(_clean(name) for name in names)


def _has_identity_shape(name: str, *, known: bool = False) -> bool:
    if not name or name.casefold() in _NON_IDENTITIES:
        return False
    if re.match(r"^(?:对着|向着|朝着|对|看向|望着|关心|笑眯眯|笑吟吟)", name):
        return False
    if re.search(r"[，,。.!！?？:：;；\n\d_{}\[\]]", name):
        return False
    if re.match(r"^(?:而后|而是|据.{0,3}所|为了|毕竟|最终|当下|当即|有一|这|那|他|她|它|再度)", name):
        return False
    if re.fullmatch(r"[A-Z][A-Za-z'’-]*(?:[ -][A-Z][A-Za-z'’-]*){0,4}", name):
        return True
    if not re.fullmatch(f"[{_CJK}·]{{1,16}}", name):
        return False
    if name in _TITLES or re.fullmatch(f"[{_CJK}]{{1,3}}" + _TITLE_SUFFIX, name):
        return True
    # A known name may use a rare surname or be a transliteration.  Action
    # phrases still fail before this branch, even if an earlier model stored one.
    if re.search(r"(?:的|地|着|则是|也是|赶紧|连忙|沉吟|轻笑|说|问|道|喊|答|抬眼|馆|宗|派|商会|神通|火焰|之焱|之炎|幡|功法|法诀|真解)$", name):
        return False
    if name.endswith(_MODIFIER_WORDS):
        return False
    if known:
        return len(name) >= 2
    if re.fullmatch(f"[阿小老][{_CJK}]{{1,2}}", name):
        return True
    if name.startswith(_COMPOUND_SURNAMES):
        return 3 <= len(name) <= 5
    return 2 <= len(name) <= 4 and name[0] in _SURNAMES


def normalize_identity_candidate(name: str | None, known_names: Iterable[str] = ()) -> str | None:
    """Strip speaking modifiers, never arbitrary suffixes or a recipient prefix.

    ``李贯一`` is distinct from known ``李贯``.  A candidate without a source
    occurrence must still be rejected by the caller, even if this check accepts it.
    """
    return _normalize_identity_candidate(_clean(name), tuple(known_names))


@lru_cache(maxsize=8192)
def _normalize_identity_candidate(value: str, known_names: tuple[str, ...]) -> str | None:
    known = _clean_known(known_names)
    recipient = re.match(r"^(?:对着|向着|朝着|对|看向|望着)", value)
    recipient = recipient or (len(value) > 2 and value[:1] in {"向", "朝"} and value[1:] in known)
    if value.casefold() in _NON_IDENTITIES or recipient:
        return None
    # Only named grammatical modifiers are removable.  Longest-first avoids
    # stripping just `是` or a final character from a real personal name.
    original = value
    while value:
        match = re.search(_MODIFIER + r"$", value)
        if not match:
            break
        value = value[:match.start()].rstrip()
    if not value:
        return None
    # Once the book has an established subject, an unparsed phrase following
    # that name is not evidence of a second, longer person.  Leave ambiguous
    # expansions to semantic analysis; number continuations are unambiguous
    # personal-name spellings such as 李贯一, not speaking actions.
    if re.fullmatch(f"[{_CJK}]+", value):
        for boundary in range(2, len(value)):
            prefix = value[:boundary]
            if prefix in known and _has_identity_shape(prefix, known=True):
                extension = value[boundary:]
                if not re.fullmatch(r"[一二三四五六七八九十百零〇两]{1,2}", extension):
                    return None
    if value[-1] in _CLIPPED_ACTION and value[:-1] in known and _has_identity_shape(value[:-1], known=True):
        return None
    if _has_identity_shape(value, known=value in known):
        return value
    # Preserve a source-vetted name which happens to end in a modifier, rather
    # than guessing that an arbitrary known prefix is its real identity.
    if original in known and _has_identity_shape(original, known=True):
        return original
    return None


@lru_cache(maxsize=256)
def _known_identities(known_names: tuple[str, ...]) -> frozenset[str]:
    return frozenset(name for raw in known_names if (name := normalize_identity_candidate(raw, known_names)))


def _candidate_names(tag: str, known_names: Iterable[str]) -> list[str]:
    known_names = tuple(known_names)
    vetted = _known_identities(known_names)
    names = set(vetted)
    # Unknown names enter only through a subject slot with a surname/title,
    # rather than through arbitrary 2–4 characters next to a speaking verb.
    head = re.sub(r"^(?:而|这时|此时|随后|于是|只见|那位|这位)\s*", "", tag)
    word_interiors = {
        index
        for match in _TAG_WORD_BOUNDARIES.finditer(head[:6 + _MAX_TAG_WORD])
        for index in range(match.start() + 1, match.start() + len(match.group(1)))
    }
    for length in range(2, min(len(head), 6) + 1):
        value = head[:length]
        # The subject boundary cannot split a grammatical word: `陆鸣苦` +
        # `笑道` or `高玉笑` + `道` are not new source-backed identities.
        if length in word_interiors:
            continue
        if len(value) > 3 and not value.startswith(_COMPOUND_SURNAMES):
            # New single-surname four-character phrases are particularly easy
            # to confuse with actions.  A known/vetted four-character name was
            # already added above; an unfamiliar one needs semantic evidence.
            continue
        if _has_identity_shape(value) and normalize_identity_candidate(value, known_names) == value:
            names.add(value)
    names.update(title for title in _TITLES if head.startswith(title))
    return sorted(names, key=lambda name: (name in vetted, len(name)), reverse=True)


def _tag_subject(tag: str, known_names: Iterable[str]) -> str | None:
    tag = tag.strip().rstrip("：:，, ")
    if not tag or _UNSAFE.search(tag):
        return None
    tag = re.sub(r"^(?:而|这时|此时|随后|于是|只见|那位|这位)\s*", "", tag)
    tag = _PERSON_AGE_PREFIX.sub("", tag)
    known_names = tuple(known_names)
    # English attribution is unambiguous only in the direct subject slot.
    english = re.fullmatch("(" + _ENGLISH_NAME + r")\s+" + _ENGLISH_SPEECH, tag)
    if english:
        return english.group(1)
    english = re.fullmatch(_ENGLISH_SPEECH + r"\s+(" + _ENGLISH_NAME + ")", tag)
    if english:
        return english.group(1)
    for name in _candidate_names(tag, known_names):
        if not tag.startswith(name):
            continue
        rest = tag[len(name):]
        if re.fullmatch(r"(?:[，,\s]|" + _MODIFIER + r")*" + _SPEECH, rest):
            return name
        verb = re.search(_SPEECH + r"$", rest)
        if not verb:
            continue
        body = rest[:verb.start()].strip("，, ")
        if not body or len(body) > 120:
            continue
        clauses = [part.strip() for part in re.split(r"[，,]", body) if part.strip()]
        # Inline multi-person actions such as '林舟望着青禾说' stay with the
        # semantic model.  A separate recipient clause has explicit grammar.
        if len(clauses) == 1 and "," not in rest and "，" not in rest:
            targets = _known_identities(known_names) | {"众人", "两人", "两女", "大家", "他", "她", "他们", "她们", "对方"}
            target_pattern = "(?:" + "|".join(re.escape(value) for value in sorted(targets, key=len, reverse=True)) + ")"
            if not re.fullmatch(r"(?:对着|向着|朝着|对|向|朝)" + target_pattern + _MODIFIER + r"*", body):
                continue
        valid = True
        for clause in clauses:
            if re.fullmatch(_MODIFIER + r"+", clause) or _ACTION.fullmatch(clause):
                continue
            object_match = _OBJECT_MARKER.search(clause)
            if not object_match:
                # Facial/voice descriptions do not introduce a new subject.
                if re.fullmatch(r"(?:面庞|脸庞|脸上|面上|眼中|声音|语气|神色|面色)[^。！？：“”]{0,45}", clause):
                    continue
                valid = False
                break
            if object_match.start() > 0 and not re.fullmatch(r"(?:目光|眼神|视线)(?:微微)?", clause[:object_match.start()]):
                valid = False
                break
            # Another named actor before the direction marker would make the
            # subject ambiguous; a name after it is the listener/object.
            if any(other != name and other in clause[:object_match.start()] for other in known_names):
                valid = False
                break
        if valid:
            return name
    return None


def source_speaker(text: str, start: int, end: int, known_names: Iterable[str] = ()) -> tuple[str, str] | None:
    """Return a speaker only when a local source tag identifies its subject.

    The returned evidence is original text.  No alternation, nearest-name or
    string-similarity inference is used, and a listener is never the subject.
    """
    known_names = tuple(known_names)
    before = text[max(0, start - 300):start]
    tag = re.split(r'[。！？；\n“”「」『』"]', before)[-1].strip()
    name = _tag_subject(tag, known_names)
    if name:
        return name, tag
    # For a postposed tag, a colon starts the next utterance and must not
    # attribute the previous quote to that next speaker.
    after = text[end:end + 180]
    english = re.match(
        r"[,，\s]*(" + _ENGLISH_NAME + r"\s+" + _ENGLISH_SPEECH + "|" + _ENGLISH_SPEECH + r"\s+" + _ENGLISH_NAME + r")(?![A-Za-z])",
        after,
    )
    if english:
        if re.match(r"\s*[:：]", after[english.end():]):
            return None
        evidence = english.group(1)
        name = _tag_subject(evidence, known_names)
        if name:
            return name, evidence
    tag = re.split(r'[。！？；\n“”「」『』"]', after)[0].lstrip("，, ")
    speech = re.search(_SPEECH, tag)
    if speech and re.match(r"\s*[：:]", tag[speech.end():]):
        return None
    if speech:
        evidence = tag[:speech.end()]
        name = _tag_subject(evidence, known_names)
        if name:
            return name, evidence
    return None


def quote_is_speech(text: str, start: int, end: int) -> bool:
    """Recognize clear quoted references while retaining short actual speech.

    Ambiguous quotations remain candidates for semantic analysis.  In
    particular, a short quote or an unfamiliar fantasy word alone is not enough
    to discard an utterance.
    """
    raw = text[start:end].strip()
    if len(raw) < 2 or _QUOTE_PAIRS.get(raw[0]) != raw[-1]:
        return True  # The caller may pass an existing unquoted utterance.
    content = raw[1:-1].strip()
    before, after = text[max(0, start - 100):start], text[end:end + 60]
    if re.search(_SPEECH + r"\s*[：:]?\s*$", before):
        return True
    if re.search(r"\b" + _ENGLISH_SPEECH + r"\s*[:：,]?\s*$", before, re.I):
        return True
    if re.match(r"[,\s]*(?:" + _ENGLISH_NAME + r"|he|she|they)\s+" + _ENGLISH_SPEECH + r"\b", after):
        return True
    if re.match(rf"[，,\s]*[{_CJK}]{{2,6}}" + _MODIFIER + r"*" + _SPEECH + r"[。！？,，\s]", after):
        return True
    if re.search(r"[。！？!?；;]", content):
        return True
    if len(content) > 40:
        return True
    reference_before = re.compile(
        r"(?:名为|称为|所谓|叫做|唤作|称作|称之为|取名|命名|名叫|修炼|施展|领悟|掌握|催动|"
        r"学会|运转|参悟|练成|进入|前往|来到|来自|这门|那门|一门|一道|这道|那道|"
        r"这部|那部|这枚|那枚|这株|那株|这片|那片|这座|那座|这把|那把|这件|那件|"
        r"石台上的|体内的|名下的|所谓的|以|用|的)\s*$"
    )
    if reference_before.search(before):
        return False
    if re.match(r"(?:的|之|者|功法|法诀|神通|法术|秘法|武学|炼法|术法|境界|区域|药材|灵药|火种|丹药|印记|手印|符文|法印)", after):
        return False
    if _QUOTED_SUBJECT_PREDICATE.match(after.lstrip()):
        return False
    # A short inline label embedded in narration is a reference, even when its
    # fantasy vocabulary is unfamiliar.  Paragraph-start utterances stay with
    # semantic analysis; quotation length alone never removes their speech.
    line_prefix = text[:start].rsplit("\n", 1)[-1]
    return not line_prefix.strip()
