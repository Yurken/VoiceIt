# ruff: noqa: RUF001
"""Recheck cast identities against source text without running another model.

This pass also applies to restored chapter checkpoints. It repairs individual
attributions rather than merging every line bearing the same polluted label.
"""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Sequence

from .character_identity import normalize_identity_candidate, quote_is_speech, source_speaker
from .character_scoring import assign_character_roles, score_character

UNKNOWN = "未识别角色"
_UNKNOWN_NAMES = {UNKNOWN, "未知角色", "未知未识别角色", "未知", "unknown", "unknown speaker"}
_NARRATORS = {"", "narrator", "narration", "旁白", "叙述者"}
_COMPOUND_NAMES = ("欧阳", "司马", "上官", "诸葛", "夏侯", "东方", "皇甫", "尉迟", "公孙", "慕容", "司徒", "司空", "令狐", "端木", "独孤", "南宫", "宇文", "长孙", "闻人", "澹台", "轩辕", "百里", "东郭", "西门", "钟离")
_PERSON = re.compile(
    r"(?:[红黑白青灰蓝金紫]衣|[红黑白青灰蓝金紫]袍|年轻|中年|老年|年老|年迈|瘦弱|魁梧|瘦小|金裙|宫装|道姑|铺主|鼠面|年纪稍轻点的|寻宝队|真炁|大神通|神通)*"
    r"(?:男子|女子|妇人|美妇|少妇|老妇|女孩|男孩|汉子|老人|队长|大师|统领|大神通者|道人|弟子)"
)
_PERSON_TITLE = re.compile(r"[\u3400-\u9fff]{0,3}(?:家主|馆主|管事|长老|前辈|统领|老师|师父|师傅|师兄|师姐|师弟|师妹|先生|夫人|姑娘|公子|小姐|大人|总管|掌柜|将军|大侠|叔|姨|姐姐|哥哥)")
_NONPERSON_END = re.compile(r"(?:馆|宗|派|商会|府|郡|城|皇域|灵禁|神通禁|境|骨|珠|幡|旗|鞭|法印|毒印|金身|玄火光|丹炉|金粉|之焱|之炎|元炁|真解)$")


def is_person_descriptor(name: str) -> bool:
    """Allow human descriptions and titles, without treating every phrase as a name."""
    if re.match(r"^(?:三位|两位|几位|数位|众|诸位|一群|多名)", name):
        return False
    return bool(_PERSON.fullmatch(name) or _PERSON_TITLE.fullmatch(name))


def source_identity_is_plausible(name: str, sources: Sequence[str], ordinary: set[str]) -> bool:
    """Require a person-shaped candidate or an explicit source introduction."""
    if _NONPERSON_END.search(name):
        return False
    if is_person_descriptor(name):
        return True
    reference = re.compile(r"(?:(?:修炼|施展|参悟|运转|催动|炼制|炼出|炼成|打出|祭出|这门|这道|一门|一道|这部|一部)\s*[“\"「『‘]?|(?:化为|化作|化成|烧成|轰成)[^。！？\n]{0,8})" + re.escape(name))
    if any(reference.search(text) for text in sources):
        actor = re.compile(r"(?:^|[。！？；\n”」』])\s*" + re.escape(name) + r"(?:轻声|低声|大声|小声|缓缓地?|冷冷地?|微笑着|笑着|平静[地的]?|沉吟着?|连忙|赶紧|[，,\s])*(?:说道|问道|笑道|回答|问|说|道)[：:，,]")
        if not any(actor.search(text) for text in sources):
            return False
    normal = normalize_identity_candidate(name)
    if normal == name and (not re.fullmatch(r"[\u3400-\u9fff]+", name) or len(name) <= 3 or name.startswith(_COMPOUND_NAMES)):
        return True
    if "·" in name:
        return True
    # A rare surname needs source evidence; being stored in an old roster is
    # not such evidence. Explicit introductions also preserve names like 青禾.
    introduction = re.compile(r"(?:[\d一二三四五六七八九十百两]{1,4}岁的|(?:他|她|少年|少女|男子|女子|姑娘)(?:的名字是|叫|名叫))" + re.escape(name))
    if any(introduction.search(text) for text in sources):
        return True
    return len(name) == 2 and sum(len(full) == 3 and full[1:] == name for full in ordinary) == 1


def refine_character_segments(
    chapter_texts: Sequence[str],
    chapter_segments: Sequence[list[dict]],
    roster: Sequence[dict],
) -> dict:
    """Mutate annotations only; preserve every source segment and its text.

    Existing character labels are candidates, not proof of identity. Source
    speaker tags can recover different people from a shared erroneous label.
    Unresolved lines remain dialogue and are explicitly flagged for review.
    """
    if len(chapter_texts) != len(chapter_segments):
        raise ValueError("角色校验的章节数量与原文不一致")
    raw_names = [str(c.get("name") or "") for c in roster]
    raw_names.extend(str(a) for c in roster for a in c.get("aliases", []))
    candidates = {
        name for raw in raw_names
        if (name := normalize_identity_candidate(raw, raw_names))
        and name not in _UNKNOWN_NAMES and name.casefold() not in _NARRATORS
    }
    ordinary = {name for name in candidates if normalize_identity_candidate(name) == name}
    candidates = {name for name in candidates if source_identity_is_plausible(name, chapter_texts, ordinary)}
    # Read chapter sources sequentially; do not concatenate a large novel.
    grounded = set()
    for text in chapter_texts:
        grounded.update(name for name in candidates - grounded if name in text)
    name_map = {str(char.get("name") or ""): None for char in roster}
    for char in roster:
        raw = str(char.get("name") or "")
        canonical = normalize_identity_candidate(raw, sorted(grounded))
        if canonical not in grounded:
            continue
        name_map[raw] = canonical
        for alias in char.get("aliases", []):
            normalized = normalize_identity_candidate(alias, sorted(grounded))
            if normalized and normalized in grounded:
                name_map[alias] = canonical
    known_names = sorted(grounded, key=len, reverse=True)
    previously_spoken = {
        segment.get("speaker") for segments in chapter_segments for segment in segments
        if segment.get("type") == "dialogue" and segment.get("speaker")
    }
    changes = []
    by_speaker = defaultdict(list)
    for chapter_index, (source, segments) in enumerate(zip(chapter_texts, chapter_segments, strict=True)):
        cursor = 0
        for segment_index, segment in enumerate(segments):
            text = segment.get("text", "")
            start = source.find(text, cursor)
            if start < 0:
                raise ValueError(f"第 {chapter_index + 1} 章的片段无法定位原文，未修改角色")
            end = start + len(text)
            cursor = end
            previous = segment.get("speaker")
            previous_type = segment.get("type")
            metadata = dict(segment.get("analysis_metadata") or {})
            reason = None
            speaker = name_map.get(previous)
            if previous in _UNKNOWN_NAMES:
                speaker = UNKNOWN
            if previous_type != "dialogue":
                speaker = None
            elif not quote_is_speech(source, start, end):
                segment["type"] = "narration"
                segment["flagged"] = False
                speaker = None
                reason = "quoted_term"
                metadata.update(method="quoted_term", speaker_verified=False, speaker_evidence=None)
            else:
                explicit = source_speaker(source, start, end, known_names)
                if explicit and explicit[0] in grounded:
                    raw_speaker, evidence = explicit
                    speaker = name_map.get(raw_speaker, raw_speaker)
                    segment["flagged"] = False
                    metadata.update(method="repair_source_tag", speaker_verified=True, speaker_evidence=evidence)
                    # The evidence used for confidence must itself be source text.
                    metadata["source_excerpt"] = source[max(0, start - 240):min(len(source), end + 160)]
                    metadata["warnings"] = []
                    reason = "source_speaker_tag" if speaker != previous else None
                elif not speaker or speaker == UNKNOWN:
                    speaker = UNKNOWN
                    segment["flagged"] = True
                    metadata.update(method="source_fallback", speaker_verified=False, speaker_evidence=None)
                    metadata["warnings"] = ["原角色名称不可靠，尚不能从原文确认说话人，请校对。"]
                    reason = "unresolved_identity" if previous != UNKNOWN else None
            segment["speaker"] = speaker
            if previous != speaker and name_map.get(previous) != speaker:
                segment["voice_stage"] = None
                metadata["voice_stage_evidence"] = None
                metadata.pop("voice_stage_inherited", None)
            if previous != speaker or previous_type != segment.get("type"):
                change = {
                    "chapter": chapter_index + 1,
                    "segment": segment_index,
                    "segment_id": segment.get("id"),
                    "from": previous,
                    "to": speaker,
                    "from_type": previous_type,
                    "to_type": segment.get("type"),
                    "reason": reason or "canonical_name",
                }
                changes.append(change)
                metadata["identity_refinement"] = change
            segment["analysis_metadata"] = metadata
            if speaker and segment.get("type") == "dialogue":
                by_speaker[speaker].append(segment)
    characters = []
    for name, segments in by_speaker.items():
        aliases = sorted({a for a, target in name_map.items() if target == name and a != name
                          and normalize_identity_candidate(a, known_names) == a})
        characters.append({
            "name": name, "aliases": aliases, "dialogue_count": len(segments),
            "confidence": score_character(segments),
        })
    characters.sort(key=lambda char: char["dialogue_count"], reverse=True)
    # A label whose old lines all proved to be references or other people's
    # speech is no longer a cast member. Unused manually added rows are not
    # covered by this rule and remain available to the database caller.
    for raw in previously_spoken:
        if name_map.get(raw) not in by_speaker:
            name_map[raw] = None
    if UNKNOWN in by_speaker:
        name_map[UNKNOWN] = UNKNOWN
    return {"characters": assign_character_roles(characters), "name_map": name_map, "changes": changes}
