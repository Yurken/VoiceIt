"""Local voice-description catalogue imported from the user's BreezeBlue CSV."""

import csv
import re
import uuid
from functools import lru_cache
from pathlib import Path

from sqlalchemy import or_

from ..database.models import VoiceDesignPrompt

CATALOGUE_PATH = Path(__file__).resolve().parent.parent / "data" / "breezeblue_voice_prompts.csv"


def infer_gender(text: str) -> str | None:
    text = text.lower()
    if re.search(r"女性|女声|女孩|女生|少女|女人|阿婆|奶奶|老太太|\b(woman|women|female|girl|grandmother)\b", text):
        return "female"
    if re.search(r"男性|男声|男孩|男生|少年|男人|爷爷|老爷子|\b(man|men|male|boy|grandfather)\b", text):
        return "male"
    return None


def infer_age(text: str) -> str | None:
    text = text.lower()
    numeric_age = re.fullmatch(r"(\d{1,3})(?:岁| years? old)?", text.strip())
    if numeric_age:
        age = int(numeric_age.group(1))
        return "child" if age < 13 else "teen" if age < 20 else "young" if age < 40 else "middle" if age < 60 else "senior"
    patterns = [
        ("child", r"儿童|小孩|幼童|小学生|六岁|七岁|八岁|九岁|\b(child|kid|little girl|little boy)\b"),
        ("senior", r"老年|老人|奶奶|爷爷|退休|六十|七十|八十|\b(elderly|senior|sixties|seventies|retired)\b"),
        ("teen", r"少年|少女|初中|高中|十[二三四五六七八九]岁|\b(teen|teenager|teenage)\b"),
        ("middle", r"中年|四十|五十|\b(middle-aged|forties|fifties)\b"),
        ("young", r"年轻|青年|二十|三十|大学|\b(young|twenties|thirties)\b"),
    ]
    return next((age for age, pattern in patterns if re.search(pattern, text)), None)


def seed_voice_prompts(db) -> int:
    """Idempotent import. Retain every source row, its prompt and provenance."""
    if not CATALOGUE_PATH.is_file():
        return 0
    existing = {row[0] for row in db.query(VoiceDesignPrompt.id).all()}
    added = 0
    with CATALOGUE_PATH.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = "breezeblue:" + row["序号"] + ":" + row["音色ID"]
            prompt_id = str(uuid.uuid5(uuid.NAMESPACE_URL, key))
            if prompt_id in existing:
                continue
            prompt = row["声音提示词（Prompt）"].strip()
            if not prompt:
                continue
            description = row["声音简介"].strip() or row["标题描述"].strip()
            evidence = row["音色名称"] + " " + description + " " + prompt
            db.add(VoiceDesignPrompt(
                id=prompt_id, name=row["音色名称"].strip(), category=row["分类"].strip(),
                description=description, prompt=prompt, sample_text=row["示例文稿"].strip(),
                source="BreezeBlue · " + row["来源"].strip(), source_voice_id=row["音色ID"].strip(),
                source_url=row["详情页链接"].strip(), gender=infer_gender(evidence),
                age_range=infer_age(evidence), language="zh",
            ))
            added += 1
    if added:
        db.commit()
    return added


def get_prompt(db, prompt_id: str) -> VoiceDesignPrompt | None:
    return db.query(VoiceDesignPrompt).filter_by(id=prompt_id).first()


def list_prompts(db, *, q: str = "", category: str = "") -> list[VoiceDesignPrompt]:
    query = db.query(VoiceDesignPrompt)
    if category:
        query = query.filter(VoiceDesignPrompt.category == category)
    if q.strip():
        escaped = q.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        query = query.filter(or_(*[
            column.ilike(pattern, escape="\\") for column in (
                VoiceDesignPrompt.name, VoiceDesignPrompt.description,
                VoiceDesignPrompt.prompt, VoiceDesignPrompt.category,
            )
        ]))
    return query.order_by(VoiceDesignPrompt.category, VoiceDesignPrompt.name).all()


@lru_cache(maxsize=2048)
def _terms(text: str) -> frozenset[str]:
    words = set(re.findall(r"[a-z]{3,}", text.lower()))
    for run in re.findall(r"[\u4e00-\u9fff]+", text):
        words.update(run[i:i + 2] for i in range(len(run) - 1))
    return frozenset(words)


def match_voice_prompt(db, *, gender=None, age_range=None, description="", is_narrator=False):
    """Choose a reproducible catalogue starting point from character evidence."""
    candidates = list_prompts(db)
    if not candidates:
        return None
    wanted_gender = infer_gender(str(gender or ""))
    wanted_age = infer_age(str(age_range or "")) or (age_range if age_range in {"child", "senior", "teen", "middle", "young"} else None)
    terms = _terms(description)

    def rank(item):
        score = 0
        if wanted_gender:
            score += 30 if item.gender == wanted_gender else (-50 if item.gender else 0)
        if wanted_age:
            score += 15 if item.age_range == wanted_age else (-15 if item.age_range else 0)
        if is_narrator and "旁白" in item.category:
            score += 20
        score += len(terms & _terms(item.description + item.prompt))
        return (-score, item.id)

    return min(candidates, key=rank)
