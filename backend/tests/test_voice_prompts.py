import csv

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from backend.database import Base, VoiceDesignPrompt
from backend.services.voice_prompts import (
    CATALOGUE_PATH,
    get_prompt,
    list_prompts,
    match_voice_prompt,
    seed_voice_prompts,
)


@pytest.fixture
def catalogue():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        yield db
    engine.dispose()


def test_import_retains_every_prompt_and_provenance_without_duplicates(catalogue):
    with CATALOGUE_PATH.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    assert len(rows) == 529
    assert seed_voice_prompts(catalogue) == len(rows)
    assert seed_voice_prompts(catalogue) == 0
    imported = catalogue.query(VoiceDesignPrompt).all()
    assert len(imported) == len(rows)
    assert sorted(item.prompt for item in imported) == sorted(row["声音提示词（Prompt）"].strip() for row in rows)
    assert all(item.source_url and item.source_voice_id and item.description for item in imported)
    assert get_prompt(catalogue, imported[0].id).prompt == imported[0].prompt


def test_catalogue_search_and_matching_use_actual_voice_traits(catalogue):
    seed_voice_prompts(catalogue)
    found = list_prompts(catalogue, q="少年")
    assert found and all("少年" in item.name + item.description + item.prompt + item.category for item in found)
    assert all(item.category == "旁白叙事" for item in list_prompts(catalogue, category="旁白叙事"))
    assert not list_prompts(catalogue, q="_%_not_a_wildcard")
    choice = match_voice_prompt(catalogue, gender="female", age_range="senior", description="温厚低沉", is_narrator=True)
    assert choice.gender == "female"
    assert choice.age_range == "senior"
    assert choice.id == match_voice_prompt(catalogue, gender="female", age_range="senior", description="温厚低沉", is_narrator=True).id
