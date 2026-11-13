import json
from typing import Literal

import pytest
from pydantic import BaseModel, Field

from backend.services.llm_structured import StructuredOutputError, generate_structured


class Item(BaseModel):
    name: str
    n: int


class _FakeLLM:
    def __init__(self, replies):
        self._replies = list(replies)
        self.calls = 0
        self.requests = []

    async def generate(self, prompt, system=None, max_tokens=512, temperature=0.7, model_size=None, examples=None):
        self.calls += 1
        self.requests.append({"prompt": prompt, "system": system, "max_tokens": max_tokens})
        return self._replies[min(self.calls - 1, len(self._replies) - 1)]


@pytest.mark.asyncio
async def test_clean_json():
    llm = _FakeLLM(['{"name": "Holston", "n": 3}'])
    out = await generate_structured("p", Item, backend=llm)
    assert out.name == "Holston"
    assert out.n == 3
    assert llm.calls == 1


@pytest.mark.asyncio
async def test_fenced_json_is_unwrapped():
    llm = _FakeLLM(['```json\n{"name": "A", "n": 1}\n```'])
    out = await generate_structured("p", Item, backend=llm)
    assert out.n == 1


@pytest.mark.asyncio
async def test_trailing_comma_repaired():
    llm = _FakeLLM(['{"name": "A", "n": 1,}'])
    out = await generate_structured("p", Item, backend=llm)
    assert out.name == "A"


@pytest.mark.asyncio
async def test_retries_then_succeeds():
    llm = _FakeLLM(["totally not json", '{"name":"B","n":2}'])
    out = await generate_structured("p", Item, backend=llm, max_retries=2)
    assert out.n == 2
    assert llm.calls == 2


@pytest.mark.asyncio
async def test_unrecoverable_raises():
    llm = _FakeLLM(["nope", "still nope", "nope again"])
    with pytest.raises(StructuredOutputError):
        await generate_structured("p", Item, backend=llm, max_retries=2)
    assert llm.calls == 3


@pytest.mark.asyncio
async def test_trailing_prose_stripped_in_one_call():
    """Balanced extraction must strip trailing prose so no retry is wasted."""
    llm = _FakeLLM(['{"name":"A","n":1} Hope this helps!'])
    out = await generate_structured("p", Item, backend=llm)
    assert out.name == "A"
    assert out.n == 1
    assert llm.calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply",
    [
        '{"name":"A',
        '{"name":"A","n":',
        '{"name":"A","n":2',
        '{"name":"A","n":tru',
        '{"name":"\\u12',
    ],
)
async def test_truncation_can_return_immediately_for_caller_batch_splitting(reply):
    llm = _FakeLLM([reply, '{"name":"A","n":2}'])
    with pytest.raises(StructuredOutputError) as caught:
        await generate_structured("p", Item, backend=llm, retry_on_truncation=False)
    assert llm.calls == 1
    assert caught.value.truncated is True
    assert caught.value.attempts == 1


@pytest.mark.asyncio
async def test_truncation_grows_budget_then_completes():
    llm = _FakeLLM(['{"name":"A', '{"name":"A","n":2}'])
    result = await generate_structured("p", Item, backend=llm, max_tokens=100)
    assert result.n == 2
    assert [request["max_tokens"] for request in llm.requests] == [100, 150]
    assert "incomplete" in llm.requests[1]["prompt"]


@pytest.mark.asyncio
async def test_truncation_stops_at_double_budget_instead_of_repeating_capped_attempts():
    llm = _FakeLLM(['{"name":"A'])
    with pytest.raises(StructuredOutputError) as caught:
        await generate_structured("p", Item, backend=llm, max_tokens=100, max_retries=6)
    assert [request["max_tokens"] for request in llm.requests] == [100, 150, 200]
    assert caught.value.truncated is True
    assert caught.value.attempts == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reply", ["{not json", "{x", "[x", '{"name":bad', '{"name":"A"e', '{"name":"A","n":"wrong"}', "{}"]
)
async def test_regular_invalid_output_keeps_retry_policy_without_growing_budget(reply):
    llm = _FakeLLM([reply, '{"name":"A","n":2}'])
    result = await generate_structured("p", Item, backend=llm, max_tokens=100, retry_on_truncation=False)
    assert result.n == 2
    assert llm.calls == 2
    assert [request["max_tokens"] for request in llm.requests] == [100, 100]


@pytest.mark.asyncio
async def test_compact_schema_keeps_nested_constraints_and_metadata_named_fields():
    class Child(BaseModel):
        title: str = Field(description="This is a real field called title", min_length=2)
        description: Literal["soft", "bright"]
        default: int = Field(ge=0)

    class Document(BaseModel):
        entries: list[Child] = Field(description="Long repeated annotation", min_length=1)
        optional: str = Field(default="saved default", description="Applied by server")

    payload = '{"entries":[{"title":"角色","description":"soft","default":1}]}'
    ordinary = _FakeLLM([payload])
    compact = _FakeLLM([payload])
    await generate_structured("p", Document, backend=ordinary)
    result = await generate_structured("p", Document, backend=compact, compact_schema=True)
    regular_text = ordinary.requests[0]["system"].split("JSON schema:\n", 1)[1]
    compact_text = compact.requests[0]["system"].split("JSON schema:\n", 1)[1]
    sent = json.loads(compact_text)
    assert len(compact_text) < len(regular_text)
    assert result.optional == "saved default"
    child = sent["$defs"]["Child"]
    assert set(child["properties"]) == {"title", "description", "default"}
    assert child["properties"]["title"]["minLength"] == 2
    assert child["properties"]["description"]["enum"] == ["soft", "bright"]
    assert child["properties"]["default"]["minimum"] == 0
    assert set(child["required"]) == {"title", "description", "default"}
    assert sent["properties"]["entries"]["items"] == {"$ref": "#/$defs/Child"}
    assert sent["properties"]["entries"]["minItems"] == 1
    assert "title" not in sent
    assert "description" not in sent["properties"]["entries"]
    assert "default" not in sent["properties"]["optional"]


def test_error_metadata_is_backward_compatible():
    error = StructuredOutputError("bad")
    assert str(error) == "bad"
    assert error.truncated is False
    assert error.attempts == 0
