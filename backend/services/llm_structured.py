"""Structured-output wrapper over the plain-string local LLM backend."""

import json
import re

from pydantic import BaseModel, ValidationError

_SYSTEM = (
    "You are a precise extraction engine. Respond with ONLY valid JSON that "
    "matches the requested schema. No prose, no markdown fences, no commentary."
)


class StructuredOutputError(RuntimeError):
    """Raised when the model never produces schema-valid JSON within the retry budget."""

    def __init__(self, message: str, *, truncated: bool = False, attempts: int = 0):
        super().__init__(message)
        self.truncated = truncated
        self.attempts = attempts


def _compact_json_schema(schema: dict | bool) -> dict | bool:
    """Remove annotations while preserving validation and arbitrary field names."""
    if not isinstance(schema, dict):
        return schema
    annotations = {"title", "description", "default", "examples", "$comment", "deprecated", "readOnly", "writeOnly"}
    schema_maps = {"properties", "$defs", "definitions", "patternProperties", "dependentSchemas"}
    schema_values = {
        "items",
        "additionalProperties",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contentSchema",
    }
    schema_lists = {"allOf", "anyOf", "oneOf", "prefixItems"}
    result = {}
    for key, value in schema.items():
        if key in annotations:
            continue
        if key in schema_maps and isinstance(value, dict):
            # The keys here are field/definition names, not schema keywords.
            result[key] = {name: _compact_json_schema(item) for name, item in value.items()}
        elif key in schema_values and isinstance(value, dict):
            result[key] = _compact_json_schema(value)
        elif key in schema_lists and isinstance(value, list):
            result[key] = [_compact_json_schema(item) for item in value]
        else:
            # Do not strip object-valued const/enum data or unknown extensions.
            result[key] = value
    return result


def _is_truncated_json(candidate: str, error: json.JSONDecodeError) -> bool:
    """Recognize an unfinished JSON prefix, rather than an arbitrary bad reply."""
    text = candidate.rstrip()
    if not text or text[0] not in "{[":
        return False
    stack = []
    in_string = False
    escaped = False
    for char in text:
        if escaped:
            escaped = False
        elif in_string:
            if char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
        elif char == '"':
            in_string = True
        elif char in "{[":
            stack.append(char)
        elif char in "}]" and (not stack or stack.pop() != {"}": "{", "]": "["}[char]):
            return False
    if not stack and not in_string:
        return False
    if error.msg.startswith("Unterminated string"):
        return True
    if error.msg.startswith("Invalid \\uXXXX escape") and re.fullmatch(r"u[0-9a-fA-F]{0,3}", text[error.pos :]):
        return True
    if error.pos >= len(text):
        return True
    # The decoder points to the beginning of an incomplete literal token.
    tail = text[error.pos :]
    if error.msg == "Expecting value" and tail in {
        "t",
        "tr",
        "tru",
        "f",
        "fa",
        "fal",
        "fals",
        "n",
        "nu",
        "nul",
        "-",
    }:
        return True
    return (
        error.msg == "Expecting ',' delimiter"
        and error.pos > 0
        and text[error.pos - 1].isdigit()
        and tail in {".", "e", "E", "e+", "e-", "E+", "E-"}
    )


def _extract_json(text: str) -> str:
    text = text.strip()
    # strip ```json ... ``` fences
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    # find the first opening brace/bracket
    start = next((i for i, c in enumerate(text) if c in "{["), None)
    if start is None:
        return text
    # walk forward tracking nesting depth to find the balanced close
    opener = text[start]
    closer = "}" if opener == "{" else "]"
    depth = 0
    in_string = False
    escape_next = False
    end = None
    for i in range(start, len(text)):
        ch = text[i]
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                end = i
                break
    if end is None:
        # no balanced close found — fall back to current behaviour
        return text[start:].strip()
    return text[start : end + 1]


def _repair(text: str) -> str:
    # drop trailing commas before } or ]
    return re.sub(r",\s*([}\]])", r"\1", text)


async def generate_structured[T: BaseModel](
    prompt: str,
    schema: type[T],
    *,
    backend=None,
    system: str | None = None,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    model_size: str | None = None,
    max_retries: int = 2,
    compact_schema: bool = False,
    retry_on_truncation: bool = True,
) -> T:
    if backend is None:
        from . import llm as llm_module

        backend = llm_module.get_llm_model()
    json_schema = schema.model_json_schema()
    if compact_schema:
        json_schema = _compact_json_schema(json_schema)
    serialized_schema = json.dumps(
        json_schema, ensure_ascii=not compact_schema, separators=(",", ":") if compact_schema else None
    )
    sys_prompt = (system or _SYSTEM) + "\n\nJSON schema:\n" + serialized_schema
    last_err: Exception | None = None
    truncated = False
    attempts = 0
    token_budget = max_tokens
    token_ceiling = max_tokens * 2
    attempt_prompt = prompt
    for attempt in range(max_retries + 1):
        attempts = attempt + 1
        raw = await backend.generate(
            attempt_prompt,
            system=sys_prompt,
            max_tokens=token_budget,
            temperature=temperature,
            model_size=model_size,
        )
        candidate = _repair(_extract_json(raw))
        try:
            data = json.loads(candidate)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError) as e:
            last_err = e
            truncated = isinstance(e, json.JSONDecodeError) and _is_truncated_json(candidate, e)
            if truncated:
                if not retry_on_truncation:
                    break
                expanded_budget = min(token_ceiling, max(token_budget + 1, (token_budget * 3 + 1) // 2))
                if expanded_budget <= token_budget:
                    break  # Another identical capped attempt would waste inference.
                token_budget = expanded_budget
                attempt_prompt = (
                    prompt + "\n\nYour previous JSON reply was incomplete. "
                    "Return a concise, complete JSON object matching the schema, with no prose."
                )
            else:
                attempt_prompt = (
                    prompt + "\n\nYour previous reply was not valid JSON for the schema. "
                    "Return ONLY the JSON object, nothing else."
                )
    raise StructuredOutputError(
        f"No valid JSON after {attempts} attempts: {last_err}", truncated=truncated, attempts=attempts
    )
