"""JSON Schema ответов моделей по ролям — режим «только наблюдение».

Схема ничего не меняет в данных и поведении: клиент LLM считает нарушения и пишет их в статистику
(LLMStats.note_schema), а ответ возвращается как есть. Поддерживается подмножество JSON Schema:
type (строка или список типов), properties, required, items, enum. additionalProperties не используется:
модель вправе добавлять поля. required стоит только на полях, без которых код не может использовать ответ;
схема не строже кода (код сам нормализует регистр, неизвестные значения и отсутствующие поля).
"""

from __future__ import annotations

from typing import Any

Schema = dict[str, Any]

_STR: Schema = {"type": "string"}
_STR_OR_NULL: Schema = {"type": ["string", "null"]}
_STRS: Schema = {"type": "array", "items": _STR}
_SEVERITY: Schema = {"type": "string", "enum": ["critical", "major", "minor"]}
_STATUS: Schema = {"type": "string", "enum": ["ok", "violation", "na"]}


def _obj(properties: dict[str, Schema], required: list[str] | None = None) -> Schema:
    return {"type": "object", "properties": properties, "required": required or []}


def _arr(item: Schema) -> Schema:
    return {"type": "array", "items": item}


# Находка ревьюера: без title код её отбрасывает (_finding_from)
_FINDING = _obj({
    "criterion": _STR, "severity": _SEVERITY, "title": _STR, "explanation": _STR, "rids": _STRS,
    "quotes": _STRS, "example": _STR, "suggestion": _STR, "question": _STR, "addressee": _STR,
}, ["title"])
_MARK = _obj({"criterion": _STR, "status": _STATUS, "comment": _STR}, ["criterion"])

SCHEMAS: dict[str, Schema] = {
    "extractor": _obj({"requirements": _arr(_obj({
        "block_ids": _STRS, "source_id": _STR_OR_NULL, "quote": _STR, "summary": _STR, "kind": _STR,
    }, ["block_ids"]))}, ["requirements"]),
    "reviewer": _obj({"assessments": _arr(_obj({
        "rid": _STR, "marks": _arr(_MARK), "findings": _arr(_FINDING),
    }, ["rid"]))}, ["assessments"]),
    "consistency": _obj({"set_marks": _arr(_MARK), "findings": _arr(_FINDING)}, ["findings"]),
    # Модель иногда отдаёт голый список находок: код такой ответ не использует, но и не падает — схема его допускает
    "gaps": {**_obj({"findings": _arr(_FINDING)}, ["findings"]), "type": ["object", "array"]},
    "critic": _obj({"verdicts": _arr(_obj({
        "fid": _STR,
        "verdict": {"type": "string", "enum": ["confirm", "downgrade", "reject", "duplicate"]},
        "severity": _SEVERITY, "duplicate_of": _STR_OR_NULL, "reason": _STR,
    }, ["fid", "verdict"]))}, ["verdicts"]),
    "analyst_checklist": _obj({
        "checklist": _arr(_obj({
            "index": {"type": ["integer", "string"]},  # код делает int(): «3» тоже принимается
            "answer": _STR, "justification": _STR, "quotes": _STRS,
        }, ["index"])),
        "trace": _arr(_obj({"rid": _STR, "acceptance": _STRS, "api": _STRS, "note": _STR}, ["rid"])),
    }, ["checklist"]),
    "analyst_tests": _obj({"tests": _arr(_obj({
        "rid": _STR, "title": _STR, "given": _STR, "when": _STR, "then": _STR,
    }, ["title"]))}, ["tests"]),
    "clarifier": _obj({"questions": _arr(_obj({
        "priority": _STR,  # без enum: код сам приводит неизвестное значение (модель иногда пишет major)
        "addressee": _STR, "question": _STR, "why": _STR, "options": _STRS, "assumption": _STR, "fids": _STRS,
    }, ["question", "fids"]))}, ["questions"]),
    "resolver": _obj({
        "self_contradictions": _arr(_obj({"qid": _STR, "part_a": _STR, "part_b": _STR}, ["qid"])),
        "effects": _arr(_obj({
            "qid": _STR, "fid": _STR,
            "status": {"type": "string", "enum": ["resolved", "partial", "open"]},
            "comment": _STR, "revised_wording": _STR, "chosen_values": _STRS, "follow_up": _STR,
        }, ["qid", "fid", "status"])),
    }, ["effects"]),
}

_TYPES: dict[str, type] = {
    "object": dict, "array": list, "string": str, "null": type(None), "boolean": bool,
}


def _is_type(value: Any, name: str) -> bool:
    if name == "integer":  # bool в Python — подкласс int, но числом не считается
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return isinstance(value, _TYPES[name])


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _check(schema: Schema, value: Any, path: str, out: list[str]) -> None:
    types = schema.get("type")
    if types is not None:
        names = [types] if isinstance(types, str) else list(types)
        if not any(_is_type(value, n) for n in names):
            out.append(f"{path or '$'}: ожидался тип {'|'.join(names)}, получено {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        out.append(f"{path or '$'}: ожидалось одно из {schema['enum']}, получено {value!r}")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                out.append(f"{_join(path, key)}: обязательное поле отсутствует")
        for key, sub in schema.get("properties", {}).items():
            if key in value:
                _check(sub, value[key], _join(path, key), out)
    elif isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            _check(schema["items"], item, f"{path}[{i}]", out)


def violations(role: str, data: Any) -> list[str]:
    """Нарушения схемы роли в виде «findings[2].severity: ожидалось одно из [...]». Неизвестная роль — пусто."""
    schema = SCHEMAS.get(role)
    if schema is None:
        return []
    out: list[str] = []
    _check(schema, data, "", out)
    return out
