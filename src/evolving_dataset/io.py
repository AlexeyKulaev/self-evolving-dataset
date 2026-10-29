"""Data remains in its original JSON record format; metadata lives outside it."""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
from pathlib import Path


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def records(path):
    path = Path(path)
    if path.suffix == ".jsonl":
        result = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        result = read_json(path)
    if not isinstance(result, list) or not result or any(not isinstance(x, dict) for x in result):
        raise ValueError("Dataset must be a nonempty JSON array or JSONL of objects")
    return result


def text_value(value):
    return value if isinstance(value, str) else canonical(value)


def input_text(record, fields):
    return "\n".join(text_value(record[field]) for field in fields)


def input_id(record, fields):
    return digest({field: record[field] for field in fields})


def object_schema(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def infer_schema(value):
    if value is None:
        return {"type": "null"}
    if isinstance(value, bool):
        return {"type": "boolean"}
    if isinstance(value, int):
        return {"type": "integer"}
    if isinstance(value, float):
        return {"type": "number"}
    if isinstance(value, str):
        return {"type": "string"}
    if isinstance(value, dict):
        return object_schema({k: infer_schema(v) for k, v in value.items()})
    if isinstance(value, list):
        if not value:
            raise ValueError("Cannot infer the item type of an empty array; supply a representative nonempty record")
        alternatives = {canonical(infer_schema(v)): infer_schema(v) for v in value}
        items = next(iter(alternatives.values())) if len(alternatives) == 1 else {"anyOf": list(alternatives.values())}
        return {"type": "array", "items": items}
    raise ValueError(f"Unsupported JSON type: {type(value).__name__}")


def validate(value, schema, path="$"):
    """Validate the deliberately small JSON Schema subset emitted by this package."""
    if "anyOf" in schema:
        for option in schema["anyOf"]:
            try:
                validate(value, option, path)
                return
            except ValueError:
                pass
        raise ValueError(f"{path}: no matching schema")
    kind = schema["type"]
    good = {"object": isinstance(value, dict), "array": isinstance(value, list),
            "string": isinstance(value, str), "boolean": isinstance(value, bool),
            "number": type(value) in (int, float) and math.isfinite(value),
            "integer": type(value) is int, "null": value is None}[kind]
    if not good:
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{path}: not in enum")
    if kind == "object":
        if set(value) != set(schema["properties"]):
            raise ValueError(f"{path}: keys must be {list(schema['properties'])}")
        for k, child in schema["properties"].items():
            validate(value[k], child, f"{path}.{k}")
    if kind == "array":
        for i, child in enumerate(value):
            validate(child, schema["items"], f"{path}[{i}]")


def tuples(value):
    return tuple(tuples(x) for x in value) if isinstance(value, list) else value


@contextlib.contextmanager
def locked(directory):
    import fcntl
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another process owns {directory}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
