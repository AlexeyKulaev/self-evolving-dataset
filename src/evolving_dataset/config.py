from __future__ import annotations

import copy
import math
from pathlib import Path
import tomllib

from .io import digest

DEFAULTS = {
    "paths": {"goal": "goal.md", "architecture": "architecture.md", "seed": "data/seed.jsonl", "output": "runs/experiment"},
    "format": {"input_fields": ["question"], "answer_field": "answer"},
    "pipeline": {"n": 10000, "k": 3, "steps": 100, "batch_size": 8, "validation_size": 100,
                 "exit_accuracy": 0.9, "delta_floor": 0.01, "delta": None, "seed": 42,
                 "generation_attempts": 8},
    "teacher": {"backend": "codex", "executable": "codex", "model": "gpt-6.1-sol",
                "reasoning_effort": "xhigh", "service_tier": "fast", "batch_size": 16,
                "timeout": 600, "retries": 2, "audit_generated": True},
    "student": {"backend": "hf", "model": "", "device": "cpu", "learning_rate": 0.00001,
                "max_length": 1024, "max_new_tokens": 128, "hidden_size": 24},
}


def load_config(path):
    path = Path(path).resolve()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    cfg = copy.deepcopy(DEFAULTS)
    for section, values in raw.items():
        if section not in cfg or not isinstance(values, dict):
            raise ValueError(f"Unknown configuration section: {section}")
        for key, value in values.items():
            if key not in cfg[section] and not (section == "student" and key == "factory"):
                raise ValueError(f"Unknown configuration key: {section}.{key}")
            cfg[section][key] = value
    for key, value in cfg["paths"].items():
        cfg["paths"][key] = str((path.parent / value).resolve())
    p = cfg["pipeline"]
    for key in ("n", "k", "steps", "batch_size", "validation_size", "generation_attempts"):
        if type(p[key]) is not int or p[key] < 1:
            raise ValueError(f"pipeline.{key} must be a positive integer")
    if p["k"] > p["n"]:
        raise ValueError("k must not exceed N")
    if p["n"] < 10:
        raise ValueError("N must be at least 10 to represent the 10% replay quota")
    if not 0 < p["exit_accuracy"] <= 1 or not 0 < p["delta_floor"] <= 1:
        raise ValueError("Invalid accuracy/delta setting")
    if p["delta"] is not None and not 0 < p["delta"] <= 1:
        raise ValueError("delta must be positive")
    t = cfg["teacher"]
    if "/" in t["executable"]:
        t["executable"] = str((path.parent / t["executable"]).resolve())
    for key in ("batch_size", "timeout", "retries"):
        if type(t[key]) is not int or t[key] < 1:
            raise ValueError(f"teacher.{key} must be a positive integer")
    f = cfg["format"]
    if (not isinstance(f["input_fields"], list) or not f["input_fields"]
            or any(not isinstance(k, str) or not k for k in f["input_fields"])
            or not isinstance(f["answer_field"], str) or not f["answer_field"]
            or len(set(f["input_fields"])) != len(f["input_fields"])
            or f["answer_field"] in f["input_fields"]):
        raise ValueError("Input fields must be distinct and must not contain the answer field")
    for key in ("max_length", "max_new_tokens", "hidden_size"):
        if type(cfg["student"][key]) is not int or cfg["student"][key] < 1:
            raise ValueError(f"student.{key} must be a positive integer")
    learning_rate = cfg["student"]["learning_rate"]
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("student.learning_rate must be finite and positive")
    goal = Path(cfg["paths"]["goal"]).read_text(encoding="utf-8")
    architecture = Path(cfg["paths"]["architecture"]).read_text(encoding="utf-8")
    if not goal.strip() or not architecture.strip():
        raise ValueError("goal.md and architecture.md must not be empty")
    cfg["context"] = {"goal": goal, "architecture": architecture}
    cfg["fingerprint"] = digest(cfg)
    return cfg
