from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import shutil
import sys

from .config import load_config
from .engine import Engine, baseline, evaluate_test, prepare
from .io import atomic_json, locked, object_schema
from .report import write_report
from .student import make_student
from .teacher import make_teacher


def main(argv=None):
    parser = argparse.ArgumentParser(description="Self-evolving dataset pipeline")
    parser.add_argument("--config", default="config.toml")
    commands = parser.add_subparsers(dest="command", required=True)
    doctor = commands.add_parser("doctor", help="Check prerequisites; optionally probe the semantic backend")
    doctor.add_argument("--live", action="store_true", help="Make one small subscription-backed model call")
    commands.add_parser("prepare", help="Partition seed data, order domains, freeze ChatGPT validation")
    run = commands.add_parser("run", help="Run or resume adaptive training")
    run.add_argument("--max-stages", type=int, help="Stop this invocation after N additional stages; resume later")
    compare = commands.add_parser("baseline", help="Train an equal-domain baseline on the shared candidate bank")
    compare.add_argument("--max-steps", type=int)
    evaluate = commands.add_parser("evaluate", help="Evaluate a final model on an external held-out test")
    evaluate.add_argument("--test", required=True)
    evaluate.add_argument("--method", choices=("adaptive", "baseline"), default="adaptive")
    commands.add_parser("report", help="Write a run report including provisional decisions")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
        for name in ("max_stages", "max_steps"):
            if getattr(args, name, None) is not None and getattr(args, name) < 1:
                raise ValueError(f"{name} must be positive")
        if args.command == "doctor":
            teacher = make_teacher(cfg)
            if cfg["teacher"]["backend"] == "codex":
                if not shutil.which(cfg["teacher"]["executable"]):
                    raise ValueError("codex executable not found")
                teacher.transport.check_auth()
                if args.live:
                    with locked(cfg["paths"]["output"]):
                        teacher.transport.request("connectivity_probe", {"instruction": "Return ok=true."},
                                                  object_schema({"ok": {"type": "boolean"}}))
            requirements = {"numpy": ["numpy"], "hf": ["torch", "transformers"], "plugin": []}
            for package in requirements[cfg["student"]["backend"]]:
                if importlib.util.find_spec(package) is None:
                    raise ValueError(f"Missing optional dependency: {package}")
            if not Path(cfg["paths"]["seed"]).exists():
                raise ValueError("Seed dataset does not exist")
            print("Prerequisites available. " + ("Live model probe succeeded." if args.live else "No model calls were made."))
            return 0
        with locked(cfg["paths"]["output"]):
            if args.command == "report":
                print(write_report(cfg))
                return 0
            teacher = make_teacher(cfg)
            if args.command in ("prepare", "run"):
                prepared = prepare(cfg, teacher)
                atomic_json(Path(cfg["paths"]["output"]) / "manifest.json", cfg)
                if args.command == "prepare":
                    print(f"Prepared {len(prepared['domains'])} domains")
                    return 0
                state = Engine(cfg, teacher, make_student(cfg)).run(prepared, args.max_stages)
                print(write_report(cfg))
                print("All domains passed." if state["complete"] else "Stage limit reached; checkpoint saved for resume.")
            elif args.command == "baseline":
                result = baseline(cfg, make_student(cfg), args.max_steps)
                print(json.dumps({k: v for k, v in result.items() if k != "rng"}))
            elif args.command == "evaluate":
                print(json.dumps(evaluate_test(cfg, teacher, make_student(cfg), args.test, args.method)))
                print(write_report(cfg))
        return 0
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
