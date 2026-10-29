from __future__ import annotations

import copy
import math
from pathlib import Path
import random
import uuid

from .io import atomic_json, digest, infer_schema, input_id, read_json, records, tuples, validate
from .policy import active_quotas, allocate, delta, progress_weights, replay_quotas, window
from .teacher import check_domains


def chunks(values, size):
    for i in range(0, len(values), size):
        yield values[i:i+size]


def catalog(domain):
    return {k: domain[k] for k in ("id", "name", "rule")}


def register(state, cfg, record, domain_id, split, source):
    validate(record, state["schema"])
    key = input_id(record, cfg["format"]["input_fields"])
    if key in state["examples"]:
        return None  # exact input duplicates never cross train/validation, even with different labels
    state["examples"][key] = {"id": key, "domain_id": domain_id, "record": record,
        "split": split, "source": source, "c": 0.0, "correct_streak": 0, "last_checked": -1}
    return key


def generate(state, cfg, teacher, domain, count, split, phase):
    if count <= 0:
        return []
    size = cfg["teacher"]["batch_size"]
    examples = [state["examples"][i]["record"] for i in domain["seed_ids"][:8]] or state["format_examples"]
    accepted = []
    max_requests = math.ceil(count / size) * cfg["pipeline"]["generation_attempts"]
    for attempt in range(max_requests):
        needed = min(size, count - len(accepted))
        if needed == 0:
            break
        batch = teacher.generate(catalog(domain), examples, needed, split,
            f"{phase}:{domain['id']}:{attempt}:{len(state['examples'])}", state["schema"])
        if len(batch) != needed:
            raise ValueError("Generation count mismatch")
        approved = teacher.audit(catalog(domain), batch) if cfg["teacher"]["audit_generated"] else [True]*len(batch)
        if len(approved) != len(batch):
            raise ValueError("Audit count mismatch")
        for record, valid in zip(batch, approved):
            if valid:
                key = register(state, cfg, record, domain["id"], split, "generated")
                if key:
                    accepted.append(key)
    if len(accepted) != count:
        raise RuntimeError(f"Could only generate {len(accepted)}/{count} unique valid {split} examples for {domain['id']}")
    return accepted


def prepare(cfg, teacher):
    destination = Path(cfg["paths"]["output"]) / "prepared.json"
    seed = records(cfg["paths"]["seed"])
    fingerprint = digest([cfg["fingerprint"], seed])
    if destination.exists():
        prepared = read_json(destination)
        if prepared["fingerprint"] != fingerprint:
            raise ValueError("Prepared data/config/context changed; use a new output directory")
        return prepared
    schema = infer_schema(seed[0])
    fmt = cfg["format"]
    unique = {}
    for record in seed:
        validate(record, schema)
        if any(k not in record for k in fmt["input_fields"] + [fmt["answer_field"]]):
            raise ValueError("Configured input/answer fields are missing from seed records")
        key = input_id(record, fmt["input_fields"])
        if key in unique and unique[key][fmt["answer_field"]] != record[fmt["answer_field"]]:
            raise ValueError("Conflicting answers for the same seed input")
        unique[key] = record
    seed = list(unique.values())
    samples = random.Random(cfg["pipeline"]["seed"]).sample(seed, min(64, len(seed)))
    domains = teacher.discover(samples)
    check_domains(domains)
    assignments = {}
    items = [{"id": key, "record": value} for key, value in unique.items()]
    for batch in chunks(items, cfg["teacher"]["batch_size"]):
        new_domains, assigned = teacher.assign(domains, batch)
        domains.extend(new_domains)
        check_domains(domains)
        if len(assigned) != len(batch) or {x["id"] for x in assigned} != {x["id"] for x in batch}:
            raise ValueError("Every seed must be assigned to exactly one domain")
        if any(x["domain_id"] not in {d["id"] for d in domains} for x in assigned):
            raise ValueError("Unknown domain in seed assignments")
        assignments.update({x["id"]: x["domain_id"] for x in assigned})
    order = teacher.order(domains)
    if len(order) != len(domains) or set(order) != {d["id"] for d in domains}:
        raise ValueError("Domain order must contain all IDs exactly once")
    state = {"fingerprint": fingerprint, "config_fingerprint": cfg["fingerprint"], "schema": schema,
             "format_examples": samples[:3], "domains": {}, "examples": {}, "order": order,
             "stage": 0, "seen_examples": 0, "seen_tokens": 0, "events": [], "dataset": [],
             "complete": False, "rng": None}
    for d in domains:
        state["domains"][d["id"]] = {**d, "seed_ids": [], "pool": [], "archive": [], "validation": [],
            "status": "unseen", "history": [], "gains": [], "reference": None,
            "forgotten_checks": 0, "exit_reason": None}
    for key, record in unique.items():
        domain = state["domains"][assignments[key]]
        identifier = register(state, cfg, record, domain["id"], "train", "seed")
        domain["seed_ids"].append(identifier)
    rng = random.Random(cfg["pipeline"]["seed"])
    for key in order:
        d = state["domains"][key]
        d["pool"] = rng.sample(d["seed_ids"], min(2*cfg["pipeline"]["n"], len(d["seed_ids"])))
        d["validation"] = generate(state, cfg, teacher, d, cfg["pipeline"]["validation_size"], "validation", "prepare")
    atomic_json(destination, state)
    return state


class Engine:
    def __init__(self, cfg, teacher, student):
        self.cfg, self.teacher, self.student = cfg, teacher, student
        self.p = cfg["pipeline"]
        self.root = Path(cfg["paths"]["output"]) / "adaptive"
        self.rng = random.Random(self.p["seed"])
        self.state = None

    def latest(self):
        pointer = self.root / "latest.json"
        return self.root / read_json(pointer)["checkpoint"] if pointer.exists() else None

    def save(self):
        self.state["rng"] = self.rng.getstate()
        directory = self.root / "checkpoints" / f"{self.state['stage']:06d}-{uuid.uuid4().hex}"
        pending = directory.with_name("." + directory.name)
        pending.mkdir(parents=True)
        self.student.save(pending / "student")
        atomic_json(pending / "state.json", self.state)
        pending.rename(directory)
        atomic_json(self.root / "latest.json", {"checkpoint": str(directory.relative_to(self.root))})

    def initialize(self, prepared):
        latest = self.latest()
        if latest:
            self.state = read_json(latest / "state.json")
            if self.state["fingerprint"] != prepared["fingerprint"]:
                raise ValueError("Checkpoint does not match prepared data")
            self.student.load(latest / "student")
            self.rng.setstate(tuples(self.state["rng"]))
            return
        self.state = copy.deepcopy(prepared)
        # Initial weights and optimizer state are immutable inputs to the baseline.
        self.student.save(self.root / "initial-student")
        active = self.state["order"][:self.p["k"]]
        for key in active:
            d = self.state["domains"][key]
            d["status"] = "active"
            self.fill_pool(d)
            d["history"].append(self.measure(d))
        quotas = allocate(self.p["n"], dict.fromkeys(active, 1), self.rng)
        for key, count in quotas.items():
            self.state["dataset"].extend(self.rng.sample(self.state["domains"][key]["pool"], count))
        self.save()

    def rows(self, ids):
        return [self.state["examples"][key]["record"] for key in ids]

    def fill_pool(self, domain):
        domain["pool"].extend(generate(self.state, self.cfg, self.teacher, domain,
            2*self.p["n"] - len(domain["pool"]), "train", f"stage-{self.state['stage']}"))

    def judge(self, domain, ids):
        output = []
        for batch in chunks(ids, self.cfg["teacher"]["batch_size"]):
            rows = self.rows(batch)
            predictions = self.student.predict(rows)
            if len(predictions) != len(rows):
                raise ValueError("Student prediction count mismatch")
            verdicts = self.teacher.judge(catalog(domain), rows, predictions)
            if len(verdicts) != len(rows) or any(type(v) is not bool for v in verdicts):
                raise ValueError("Judge must return one boolean per record")
            output.extend(verdicts)
        return output

    def measure(self, domain):
        losses = self.student.losses(self.rows(domain["validation"]))
        if len(losses) != len(domain["validation"]) or any(not math.isfinite(v) or v < 0 for v in losses):
            raise ValueError("Student must return one finite nonnegative validation loss per example")
        correct = self.judge(domain, domain["validation"])
        return {"stage": self.state["stage"], "loss": sum(losses)/len(losses),
                "accuracy": sum(correct)/len(correct), "correct": correct}

    def threshold(self, before, after):
        return delta(before["correct"], after["correct"], self.p["delta"], self.p["delta_floor"])

    def drop_learned(self, domain):
        # Deliberate provisional Q6 choice: check the whole participating pool.
        verdicts = self.judge(domain, domain["pool"])
        retained = []
        for key, correct in zip(domain["pool"], verdicts):
            example = self.state["examples"][key]
            consecutive = example["last_checked"] == self.state["stage"] - 1
            example["correct_streak"] = (example["correct_streak"] + 1 if consecutive else 1) if correct else 0
            example["last_checked"] = self.state["stage"]
            if example["correct_streak"] >= 2:
                if key not in domain["archive"]:
                    domain["archive"].append(key)
            else:
                retained.append(key)
        domain["pool"] = retained

    def step(self):
        state, domains = self.state, self.state["domains"]
        if state["complete"]:
            return
        active = [key for key in state["order"] if domains[key]["status"] == "active"]
        participating = [key for key in state["order"] if domains[key]["status"] in ("active", "recovering")]
        if not state["dataset"]:
            raise ValueError("No training data; N may be too small for a replay quota")
        before = {key: domains[key]["history"][-1] for key in active}
        state["stage"] += 1
        actual = {}
        for _ in range(self.p["steps"]):
            selected = self.rng.choices(state["dataset"], k=self.p["batch_size"])
            state["seen_tokens"] += self.student.train_batch(self.rows(selected))
            state["seen_examples"] += len(selected)
            for key in selected:
                domain_id = state["examples"][key]["domain_id"]
                actual[domain_id] = actual.get(domain_id, 0) + 1
        for d in domains.values():
            if d["status"] != "unseen":
                d["history"].append(self.measure(d))
        weights = progress_weights({k: v["loss"] for k, v in before.items()},
                                   {k: domains[k]["history"][-1]["loss"] for k in active}) if active else {}
        for key in participating:
            self.drop_learned(domains[key])
        exited, returned = [], []
        for key in active:
            d, after = domains[key], domains[key]["history"][-1]
            d["gains"].append({"gain": after["accuracy"] - before[key]["accuracy"],
                               "delta": self.threshold(before[key], after)})
            mastered = after["accuracy"] >= self.p["exit_accuracy"]
            plateau = len(d["gains"]) >= 2 and all(x["gain"] < x["delta"] for x in d["gains"][-2:])
            if mastered or plateau:
                d["status"], d["reference"] = "passed", copy.deepcopy(after)
                d["exit_reason"] = "accuracy" if mastered else "plateau"
                d["forgotten_checks"] = 0
                exited.append(key)
        for key, d in domains.items():
            if d["status"] not in ("passed", "recovering") or key in exited:
                continue
            current = d["history"][-1]
            threshold = self.threshold(d["reference"], current)
            drop = d["reference"]["accuracy"] - current["accuracy"]
            if d["status"] == "recovering":
                if drop < threshold:
                    d["status"], d["forgotten_checks"] = "passed", 0
                continue
            d["forgotten_checks"] = d["forgotten_checks"] + 1 if drop > threshold else 0
            if d["forgotten_checks"] >= 2:
                d["status"] = "recovering"
                returned.append(key)
                # Restore previously mastered examples first; preserve the full archive.
                d["pool"] = list(dict.fromkeys(d["archive"] + d["pool"]))[:2*self.p["n"]]
                for identifier in d["pool"]:
                    state["examples"][identifier]["correct_streak"] = 0
                    state["examples"][identifier]["last_checked"] = -1
        active = [key for key in state["order"] if domains[key]["status"] == "active"]
        newcomers = []
        for key in state["order"]:
            if len(active) >= self.p["k"]:
                break
            d = domains[key]
            if d["status"] == "unseen":
                d["status"] = "active"
                d["history"].append(self.measure(d))
                active.append(key)
                newcomers.append(key)
        recovery = {}
        for key, d in domains.items():
            if d["status"] == "recovering":
                current = d["history"][-1]
                recovery[key] = (d["reference"]["accuracy"] - current["accuracy"], self.threshold(d["reference"], current))
        state["complete"] = all(d["status"] == "passed" for d in domains.values())
        state["dataset"] = []
        quotas = active_quotas(self.p["n"], active, newcomers, weights, self.rng) if active else {}
        quotas.update(replay_quotas(self.p["n"], recovery, self.rng))
        for key, count in quotas.items():
            d = domains[key]
            self.fill_pool(d)
            scores = self.student.scores(self.rows(d["pool"]))
            if len(scores) != len(d["pool"]) or any(not math.isfinite(v) for v in scores):
                raise ValueError("Student must return a finite C for each pool example")
            for identifier, score in zip(d["pool"], scores):
                state["examples"][identifier]["c"] = float(score)
            state["dataset"].extend(window(d["pool"], count, d["history"][-1]["accuracy"], state["examples"]))
        state["events"].append({"stage": state["stage"], "actual_presentations": actual, "weights": weights,
            "next_quotas": quotas, "exited": exited, "introduced": newcomers, "returned": returned,
            "seen_examples": state["seen_examples"], "seen_tokens": state["seen_tokens"],
            "accuracy": {k: d["history"][-1]["accuracy"] for k, d in domains.items() if d["history"]}})
        self.save()

    def run(self, prepared, max_stages=None):
        self.initialize(prepared)
        performed = 0
        while not self.state["complete"] and (max_stages is None or performed < max_stages):
            self.step()
            performed += 1
            print(f"stage={self.state['stage']} seen={self.state['seen_examples']} complete={self.state['complete']}", flush=True)
        return self.state


def load_state(cfg):
    root = Path(cfg["paths"]["output"]) / "adaptive"
    directory = root / read_json(root / "latest.json")["checkpoint"]
    state = read_json(directory / "state.json")
    if state["config_fingerprint"] != cfg["fingerprint"]:
        raise ValueError("Run configuration/context changed; use the original config or a new run")
    return state, directory


def baseline(cfg, student, max_steps=None):
    """Uniform all-domain comparator, sharing the completed run's candidate bank."""
    state, _ = load_state(cfg)
    if not state["complete"]:
        raise ValueError("Finish the adaptive run before constructing the shared-candidate baseline")
    root = Path(cfg["paths"]["output"])
    target = root / "baseline"
    pools = {k: [] for k in state["order"]}
    for item in state["examples"].values():
        if item["split"] == "train":
            pools[item["domain_id"]].append(item["record"])
    rng = random.Random(cfg["pipeline"]["seed"])
    pointer = target / "latest.json"
    if pointer.exists():
        directory = target / read_json(pointer)["checkpoint"]
        progress = read_json(directory / "state.json")
        if progress["adaptive_fingerprint"] != digest(state):
            raise ValueError("Adaptive source has changed since baseline initialization")
        student.load(directory / "student")
        rng.setstate(tuples(progress["rng"]))
    else:
        student.load(root / "adaptive" / "initial-student")
        progress = {"adaptive_fingerprint": digest(state), "seen_examples": 0, "seen_tokens": 0,
                    "target_examples": state["seen_examples"], "complete": False}
    steps = 0
    while progress["seen_examples"] < progress["target_examples"] and (max_steps is None or steps < max_steps):
        count = min(cfg["pipeline"]["batch_size"], progress["target_examples"] - progress["seen_examples"])
        # Equal domain probability; random sample within each domain, with replacement.
        batch = [rng.choice(pools[rng.choice(state["order"])]) for _ in range(count)]
        progress["seen_tokens"] += student.train_batch(batch)
        progress["seen_examples"] += count
        steps += 1
    progress["rng"] = rng.getstate()
    progress["complete"] = progress["seen_examples"] == progress["target_examples"]
    directory = target / "checkpoints" / uuid.uuid4().hex
    pending = directory.with_name("." + directory.name)
    student.save(pending / "student")
    atomic_json(pending / "state.json", progress)
    pending.rename(directory)
    atomic_json(pointer, {"checkpoint": str(directory.relative_to(target))})
    return progress


def evaluate_test(cfg, teacher, student, test_path, method="adaptive"):
    # This function is the only place test data enters; it never changes scheduler state.
    state, adaptive_directory = load_state(cfg)
    if not state["complete"]:
        raise ValueError("Test evaluation is available only after all domains have exited")
    root = Path(cfg["paths"]["output"])
    directory = adaptive_directory
    if method == "baseline":
        directory = root / "baseline" / read_json(root / "baseline" / "latest.json")["checkpoint"]
        if not read_json(directory / "state.json")["complete"]:
            raise ValueError("Baseline has not finished its matched presentation budget")
    student.load(directory / "student")
    test = records(test_path)
    identifiers = [input_id(row, cfg["format"]["input_fields"]) for row in test]
    if len(set(identifiers)) != len(identifiers) or any(key in state["examples"] for key in identifiers):
        raise ValueError("Test contains duplicate inputs or overlaps with train/validation")
    correct = []
    task = {"id": "external_test", "name": "External test", "rule": cfg["context"]["goal"]}
    for batch in chunks(test, cfg["teacher"]["batch_size"]):
        for row in batch:
            validate(row, state["schema"])
        verdicts = teacher.judge(task, batch, student.predict(batch))
        if len(verdicts) != len(batch) or any(type(v) is not bool for v in verdicts):
            raise ValueError("Invalid test verdicts")
        correct.extend(verdicts)
    result = {"method": method, "test_hash": digest(test), "count": len(test),
              "accuracy": sum(correct)/len(correct), "correct": correct}
    atomic_json(root / f"test-{method}.json", result)
    return result
