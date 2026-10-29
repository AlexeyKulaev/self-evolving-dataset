import copy
import json
import math
import os
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from evolving_dataset.config import DEFAULTS, load_config
from evolving_dataset.engine import Engine, baseline, evaluate_test, load_state, prepare
from evolving_dataset.io import atomic_json, digest, infer_schema, input_id, read_json, validate
from evolving_dataset.policy import active_quotas, allocate, angle_concentration, delta, progress_weights, replay_quotas, window
from evolving_dataset.teacher import CodexTeacher, CodexTransport, DemoTeacher, verdicts


def fixture(root):
    cfg = copy.deepcopy(DEFAULTS)
    seed = Path(root) / "seed.jsonl"
    seed.write_text('\n'.join(json.dumps({"question": question, "answer": answer}) for question, answer in
                    [("1+2", "3"), ("7-2", "5"), ("2*3", "6")]) + '\n')
    cfg["paths"] = {"seed": str(seed), "output": str(Path(root) / "run")}
    cfg["context"] = {"goal": "GOAL VERBATIM: arithmetic modulo 10", "architecture": "ARCH VERBATIM: fixed RNN"}
    cfg["pipeline"].update(n=20, k=2, steps=1, batch_size=2, validation_size=4, delta=.05)
    cfg["teacher"].update(backend="demo", batch_size=8, retries=1)
    cfg["fingerprint"] = digest(cfg)
    return cfg


class ScheduledStudent:
    """A controlled observation fixture for lifecycle tests, not a numerical model."""
    def __init__(self):
        self.step = 0
    def train_batch(self, records):
        self.step += 1
        return sum(len(r["question"]) for r in records)
    def losses(self, records):
        return [1 / (self.step + 1)] * len(records)
    def predict(self, records):
        return [r["answer"] if "+" in r["question"] and self.step in (1, 4) else "wrong" for r in records]
    def scores(self, records):
        return [len(r["question"]) / 10 for r in records]
    def save(self, directory):
        atomic_json(Path(directory) / "fake.json", {"step": self.step})
    def load(self, directory):
        self.step = read_json(Path(directory) / "fake.json")["step"]


class PolicyTests(unittest.TestCase):
    def test_document_loss_example(self):
        weights = progress_weights({"a": 1, "b": .8, "c": 1}, {"a": .5, "b": .2, "c": .8})
        self.assertAlmostEqual(weights["b"], 4/7.25)
        quotas = allocate(10000, weights, random.Random(1))
        self.assertEqual(quotas, {"a": 2759, "b": 5517, "c": 1724})

    def test_zero_losses_are_finite(self):
        self.assertEqual(progress_weights({"x": 0}, {"x": 0}), {"x": 1})
        with self.assertRaises(ValueError):
            progress_weights({"x": 1}, {"x": float("nan")})

    def test_newcomers_share_half(self):
        q = active_quotas(101, ["old", "new1", "new2"], ["new1", "new2"], {"old": .4}, random.Random(1))
        self.assertEqual(q, {"new1": 25, "new2": 25, "old": 51})
        self.assertEqual(active_quotas(100, ["new"], ["new"], {}, random.Random(1)), {"new": 100})

    def test_replay_cap(self):
        self.assertEqual(replay_quotas(100, {"a": (.1, .08)}, random.Random(1)), {"a": 10})
        self.assertEqual(replay_quotas(100, {"a": (.3, .08)}, random.Random(1)), {"a": 20})
        self.assertEqual(sum(replay_quotas(100, {"a": (.3,.1), "b": (.2,.1), "c": (.1,.05)}, random.Random(1)).values()), 20)

    def test_window_boundary(self):
        pool = [str(i) for i in range(20)]
        examples = {key: {"c": 20-int(key)} for key in pool}
        self.assertEqual(window(pool, 5, .3, examples), ["6", "7", "8", "9", "10"])
        self.assertEqual(window(pool, 5, .8, examples), ["15", "16", "17", "18", "19"])
        self.assertEqual(window(pool, 0, 1, examples), [])

    def test_c_matches_explicit_pairs(self):
        vectors = [[1, 2], [3, -4], [-1, 2]]
        pairs = []
        for i in range(3):
            for j in range(i+1, 3):
                pairs.append(sum(x*y for x,y in zip(vectors[i], vectors[j])) /
                             math.sqrt(sum(x*x for x in vectors[i])*sum(x*x for x in vectors[j])))
        self.assertAlmostEqual(angle_concentration(vectors), sum(pairs)/3)
        self.assertAlmostEqual(angle_concentration([[1,0], [-1,0]]), -1)
        self.assertEqual(angle_concentration([[1,0]]), 0)
        with self.assertRaises(ValueError):
            angle_concentration([[0, 0]])

    def test_delta_uses_pairs(self):
        self.assertEqual(delta([True, False], [True, False]), .01)
        self.assertGreater(delta([True]*4, [False, True, True, True]), .01)
        self.assertEqual(delta([True], [False], fixed=.04), .04)


class LifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.cfg = fixture(self.temp.name)
        self.teacher = DemoTeacher(self.cfg)
        self.prepared = prepare(self.cfg, self.teacher)

    def test_seed_partition_and_frozen_holdout(self):
        self.assertEqual(len(self.prepared["domains"]), 3)
        seed_ids = [i for d in self.prepared["domains"].values() for i in d["seed_ids"]]
        validation = [i for d in self.prepared["domains"].values() for i in d["validation"]]
        self.assertEqual(len(seed_ids), 3)
        self.assertEqual(len(validation), 12)
        self.assertFalse(set(seed_ids) & set(validation))
        teacher = SimpleNamespace(discover=lambda *_: self.fail("Must reuse frozen preparation"))
        self.assertEqual(prepare(self.cfg, teacher), self.prepared)

    def test_forgetting_recovery_terminal_state_and_new_weights(self):
        engine = Engine(self.cfg, self.teacher, ScheduledStudent())
        engine.initialize(self.prepared)
        self.assertEqual(len(engine.state["dataset"]), 20)
        engine.step()
        self.assertEqual(engine.state["events"][-1]["next_quotas"], {"multiplication": 10, "subtraction": 10})
        engine.step()
        self.assertEqual(engine.state["domains"]["addition"]["forgotten_checks"], 1)
        engine.step()
        self.assertFalse(engine.state["complete"])
        self.assertEqual(engine.state["domains"]["addition"]["status"], "recovering")
        self.assertEqual(len(engine.state["dataset"]), 4)
        self.assertEqual(engine.state["domains"]["multiplication"]["exit_reason"], "plateau")
        engine.step()
        self.assertTrue(engine.state["complete"])
        self.assertEqual(engine.state["seen_examples"], 8)
        self.assertEqual(engine.state["dataset"], [])

    def test_resume_is_identical_to_uninterrupted(self):
        reference = Engine(self.cfg, self.teacher, ScheduledStudent()).run(self.prepared)
        other_cfg = copy.deepcopy(self.cfg)
        other_cfg["paths"]["output"] = str(Path(self.temp.name) / "resumed")
        engine = Engine(other_cfg, self.teacher, ScheduledStudent())
        engine.run(self.prepared, max_stages=2)
        resumed = Engine(other_cfg, self.teacher, ScheduledStudent()).run(self.prepared)
        # JSON serialization turns PRNG tuples into lists; normalize both representations.
        self.assertEqual(json.loads(json.dumps(reference)), json.loads(json.dumps(resumed)))

    def test_failure_does_not_publish_partial_stage(self):
        engine = Engine(self.cfg, self.teacher, ScheduledStudent())
        engine.initialize(self.prepared)
        with patch.object(self.teacher, "judge", side_effect=RuntimeError("interrupted")):
            with self.assertRaises(RuntimeError):
                engine.step()
        state, _ = load_state(self.cfg)
        self.assertEqual(state["stage"], 0)
        recovered = Engine(self.cfg, self.teacher, ScheduledStudent())
        recovered.initialize(self.prepared)
        self.assertEqual(recovered.student.step, 0)
        self.assertEqual(recovered.state["seen_examples"], 0)

    def test_whole_pool_mastery_archives_after_two_consecutive_checks(self):
        engine = Engine(self.cfg, self.teacher, ScheduledStudent())
        engine.initialize(self.prepared)
        domain = engine.state["domains"]["addition"]
        original = domain["pool"].copy()
        engine.student.step = 1
        engine.state["stage"] = 1
        engine.drop_learned(domain)
        self.assertEqual(domain["pool"], original)
        engine.state["stage"] = 2
        engine.drop_learned(domain)
        self.assertEqual(domain["pool"], [])
        self.assertEqual(domain["archive"], original)
        self.assertTrue(all(i in engine.state["examples"] for i in original))

    def test_baseline_matches_presentations_and_resumes(self):
        adaptive = Engine(self.cfg, self.teacher, ScheduledStudent()).run(self.prepared)
        partial = baseline(self.cfg, ScheduledStudent(), max_steps=1)
        self.assertFalse(partial["complete"])
        student = ScheduledStudent()
        result = baseline(self.cfg, student)
        self.assertTrue(result["complete"])
        self.assertEqual(result["seen_examples"], adaptive["seen_examples"])
        self.assertEqual(student.step, 4)  # initial weights, not the final adaptive checkpoint

    def test_test_set_not_used_until_finish_and_overlap_rejected(self):
        engine = Engine(self.cfg, self.teacher, ScheduledStudent())
        engine.run(self.prepared, max_stages=1)
        with self.assertRaisesRegex(ValueError, "only after"):
            evaluate_test(self.cfg, self.teacher, ScheduledStudent(), "/does/not/exist")
        engine.run(self.prepared)
        with self.assertRaisesRegex(ValueError, "overlaps"):
            evaluate_test(self.cfg, self.teacher, ScheduledStudent(), self.cfg["paths"]["seed"])

    def test_changed_seed_refuses_existing_preparation(self):
        Path(self.cfg["paths"]["seed"]).write_text('{"question":"8+2","answer":"0"}\n')
        with self.assertRaisesRegex(ValueError, "changed"):
            prepare(self.cfg, self.teacher)

    def test_duplicate_generation_fails_without_cross_split_leakage(self):
        class DuplicateTeacher(DemoTeacher):
            def generate(self, domain, seeds, count, split, nonce, schema):
                return [seeds[0]] * count
        cfg = copy.deepcopy(self.cfg)
        cfg["paths"]["output"] = str(Path(self.temp.name) / "duplicates")
        with self.assertRaisesRegex(RuntimeError, "unique valid"):
            prepare(cfg, DuplicateTeacher(cfg))


class FormatAndTransportTests(unittest.TestCase):
    def test_nested_table_format(self):
        example = {"question": "sum", "table": [[1,2], [3,4]], "answer": 10}
        schema = infer_schema(example)
        validate({"question": "sum", "table": [[5,6]], "answer": 11}, schema)
        with self.assertRaises(ValueError):
            validate({"question": "sum", "table": [[5,6]], "answer": "11"}, schema)
        with self.assertRaises(ValueError):
            validate({"question": "sum", "table": [[5,6]], "answer": 11.5}, schema)
        with self.assertRaises(ValueError):
            validate({**example, "domain": "extra"}, schema)

    def test_judge_indices_must_be_complete(self):
        with self.assertRaises(ValueError):
            verdicts([{"index": 0, "correct": True}]*2, 2, "correct")

    def test_codex_command_context_auth_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = fixture(root)
            transport = CodexTransport(cfg)
            calls = []
            def run(command, **kwargs):
                calls.append(command)
                self.assertNotIn("OPENAI_API_KEY", kwargs["env"])
                self.assertNotIn("CODEX_API_KEY", kwargs["env"])
                if command[1:3] == ["login", "status"]:
                    return SimpleNamespace(returncode=0, stdout="Logged in using ChatGPT", stderr="")
                self.assertIn("gpt-6.1-sol", command)
                self.assertIn('service_tier="fast"', command)
                self.assertIn('model_reasoning_effort="xhigh"', command)
                self.assertIn("read-only", command)
                self.assertIn(cfg["context"]["goal"], kwargs["input"])
                self.assertIn(cfg["context"]["architecture"], kwargs["input"])
                atomic_json(command[command.index("--output-last-message")+1], {"ok": True})
                return SimpleNamespace(returncode=0, stdout="ignored events", stderr="")
            schema = infer_schema({"ok": True})
            with patch.dict(os.environ, {"OPENAI_API_KEY": "not-a-real-key", "CODEX_API_KEY": "not-a-real-key"}), patch("subprocess.run", side_effect=run):
                self.assertEqual(transport.request("probe", {}, schema), {"ok": True})
                self.assertEqual(transport.request("probe", {}, schema), {"ok": True})
            self.assertEqual(len(calls), 2)

    def test_api_auth_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            transport = CodexTransport(fixture(root))
            with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="Logged in using an API key", stderr="")):
                with self.assertRaisesRegex(RuntimeError, "ChatGPT"):
                    transport.check_auth()

    def test_silent_fast_fallback_is_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            transport = CodexTransport(fixture(root))
            transport.auth_checked = True
            with patch("subprocess.run", return_value=SimpleNamespace(returncode=0, stdout="", stderr=
                       "warning: Configured service tier priority will be omitted from requests")):
                with self.assertRaisesRegex(RuntimeError, "silent tier fallback"):
                    transport.request("probe", {}, infer_schema({"ok": True}))


if __name__ == "__main__":
    unittest.main()
