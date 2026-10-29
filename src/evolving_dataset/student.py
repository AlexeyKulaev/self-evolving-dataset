"""Numerical learner interface. Semantic judging belongs to the teacher."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Protocol

from .io import input_text, text_value
from .policy import angle_concentration


class Student(Protocol):
    def train_batch(self, records: list[dict]) -> int: ...  # actual processed tokens
    def losses(self, records: list[dict]) -> list[float]: ...
    def predict(self, records: list[dict]) -> list[str]: ...
    def scores(self, records: list[dict]) -> list[float]: ...
    def save(self, directory: Path) -> None: ...
    def load(self, directory: Path) -> None: ...


class NumpyRNN:
    """Small, actually trainable byte-level tanh RNN for CPU integration demos.

    SFT with teacher forcing, answer-only cross entropy, full BPTT and SGD.
    It is a demonstration architecture, not a substitute for architecture.md.
    """
    BOS, SEP, EOS, VOCAB = 256, 257, 258, 259

    def __init__(self, cfg):
        import numpy as np
        self.np, self.cfg = np, cfg
        self.opts, self.fmt = cfg["student"], cfg["format"]
        rng = np.random.default_rng(cfg["pipeline"]["seed"])
        h = self.opts["hidden_size"]
        self.params = {"E": rng.normal(0, .1, (self.VOCAB, h)),
                       "U": rng.normal(0, .1, (h, h)), "W": rng.normal(0, .1, (self.VOCAB, h)),
                       "b": np.zeros(h), "c": np.zeros(self.VOCAB)}

    def question(self, record):
        data = list(input_text(record, self.fmt["input_fields"]).encode("utf-8"))
        if not data or len(data) > self.opts["max_length"]:
            raise ValueError("Input empty or exceeds max_length; refusing silent truncation")
        return data

    def sequence(self, record):
        prefix = [self.BOS] + self.question(record) + [self.SEP]
        answer = list(text_value(record[self.fmt["answer_field"]]).encode("utf-8")) + [self.EOS]
        if len(prefix) + len(answer) > self.opts["max_length"]:
            raise ValueError("Training sequence exceeds max_length")
        return prefix + answer, len(prefix) - 1

    def forward(self, tokens):
        np, p = self.np, self.params
        h = np.zeros_like(p["b"])
        states, probabilities = [], []
        for token in tokens:
            h = np.tanh(p["E"][token] + p["U"] @ h + p["b"])
            logits = p["W"] @ h + p["c"]
            exp = np.exp(logits - logits.max())
            states.append(h)
            probabilities.append(exp / exp.sum())
        return states, probabilities

    def objective(self, record, backward=False):
        np, p = self.np, self.params
        sequence, first = self.sequence(record)
        states, probs = self.forward(sequence[:-1])
        count = len(probs) - first
        loss = -sum(float(np.log(max(probs[t][sequence[t+1]], 1e-300)))
                    for t in range(first, len(probs))) / count
        if not backward:
            return loss
        gradients = {k: np.zeros_like(v) for k, v in p.items()}
        carry = np.zeros_like(p["b"])
        for t in range(len(probs)-1, -1, -1):
            dy = np.zeros(self.VOCAB)
            if t >= first:
                dy = probs[t].copy()
                dy[sequence[t+1]] -= 1
                dy /= count
            gradients["W"] += np.outer(dy, states[t])
            gradients["c"] += dy
            dh = (p["W"].T @ dy + carry) * (1 - states[t] ** 2)
            previous = states[t-1] if t else np.zeros_like(dh)
            gradients["U"] += np.outer(dh, previous)
            gradients["E"][sequence[t]] += dh
            gradients["b"] += dh
            carry = p["U"].T @ dh
        return loss, gradients, len(sequence)-1

    def train_batch(self, records):
        np = self.np
        gradients = {k: np.zeros_like(v) for k, v in self.params.items()}
        tokens = 0
        for record in records:
            _, g, n = self.objective(record, True)
            tokens += n
            for key in gradients:
                gradients[key] += g[key] / len(records)
        norm = np.sqrt(sum(float(np.sum(g*g)) for g in gradients.values()))
        scale = min(1.0, 5.0 / max(norm, 1e-12))
        for key in self.params:
            self.params[key] -= self.opts["learning_rate"] * scale * gradients[key]
        return tokens

    def losses(self, records):
        return [self.objective(record) for record in records]

    def predict(self, records):
        np, p = self.np, self.params
        answers = []
        for record in records:
            states, _ = self.forward([self.BOS] + self.question(record) + [self.SEP])
            h, generated = states[-1], []
            for _ in range(self.opts["max_new_tokens"]):
                token = int(np.argmax(p["W"] @ h + p["c"]))
                if token == self.EOS:
                    break
                if token < 256:
                    generated.append(token)
                h = np.tanh(p["E"][token] + p["U"] @ h + p["b"])
            answers.append(bytes(generated).decode("utf-8", errors="replace"))
        return answers

    def scores(self, records):
        return [angle_concentration(self.forward(self.question(r))[0]) for r in records]

    def save(self, directory):
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.np.savez(Path(directory) / "rnn.npz", **self.params)

    def load(self, directory):
        with self.np.load(Path(directory) / "rnn.npz", allow_pickle=False) as values:
            self.params = {key: values[key].copy() for key in self.params}


class HuggingFaceStudent:
    """Full-parameter causal-LM SFT. No arbitrary remote model code is executed."""
    def __init__(self, cfg):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.cfg = torch, cfg
        self.opts, self.fmt = cfg["student"], cfg["format"]
        if not self.opts["model"]:
            raise ValueError("Set student.model to the checkpoint described in architecture.md")
        torch.manual_seed(cfg["pipeline"]["seed"])
        self.tokenizer = AutoTokenizer.from_pretrained(self.opts["model"], trust_remote_code=False)
        self.model = AutoModelForCausalLM.from_pretrained(self.opts["model"], trust_remote_code=False).to(self.opts["device"])
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.opts["learning_rate"])

    def prompt_ids(self, record):
        return self.tokenizer.encode(input_text(record, self.fmt["input_fields"]) + "\nAnswer:\n", add_special_tokens=True)

    def encoded(self, record):
        prefix = self.prompt_ids(record)
        target = self.tokenizer.encode(text_value(record[self.fmt["answer_field"]]), add_special_tokens=False)
        if self.tokenizer.eos_token_id is not None:
            target.append(self.tokenizer.eos_token_id)
        ids = prefix + target
        if not target or len(ids) > self.opts["max_length"]:
            raise ValueError("Empty target or sequence exceeds max_length; refusing silent truncation")
        tensor = self.torch.tensor([ids], device=self.opts["device"])
        labels = tensor.clone()
        labels[:, :len(prefix)] = -100
        return tensor, labels

    def train_batch(self, records):
        self.model.train()
        self.optimizer.zero_grad()
        tokens = 0
        for record in records:
            ids, labels = self.encoded(record)
            loss = self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids), labels=labels).loss / len(records)
            loss.backward()
            tokens += ids.numel()
        self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return tokens

    def losses(self, records):
        self.model.eval()
        with self.torch.no_grad():
            return [float(self.model(input_ids=ids, attention_mask=self.torch.ones_like(ids), labels=labels).loss.item())
                    for ids, labels in (self.encoded(r) for r in records)]

    def predict(self, records):
        self.model.eval()
        answers = []
        with self.torch.no_grad():
            for record in records:
                ids = self.prompt_ids(record)
                if len(ids) + self.opts["max_new_tokens"] > self.opts["max_length"]:
                    raise ValueError("Prompt plus generation budget exceeds max_length")
                tensor = self.torch.tensor([ids], device=self.opts["device"])
                output = self.model.generate(input_ids=tensor, attention_mask=self.torch.ones_like(tensor),
                    max_new_tokens=self.opts["max_new_tokens"], do_sample=False,
                    pad_token_id=self.tokenizer.eos_token_id)
                answers.append(self.tokenizer.decode(output[0, len(ids):], skip_special_tokens=True))
        return answers

    def scores(self, records):
        self.model.eval()
        result = []
        with self.torch.no_grad():
            for record in records:
                # No answer, goal, system instructions, separator, or special tokens.
                ids = self.tokenizer.encode(input_text(record, self.fmt["input_fields"]), add_special_tokens=False)
                if not ids or len(ids) > self.opts["max_length"]:
                    raise ValueError("C input empty or exceeds max_length")
                output = self.model(input_ids=self.torch.tensor([ids], device=self.opts["device"]),
                                    output_hidden_states=True, use_cache=False)
                result.append(angle_concentration(output.hidden_states[-1][0].float().cpu().tolist()))
        return result

    def save(self, directory):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory)
        self.tokenizer.save_pretrained(directory)
        self.torch.save({"optimizer": self.optimizer.state_dict(), "rng": self.torch.get_rng_state(),
                        "cuda_rng": self.torch.cuda.get_rng_state_all() if self.torch.cuda.is_available() else []},
                       directory / "training.pt")

    def load(self, directory):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model = AutoModelForCausalLM.from_pretrained(directory, trust_remote_code=False).to(self.opts["device"])
        self.tokenizer = AutoTokenizer.from_pretrained(directory, trust_remote_code=False)
        self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=self.opts["learning_rate"])
        state = self.torch.load(Path(directory) / "training.pt", map_location="cpu", weights_only=True)
        self.optimizer.load_state_dict(state["optimizer"])
        self.torch.set_rng_state(state["rng"])
        if state["cuda_rng"] and self.torch.cuda.is_available():
            self.torch.cuda.set_rng_state_all(state["cuda_rng"])


def make_student(cfg):
    backend = cfg["student"]["backend"]
    if backend == "numpy":
        return NumpyRNN(cfg)
    if backend == "hf":
        return HuggingFaceStudent(cfg)
    if backend == "plugin":
        module, name = cfg["student"]["factory"].split(":")
        return getattr(importlib.import_module(module), name)(cfg)
    raise ValueError(f"Unknown student backend: {backend}")
