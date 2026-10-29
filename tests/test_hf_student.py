import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

from evolving_dataset.config import DEFAULTS
from evolving_dataset.student import HuggingFaceStudent


AVAILABLE = all(importlib.util.find_spec(name) for name in ("torch", "transformers", "tokenizers"))


@unittest.skipUnless(AVAILABLE, "Install the hf extra to test the real Transformers adapter")
class HuggingFaceTests(unittest.TestCase):
    def setUp(self):
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        torch.set_num_threads(1)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        directory = Path(self.temp.name) / "initial"
        tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "[EOS]": 1, "2": 2, "+": 3,
            "3": 4, "5": 5, "Answer": 6, ":": 7, "7": 8}, unk_token="[UNK]"))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
        wrapped = PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="[UNK]", eos_token="[EOS]", pad_token="[EOS]")
        wrapped.save_pretrained(directory)
        torch.manual_seed(42)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=9, n_positions=64, n_embd=16, n_layer=1, n_head=2,
                               bos_token_id=1, eos_token_id=1, pad_token_id=1))
        model.save_pretrained(directory)
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["student"].update(model=str(directory), max_length=64, max_new_tokens=2, learning_rate=.01)
        self.student = HuggingFaceStudent(self.cfg)
        self.row = {"question": "2+3", "answer": "5"}

    def test_training_scores_and_generation_without_remote_weights(self):
        before = self.student.losses([self.row])[0]
        for _ in range(8):
            self.assertGreater(self.student.train_batch([self.row]), 0)
        self.assertLess(self.student.losses([self.row])[0], before)
        score = self.student.scores([self.row])[0]
        self.assertTrue(-1 <= score <= 1)
        self.assertEqual(score, self.student.scores([{**self.row, "answer": "7"}])[0])
        self.assertEqual(len(self.student.predict([self.row])), 1)

    def test_checkpoint_restores_optimizer_and_dropout_rng(self):
        import torch
        self.student.train_batch([self.row])
        directory = Path(self.temp.name) / "saved"
        self.student.save(directory)
        self.student.train_batch([self.row])
        expected = {k: v.clone() for k,v in self.student.model.state_dict().items()}
        other = HuggingFaceStudent(self.cfg)
        other.load(directory)
        other.train_batch([self.row])
        for key, value in other.model.state_dict().items():
            self.assertTrue(torch.equal(value, expected[key]), key)

    def test_long_inputs_are_not_silently_truncated(self):
        with self.assertRaisesRegex(ValueError, "max_length"):
            self.student.scores([{**self.row, "question": "2 " * 100}])


if __name__ == "__main__":
    unittest.main()
