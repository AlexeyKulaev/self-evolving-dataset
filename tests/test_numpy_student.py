import copy
import importlib.util
import tempfile
import unittest

from evolving_dataset.config import DEFAULTS
from evolving_dataset.student import NumpyRNN


@unittest.skipUnless(importlib.util.find_spec("numpy"), "Install the demo extra to test the numerical learner")
class NumericalTests(unittest.TestCase):
    def setUp(self):
        self.cfg = copy.deepcopy(DEFAULTS)
        self.cfg["student"].update(hidden_size=8, learning_rate=.1, max_length=128, max_new_tokens=2)
        self.student = NumpyRNN(self.cfg)
        self.record = {"question": "2+3", "answer": "5"}

    def test_gradient_matches_finite_differences(self):
        _, gradient, _ = self.student.objective(self.record, True)
        for key, index in [("E", (50, 0)), ("U", (0,1)), ("W", (53,2)), ("b", (1,)), ("c", (53,))]:
            array = self.student.params[key]
            original, eps = float(array[index]), 1e-5
            array[index] = original + eps
            high = self.student.objective(self.record)
            array[index] = original - eps
            low = self.student.objective(self.record)
            array[index] = original
            self.assertAlmostEqual((high-low)/(2*eps), gradient[key][index], places=6)

    def test_training_reduces_loss_and_checkpoint_roundtrip(self):
        before = self.student.losses([self.record])[0]
        for _ in range(20):
            self.student.train_batch([self.record])
        self.assertLess(self.student.losses([self.record])[0], before)
        with tempfile.TemporaryDirectory() as directory:
            self.student.save(directory)
            other = NumpyRNN(self.cfg)
            other.load(directory)
            self.assertEqual(self.student.losses([self.record]), other.losses([self.record]))
            self.assertEqual(self.student.predict([self.record]), other.predict([self.record]))

    def test_c_does_not_see_answer_or_context(self):
        score = self.student.scores([self.record])[0]
        other = {**self.record, "answer": "A completely different target"}
        self.assertEqual(score, self.student.scores([other])[0])


if __name__ == "__main__":
    unittest.main()
