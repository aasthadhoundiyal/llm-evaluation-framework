import unittest
from eval_framework.evaluators.deterministic import (
    DeterministicEvaluator,
    compute_exact_match,
    compute_token_f1,
)


class TestDeterministicEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = DeterministicEvaluator()

    def test_exact_match_normalized(self):
        self.assertTrue(compute_exact_match("Denver Broncos", "The Denver Broncos"))
        self.assertTrue(compute_exact_match("1969", "1969"))
        self.assertFalse(compute_exact_match("Carolina Panthers", "Denver Broncos"))

    def test_token_f1(self):
        f1 = compute_token_f1("The Denver Broncos won Super Bowl 50", "Denver Broncos")
        self.assertGreater(f1, 0.4)
        self.assertEqual(compute_token_f1("Denver Broncos", "Denver Broncos"), 1.0)

    def test_keyword_assertions(self):
        sample = {
            "id": "T-1",
            "prompt": "Explain Rayleigh scattering",
            "expected_keywords": ["scattering", "rayleigh"],
            "forbidden_keywords": ["green sky"],
        }
        res_pass = self.evaluator.evaluate(sample, "Rayleigh scattering causes the blue appearance.")
        self.assertTrue(res_pass.passed)

        res_fail = self.evaluator.evaluate(sample, "The sky looks blue because of green sky.")
        self.assertFalse(res_fail.passed)


if __name__ == "__main__":
    unittest.main()
