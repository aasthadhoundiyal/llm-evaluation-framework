import unittest
from eval_framework.client import ModelClient
from eval_framework.evaluators.judge import LLMJudgeEvaluator


class TestLLMJudgeEvaluator(unittest.TestCase):
    def setUp(self):
        self.client = ModelClient(provider="mock", model="gpt-4o-mini")
        self.judge = LLMJudgeEvaluator(client=self.client)

    def test_rubric_scoring(self):
        sample = {
            "id": "J-1",
            "prompt": "How do I dispute a fraudulent transaction?",
            "ideal_answer": "Contact fraud department or dispute in mobile app.",
        }
        res = self.judge.evaluate(
            sample,
            "You can dispute transactions through our mobile app or by calling our fraud hotline.",
        )
        self.assertTrue(res.passed)
        self.assertGreater(len(res.metrics), 0)


if __name__ == "__main__":
    unittest.main()
