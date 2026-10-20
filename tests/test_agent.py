import unittest
from eval_framework.evaluators.agent import AgentEvaluator, TrajectoryEvaluator, TrajectoryStep


class TestAgentEvaluator(unittest.TestCase):
    def setUp(self):
        self.traj_eval = TrajectoryEvaluator(max_allowed_steps=5)
        self.agent_eval = AgentEvaluator()

    def test_trajectory_metrics(self):
        steps = [
            TrajectoryStep(1, "Check balance", "get_account_balance", {"account_id": "A1"}, {"balance": 500}),
            TrajectoryStep(2, "Transfer funds", "execute_fund_transfer", {"from": "A1", "amount": 100}, {"status": "ok"}),
        ]
        res = self.traj_eval.evaluate_trajectory(
            steps,
            expected_tools=["get_account_balance", "execute_fund_transfer"],
            goal_achieved=True,
        )
        self.assertTrue(res["passed"])
        self.assertEqual(res["loops_detected"], 0)
        self.assertEqual(res["tool_recall"], 1.0)

    def test_loop_detection(self):
        # Repeated identical tool call with same arguments
        steps = [
            TrajectoryStep(1, "Check balance", "get_balance", {"id": "A1"}, 500),
            TrajectoryStep(2, "Check balance again", "get_balance", {"id": "A1"}, 500),
        ]
        res = self.traj_eval.evaluate_trajectory(steps, expected_tools=["get_balance"], goal_achieved=True)
        self.assertGreater(res["loops_detected"], 0)


if __name__ == "__main__":
    unittest.main()
