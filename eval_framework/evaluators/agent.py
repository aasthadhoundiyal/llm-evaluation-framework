"""
eval_framework.evaluators.agent
===============================
Autonomous Agent & Trajectory Evaluation.
Assesses multi-step agent behaviors, tool call accuracy, looping detection,
redundant action penalties, and goal fulfillment.
"""

from typing import Any, Dict, List, Optional
from eval_framework.evaluators.base import BaseEvaluator, EvaluationResult, MetricResult


class TrajectoryStep:
    """Represents a single step in an agent's trajectory."""

    def __init__(
        self,
        step_index: int,
        thought: str,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_result: Any,
    ):
        self.step_index = step_index
        self.thought = thought
        self.tool_name = tool_name
        self.tool_args = tool_args
        self.tool_result = tool_result

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step_index,
            "thought": self.thought,
            "tool": self.tool_name,
            "args": self.tool_args,
            "result": self.tool_result,
        }


class TrajectoryEvaluator:
    """Evaluates agent trajectories for efficiency, correct tool sequences, and loops."""

    def __init__(
        self,
        max_allowed_steps: int = 5,
        penalize_loops: bool = True,
    ):
        self.max_allowed_steps = max_allowed_steps
        self.penalize_loops = penalize_loops

    def evaluate_trajectory(
        self,
        trajectory: List[TrajectoryStep],
        expected_tools: List[str],
        goal_achieved: bool = True,
    ) -> Dict[str, Any]:
        actual_tools = [s.tool_name for s in trajectory]
        num_steps = len(trajectory)

        # 1. Tool Recall: Did the agent call all required tools?
        missing_tools = [t for t in expected_tools if t not in actual_tools]
        tool_recall = 1.0 - (len(missing_tools) / max(len(expected_tools), 1))

        # 2. Tool Precision: Did the agent avoid extraneous tools?
        unexpected_tools = [t for t in actual_tools if t not in expected_tools]
        tool_precision = 1.0 - (len(unexpected_tools) / max(len(actual_tools), 1))

        # 3. Loop Detection: Repeated identical tool calls with same args
        seen_calls = set()
        loops_detected = 0
        for step in trajectory:
            call_sig = (step.tool_name, str(sorted(step.tool_args.items())))
            if call_sig in seen_calls:
                loops_detected += 1
            seen_calls.add(call_sig)

        # 4. Step Efficiency: Actual steps vs max allowed
        step_efficiency = max(0.0, 1.0 - (max(0, num_steps - len(expected_tools)) / self.max_allowed_steps))

        # Composite score
        loop_penalty = (0.25 * loops_detected) if self.penalize_loops else 0.0
        composite = (
            (tool_recall * 0.35)
            + (tool_precision * 0.25)
            + (step_efficiency * 0.20)
            + ((1.0 if goal_achieved else 0.0) * 0.20)
            - loop_penalty
        )
        composite = max(0.0, min(round(composite, 3), 1.0))
        passed = (composite >= 0.70) and (loops_detected == 0) and goal_achieved

        return {
            "num_steps": num_steps,
            "actual_tools": actual_tools,
            "expected_tools": expected_tools,
            "missing_tools": missing_tools,
            "unexpected_tools": unexpected_tools,
            "tool_recall": round(tool_recall, 3),
            "tool_precision": round(tool_precision, 3),
            "loops_detected": loops_detected,
            "step_efficiency": round(step_efficiency, 3),
            "goal_achieved": goal_achieved,
            "composite_score": composite,
            "passed": passed,
        }


class AgentEvaluator(BaseEvaluator):
    """Evaluates agent interactions and tool invocations."""

    def __init__(
        self,
        name: str = "agent_evaluator",
        max_allowed_steps: int = 5,
    ):
        super().__init__(name)
        self.traj_eval = TrajectoryEvaluator(max_allowed_steps=max_allowed_steps)

    def evaluate(self, sample: Dict[str, Any], response: str) -> EvaluationResult:
        test_id = str(sample.get("id", "AGENT-000"))
        prompt = sample.get("prompt", "")
        expected_tools = sample.get("expected_tools", [])

        # Build trajectory from sample if present
        raw_steps = sample.get("trajectory", [])
        steps = []
        for i, s in enumerate(raw_steps):
            steps.append(
                TrajectoryStep(
                    step_index=i + 1,
                    thought=s.get("thought", ""),
                    tool_name=s.get("tool", ""),
                    tool_args=s.get("args", {}),
                    tool_result=s.get("result", None),
                )
            )

        goal_achieved = sample.get("goal_achieved", True)
        analysis = self.traj_eval.evaluate_trajectory(steps, expected_tools, goal_achieved)

        metrics = [
            MetricResult(
                name="tool_call_recall",
                score=analysis["tool_recall"],
                passed=analysis["tool_recall"] >= 0.80,
                threshold=0.80,
                reasoning=f"Required tools called: {len(expected_tools) - len(analysis['missing_tools'])}/{len(expected_tools)}",
            ),
            MetricResult(
                name="loop_freedom",
                score=1.0 if analysis["loops_detected"] == 0 else 0.0,
                passed=analysis["loops_detected"] == 0,
                threshold=1.0,
                reasoning=f"Loops detected: {analysis['loops_detected']}",
            ),
            MetricResult(
                name="trajectory_composite",
                score=analysis["composite_score"],
                passed=analysis["passed"],
                threshold=0.70,
                reasoning="Composite trajectory efficiency and tool accuracy",
                details=analysis,
            ),
        ]

        return EvaluationResult(
            test_id=test_id,
            prompt=prompt,
            actual_output=response,
            metrics=metrics,
            passed=analysis["passed"],
            metadata={"trajectory": [s.to_dict() for s in steps]},
        )
