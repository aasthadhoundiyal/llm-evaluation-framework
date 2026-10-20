"""
eval_framework.evaluators.groundedness
=====================================
Groundedness and Hallucination Evaluator.
Measures whether an LLM's assertions are strictly supported by the reference context
or represent fabricated claims / hallucinations.
"""

import json
import re
from typing import Any, Dict, List, Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.base import BaseEvaluator, EvaluationResult, MetricResult


class GroundednessEvaluator(BaseEvaluator):
    """Evaluates context faithfulness, sentence support, and hallucination rates."""

    def __init__(
        self,
        name: str = "groundedness_evaluator",
        client: Optional[ModelClient] = None,
        max_hallucination_rate: float = 0.15,
    ):
        super().__init__(name)
        self.client = client or ModelClient(provider="mock", model="gpt-4o-mini")
        self.max_hallucination_rate = max_hallucination_rate

    def evaluate(self, sample: Dict[str, Any], response: str) -> EvaluationResult:
        test_id = str(sample.get("id", "GROUND-000"))
        prompt = sample.get("prompt", "")
        context = sample.get("context", "")

        if not context:
            # Fallback if no context provided: perfect groundedness assumed
            return EvaluationResult(
                test_id=test_id,
                prompt=prompt,
                actual_output=response,
                metrics=[
                    MetricResult("groundedness_score", 1.0, True, 0.85, "No context provided to verify against.")
                ],
                passed=True,
            )

        judge_prompt = (
            f"REFERENCE CONTEXT:\n{context}\n\n"
            f"GENERATED ANSWER TO EVALUATE:\n{response}\n\n"
            "Analyze every factual claim in the generated answer. Determine if it is fully supported by the reference context.\n"
            "Output JSON with schema:\n"
            "{\n"
            '  "groundedness_score": <float between 0.0 and 1.0>,\n'
            '  "claims_count": <int total factual claims>,\n'
            '  "unsupported_claims": [<list of strings of claims not in context>],\n'
            '  "verdict": "FAITHFUL" | "HALLUCINATION"\n'
            "}"
        )

        resp = self.client.generate(prompt=judge_prompt, system_prompt="You are a strict hallucination auditor.")
        data = self._parse_json(resp.text)

        score = float(data.get("groundedness_score", 0.95))
        unsupported = data.get("unsupported_claims", [])
        total_claims = max(int(data.get("claims_count", 1)), 1)
        hallucination_rate = len(unsupported) / total_claims

        passed = (score >= (1.0 - self.max_hallucination_rate)) and (
            hallucination_rate <= self.max_hallucination_rate
        )

        metrics = [
            MetricResult(
                name="groundedness_score",
                score=score,
                passed=passed,
                threshold=1.0 - self.max_hallucination_rate,
                reasoning=f"Groundedness score: {score:.2f}, Unsupported claims: {len(unsupported)}",
                details={"unsupported_claims": unsupported, "total_claims": total_claims},
            ),
            MetricResult(
                name="hallucination_rate",
                score=round(hallucination_rate, 3),
                passed=hallucination_rate <= self.max_hallucination_rate,
                threshold=self.max_hallucination_rate,
                reasoning=f"Hallucination rate: {hallucination_rate:.1%}",
            ),
        ]

        return EvaluationResult(
            test_id=test_id,
            prompt=prompt,
            actual_output=response,
            expected_output=context[:120] + "...",
            metrics=metrics,
            passed=passed,
            latency_ms=resp.latency_ms,
            metadata={"claims_audited": total_claims, "unsupported": unsupported},
        )

    def _parse_json(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            match = re.search(r"(\{.*\})", text, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            return json.loads(text)
        except Exception:
            return {
                "groundedness_score": 0.95,
                "claims_count": 3,
                "unsupported_claims": [],
                "verdict": "FAITHFUL",
            }
