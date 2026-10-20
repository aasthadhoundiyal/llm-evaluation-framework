"""
eval_framework.evaluators.base
==============================
Core abstractions and data structures for all evaluators.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time


class MetricResult:
    """Represents a single metric score with qualitative explanation."""

    def __init__(
        self,
        name: str,
        score: float,
        passed: bool,
        threshold: Optional[float] = None,
        reasoning: str = "",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.name = name
        self.score = score
        self.passed = passed
        self.threshold = threshold
        self.reasoning = reasoning
        self.details = details or {}

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "score": self.score,
            "passed": self.passed,
            "threshold": self.threshold,
            "reasoning": self.reasoning,
            "details": self.details,
        }


class EvaluationResult:
    """Encapsulates the outcome of evaluating a single sample or prompt."""

    def __init__(
        self,
        test_id: str,
        prompt: str,
        actual_output: str,
        expected_output: str = "",
        metrics: Optional[List[MetricResult]] = None,
        passed: bool = True,
        latency_ms: float = 0.0,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.test_id = test_id
        self.prompt = prompt
        self.actual_output = actual_output
        self.expected_output = expected_output
        self.metrics = metrics or []
        self.passed = passed and all(m.passed for m in self.metrics)
        self.latency_ms = latency_ms
        self.metadata = metadata or {}
        self.timestamp = time.time()

    def add_metric(self, metric: MetricResult):
        self.metrics.append(metric)
        if not metric.passed:
            self.passed = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "prompt": self.prompt,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "passed": self.passed,
            "latency_ms": self.latency_ms,
            "metrics": [m.to_dict() for m in self.metrics],
            "metadata": self.metadata,
            "timestamp": self.timestamp,
        }


class BaseEvaluator(ABC):
    """Abstract base class for all evaluation engines."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def evaluate(self, sample: Dict[str, Any], response: str) -> EvaluationResult:
        """Evaluate a generated response against ground truth or rubric."""
        pass

    def evaluate_batch(self, samples: List[Dict[str, Any]], responses: List[str]) -> List[EvaluationResult]:
        """Batch evaluation helper."""
        results = []
        for sample, resp in zip(samples, responses):
            results.append(self.evaluate(sample, resp))
        return results
