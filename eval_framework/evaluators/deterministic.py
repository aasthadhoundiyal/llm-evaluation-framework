"""
eval_framework.evaluators.deterministic
=======================================
Deterministic, rule-based, and exact-match evaluators.
Includes token-level F1, Exact Match (EM), Regex assertions, and Forbidden Keyword guards.
"""

import re
import json
from collections import Counter
from typing import Any, Callable, Dict, List, Optional
from eval_framework.evaluators.base import BaseEvaluator, EvaluationResult, MetricResult


def normalize_text(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    s = s.lower()
    s = re.sub(r"[^\w\s]", "", s)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def compute_exact_match(prediction: str, truth: str) -> bool:
    """Returns True if normalized prediction exactly equals normalized ground truth."""
    return normalize_text(prediction) == normalize_text(truth)


def compute_token_f1(prediction: str, truth: str) -> float:
    """Computes token-level precision, recall, and harmonic F1 score."""
    pred_tokens = normalize_text(prediction).split()
    truth_tokens = normalize_text(truth).split()

    if not pred_tokens or not truth_tokens:
        return 1.0 if pred_tokens == truth_tokens else 0.0

    common = Counter(pred_tokens) & Counter(truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1, 4)


class DeterministicEvaluator(BaseEvaluator):
    """Evaluates models deterministically using assertions, regex, and vocabulary guards."""

    def __init__(self, name: str = "deterministic_evaluator"):
        super().__init__(name)

    def evaluate(self, sample: Dict[str, Any], response: str) -> EvaluationResult:
        test_id = str(sample.get("id", "TEST-000"))
        prompt = sample.get("prompt", "")
        expected = sample.get("expected_output", sample.get("ideal_answer", ""))
        metrics: List[MetricResult] = []

        # 1. Expected Keywords Check
        expected_keywords = sample.get("expected_keywords", [])
        if expected_keywords:
            missing = [kw for kw in expected_keywords if kw.lower() not in response.lower()]
            passed = len(missing) == 0
            score = 1.0 if passed else (len(expected_keywords) - len(missing)) / len(expected_keywords)
            metrics.append(
                MetricResult(
                    name="expected_keywords",
                    score=score,
                    passed=passed,
                    threshold=1.0,
                    reasoning=f"Missing keywords: {missing}" if missing else "All expected keywords present.",
                    details={"missing": missing, "expected": expected_keywords},
                )
            )

        # 2. Forbidden Keywords Check (Negative assertion for safety/leakage)
        forbidden_keywords = sample.get("forbidden_keywords", [])
        if forbidden_keywords:
            found = [kw for kw in forbidden_keywords if kw.lower() in response.lower()]
            passed = len(found) == 0
            score = 1.0 if passed else 0.0
            metrics.append(
                MetricResult(
                    name="forbidden_keywords",
                    score=score,
                    passed=passed,
                    threshold=1.0,
                    reasoning=f"Found forbidden tokens: {found}" if found else "No forbidden tokens detected.",
                    details={"detected_forbidden": found},
                )
            )

        # 3. Regex Validator
        regex_pattern = sample.get("regex_pattern")
        if regex_pattern:
            match = bool(re.search(regex_pattern, response, re.IGNORECASE))
            metrics.append(
                MetricResult(
                    name="regex_pattern",
                    score=1.0 if match else 0.0,
                    passed=match,
                    threshold=1.0,
                    reasoning=f"Pattern '{regex_pattern}' matched: {match}",
                )
            )

        # 4. Custom Callable Validator
        validator: Optional[Callable[[str], bool]] = sample.get("validator")
        if validator:
            v_passed = bool(validator(response))
            metrics.append(
                MetricResult(
                    name="custom_validator",
                    score=1.0 if v_passed else 0.0,
                    passed=v_passed,
                    threshold=1.0,
                    reasoning="Custom function validation",
                )
            )

        # 5. Token F1 / EM (Evaluated when requested or informatively)
        if expected and sample.get("require_f1", False):
            f1 = compute_token_f1(response, expected)
            f1_threshold = sample.get("f1_threshold", 0.35)
            metrics.append(
                MetricResult(
                    name="token_f1",
                    score=f1,
                    passed=f1 >= f1_threshold,
                    threshold=f1_threshold,
                    reasoning=f"Token F1 score: {f1} (threshold: {f1_threshold})",
                )
            )
        elif expected:
            f1 = compute_token_f1(response, expected)
            metrics.append(
                MetricResult(
                    name="token_f1_informative",
                    score=f1,
                    passed=True,
                    threshold=None,
                    reasoning=f"Informative token F1 score: {f1}",
                )
            )

        overall_passed = all(m.passed for m in metrics) if metrics else True
        return EvaluationResult(
            test_id=test_id,
            prompt=prompt,
            actual_output=response,
            expected_output=expected,
            metrics=metrics,
            passed=overall_passed,
            metadata=sample,
        )
