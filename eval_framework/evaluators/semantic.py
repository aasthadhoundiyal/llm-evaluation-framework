"""
eval_framework.evaluators.semantic
==================================
Semantic similarity evaluation including Cosine Similarity, BLEU, and ROUGE-L.
Provides built-in mathematical fallbacks so tests run with zero external heavy dependencies.
"""

import math
from collections import Counter
from typing import Any, Dict, List, Optional
from eval_framework.evaluators.base import BaseEvaluator, EvaluationResult, MetricResult


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in text.split() if w.strip()]


def compute_cosine_similarity(text_a: str, text_b: str) -> float:
    """Computes cosine similarity between two texts using term frequency vectors."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    vec_a = Counter(tokens_a)
    vec_b = Counter(tokens_b)

    intersection = set(vec_a.keys()) & set(vec_b.keys())
    numerator = sum(vec_a[x] * vec_b[x] for x in intersection)

    sum1 = sum(vec_a[x] ** 2 for x in vec_a.keys())
    sum2 = sum(vec_b[x] ** 2 for x in vec_b.keys())
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    return round(float(numerator) / denominator, 4)


def compute_lcs_length(tokens_a: List[str], tokens_b: List[str]) -> int:
    """Computes Longest Common Subsequence length for ROUGE-L."""
    m, n = len(tokens_a), len(tokens_b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if tokens_a[i - 1] == tokens_b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(candidate: str, reference: str) -> float:
    """Calculates ROUGE-L F1 score based on Longest Common Subsequence."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    lcs = compute_lcs_length(cand_tokens, ref_tokens)
    if lcs == 0:
        return 0.0

    r_lcs = lcs / len(ref_tokens)
    p_lcs = lcs / len(cand_tokens)
    f_lcs = (2 * r_lcs * p_lcs) / (r_lcs + p_lcs)
    return round(f_lcs, 4)


def compute_simple_bleu(candidate: str, reference: str, max_n: int = 4) -> float:
    """Computes smoothed BLEU score up to 4-grams."""
    cand_tokens = _tokenize(candidate)
    ref_tokens = _tokenize(reference)

    if not cand_tokens or not ref_tokens:
        return 0.0

    precisions = []
    for n in range(1, max_n + 1):
        cand_ngrams = [tuple(cand_tokens[i : i + n]) for i in range(len(cand_tokens) - n + 1)]
        ref_ngrams = [tuple(ref_tokens[i : i + n]) for i in range(len(ref_tokens) - n + 1)]

        if not cand_ngrams:
            precisions.append(1e-5)
            continue

        cand_counts = Counter(cand_ngrams)
        ref_counts = Counter(ref_ngrams)
        overlap = sum(min(cand_counts[ng], ref_counts.get(ng, 0)) for ng in cand_counts)
        precisions.append(max(overlap / len(cand_ngrams), 1e-5))

    # Brevity penalty
    c = len(cand_tokens)
    r = len(ref_tokens)
    bp = 1.0 if c > r else math.exp(1 - (r / (c + 1e-9)))

    # Geometric mean
    log_precisions = sum(math.log(p) for p in precisions) / max_n
    bleu = bp * math.exp(log_precisions)
    return round(bleu, 4)


class SemanticEvaluator(BaseEvaluator):
    """Evaluator assessing semantic alignment via Cosine Similarity, ROUGE-L, and BLEU."""

    def __init__(
        self,
        name: str = "semantic_evaluator",
        cosine_threshold: float = 0.70,
        rouge_threshold: float = 0.50,
    ):
        super().__init__(name)
        self.cosine_threshold = cosine_threshold
        self.rouge_threshold = rouge_threshold

    def evaluate(self, sample: Dict[str, Any], response: str) -> EvaluationResult:
        test_id = str(sample.get("id", "SEM-000"))
        prompt = sample.get("prompt", "")
        reference = sample.get("reference_answer", sample.get("ideal_answer", ""))

        cosine = compute_cosine_similarity(response, reference)
        rouge_l = compute_rouge_l(response, reference)
        bleu = compute_simple_bleu(response, reference)

        metrics = [
            MetricResult(
                name="cosine_similarity",
                score=cosine,
                passed=cosine >= self.cosine_threshold,
                threshold=self.cosine_threshold,
                reasoning=f"Cosine similarity: {cosine:.3f} (target >= {self.cosine_threshold})",
            ),
            MetricResult(
                name="rouge_l",
                score=rouge_l,
                passed=rouge_l >= self.rouge_threshold,
                threshold=self.rouge_threshold,
                reasoning=f"ROUGE-L LCS score: {rouge_l:.3f} (target >= {self.rouge_threshold})",
            ),
            MetricResult(
                name="bleu_score",
                score=bleu,
                passed=True,  # BLEU recorded informatively
                threshold=0.20,
                reasoning=f"Smoothed BLEU-4 score: {bleu:.3f}",
            ),
        ]

        # Optional Medical Disclaimer check for healthcare datasets
        if sample.get("mandatory_disclaimer"):
            disclaimer_terms = ["disclaimer", "doctor", "physician", "professional", "emergency", "911"]
            has_disclaimer = any(term in response.lower() for term in disclaimer_terms)
            metrics.append(
                MetricResult(
                    name="medical_disclaimer_presence",
                    score=1.0 if has_disclaimer else 0.0,
                    passed=has_disclaimer,
                    threshold=1.0,
                    reasoning="Mandatory medical advisory disclaimer presence check",
                )
            )

        overall_passed = all(m.passed for m in metrics)
        return EvaluationResult(
            test_id=test_id,
            prompt=prompt,
            actual_output=response,
            expected_output=reference,
            metrics=metrics,
            passed=overall_passed,
            metadata=sample,
        )
