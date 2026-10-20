"""
Phase 3: Temperature Sensitivity & Stochasticity Stress Lab
===========================================================
Measures output variance, semantic drift, and hallucination emergence
across temperature levels (0.0, 0.3, 0.7, 1.0).
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import os
from typing import Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.base import EvaluationResult, MetricResult
from eval_framework.evaluators.sensitivity import SensitivityEvaluator
from eval_framework.reporting.html_generator import HTMLReportGenerator

STRESS_PROMPT = "What is the exact overdraft fee and daily maximum fee limit for LearnFund checking accounts?"


def run_temp_stress(client: Optional[ModelClient] = None, results_dir: str = "results"):
    eval_client = client or ModelClient(provider="mock", model="llama3.1:8b")
    sens_eval = SensitivityEvaluator(client=eval_client)

    temperatures = [0.0, 0.3, 0.7, 1.0]
    print(f"\n[Phase 3 - Temperature Sensitivity] Testing stability across temperatures: {temperatures}...")

    analysis = sens_eval.evaluate_temperature_sweep(
        prompt=STRESS_PROMPT,
        temperatures=temperatures,
        runs_per_temp=2,
    )

    results = []
    for item in analysis["sweep"]:
        temp = item["temperature"]
        sim = item["avg_cosine_to_baseline"]
        passed = sim >= 0.70

        res = EvaluationResult(
            test_id=f"TEMP-{int(temp*10)}",
            prompt=f"[Temp {temp}] {STRESS_PROMPT}",
            actual_output=item["sample_output"],
            metrics=[
                MetricResult(
                    name="semantic_stability_to_baseline",
                    score=sim,
                    passed=passed,
                    threshold=0.70,
                    reasoning=f"Cosine similarity to greedy baseline: {sim:.3f}",
                )
            ],
            passed=passed,
            metadata={"category": "Temperature Sensitivity", "temperature": temp},
        )
        results.append(res)
        status = "✓ STABLE" if passed else "⚠ DRIFT"
        print(f"  • Temp {temp:.1f} — Cosine Sim to Baseline: {sim:.3f} | {status}")

    report_gen = HTMLReportGenerator(
        title="Phase 3: Temperature Sensitivity & Variance Report",
        subtitle="Stochasticity analysis and stability decay across temperature tiers",
    )
    out_html = os.path.join(results_dir, "phase3_temperature_report.html")
    report_gen.generate_report(results, out_html, metadata={"model": eval_client.model})
    print(f"  ✓ Saved visual temperature report: {out_html}")


if __name__ == "__main__":
    run_temp_stress()
