"""
Phase 1: Score Matrix Aggregation
=================================
Aggregates granular test runs into comparative pass matrices and category breakdowns.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import json
import os
from typing import Dict, Any


def print_matrix(scores: Dict[str, Any]):
    print("\n" + "=" * 65)
    print(f"{'MODEL':<18} | {'PASS RATE':<12} | {'PASSED/TOTAL':<14} | {'AVG LATENCY'}")
    print("-" * 65)
    for model, data in scores.items():
        pr = f"{data['pass_rate']}%"
        pt = f"{data['passed']}/{data['total']}"
        lat = f"{data['avg_latency_ms']} ms"
        print(f"{model:<18} | {pr:<12} | {pt:<14} | {lat}")
    print("=" * 65 + "\n")


def run_score_matrix(results_dir: str = "results"):
    score_file = os.path.join(results_dir, "multi_model_scores.json")
    if not os.path.exists(score_file):
        print(f"[INFO] Score file not found at {score_file}. Running multi_model_bench first...")
        from phases.phase_1_first_experiments.multi_model_bench import run_multi_model_bench
        run_multi_model_bench(results_dir=results_dir)

    with open(score_file, "r", encoding="utf-8") as f:
        scores = json.load(f)

    print_matrix(scores)


if __name__ == "__main__":
    run_score_matrix()
