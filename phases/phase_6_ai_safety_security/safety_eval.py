"""
Phase 6: AI Safety & Security Evaluation Suite
==============================================
Evaluates models against adversarial prompt injections, PII leakage,
and harmful instruction compliance.

Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
GitHub: https://github.com/aasthadhoundiyal/llm-evaluation-framework
"""

import json
import os
import time
from typing import Any, Dict, List, Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.safety import SafetyEvaluator


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def run_safety_eval(client: Optional[ModelClient] = None, results_dir: str = "results") -> Dict[str, Any]:
    """Execute the Phase 6 AI Safety & Security benchmark suite."""
    eval_client = client or ModelClient(provider="mock")
    evaluator = SafetyEvaluator()

    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    dataset_path = os.path.join(root_dir, "datasets", "safety_security_cases.json")

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Safety dataset not found at: {dataset_path}")

    with open(dataset_path, "r", encoding="utf-8") as f:
        test_cases: List[Dict[str, Any]] = json.load(f)

    print(c("1;36", "\n[PHASE 6] Running AI Safety, PII Leakage & Prompt Injection Benchmarks..."))
    print(c("37", f"Evaluating {len(test_cases)} adversarial test vectors with provider '{eval_client.provider}'..."))

    evaluation_records = []
    pii_passed_count = 0
    injection_passed_count = 0
    total_injections = 0

    for sample in test_cases:
        t0 = time.time()
        response = eval_client.generate(prompt=sample["prompt"], temperature=0.0)
        latency_ms = (time.time() - t0) * 1000

        eval_result = evaluator.evaluate(sample, response.text)
        eval_result.latency_ms = latency_ms

        # Track category stats
        is_injection_case = "injection" in sample["category"].lower() or "jailbreak" in sample["category"].lower()
        if is_injection_case:
            total_injections += 1
            if eval_result.passed:
                injection_passed_count += 1

        # Check PII metric
        pii_metric = next((m for m in eval_result.metrics if m.name == "pii_defense"), None)
        if pii_metric and pii_metric.passed:
            pii_passed_count += 1

        status_str = c("1;32", "DEFENDED") if eval_result.passed else c("1;31", "BREACHED")
        print(f"  • [{sample['id']}] {sample['category'][:22].ljust(22)}: {status_str} (latency: {latency_ms:.0f}ms)")

        record = {
            "id": sample["id"],
            "category": sample["category"],
            "prompt": sample["prompt"],
            "model_response": response.text,
            "passed": eval_result.passed,
            "metrics": [m.to_dict() for m in eval_result.metrics],
            "latency_ms": latency_ms,
        }
        evaluation_records.append(record)

    total_samples = len(test_cases)
    pii_rate = round((pii_passed_count / total_samples) * 100, 1)
    injection_rate = round((injection_passed_count / total_injections * 100) if total_injections > 0 else 100.0, 1)
    overall_pass_rate = round((sum(1 for r in evaluation_records if r["passed"]) / total_samples) * 100, 1)

    summary = {
        "author": "Aastha Dhoundiyal",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "provider": eval_client.provider,
        "model": eval_client.model,
        "total_test_cases": total_samples,
        "pii_defense_rate_pct": pii_rate,
        "injection_resistance_rate_pct": injection_rate,
        "overall_safety_score_pct": overall_pass_rate,
        "detailed_results": evaluation_records,
    }

    os.makedirs(results_dir, exist_ok=True)
    out_file = os.path.join(results_dir, "phase6_safety_report.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(c("1;32", f"\n  ✓ Phase 6 Safety Evaluation Completed!"))
    print(c("1;97", f"    - PII Leak Defense: {pii_rate}%"))
    print(c("1;97", f"    - Prompt Injection Resistance: {injection_rate}%"))
    print(c("1;97", f"    - Overall Safety Score: {overall_pass_rate}%"))
    print(c("36", f"    - Report saved to: {os.path.abspath(out_file)}\n"))

    return summary


if __name__ == "__main__":
    run_safety_eval()
