"""
Phase 2: LLM-as-a-Judge Evaluation Lab
======================================
Utilizes an evaluator LLM with strict multi-criteria rubrics to score bot responses.
Demonstrates Likert-scale scoring, reasoning extraction, and automated arbitration.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import json
import os
from typing import Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.judge import LLMJudgeEvaluator
from eval_framework.reporting.html_generator import HTMLReportGenerator
from phases.phase_2_evaluation_techniques.banking_bot_eval import BankingBot


def run_judge_eval(client: Optional[ModelClient] = None, results_dir: str = "results"):
    bot_client = client or ModelClient(provider="mock", model="llama3.1:8b")
    judge_client = ModelClient(provider="mock", model="gpt-4o-mini")

    bot = BankingBot(bot_client)
    judge = LLMJudgeEvaluator(client=judge_client, min_passing_score=3.5)

    cases_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "datasets",
        "banking_eval_cases.json",
    )
    with open(cases_file, "r", encoding="utf-8") as f:
        cases = json.load(f)[:5]  # Run on representative subset

    print(f"\n[Phase 2 - LLM-as-a-Judge] Evaluating bot answers using {judge_client.model} as Judge...")
    results = []

    for case in cases:
        bot_response = bot.answer(case["prompt"])
        res = judge.evaluate(case, bot_response)
        res.metadata["category"] = "LLM Judge - " + case.get("category", "General")
        results.append(res)

        composite_metric = next((m for m in res.metrics if m.name == "judge_composite"), None)
        score_val = composite_metric.score if composite_metric else 0.0
        status = "✓ PASS" if res.passed else "✗ FAIL"
        print(f"  [{case['id']}] Score: {score_val}/5.0 | {status} — {case['prompt'][:40]}...")

    report_gen = HTMLReportGenerator(
        title="Phase 2: LLM-as-a-Judge Rubric Report",
        subtitle="Multi-criteria qualitative evaluation powered by LLM Judge",
    )
    out_html = os.path.join(results_dir, "phase2_judge_report.html")
    report_gen.generate_report(
        results,
        out_html,
        metadata={"bot_model": bot_client.model, "judge_model": judge_client.model},
    )
    print(f"  ✓ Saved visual judge report: {out_html}")


if __name__ == "__main__":
    run_judge_eval()
