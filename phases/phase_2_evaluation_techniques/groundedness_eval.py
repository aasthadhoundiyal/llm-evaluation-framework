"""
Phase 2: Groundedness & Hallucination Scoring Lab
================================================
Detects fabricated claims and scores sentence-level attribution against reference knowledge.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import json
import os
from typing import Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.groundedness import GroundednessEvaluator
from eval_framework.reporting.html_generator import HTMLReportGenerator

GROUNDEDNESS_TEST_SAMPLES = [
    {
        "id": "GROUND-01",
        "category": "Credit Card Policy",
        "prompt": "What are the key benefits of the Premier Travel Advantage card?",
        "context": (
            "The Premier Travel Advantage card features a 60,000 bonus point signup reward after $4,000 spend "
            "in 90 days. It earns 5x points on flights booked directly with airlines and 3x on hotels. "
            "It has a $95 annual fee and $0 foreign transaction fees."
        ),
    },
    {
        "id": "GROUND-02",
        "category": "Loan Prepayment",
        "prompt": "Is there a penalty if I pay off my personal loan early?",
        "context": (
            "LearnFund Personal Loans range from $2,500 to $50,000 with terms up to 60 months. "
            "Prepayment Penalty is $0. Borrowers may pay down balance or settle loan early without any fee."
        ),
    },
    {
        "id": "GROUND-03",
        "category": "Check Deposit Funds Availability",
        "prompt": "When will funds from my deposited check become available?",
        "context": (
            "Under Regulation CC, the first $225 of check deposits is available the next business day. "
            "The remaining balance of standard checks is available on the 2nd business day following deposit."
        ),
    },
]


def run_groundedness_eval(client: Optional[ModelClient] = None, results_dir: str = "results"):
    eval_client = client or ModelClient(provider="mock", model="llama3.1:8b")
    ground_evaluator = GroundednessEvaluator(client=eval_client, max_hallucination_rate=0.15)

    print(f"\n[Phase 2 - Groundedness] Auditing {len(GROUNDEDNESS_TEST_SAMPLES)} context-grounded responses...")
    results = []

    for sample in GROUNDEDNESS_TEST_SAMPLES:
        prompt = f"Answer strictly using ONLY this context:\n{sample['context']}\n\nQuestion: {sample['prompt']}"
        resp = eval_client.generate(prompt=prompt)
        res = ground_evaluator.evaluate(sample, resp.text)
        res.metadata["category"] = sample.get("category", "Grounded QA")
        results.append(res)

        g_score = next((m.score for m in res.metrics if m.name == "groundedness_score"), 1.0)
        status = "✓ FAITHFUL" if res.passed else "✗ HALLUCINATED"
        print(f"  [{sample['id']}] Groundedness: {g_score:.2f} | {status} — {sample['prompt'][:35]}...")

    report_gen = HTMLReportGenerator(
        title="Phase 2: Groundedness & Hallucination Audit Report",
        subtitle="Claim-level context verification and factual consistency scoring",
    )
    out_html = os.path.join(results_dir, "phase2_groundedness_report.html")
    report_gen.generate_report(results, out_html, metadata={"model": eval_client.model, "provider": eval_client.provider})
    print(f"  ✓ Saved visual groundedness report: {out_html}")


if __name__ == "__main__":
    run_groundedness_eval()
