"""
Phase 5: Agent Trajectory Evaluation Lab
=======================================
Evaluates multi-turn reasoning traces, tool-calling precision/recall,
infinite loop detection, and goal completion efficiency.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import json
import os
from typing import Optional
from eval_framework.evaluators.agent import AgentEvaluator
from eval_framework.reporting.html_generator import HTMLReportGenerator


def run_trajectory_eval(results_dir: str = "results"):
    evaluator = AgentEvaluator(max_allowed_steps=5)
    traj_path = os.path.join(results_dir, "agent_trajectories.json")

    if not os.path.exists(traj_path):
        print("[INFO] Trajectory file not found. Generating trajectories via agent simulation...")
        from phases.phase_5_production_agents.agent_simulation import run_agent_simulation
        scenarios = run_agent_simulation(results_dir=results_dir)
    else:
        with open(traj_path, "r", encoding="utf-8") as f:
            scenarios = json.load(f)

    print(f"\n[Phase 5 - Trajectory Evaluation] Auditing {len(scenarios)} agent execution trajectories...")
    results = []

    for scenario in scenarios:
        res = evaluator.evaluate(scenario, response="Agent execution completed successfully.")
        res.metadata["category"] = "Agent Trajectory"
        results.append(res)

        score_m = next((m for m in res.metrics if m.name == "trajectory_composite"), None)
        score_val = score_m.score if score_m else 1.0
        status = "✓ PASS" if res.passed else "✗ FAIL"
        print(f"  [{scenario['id']}] Composite: {score_val:.2f} | {status} — {scenario['prompt'][:40]}...")

    report_gen = HTMLReportGenerator(
        title="Phase 5: Agent Trajectory & Efficiency Audit",
        subtitle="Step efficiency, tool call precision, loop absence, and goal attainment",
    )
    out_html = os.path.join(results_dir, "phase5_trajectory_report.html")
    report_gen.generate_report(results, out_html)
    print(f"  ✓ Saved visual trajectory report: {out_html}")


if __name__ == "__main__":
    run_trajectory_eval()
