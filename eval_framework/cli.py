"""
eval_framework.cli
==================
Command-line interface for the LLM Evaluation Framework.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
GitHub: https://github.com/aasthadhoundiyal/llm-evaluation-framework
"""

import argparse
import http.server
import json
import os
import socketserver
import sys
import webbrowser
from typing import List

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval_framework.client import ModelClient
from eval_framework.config import load_config
from eval_framework.evaluators.deterministic import DeterministicEvaluator
from eval_framework.evaluators.judge import LLMJudgeEvaluator
from eval_framework.evaluators.semantic import SemanticEvaluator
from eval_framework.evaluators.groundedness import GroundednessEvaluator
from eval_framework.evaluators.rag import RAGEvaluator
from eval_framework.evaluators.sensitivity import SensitivityEvaluator
from eval_framework.evaluators.agent import AgentEvaluator
from eval_framework.reporting.html_generator import HTMLReportGenerator


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def print_banner():
    print(c("1;36", "=" * 70))
    print(c("1;97", "  LLM EVALUATION FRAMEWORK - Production GenAI Testing Suite"))
    print(c("35", "  Author: Aastha Dhoundiyal  |  GitHub: @aasthadhoundiyal"))
    print(c("1;36", "=" * 70))


def run_phase_0(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 0] Foundations & Deterministic Baselines..."))
    from phases.phase_0_foundations.squad_evaluation import run_squad_baseline
    from phases.phase_0_foundations.deterministic_checks import run_deterministic_lab

    run_squad_baseline()
    run_deterministic_lab(client)


def run_phase_1(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 1] First Experiments & Multi-Model Benchmarking..."))
    from phases.phase_1_first_experiments.multi_model_bench import run_multi_model_bench

    run_multi_model_bench(client, results_dir)


def run_phase_2(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 2] Core Evaluation Techniques (Judge, Semantic, Groundedness)..."))
    from phases.phase_2_evaluation_techniques.banking_bot_eval import run_banking_eval
    from phases.phase_2_evaluation_techniques.llm_judge_eval import run_judge_eval
    from phases.phase_2_evaluation_techniques.semantic_eval import run_semantic_eval
    from phases.phase_2_evaluation_techniques.groundedness_eval import run_groundedness_eval

    run_banking_eval(client, results_dir)
    run_judge_eval(client, results_dir)
    run_semantic_eval(client, results_dir)
    run_groundedness_eval(client, results_dir)


def run_phase_3(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 3] System-Level Evaluations (RAG Triad & Sensitivity)..."))
    from phases.phase_3_system_evaluations.rag_pipeline_eval import run_rag_eval
    from phases.phase_3_system_evaluations.temperature_stress import run_temp_stress

    run_rag_eval(client, results_dir)
    run_temp_stress(client, results_dir)


def run_phase_4(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 4] Industry Framework Adapters (Ragas & DeepEval)..."))
    from phases.phase_4_industry_frameworks.ragas_adapter import run_ragas_comparison
    from phases.phase_4_industry_frameworks.deepeval_adapter import run_deepeval_comparison

    run_ragas_comparison(results_dir)
    run_deepeval_comparison(results_dir)


def run_phase_5(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 5] Autonomous Agents & Observability Trajectories..."))
    from phases.phase_5_production_agents.agent_simulation import run_agent_simulation
    from phases.phase_5_production_agents.trajectory_eval import run_trajectory_eval
    from phases.phase_5_production_agents.observability_trace import run_observability_trace

    run_agent_simulation(client, results_dir)
    run_trajectory_eval(results_dir)
    run_observability_trace(results_dir)


def run_phase_6(client: ModelClient, results_dir: str):
    print(c("1;33", "\n[PHASE 6] AI Safety & Security (PII Leaks & Prompt Injection Defense)..."))
    from phases.phase_6_ai_safety_security.safety_eval import run_safety_eval

    run_safety_eval(client, results_dir)


def run_all_phases(provider: str, model: str, results_dir: str):
    print_banner()
    client = ModelClient(provider=provider, model=model)
    os.makedirs(results_dir, exist_ok=True)

    run_phase_0(client, results_dir)
    run_phase_1(client, results_dir)
    run_phase_2(client, results_dir)
    run_phase_3(client, results_dir)
    run_phase_4(client, results_dir)
    run_phase_5(client, results_dir)
    run_phase_6(client, results_dir)

    print(c("1;32", "\n" + "=" * 70))
    print(c("1;97", "  [SUCCESS] ALL EVALUATION PHASES (0-6) COMPLETED!"))
    print(c("36", f"  Reports & Artifacts stored in: {os.path.abspath(results_dir)}"))
    print(c("1;32", "=" * 70 + "\n"))


class NoCacheHandler(http.server.SimpleHTTPRequestHandler):
    """Disable aggressive browser caching for local dashboard development."""
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        self.send_header("Pragma", "no-cache")
        self.send_header("Expires", "0")
        super().end_headers()


def serve_dashboard(port: int = 8080):
    print_banner()
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(repo_root)

    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("", port), NoCacheHandler) as httpd:
        url = f"http://localhost:{port}/dashboard/index.html"
        print(c("1;32", f"Serving Dashboard at: {url}"))
        print(c("33", "Press Ctrl+C to stop the server."))
        try:
            webbrowser.open(url)
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down dashboard server.")


def main():
    parser = argparse.ArgumentParser(
        description="LLM Evaluation Framework - Production-grade AI evaluation suite"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # run-all
    run_all_parser = subparsers.add_parser("run-all", help="Execute all evaluation phases (0-6)")
    run_all_parser.add_argument("--provider", default="mock", help="Provider (mock, ollama, openai, gemini, anthropic, groq)")
    run_all_parser.add_argument("--model", default="llama3.1:8b", help="Target model name")
    run_all_parser.add_argument("--results-dir", default="results", help="Destination results directory")

    # run-phase
    phase_parser = subparsers.add_parser("run-phase", help="Run a specific evaluation phase")
    phase_parser.add_argument("--phase", type=int, choices=[0, 1, 2, 3, 4, 5, 6], required=True)
    phase_parser.add_argument("--provider", default="mock")
    phase_parser.add_argument("--model", default="llama3.1:8b")
    phase_parser.add_argument("--results-dir", default="results")

    # serve
    serve_parser = subparsers.add_parser("serve", help="Launch local interactive dashboard")
    serve_parser.add_argument("--port", type=int, default=8080)

    args = parser.parse_args()

    if not args.command or args.command == "run-all":
        prov = getattr(args, "provider", "mock")
        mdl = getattr(args, "model", "llama3.1:8b")
        rdir = getattr(args, "results_dir", "results")
        run_all_phases(prov, mdl, rdir)

    elif args.command == "run-phase":
        client = ModelClient(provider=args.provider, model=args.model)
        p = args.phase
        if p == 0:
            run_phase_0(client, args.results_dir)
        elif p == 1:
            run_phase_1(client, args.results_dir)
        elif p == 2:
            run_phase_2(client, args.results_dir)
        elif p == 3:
            run_phase_3(client, args.results_dir)
        elif p == 4:
            run_phase_4(client, args.results_dir)
        elif p == 5:
            run_phase_5(client, args.results_dir)
        elif p == 6:
            run_phase_6(client, args.results_dir)

    elif args.command == "serve":
        serve_dashboard(args.port)


if __name__ == "__main__":
    main()
