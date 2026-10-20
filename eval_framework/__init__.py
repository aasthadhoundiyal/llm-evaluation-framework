"""
LLM Evaluation Framework
========================
A production-grade, extensible evaluation suite for LLMs, RAG pipelines,
and autonomous AI agents.

Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
GitHub: https://github.com/aasthadhoundiyal/llm-evaluation-framework
"""

__version__ = "1.1.0"
__author__ = "Aastha Dhoundiyal"
__email__ = "dhoundiyalaastha@gmail.com"

from eval_framework.client import ModelClient
from eval_framework.config import load_config
from eval_framework.evaluators.base import BaseEvaluator, EvaluationResult, MetricResult
from eval_framework.evaluators.deterministic import DeterministicEvaluator
from eval_framework.evaluators.judge import LLMJudgeEvaluator
from eval_framework.evaluators.semantic import SemanticEvaluator
from eval_framework.evaluators.groundedness import GroundednessEvaluator
from eval_framework.evaluators.rag import RAGEvaluator
from eval_framework.evaluators.sensitivity import SensitivityEvaluator
from eval_framework.evaluators.agent import AgentEvaluator, TrajectoryEvaluator
from eval_framework.evaluators.safety import SafetyEvaluator
from eval_framework.reporting.html_generator import HTMLReportGenerator

__all__ = [
    "ModelClient",
    "load_config",
    "BaseEvaluator",
    "EvaluationResult",
    "MetricResult",
    "DeterministicEvaluator",
    "LLMJudgeEvaluator",
    "SemanticEvaluator",
    "GroundednessEvaluator",
    "RAGEvaluator",
    "SensitivityEvaluator",
    "AgentEvaluator",
    "TrajectoryEvaluator",
    "SafetyEvaluator",
    "HTMLReportGenerator",
]
