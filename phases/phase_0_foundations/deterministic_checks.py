"""
Phase 0: Deterministic Evaluation Lab
=====================================
Executes foundational assertions (keyword inclusion, regex validators) against an LLM.
Highlights failure modes: False Positives (Quality Failures) and False Negatives (Format Failures).
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import re
import sys
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval_framework.client import ModelClient
from eval_framework.evaluators.deterministic import DeterministicEvaluator


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


def validate_math(text: str) -> bool:
    """Matches 4 or four surrounded by word boundaries."""
    return bool(re.search(r"\b(4|four)\b", text.lower()))


FOUNDATIONAL_TESTS = [
    {
        "id": "FOUND-01",
        "title": "Task 1 - Creative Generation (Keyword Check)",
        "prompt": "Tell a joke about a robot.",
        "expected_keywords": ["robot"],
        "explanation": "Validates that the model stays on topic and references robots.",
        "limitation": "The model could write an awful joke or nonsense and still PASS.\n"
        "                Keyword checks evaluate topic presence, not creative quality (Quality Failure risk).",
    },
    {
        "id": "FOUND-02",
        "title": "Task 2 - Summarization (Keyword Check)",
        "prompt": "Summarize in one sentence: The sky appears blue because of Rayleigh scattering in the atmosphere.",
        "expected_keywords": ["scattering"],
        "explanation": "Verifies that core scientific terminology is preserved in the summary.",
        "limitation": "If the model says 'Rayleigh dispersion', it fails despite being semantically valid (Synonym Failure).",
    },
    {
        "id": "FOUND-03",
        "title": "Task 3 - Strict Math Validation (Regex Boundary)",
        "prompt": "What is 2+2? Answer in one word or digit.",
        "regex_pattern": r"\b(4|four)\b",
        "explanation": "Uses regex boundary checking to prevent partial substring matches like '40' or 'fourteen'.",
        "limitation": "If the model outputs 'The answer is IV', regex fails unless roman numerals are handled.",
    },
]


def run_deterministic_lab(client: Optional[ModelClient] = None):
    eval_client = client or ModelClient(provider="mock", model="llama3.1:8b")
    evaluator = DeterministicEvaluator()

    print(c("1;36", "=" * 70))
    print(c("1;97", "  Phase 0: Deterministic Evaluation & Failure Mode Analysis"))
    print(c("1;36", "=" * 70))

    for item in FOUNDATIONAL_TESTS:
        print(c("1;33", f"\n>> {item['title']}"))
        print(f"  {c('37', 'Prompt:')}      {item['prompt']}")

        response = eval_client.generate(item["prompt"])
        print(f"  {c('36', 'Model Output:')} {response.text.strip()}")

        result = evaluator.evaluate(item, response.text)
        status = c("1;32", "[PASS]") if result.passed else c("1;31", "[FAIL]")
        print(f"  {c('37', 'Eval Result:')}  {status}")

        print(f"  {c('34', '* Intent:')}     {item['explanation']}")
        print(f"  {c('35', '! Limitation:')} {item['limitation']}")

    print(c("1;32", "\n" + "-" * 70))
    print(c("1;97", "  Summary: Deterministic checks are instant ($0.00 cost) and catch 80% of regressions,"))
    print(c("1;97", "  but must be paired with LLM-as-a-Judge and Semantic metrics for comprehensive QA."))
    print(c("1;32", "-" * 70))


if __name__ == "__main__":
    run_deterministic_lab()
