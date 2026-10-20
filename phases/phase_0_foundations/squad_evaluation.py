"""
Phase 0: SQuAD Dataset & Baseline NLP Metrics
=============================================
Demonstrates classical NLP evaluation metrics: Exact Match (EM) and Token-level F1.
Examines why strict lexical metrics fail on generative LLMs due to non-determinism and verbosity.
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from eval_framework.evaluators.deterministic import compute_exact_match, compute_token_f1


def c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m"


SAMPLE_SQUAD_QA = [
    {
        "id": "squad-01",
        "context": "Super Bowl 50 was an American football game to determine the champion of the National Football League (NFL) for the 2015 season. The American Football Conference (AFC) champion Denver Broncos defeated the National Football Conference (NFC) champion Carolina Panthers 24-10.",
        "question": "Which NFL team won Super Bowl 50?",
        "ground_truths": ["Denver Broncos", "Denver Broncos", "Broncos"],
        "model_pred": "The Denver Broncos won Super Bowl 50 by defeating Carolina.",
    },
    {
        "id": "squad-02",
        "context": "The Apollo program was the third United States human spaceflight program carried out by NASA, which succeeded in landing the first humans on the Moon in 1969.",
        "question": "What year did the Apollo program land the first humans on the Moon?",
        "ground_truths": ["1969"],
        "model_pred": "1969",
    },
    {
        "id": "squad-03",
        "context": "Oxygen is the chemical element with the symbol O and atomic number 8. It is a member of the chalcogen group in the periodic table.",
        "question": "What is the atomic number of oxygen?",
        "ground_truths": ["8", "eight"],
        "model_pred": "Oxygen has an atomic number of 8.",
    },
]


def run_squad_baseline():
    print(c("1;36", "=" * 70))
    print(c("1;97", "  Phase 0: SQuAD Exploration - Exact Match (EM) vs Token F1"))
    print(c("1;36", "=" * 70))

    total_em = 0.0
    total_f1 = 0.0
    count = len(SAMPLE_SQUAD_QA)

    for i, item in enumerate(SAMPLE_SQUAD_QA, 1):
        pred = item["model_pred"]
        truths = item["ground_truths"]

        em_scores = [compute_exact_match(pred, t) for t in truths]
        f1_scores = [compute_token_f1(pred, t) for t in truths]

        best_em = max(em_scores)
        best_f1 = max(f1_scores)

        total_em += 1.0 if best_em else 0.0
        total_f1 += best_f1

        print(c("1;33", f"\n[Sample {i}] Question: {item['question']}"))
        print(f"  {c('37', 'Ground Truth:')} {truths[0]}")
        print(f"  {c('36', 'Model Output:')} {pred}")
        print(
            f"  {c('32' if best_em else '31', f'Exact Match (EM): {best_em}')} | "
            f"{c('32' if best_f1 >= 0.5 else '33', f'Token F1: {best_f1:.2f}')}"
        )

        if not best_em and best_f1 > 0.3:
            print(
                c(
                    "35",
                    "  * Key Insight: Model answered correctly in natural language, but EM scored 0.\n"
                    "    Lexical exact matching penalizes conversational verbosity in generative LLMs.",
                )
            )

    avg_em = (total_em / count) * 100
    avg_f1 = (total_f1 / count) * 100

    print(c("1;34", "\n" + "-" * 70))
    print(f"  Average Exact Match (EM): {avg_em:.1f}%")
    print(f"  Average Token-level F1:   {avg_f1:.1f}%")
    print(c("1;34", "-" * 70))


if __name__ == "__main__":
    run_squad_baseline()
