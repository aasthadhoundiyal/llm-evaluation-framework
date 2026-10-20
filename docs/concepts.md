# Core Concepts in LLM Evaluation

Authored by **Aastha Dhoundiyal** (`dhoundiyalaastha@gmail.com`)

---

## 1. The Challenge of Non-Determinism

Traditional unit tests expect identical outputs for identical inputs (`assert f(2) == 4`). Generative LLMs, however, are non-deterministic due to:
- **Temperature Sampling**: Probability distribution flattening during token decoding.
- **Top-p / Nucleus Sampling**: Varying token candidate pools.
- **Natural Language Variance**: Multiple distinct sentences can convey identical semantic meaning with totally different token vocabulary.

Therefore, GenAI QA shifts from **literal equality** to **distributional and semantic verification**.

---

## 2. The Evaluation Pyramid

A robust evaluation strategy layers multiple evaluation types:

```
          ▲
         / \     Phase 6: AI Safety & Adversarial Injection Defense
        /   \    Phase 5: Agent Trajectory & Observability Tracing
       /     \   Phase 4: Industry Standards (Ragas / DeepEval)
      /       \  Phase 3: RAG Triad & Temperature Sweeps
     /         \ Phase 2: LLM-as-a-Judge & Semantic Similarity
    /           \Phase 1: Multi-Model Comparative Benchmarking
   /─────────────\Phase 0: Deterministic Assertions & Exact Match
```

1. **Deterministic Layer (Bottom)**:
   - High speed, $0 cost, zero hallucinations.
   - Regex patterns, keyword containment, forbidden token absence, schema validation.
2. **Semantic Layer (Middle)**:
   - Cosine similarity, BLEU, ROUGE-L, sentence embedding distance.
   - Tolerates paraphrasing and grammatical variance.
3. **LLM-as-a-Judge & RAG Triad (Upper)**:
   - Evaluator model graded against multi-criteria rubrics (Accuracy, Tone, Policy, Reasoning).
   - RAG Triad: Context Relevance, Groundedness (Faithfulness), Answer Relevance.
4. **Trajectory & Observability (Top)**:
   - Multi-step tool calls, infinite loop detection, step efficiency, and distributed tracing.

---

## 3. Failure Mode Taxonomy

When testing GenAI, failures generally fall into four categories:

1. **Quality Failure**: The model generates poor, hallucinated, or unhelpful answers that coincidentally satisfy basic keyword checks.
2. **Refusal Failure**: The model complies with an adversarial prompt injection or jailbreak that should have been rejected.
3. **Format / Boundary Failure**: The model provides correct information but buries it in conversational filler, failing strict regex or schema assertions.
4. **Trajectory Failure**: An autonomous agent gets trapped in a tool-calling loop or takes unnecessary steps, exhausting token budgets without reaching the goal.

---

## 4. The RAG Triad

Retrieval-Augmented Generation systems require auditing three distinct vectors:
- **Context Relevance**: Did the retriever pull information that directly answers the user query?
- **Groundedness / Faithfulness**: Are the claims made in the answer strictly backed by the retrieved context chunks (zero hallucinations)?
- **Answer Relevance**: Does the final answer actually solve the user's initial prompt, or does it wander into unrelated tangents?
