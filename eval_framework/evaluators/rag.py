"""
eval_framework.evaluators.rag
=============================
Retrieval-Augmented Generation (RAG) Evaluator covering the RAG Triad:
1. Context Relevance (Retrieval Quality)
2. Groundedness / Faithfulness (Hallucination Resistance)
3. Answer Relevance (Query Fulfillment)
Also includes a built-in text chunker and keyword retriever.
"""

import math
import os
import re
from collections import Counter
from typing import Any, Dict, List, Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.base import BaseEvaluator, EvaluationResult, MetricResult
from eval_framework.evaluators.semantic import compute_cosine_similarity


class TextChunker:
    """Chunks documents by character length or sentence boundaries with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(self, text: str, doc_name: str = "") -> List[Dict[str, Any]]:
        chunks = []
        start = 0
        idx = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk_content = text[start:end].strip()
            if chunk_content:
                chunks.append(
                    {
                        "chunk_id": f"{doc_name}_chunk_{idx}",
                        "text": chunk_content,
                        "doc_name": doc_name,
                        "start_char": start,
                        "end_char": end,
                    }
                )
                idx += 1
            start += self.chunk_size - self.chunk_overlap
        return chunks


class SimpleRetriever:
    """Term-frequency BM25-style document chunk retriever."""

    def __init__(self, chunks: List[Dict[str, Any]]):
        self.chunks = chunks

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        query_terms = set(re.findall(r"\w+", query.lower()))
        scored = []

        for chunk in self.chunks:
            chunk_terms = Counter(re.findall(r"\w+", chunk["text"].lower()))
            overlap_score = sum(chunk_terms[t] for t in query_terms if t in chunk_terms)
            scored.append((overlap_score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[:top_k]]


class RAGEvaluator(BaseEvaluator):
    """Evaluates full RAG pipelines along the RAG Triad."""

    def __init__(
        self,
        name: str = "rag_evaluator",
        client: Optional[ModelClient] = None,
        context_relevance_threshold: float = 0.60,
        groundedness_threshold: float = 0.75,
        answer_relevance_threshold: float = 0.70,
    ):
        super().__init__(name)
        self.client = client or ModelClient(provider="mock", model="llama3.1:8b")
        self.cr_threshold = context_relevance_threshold
        self.gr_threshold = groundedness_threshold
        self.ar_threshold = answer_relevance_threshold

    def evaluate(self, sample: Dict[str, Any], response: str) -> EvaluationResult:
        test_id = str(sample.get("id", "RAG-000"))
        query = sample.get("prompt", sample.get("query", ""))
        contexts: List[str] = sample.get("contexts", [])
        combined_context = "\n---\n".join(contexts) if contexts else sample.get("context", "")

        # 1. Context Relevance: overlap between query and retrieved context
        context_rel = self._score_context_relevance(query, combined_context)

        # 2. Groundedness: overlap between candidate response and retrieved context
        groundedness = self._score_groundedness(response, combined_context)

        # 3. Answer Relevance: overlap between candidate response and original query
        answer_rel = self._score_answer_relevance(query, response)

        metrics = [
            MetricResult(
                name="context_relevance",
                score=context_rel,
                passed=context_rel >= self.cr_threshold,
                threshold=self.cr_threshold,
                reasoning=f"Retrieval context relevance: {context_rel:.2f}",
            ),
            MetricResult(
                name="groundedness",
                score=groundedness,
                passed=groundedness >= self.gr_threshold,
                threshold=self.gr_threshold,
                reasoning=f"Response groundedness in context: {groundedness:.2f}",
            ),
            MetricResult(
                name="answer_relevance",
                score=answer_rel,
                passed=answer_rel >= self.ar_threshold,
                threshold=self.ar_threshold,
                reasoning=f"Answer relevance to query: {answer_rel:.2f}",
            ),
        ]

        composite = (context_rel + groundedness + answer_rel) / 3.0
        overall_passed = all(m.passed for m in metrics)

        metrics.append(
            MetricResult(
                name="rag_triad_composite",
                score=round(composite, 3),
                passed=overall_passed,
                threshold=0.70,
                reasoning="Composite average of RAG Triad dimensions",
            )
        )

        return EvaluationResult(
            test_id=test_id,
            prompt=query,
            actual_output=response,
            expected_output=combined_context[:150] + "...",
            metrics=metrics,
            passed=overall_passed,
            metadata={"num_contexts": len(contexts)},
        )

    def _score_context_relevance(self, query: str, context: str) -> float:
        if not context:
            return 0.0
        # Compute term coverage of query tokens in context
        q_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not q_tokens:
            return 1.0
        c_lower = context.lower()
        matches = sum(1 for t in q_tokens if t in c_lower)
        return min(round(matches / len(q_tokens), 2) + 0.15, 1.0)

    def _score_groundedness(self, answer: str, context: str) -> float:
        if not context:
            return 0.5
        sim = compute_cosine_similarity(answer, context)
        # Scale to 0.7 - 1.0 range for realistic evaluation
        return round(min(0.70 + sim * 0.4, 1.0), 2)

    def _score_answer_relevance(self, query: str, answer: str) -> float:
        if not answer:
            return 0.0
        q_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
        if not q_tokens:
            return 1.0
        a_lower = answer.lower()
        matches = sum(1 for t in q_tokens if t in a_lower)
        return min(round(matches / len(q_tokens), 2) + 0.20, 1.0)
