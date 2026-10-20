"""
Phase 3: End-to-End RAG Pipeline Evaluation Lab
===============================================
Indexes domain knowledge base documents, chunks content, performs BM25 retrieval,
and scores the RAG Triad:
  1. Context Relevance (Retrieval Quality)
  2. Groundedness / Faithfulness (Hallucination Resistance)
  3. Answer Relevance (Query Fulfillment)
Author: Aastha Dhoundiyal (dhoundiyalaastha@gmail.com)
"""

import glob
import os
from typing import Optional
from eval_framework.client import ModelClient
from eval_framework.evaluators.rag import RAGEvaluator, SimpleRetriever, TextChunker
from eval_framework.reporting.html_generator import HTMLReportGenerator

RAG_BENCHMARK_QUERIES = [
    {
        "id": "RAG-QUERY-01",
        "category": "Travel Cards",
        "prompt": "What are the perks and annual fee for the Premier Travel Advantage card?",
    },
    {
        "id": "RAG-QUERY-02",
        "category": "Personal Loans",
        "prompt": "What are the loan amounts and terms for personal loans, and is there a prepayment penalty?",
    },
    {
        "id": "RAG-QUERY-03",
        "category": "Anti-Money Laundering",
        "prompt": "When is a Currency Transaction Report (CTR) legally required, and what is the policy on structuring?",
    },
    {
        "id": "RAG-QUERY-04",
        "category": "Card Fraud",
        "prompt": "How many days do I have to dispute an unauthorized transaction and when is provisional credit given?",
    },
]


def load_knowledge_base() -> list:
    kb_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "datasets",
        "rag_knowledge_base",
    )
    chunker = TextChunker(chunk_size=400, chunk_overlap=80)
    all_chunks = []

    for file_path in glob.glob(os.path.join(kb_dir, "*.txt")):
        doc_name = os.path.basename(file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        chunks = chunker.chunk_text(text, doc_name=doc_name)
        all_chunks.extend(chunks)

    return all_chunks


def run_rag_eval(client: Optional[ModelClient] = None, results_dir: str = "results"):
    eval_client = client or ModelClient(provider="mock", model="llama3.1:8b")
    rag_evaluator = RAGEvaluator(client=eval_client)

    chunks = load_knowledge_base()
    retriever = SimpleRetriever(chunks)
    print(f"\n[Phase 3 - RAG Pipeline] Indexed {len(chunks)} chunks across financial knowledge base.")

    results = []
    for query_item in RAG_BENCHMARK_QUERIES:
        q = query_item["prompt"]
        retrieved = retriever.retrieve(q, top_k=2)
        retrieved_texts = [c["text"] for c in retrieved]
        context_block = "\n---\n".join(retrieved_texts)

        rag_prompt = (
            f"You are a helpful banking assistant. Answer the user question based strictly on the retrieved context below.\n\n"
            f"CONTEXT:\n{context_block}\n\n"
            f"QUESTION:\n{q}\n"
        )

        resp = eval_client.generate(prompt=rag_prompt)

        sample = {
            "id": query_item["id"],
            "prompt": q,
            "contexts": retrieved_texts,
            "category": query_item["category"],
        }
        res = rag_evaluator.evaluate(sample, resp.text)
        res.metadata["category"] = "RAG - " + query_item["category"]
        results.append(res)

        cr = next((m.score for m in res.metrics if m.name == "context_relevance"), 0)
        gr = next((m.score for m in res.metrics if m.name == "groundedness"), 0)
        ar = next((m.score for m in res.metrics if m.name == "answer_relevance"), 0)
        status = "✓ PASS" if res.passed else "✗ FAIL"
        print(f"  [{query_item['id']}] CR: {cr:.2f} | GR: {gr:.2f} | AR: {ar:.2f} | {status} — {q[:35]}...")

    report_gen = HTMLReportGenerator(
        title="Phase 3: RAG Triad Pipeline Evaluation",
        subtitle="Context Relevance, Groundedness, and Answer Relevance across Knowledge Base Chunks",
    )
    out_html = os.path.join(results_dir, "phase3_rag_report.html")
    report_gen.generate_report(results, out_html, metadata={"model": eval_client.model, "chunks_indexed": len(chunks)})
    print(f"  ✓ Saved visual RAG report: {out_html}")


if __name__ == "__main__":
    run_rag_eval()
