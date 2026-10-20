import unittest
from eval_framework.evaluators.rag import RAGEvaluator, TextChunker, SimpleRetriever


class TestRAGEvaluator(unittest.TestCase):
    def setUp(self):
        self.evaluator = RAGEvaluator()

    def test_chunking_and_retrieval(self):
        text = "LearnFund wire transfer fees are $45 for international wires and $25 for domestic wires."
        chunker = TextChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk_text(text, doc_name="fees")
        self.assertGreaterEqual(len(chunks), 1)

        retriever = SimpleRetriever(chunks)
        results = retriever.retrieve("international wire fee", top_k=1)
        self.assertEqual(len(results), 1)
        self.assertIn("international", results[0]["text"].lower())

    def test_rag_triad_evaluation(self):
        sample = {
            "id": "RAG-T-1",
            "prompt": "What is the international wire fee?",
            "contexts": ["International wire fee is $45 per wire."],
        }
        res = self.evaluator.evaluate(sample, "The international wire transfer fee is $45.")
        self.assertTrue(res.passed)
        metric_names = [m.name for m in res.metrics]
        self.assertIn("context_relevance", metric_names)
        self.assertIn("groundedness", metric_names)
        self.assertIn("answer_relevance", metric_names)


if __name__ == "__main__":
    unittest.main()
