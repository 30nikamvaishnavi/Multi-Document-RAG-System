"""Offline tests for ingestion and agent routing without calling Gemini."""
import unittest
from hashlib import sha256
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional
from unittest.mock import patch

from agentic_rag import AgenticRAG
from ingestion import PDFIngestion
from rag_engine import RAGEngine, response_to_text
from vectorstore import VectorStore
from docx import Document as WordDocument
from openpyxl import Workbook
from pptx import Presentation
from pptx.util import Inches


class FakeLLM:
    def __init__(self, ground_checks: Optional[List[str]] = None):
        self.ground_checks = list(ground_checks or ["yes"])
        self.rewrite_count = 0

    def invoke(self, prompt: str) -> SimpleNamespace:
        if prompt.startswith("Decide whether"):
            content = "no" if "Hello" in prompt else "yes"
        elif prompt.startswith("You are a query rewriting assistant"):
            self.rewrite_count += 1
            content = f"refined query {self.rewrite_count}"
        elif prompt.startswith("You are a strict fact-checker"):
            content = self.ground_checks.pop(0)
        else:
            content = "A general response."
        return SimpleNamespace(content=content)


class FakeRAG:
    k = 3

    def __init__(self):
        self.queries = []

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        self.queries.append(query)
        return [{"text": "Evidence", "source": "guide.pdf", "page": 2}]

    def _format_retrieved(self, chunks: List[Dict[str, Any]]) -> str:
        return "\n".join(chunk["text"] for chunk in chunks)

    def generate(self, query: str, context: str) -> str:
        return "The document says this. [guide.pdf, page 2]"


def make_agent(ground_checks: Optional[List[str]] = None) -> AgenticRAG:
    agent = object.__new__(AgenticRAG)
    agent.llm = FakeLLM(ground_checks)
    agent.rag = FakeRAG()
    agent.max_retries = 1
    agent.graph = agent._build_graph()
    return agent


class IngestionTests(unittest.TestCase):
    def test_chunks_have_requested_overlap_and_keep_page_metadata(self):
        ingestion = PDFIngestion(chunk_size=6, chunk_overlap=2)

        chunks = ingestion.chunk_pages(
            [{"source": "guide.pdf", "page": 4, "text": "abcdefghijkl"}]
        )

        self.assertEqual([chunk["text"] for chunk in chunks], ["abcdef", "efghij", "ijkl"])
        self.assertTrue(all(chunk["page"] == 4 for chunk in chunks))
        self.assertTrue(all(chunk["source"] == "guide.pdf" for chunk in chunks))

    def test_rejects_invalid_chunk_configuration(self):
        with self.assertRaises(ValueError):
            PDFIngestion(chunk_size=10, chunk_overlap=10)

    def test_response_to_text_accepts_string_and_text_blocks(self):
        self.assertEqual(response_to_text(SimpleNamespace(content="hello")), "hello")
        self.assertEqual(
            response_to_text(
                SimpleNamespace(
                    content=[
                        {"type": "text", "text": "hello "},
                        {"type": "text", "text": "world"},
                    ]
                )
            ),
            "hello world",
        )

    def test_extracts_common_document_formats_and_locations(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            folder = Path(temp_dir)
            (folder / "notes.txt").write_text("Plain text evidence.", encoding="utf-8")
            (folder / "notes.md").write_text("# Markdown evidence", encoding="utf-8")
            (folder / "page.html").write_text(
                "<h1>HTML evidence</h1><script>ignore this</script>", encoding="utf-8"
            )
            (folder / "data.csv").write_text("name,value\nalpha,42\n", encoding="utf-8")

            word = WordDocument()
            word.add_paragraph("Word document evidence.")
            word.save(folder / "notes.docx")

            workbook = Workbook()
            sheet = workbook.active
            sheet.title = "Facts"
            sheet.append(["name", "value"])
            sheet.append(["answer", "42"])
            workbook.save(folder / "data.xlsx")

            presentation = Presentation()
            slide = presentation.slides.add_slide(presentation.slide_layouts[5])
            slide.shapes.title.text = "Presentation evidence"
            presentation.save(folder / "slides.pptx")

            ingestion = PDFIngestion()
            pages = ingestion.extract_all(str(folder))
            by_source = {page["source"]: page for page in pages}

            self.assertEqual(set(by_source), {
                "data.csv", "data.xlsx", "notes.docx", "notes.md",
                "notes.txt", "page.html", "slides.pptx",
            })
            self.assertIn("Plain text evidence", by_source["notes.txt"]["text"])
            self.assertIn("Word document evidence", by_source["notes.docx"]["text"])
            self.assertNotIn("ignore this", by_source["page.html"]["text"])
            self.assertEqual(by_source["data.xlsx"]["location"], "Sheet: Facts")
            self.assertEqual(by_source["slides.pptx"]["location"], "Slide 1")

            chunks = ingestion.chunk_pages(pages)
            self.assertTrue(all("location" in chunk for chunk in chunks))


class AgentTests(unittest.TestCase):
    def test_routes_general_message_without_retrieval(self):
        agent = make_agent()

        result = agent.invoke("Hello")

        self.assertFalse(result["retrieval_used"])
        self.assertIsNone(result["grounded"])
        self.assertEqual(result["sources"], [])
        self.assertEqual(result["answer"], "A general response.")

    def test_retrieves_and_retries_an_ungrounded_answer(self):
        agent = make_agent(["no", "yes"])

        result = agent.invoke("What does my document say?")

        self.assertTrue(result["retrieval_used"])
        self.assertTrue(result["grounded"])
        self.assertEqual(result["retry_count"], 1)
        self.assertEqual(agent.rag.queries, ["refined query 1", "refined query 2"])
        self.assertEqual(result["sources"][0]["page"], 2)

    def test_rejects_an_empty_question(self):
        agent = make_agent()

        with self.assertRaises(ValueError):
            agent.invoke("  ")


class FakeEmbeddings:
    def __init__(self):
        self.batch_sizes = []
        self.fail_once = False

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.batch_sizes.append(len(texts))
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        return [[float(len(text)), 1.0, 0.0] for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return [float(len(text)), 1.0, 0.0]


class VectorStoreTests(unittest.TestCase):
    def test_persists_searchable_page_metadata_and_clears_collection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                embedding_patch = patch(
                    "vectorstore.GoogleGenerativeAIEmbeddings",
                    return_value=FakeEmbeddings(),
                )
                embedding_patch.start()
                self.addCleanup(embedding_patch.stop)
                vectorstore = VectorStore(persist_dir=temp_dir)
                try:
                    vectorstore.clear()
                    vectorstore.add_documents(
                        [{"text": "important content", "source": "guide.pdf", "page": 3}]
                    )

                    results = vectorstore.search("important content", k=5)

                    self.assertEqual(len(results), 1)
                    self.assertEqual(results[0]["source"], "guide.pdf")
                    self.assertEqual(results[0]["page"], 3)
                    self.assertTrue(VectorStore.has_documents(persist_dir=temp_dir))

                    vectorstore.clear()

                    self.assertFalse(VectorStore.has_documents(persist_dir=temp_dir))
                finally:
                    vectorstore.close()

    def test_embeddings_are_batched_and_rate_limits_are_retried(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_embeddings = FakeEmbeddings()
            fake_embeddings.fail_once = True
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                with patch(
                    "vectorstore.GoogleGenerativeAIEmbeddings",
                    return_value=fake_embeddings,
                ), patch("vectorstore.time.sleep") as sleep:
                    vectorstore = VectorStore(persist_dir=temp_dir)
                    vectorstore.EMBEDDING_BATCH_SIZE = 2
                    try:
                        vectorstore.add_documents(
                            [
                                {"text": f"item {index}", "source": "batch.txt", "page": 1}
                                for index in range(5)
                            ]
                        )
                    finally:
                        vectorstore.close()

                    self.assertEqual(fake_embeddings.batch_sizes, [2, 2, 2, 1])
                    sleep.assert_called_once_with(VectorStore.RATE_LIMIT_WAIT_SECONDS)

    def test_incomplete_index_can_resume_without_reembedding_stored_chunks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            fake_embeddings = FakeEmbeddings()
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                with patch(
                    "vectorstore.GoogleGenerativeAIEmbeddings",
                    return_value=fake_embeddings,
                ):
                    vectorstore = VectorStore(persist_dir=temp_dir)
                    vectorstore.EMBEDDING_BATCH_SIZE = 2
                    documents = [
                        {"text": f"resume {i}", "source": "notes.txt", "page": 1}
                        for i in range(3)
                    ]
                    try:
                        collection = vectorstore.client.get_or_create_collection(
                            vectorstore.collection_name
                        )
                        first_document = documents[0]
                        first_id = sha256(
                            f"{first_document['source']}\0{first_document['page']}\0"
                            f"{first_document['text']}".encode("utf-8")
                        ).hexdigest()
                        collection.upsert(
                            ids=[first_id],
                            documents=[first_document["text"]],
                            embeddings=[[1.0, 0.0, 0.0]],
                            metadatas=[
                                {
                                    "source": "notes.txt",
                                    "page": 1,
                                    "location": "Document",
                                }
                            ],
                        )

                        vectorstore.add_documents(documents)

                        self.assertEqual(fake_embeddings.batch_sizes, [2])
                        self.assertTrue(VectorStore.has_documents(persist_dir=temp_dir))
                        self.assertEqual(collection.count(), len(documents))
                    finally:
                        vectorstore.close()

    def test_rag_engine_reads_chunks_created_by_vectorstore(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
                with patch(
                    "vectorstore.GoogleGenerativeAIEmbeddings",
                    return_value=FakeEmbeddings(),
                ), patch(
                    "rag_engine.GoogleGenerativeAIEmbeddings",
                    return_value=FakeEmbeddings(),
                ), patch(
                    "rag_engine.ChatGoogleGenerativeAI",
                    return_value=FakeLLM(),
                ):
                    vectorstore = VectorStore(persist_dir=temp_dir)
                    engine = None
                    try:
                        vectorstore.add_documents(
                            [{"text": "searchable text", "source": "notes.pdf", "page": 5}]
                        )
                        engine = RAGEngine(
                            persist_dir=Path(temp_dir),
                            chroma_client=vectorstore.client,
                        )
                        chunks = engine.retrieve("searchable text")
                        self.assertEqual(chunks[0]["source"], "notes.pdf")
                        self.assertEqual(chunks[0]["page"], 5)
                    finally:
                        vectorstore.close()


if __name__ == "__main__":
    unittest.main()
