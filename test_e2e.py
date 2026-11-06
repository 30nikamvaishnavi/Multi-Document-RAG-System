"""
End-to-end test for the Multi-Document Agentic RAG system.
Generates two sample PDFs, builds the index, and runs a sample query.
"""
import os
import sys
import tempfile
from datetime import datetime

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ingestion import PDFIngestion
from vectorstore import VectorStore
from agentic_rag import AgenticRAG
from dotenv import load_dotenv

load_dotenv()


def create_sample_pdf(path: str, title: str, content: str) -> None:
    """Create a minimal valid PDF using reportlab if available, else write a text file."""
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(path, pagesize=letter)
        width, height = letter
        y = height - 72
        c.setFont("Helvetica-Bold", 14)
        c.drawString(72, y, title)
        y -= 24
        c.setFont("Helvetica", 10)
        for line in content.split("\n"):
            if y < 72:
                c.showPage()
                y = height - 72
                c.setFont("Helvetica", 10)
            c.drawString(72, y, line)
            y -= 14
        c.save()
        print(f"  Created PDF: {path}")
    except ImportError:
        # Fallback: write a .txt file (won't be picked up as PDF, but shows the flow)
        with open(path.replace(".pdf", ".txt"), "w") as f:
            f.write(f"{title}\n{content}")
        print(f"  reportlab not installed; wrote text file: {path.replace('.pdf', '.txt')}")


SAMPLE_CONTENT = {
    "ai_overview.txt": """
Artificial Intelligence Overview
Page 1
Artificial Intelligence (AI) is the simulation of human intelligence in machines.
Key subfields include Machine Learning, Natural Language Processing, and Computer Vision.
Machine Learning enables systems to learn from data without explicit programming.
Page 2
Deep Learning uses neural networks with many layers to model complex patterns.
Large Language Models (LLMs) are a class of deep learning models trained on vast text corpora.
Generative AI refers to models that can create new content such as text, images, and code.
Page 3
Agentic AI extends generative AI by giving models the ability to plan, use tools, and act autonomously.
Agents can call APIs, retrieve information from documents, and verify their own outputs.
This is the foundation of modern Retrieval-Augmented Generation (RAG) systems.
""",
    "company_policy.txt": """
Company Remote Work Policy
Page 1
This document outlines the remote work guidelines for all employees.
Employees may work remotely up to 3 days per week with manager approval.
Page 2
All remote work must be conducted using company-provided equipment and VPN.
Security protocols require encrypted connections at all times.
Page 3
Expenses for home office setup are reimbursable up to $500 per year.
Employees must attend weekly check-ins with their manager.
""",
}


def main():
    print("=" * 60)
    print("Multi-Document Agentic RAG — End-to-End Test")
    print("=" * 60)

    # 1. Create sample PDFs
    print("\n[1] Creating sample PDFs...")
    with tempfile.TemporaryDirectory() as tmpdir:
        for fname, content in SAMPLE_CONTENT.items():
            pdf_path = os.path.join(tmpdir, fname.replace(".txt", ".pdf"))
            create_sample_pdf(pdf_path, fname.replace(".txt", ""), content)

        # 2. Ingest
        print("\n[2] Ingesting PDFs...")
        ingestion = PDFIngestion()
        chunks = ingestion.load_and_chunk(tmpdir)
        print(f"  Generated {len(chunks)} chunks from {len(SAMPLE_CONTENT)} PDFs.")
        for c in chunks[:3]:
            print(f"    - {c['source']} page {c['page']}: {c['text'][:60]}...")

        # 3. Build vector store
        print("\n[3] Building Chroma index...")
        chroma_dir = os.path.join(tmpdir, "chroma")
        vs = VectorStore(persist_dir=chroma_dir)
        vs.clear()
        vs.add_documents(chunks)
        print("  Index built.")

        # 4. Test retrieval
        print("\n[4] Testing retrieval...")
        results = vs.search("What is Generative AI?", k=2)
        for r in results:
            print(f"    - {r['source']} page {r['page']} (dist={r['distance']:.3f}): {r['text'][:80]}...")

        # 5. Run agentic RAG
        print("\n[5] Running Agentic RAG query...")
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your-api-key-here":
            print("  ⚠️  GOOGLE_API_KEY not set. Skipping LLM-based test.")
            print("  Set your key in .env to run the full agentic test.")
        else:
            agent = AgenticRAG(persist_dir=chroma_dir, k=3, max_retries=2)
            result = agent.invoke("What is Agentic AI and how does it relate to RAG?")
            print(f"\n  Answer:\n  {result['answer']}")
            print(f"\n  Rewritten query: {result['rewritten_question']}")
            print(f"  Grounded: {result['grounded']}")
            print(f"  Retries: {result['retry_count']}")
            print(f"  Sources:")
            for s in result["sources"]:
                print(f"    - {s['source']} page {s['page']}")

    print("\n" + "=" * 60)
    print("Test complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()